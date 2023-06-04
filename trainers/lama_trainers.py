import os
import time

import cv2
from tqdm import tqdm

import utils
from base import BaseTrainer
from inpainting_metric import get_inpainting_metrics
from networks.losses import *
from networks.pcp import PerceptualLoss, ResNetPL
from utils import *
from utils import stitch_images


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class LaMaTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, G, D, g_opt, d_opt, config, train_loader, val_loader=None, sample_iterator=None, train_sampler=None,
                 g_sche=None, d_sche=None, writer=None, rank=0, ddp=False, total_rank=1):
        super().__init__(G, D, g_opt, d_opt, g_sche, d_sche, config, total_rank, writer=writer, rank=rank)
        self.config = config['trainer']
        self.test_path = config['test_path']
        self.ddp = ddp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sample_iterator = sample_iterator
        self.train_sampler = train_sampler
        self.log_step = self.config['logging_every']
        self.D_reg_interval = self.config.get('D_reg_interval', 1)

        # adv loss
        self.adv_args = self.config['adversarial']

        # loss
        if self.config.get("perceptual", {"weight": 0})['weight'] > 0:
            self.loss_pl = PerceptualLoss(layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)).to(self.device)
        else:
            self.loss_pl = None
        if self.config.get("resnet_pl", {"weight": 0})['weight'] > 0:
            self.loss_resnet_pl = ResNetPL(**self.config['resnet_pl']).to(self.device)
        else:
            self.loss_resnet_pl = None

        self.total_rank = total_rank
        if config['fp16'] is True:
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False

        self.autocast = torch.cuda.amp.autocast if self.fp16 else identity_with

    def run_G(self, img, mask):
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        out = self.G(masked_img)
        combined_out = img * (1 - mask) + out * mask

        return combined_out, out

    def run_G_ema(self, img, mask):
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        out = self.G_ema(masked_img)
        combined_out = img * (1 - mask) + out * mask

        return combined_out, out

    def postprocess(self, img, range=[-1, 1]):
        img = (img - range[0]) / (range[1] - range[0])
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def train_G(self, items):
        self.g_opt.zero_grad(set_to_none=True)
        self.G.requires_grad_(True)

        loss_dict = dict()
        loss_G_all = torch.tensor(0.0, device=self.device)
        with self.autocast():
            _, gen_img = self.run_G(items['img'], items['mask'])
            gen_logits, gen_feats = self.D.forward(gen_img)
        adv_gen_loss = generator_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_G_all += adv_gen_loss
        loss_dict['g_fake'] = adv_gen_loss

        # perceptual loss
        if self.loss_resnet_pl is not None:
            pcp_loss = self.loss_resnet_pl(gen_img, items['img'])
            loss_G_all += pcp_loss
            loss_dict['pcp'] = pcp_loss

        # feature matching
        if self.config['feature_matching']['weight'] > 0:
            _, real_feats = self.D(items['img'])
            fm_loss = feature_matching_loss(gen_feats, real_feats, mask=None) * self.config['feature_matching']['weight']
            loss_G_all += fm_loss
            loss_dict['fm'] = fm_loss

        # L1 loss
        if self.config['l1']['use_l1']:
            per_pixel_l1 = F.l1_loss(gen_img, items['img'], reduction='none')
            l1_mask = items['mask'] * self.config['l1']['weight_missing'] + (1 - items['mask']) * self.config['l1']['weight_known']
            l1_loss = (per_pixel_l1 * l1_mask).mean()
            loss_G_all += l1_loss
            loss_dict['l1'] = l1_loss

        if self.fp16:
            self.scaler.scale(loss_G_all).backward()
        else:
            loss_G_all.mean().backward()

        self.G.requires_grad_(False)
        for param in self.G.parameters():
            if param.grad is not None:
                utils.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        if self.fp16:
            self.scaler.step(self.g_opt)
            self.scaler.update()
        else:
            self.g_opt.step()

        if self.g_sche is not None:
            self.g_sche.step()

        return loss_dict, gen_img

    def train_D(self, items, gen_img, do_GP=True):
        self.d_opt.zero_grad(set_to_none=True)
        self.D.requires_grad_(True)
        real_img_tmp = items['img'].detach().requires_grad_(do_GP)
        with self.autocast():
            real_logits, _ = self.D.forward(real_img_tmp)
            gen_logits, _ = self.D.forward(gen_img.detach())
        dis_real_loss, grad_penalty = discriminator_real_loss(real_batch=real_img_tmp, discr_real_pred=real_logits,
                                                              gp_coef=self.adv_args['gp_coef'], do_GP=do_GP)
        dis_fake_loss = discriminator_fake_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_dict = {'d_real': dis_real_loss, 'd_fake': dis_fake_loss}
        if do_GP:
            grad_penalty = grad_penalty * self.D_reg_interval
            loss_dict['gp'] = grad_penalty
        loss_D_all = dis_real_loss + dis_fake_loss + grad_penalty

        if self.fp16:
            self.scaler.scale(loss_D_all.mean()).backward()
        else:
            loss_D_all.mean().backward()

        self.D.requires_grad_(False)
        for param in self.D.parameters():
            if param.grad is not None:
                utils.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        if self.fp16:
            self.scaler.step(self.d_opt)
            self.scaler.update()
        else:
            self.d_opt.step()

        if self.d_sche is not None:
            self.d_sche.step()

        return loss_dict

    def train(self):

        epoch = 0
        keep_training = True
        scalar_outputs = dict()
        loss_dict = dict()

        while keep_training:
            epoch += 1
            if self.ddp:
                self.train_sampler.set_epoch(epoch)  # Shuffle each epoch

            # training
            for batch_idx, items in enumerate(self.train_loader):
                self.G.train()
                self.D.train()
                start_time = time.time()

                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                g_loss_dict, gen_img = self.train_G(items)
                loss_dict.update(g_loss_dict)

                do_GP = True if batch_idx % self.D_reg_interval == 0 else False
                d_loss_dict = self.train_D(items, gen_img, do_GP)
                loss_dict.update(d_loss_dict)

                # Update G_ema.
                with torch.no_grad():
                    for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                        p_ema.copy_(p.lerp(p_ema, self.config['ema_beta']))
                    for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                        b_ema.copy_(b)

                self.global_step += 1

                if self.global_step % self.log_step == 0 and self.rank == 0:
                    for k in loss_dict:
                        scalar_outputs[k] = loss_dict[k].item()
                    save_scalars(self.writer, 'train', scalar_outputs, self.global_step)
                    print_str = "Epoch {}, Iter {}/{}k, g_lr={:.1e}, d_lr={:.1e}, ".format(epoch, self.global_step, int(self.total_step / 1000),
                                                                                           self.g_opt.param_groups[0]["lr"], self.d_opt.param_groups[0]["lr"])
                    for k in scalar_outputs:
                        if k == 'gp':
                            print_str += "{}={:.1e}, ".format(k, scalar_outputs[k])
                        else:
                            print_str += "{}={:.3f}, ".format(k, scalar_outputs[k])
                    print_str += "time={:.2f}".format(time.time() - start_time)
                    if self.fp16:
                        print_str += ', scale={:d}'.format(int(self.scaler.get_scale()))
                    print(print_str)

                if self.global_step % self.sample_period == 0:
                    self._sample()

                if self.global_step % self.eval_period == 0:
                    self._eval()

                # STOP training
                if self.global_step >= self.total_step:
                    keep_training = False
                    break

        print('End Training Rank:', self.rank)

    def _sample(self):
        if len(self.val_loader) == 0:
            return

        self.G.eval()
        self.G_ema.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)

            if self.config.get('sample_with_center_mask', False):
                items['mask'] = torch.zeros_like(items['mask'])
                h, w = items['mask'].shape[2], items['mask'].shape[3]
                r = int(np.sqrt(h * w * 0.4))
                items['mask'][:, :, int((h - r) / 2):int((h - r) / 2) + r, int((w - r) / 2):int((w - r) / 2) + r] = 1

            combined_gen_img, _ = self.run_G(items['img'], items['mask'])
            combined_gen_img = torch.clamp(combined_gen_img, -1, 1)
            combined_gen_ema_img, _ = self.run_G_ema(items['img'], items['mask'])
            combined_gen_ema_img = torch.clamp(combined_gen_ema_img, -1, 1)

            image_per_row = 2
            images = stitch_images(
                self.postprocess((items['img']).cpu()),
                self.postprocess((items['img'] * (1 - items['mask'])).cpu()),
                self.postprocess((combined_gen_img).cpu()),
                self.postprocess((combined_gen_ema_img).cpu()),
                img_per_row=image_per_row
            )

            if self.rank == 0:
                name = os.path.join(self.sample_path, str(self.global_step).zfill(6) + ".jpg")
                print('saving sample ' + name)
                images.save(name)

    def _eval(self):
        self.G_ema.eval()

        with torch.no_grad():
            for items in tqdm(self.val_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                gen_ema_img, _ = self.run_G_ema(items['img'], items['mask'])
                gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
                gen_ema_img = (gen_ema_img + 1) / 2 * 255.0
                gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
                if self.rank == 0:
                    for img_num in range(items['img'].shape[0]):
                        cv2.imwrite(self.eval_path + '/' + items['name'][img_num], gen_ema_img[img_num, :, :, ::-1])

            if self.rank == 0:
                self.metric = get_inpainting_metrics(self.eval_path, self.test_path)

                print(f'Steps:{self.global_step}')
                for m in self.metric:
                    if m in ['PSNR', 'SSIM', 'FID', 'LPIPS']:
                        print(m, self.metric[m])

                save_scalars(self.writer, 'val', self.metric, self.global_step)

                if 'FID' not in self.best_metric or self.metric['FID'] <= self.best_metric['FID']:
                    self.best_metric = self.metric
                    self._save_checkpoint(postfix='best')

                self._save_checkpoint(postfix='last')


class OldLaMaTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, G, D, g_opt, d_opt, config, train_loader, val_loader=None, sample_iterator=None,
                 train_sampler=None,
                 g_sche=None, d_sche=None, writer=None, rank=0, ddp=False, total_rank=1):
        super().__init__(G, D, g_opt, d_opt, g_sche, d_sche, config, total_rank, writer=writer, rank=rank)
        self.config = config['trainer']
        self.test_path = config['test_path']
        self.ddp = ddp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sample_iterator = sample_iterator
        self.train_sampler = train_sampler
        self.log_step = self.config['logging_every']
        self.D_reg_interval = self.config.get('D_reg_interval', 1)
        self.G.requires_grad_(True)
        self.D.requires_grad_(True)

        # adv loss
        self.adv_args = self.config['adversarial']

        # loss
        if self.config.get("perceptual", {"weight": 0})['weight'] > 0:
            self.loss_pl = PerceptualLoss(layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)).to(self.device)
        else:
            self.loss_pl = None
        if self.config.get("resnet_pl", {"weight": 0})['weight'] > 0:
            self.loss_resnet_pl = ResNetPL(**self.config['resnet_pl']).to(self.device)
        else:
            self.loss_resnet_pl = None

        self.total_rank = total_rank
        if config['fp16'] is True:
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False

        self.autocast = torch.cuda.amp.autocast if self.fp16 else identity_with

    def run_G(self, img, mask):
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        out = self.G(masked_img)
        combined_out = img * (1 - mask) + out * mask

        return combined_out, out

    def run_G_ema(self, img, mask):
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        out = self.G_ema(masked_img)
        combined_out = img * (1 - mask) + out * mask

        return combined_out, out

    def postprocess(self, img, range=[-1, 1]):
        img = (img - range[0]) / (range[1] - range[0])
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generator_loss(self, batch):
        self.g_opt.zero_grad()

        gen_img = batch['predicted_image']

        loss_dict = dict()
        loss_G_all = torch.tensor(0.0, device=self.device)
        with self.autocast():
            gen_logits, gen_feats = self.D.forward(gen_img)
        adv_gen_loss = generator_loss(discr_fake_pred=gen_logits, mask=batch['mask'], args=self.adv_args)
        loss_G_all += adv_gen_loss
        loss_dict['g_fake'] = adv_gen_loss

        # perceptual loss
        if self.loss_resnet_pl is not None:
            pcp_loss = self.loss_resnet_pl(gen_img, batch['img'])
            loss_G_all += pcp_loss
            loss_dict['pcp'] = pcp_loss

        # feature matching
        if self.config['feature_matching']['weight'] > 0:
            _, real_feats = self.D(batch['img'])
            fm_loss = feature_matching_loss(gen_feats, real_feats, mask=None) * self.config['feature_matching'][
                'weight']
            loss_G_all += fm_loss
            loss_dict['fm'] = fm_loss

        # L1 loss
        if self.config['l1']['use_l1']:
            per_pixel_l1 = F.l1_loss(gen_img, batch['img'], reduction='none')
            l1_mask = batch['mask'] * self.config['l1']['weight_missing'] + (1 - batch['mask']) * self.config['l1'][
                'weight_known']
            l1_loss = (per_pixel_l1 * l1_mask).mean()
            loss_G_all += l1_loss
            loss_dict['l1'] = l1_loss

        if self.fp16:
            self.scaler.scale(loss_G_all).backward()
        else:
            loss_G_all.mean().backward()

        if self.fp16:
            self.scaler.step(self.g_opt)
            self.scaler.update()
        else:
            self.g_opt.step()

        if self.g_sche is not None:
            self.g_sche.step()

        return loss_dict


    def discriminator_loss(self, batch):
        self.d_opt.zero_grad()

        batch['img'].requires_grad = True
        with self.autocast():
            real_logits, _ = self.D.forward(batch['img'])

        dis_real_loss, grad_penalty = discriminator_real_loss(real_batch=batch['img'], discr_real_pred=real_logits,
                                                              gp_coef=self.adv_args['gp_coef'], do_GP=True)
        with self.autocast():
            _, gen_img = self.run_G(batch['img'], batch['mask'])
            batch['predicted_image'] = gen_img
        predicted_img = batch['predicted_image'].detach()
        gen_logits, _ = self.D.forward(predicted_img)
        dis_fake_loss = discriminator_fake_loss(discr_fake_pred=gen_logits, mask=batch['mask'], args=self.adv_args)
        loss_dict = {'d_real': dis_real_loss, 'd_fake': dis_fake_loss}
        grad_penalty = grad_penalty * self.D_reg_interval
        loss_dict['gp'] = grad_penalty
        loss_D_all = dis_real_loss + dis_fake_loss + grad_penalty

        if self.fp16:
            self.scaler.scale(loss_D_all.mean()).backward()
        else:
            loss_D_all.mean().backward()

        if self.fp16:
            self.scaler.step(self.d_opt)
            self.scaler.update()
        else:
            self.d_opt.step()

        if self.d_sche is not None:
            self.d_sche.step()

        return loss_dict, batch

    def train(self):

        epoch = 0
        keep_training = True
        scalar_outputs = dict()
        loss_dict = dict()

        while keep_training:
            epoch += 1
            if self.ddp:
                self.train_sampler.set_epoch(epoch)  # Shuffle each epoch

            # training
            for batch_idx, items in enumerate(self.train_loader):
                self.G.train()
                self.D.train()
                start_time = time.time()

                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                d_loss_dict, batch = self.discriminator_loss(items)
                loss_dict.update(d_loss_dict)

                g_loss_dict = self.generator_loss(batch)
                loss_dict.update(g_loss_dict)

                # Update G_ema.
                with torch.no_grad():
                    for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                        p_ema.copy_(p.lerp(p_ema, self.config['ema_beta']))
                    for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                        b_ema.copy_(b)

                self.global_step += 1

                if self.global_step % self.log_step == 0 and self.rank == 0:
                    for k in loss_dict:
                        scalar_outputs[k] = loss_dict[k].item()
                    save_scalars(self.writer, 'train', scalar_outputs, self.global_step)
                    print_str = "Epoch {}, Iter {}/{}k, g_lr={:.1e}, d_lr={:.1e}, ".format(epoch, self.global_step,
                                                                                           int(self.total_step / 1000),
                                                                                           self.g_opt.param_groups[0][
                                                                                               "lr"],
                                                                                           self.d_opt.param_groups[0][
                                                                                               "lr"])
                    for k in scalar_outputs:
                        if k == 'gp':
                            print_str += "{}={:.1e}, ".format(k, scalar_outputs[k])
                        else:
                            print_str += "{}={:.3f}, ".format(k, scalar_outputs[k])
                    print_str += "time={:.2f}".format(time.time() - start_time)
                    if self.fp16:
                        print_str += ', scale={:d}'.format(int(self.scaler.get_scale()))
                    print(print_str)

                if self.global_step % self.sample_period == 0:
                    self._sample()

                if self.global_step % self.eval_period == 0:
                    self._eval()

                # STOP training
                if self.global_step >= self.total_step:
                    keep_training = False
                    break

        print('End Training Rank:', self.rank)

    def _sample(self):
        if len(self.val_loader) == 0:
            return

        self.G.eval()
        self.G_ema.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)

            if self.config.get('sample_with_center_mask', False):
                items['mask'] = torch.zeros_like(items['mask'])
                h, w = items['mask'].shape[2], items['mask'].shape[3]
                r = int(np.sqrt(h * w * 0.4))
                items['mask'][:, :, int((h - r) / 2):int((h - r) / 2) + r, int((w - r) / 2):int((w - r) / 2) + r] = 1

            combined_gen_img, _ = self.run_G(items['img'], items['mask'])
            combined_gen_img = torch.clamp(combined_gen_img, -1, 1)
            combined_gen_ema_img, _ = self.run_G_ema(items['img'], items['mask'])
            combined_gen_ema_img = torch.clamp(combined_gen_ema_img, -1, 1)

            image_per_row = 2
            images = stitch_images(
                self.postprocess((items['img']).cpu()),
                self.postprocess((items['img'] * (1 - items['mask'])).cpu()),
                self.postprocess((combined_gen_img).cpu()),
                self.postprocess((combined_gen_ema_img).cpu()),
                img_per_row=image_per_row
            )

            if self.rank == 0:
                name = os.path.join(self.sample_path, str(self.global_step).zfill(6) + ".jpg")
                print('saving sample ' + name)
                images.save(name)

    def _eval(self):
        self.G_ema.eval()

        with torch.no_grad():
            for items in tqdm(self.val_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                gen_ema_img, _ = self.run_G_ema(items['img'], items['mask'])
                gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
                gen_ema_img = (gen_ema_img + 1) / 2 * 255.0
                gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
                if self.rank == 0:
                    for img_num in range(items['img'].shape[0]):
                        cv2.imwrite(self.eval_path + '/' + items['name'][img_num], gen_ema_img[img_num, :, :, ::-1])

            if self.rank == 0:
                self.metric = get_inpainting_metrics(self.eval_path, self.test_path)

                print(f'Steps:{self.global_step}')
                for m in self.metric:
                    if m in ['PSNR', 'SSIM', 'FID', 'LPIPS']:
                        print(m, self.metric[m])

                save_scalars(self.writer, 'val', self.metric, self.global_step)

                if 'FID' not in self.best_metric or self.metric['FID'] <= self.best_metric['FID']:
                    self.best_metric = self.metric
                    self._save_checkpoint(postfix='best')

                self._save_checkpoint(postfix='last')
