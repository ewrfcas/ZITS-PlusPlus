import os
import time

import cv2
from tqdm import tqdm

import utils
from base import BaseTrainer
from inpainting_metric import get_inpainting_metrics
from networks.losses import *
from networks.pcp import PerceptualLoss, ResNetPL
from torch_utils.ops import conv2d_gradfix
from utils import *
from utils import stitch_images


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, G, D, g_opt, d_opt, config, train_loader, val_loader=None, sample_iterator=None, train_sampler=None,
                 g_sche=None, d_sche=None, writer=None, rank=0, ddp=False, total_rank=1):
        super().__init__(G, D, g_opt, d_opt, g_sche, d_sche, config, total_rank, writer=writer, rank=rank)
        self.config = config
        self.ddp = ddp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sample_iterator = sample_iterator
        self.train_sampler = train_sampler
        self.log_step = config['trainer']['logging_every']
        self.resnet_pl = config['g_args'].get('resnet_pl', False)
        if self.resnet_pl:
            self.perceptual = ResNetPL(weight=config['pcp_ratio'], weights_path=config['g_args']['resnet_path']).to(self.device)
        else:
            self.perceptual = PerceptualLoss(layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)).to(self.device)
        self.g_sche = g_sche
        self.d_sche = d_sche
        self.total_rank = total_rank
        self.ema_rampup = None
        self.l1_config = config['g_args'].get('l1_args', None)
        if config['fp16'] is True:
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False

        self.autocast = torch.cuda.amp.autocast if self.fp16 else identity_with

    def run_G(self, img_in, mask_in, z, c):
        ws = self.G.mapping(z, c, truncation_psi=self.config['truncation_psi'])
        if self.config['style_mixing_prob'] > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.config['style_mixing_prob'], cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, truncation_psi=self.config['truncation_psi'], skip_w_avg_update=True)[:, cutoff:]
        combined_out, out = self.G.synthesis(img_in, mask_in, ws)
        return combined_out, out

    def postprocess(self, img, range=[-1, 1]):
        img = (img - range[0]) / (range[1] - range[0])
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def train(self):

        epoch = 0
        keep_training = True
        scalar_outputs = dict()

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
                gen_z = torch.randn([items['img'].shape[0], self.G.z_dim], device=self.device)

                # Gmain: Maximize logits for generated images.
                self.g_opt.zero_grad(set_to_none=True)
                self.G.requires_grad_(True)

                with self.autocast():
                    gen_img, gen_img_no_combined = self.run_G(items['img'], items['mask'], z=gen_z, c=None)
                    gen_logits = self.D.forward(gen_img, items['mask'])
                loss_Gmain = F.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                # perceptual loss
                if self.resnet_pl:
                    pcp_loss = self.perceptual(gen_img, items['img'])
                elif self.config['pcp_ratio'] > 0:
                    pcp_loss, _ = self.perceptual(gen_img, items['img'])
                    pcp_loss *= self.config['pcp_ratio']
                else:
                    pcp_loss = torch.tensor(0.0).to(self.device)
                # L1 loss
                if self.l1_config is not None and self.l1_config['use_l1']:
                    per_pixel_l1 = F.l1_loss(gen_img_no_combined, items['img'], reduction='none')
                    l1_mask = items['mask'] * self.l1_config['weight_missing'] + (1 - items['mask']) * self.l1_config['weight_known']
                    l1_loss = (per_pixel_l1 * l1_mask).mean()
                else:
                    l1_loss = 0
                loss_Gmain_all = loss_Gmain + pcp_loss + l1_loss
                if self.fp16:
                    self.scaler.scale(loss_Gmain_all.mean()).backward()
                else:
                    loss_Gmain_all.mean().backward()

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

                # Dmain: Minimize logits for generated images.
                self.d_opt.zero_grad(set_to_none=True)
                self.D.requires_grad_(True)
                do_GP = True if batch_idx % self.config['D_reg_interval'] == 0 else False
                with self.autocast():
                    gen_logits = self.D.forward(gen_img.detach(), items['mask'])
                loss_Dgen = F.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
                loss_Dgen_all = loss_Dgen
                if self.fp16:
                    self.scaler.scale(loss_Dgen_all.mean()).backward()
                else:
                    loss_Dgen_all.mean().backward()

                real_img_tmp = items['img'].detach().requires_grad_(do_GP)
                with self.autocast():
                    real_logits = self.D.forward(real_img_tmp, items['mask'])
                loss_Dreal = F.softplus(-real_logits)  # -log(sigmoid(real_logits))

                # D_GP
                if do_GP:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    r1_penalty = r1_penalty * (self.config['r1_gamma'] / 2)

                with torch.autograd.profiler.record_function('Dreal_backward'):
                    d_loss = loss_Dreal.mean()
                    if do_GP:
                        d_loss += r1_penalty.mean().mul(self.config['D_reg_interval'])

                if self.fp16:
                    self.scaler.scale(d_loss).backward()
                else:
                    d_loss.backward()

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

                # Update G_ema.
                with torch.no_grad():
                    ema_nimg = self.config['ema_nimg'] * self.config['batch_size'] / 32
                    ema_beta = 0.5 ** (self.config['batch_size'] / max(ema_nimg, 1e-8))
                    for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                    for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                        b_ema.copy_(b)

                self.global_step += 1

                if self.global_step % self.log_step == 0 and self.rank == 0:
                    scalar_outputs['g_fake'] = loss_Gmain.mean().item()
                    scalar_outputs['pcp'] = pcp_loss.mean().item()
                    scalar_outputs['d_fake'] = loss_Dgen.mean().item()
                    scalar_outputs['d_real'] = loss_Dreal.mean().item()
                    scalar_outputs['gp'] = r1_penalty.mean().item()
                    if self.l1_config is not None and self.l1_config['use_l1']:
                        scalar_outputs['l1'] = l1_loss.mean().item()

                    save_scalars(self.writer, 'train', scalar_outputs, self.global_step)
                    print_str = "Epoch {}, Iter {}/{}k, lr={:.1e}, ".format(epoch, self.global_step, int(self.total_step / 1000),
                                                                            self.g_opt.param_groups[0]["lr"])
                    for k in scalar_outputs:
                        if k == 'gp':
                            print_str += "{}={:.1e}, ".format(k, scalar_outputs[k])
                        else:
                            print_str += "{}={:.3f}, ".format(k, scalar_outputs[k])
                    print_str += "time={:.2f}".format(time.time() - start_time)
                    if self.fp16:
                        print_str += ', scale={:d}'.format(int(self.scaler.get_scale()))
                    print(print_str)

                if self.global_step % self.sample_period == 0 and self.rank == 0:
                    self._sample()

                if self.global_step % self.eval_period == 0 and self.rank == 0:
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

            if self.config['g_args'].get('sample_with_center_mask', False):
                items['mask'] = torch.zeros_like(items['mask'])
                h, w = items['mask'].shape[2], items['mask'].shape[3]
                r = int(np.sqrt(h * w * 0.4))
                items['mask'][:, :, int((h - r) / 2):int((h - r) / 2) + r, int((w - r) / 2):int((w - r) / 2) + r] = 1

            total_gens = []
            gen_img = None
            for _ in range(3):  # sample for different results
                gen_z = torch.randn([items['img'].shape[0], self.G.z_dim], device=self.device)
                if gen_img is None:
                    gen_img, _ = self.G(items['img'], items['mask'], gen_z, c=None, noise_mode='const')
                    gen_img = torch.clamp(gen_img, -1, 1)
                gen_ema_img, _ = self.G_ema(items['img'], items['mask'], gen_z, c=None, noise_mode='const')
                gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
                total_gens.append(gen_ema_img)

            if self.config['sample_size'] <= 6:
                image_per_row = 1
            else:
                image_per_row = 2
            images = stitch_images(
                self.postprocess((items['img']).cpu()),
                self.postprocess((items['img'] * (1 - items['mask'])).cpu()),
                self.postprocess((gen_img).cpu()),
                self.postprocess((total_gens[0]).cpu()),
                self.postprocess((total_gens[1]).cpu()),
                self.postprocess((total_gens[2]).cpu()),
                img_per_row=image_per_row
            )

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

                gen_z = torch.randn([items['img'].shape[0], self.G.z_dim], device=self.device)
                gen_ema_img, _ = self.G_ema(items['img'], items['mask'], gen_z, c=None, noise_mode='const')
                gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
                gen_ema_img = (gen_ema_img + 1) / 2 * 255.0
                gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
                for img_num in range(items['img'].shape[0]):
                    cv2.imwrite(self.eval_path + '/' + items['name'][img_num], gen_ema_img[img_num, :, :, ::-1])

            self.metric = get_inpainting_metrics(self.eval_path, self.config['test_path'])

            print(f'Steps:{self.global_step}')
            for m in self.metric:
                if m in ['PSNR', 'SSIM', 'FID', 'LPIPS']:
                    print(m, self.metric[m])

            save_scalars(self.writer, 'val', self.metric, self.global_step)

            if 'FID' not in self.best_metric or self.metric['FID'] <= self.best_metric['FID']:
                self.best_metric = self.metric
                self._save_checkpoint(postfix='best')

            self._save_checkpoint(postfix='last')
