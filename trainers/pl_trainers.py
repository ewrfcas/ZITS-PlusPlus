import copy
import glob
import os
import pickle
import random
import time

import cv2
import numpy as np
import pytorch_lightning as ptl
import skimage
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dnnlib.util import get_obj_by_name
from inpainting_metric import get_inpainting_metrics
from networks.losses import *
from networks.pcp import PerceptualLoss, ResNetPL
from utils import get_lr_milestone_decay_with_warmup
from utils import stitch_images
from .nms_temp import get_nms as get_np_nms
from .nms_torch import get_nms as get_torch_nms


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


def to_int(x):
    return tuple(map(int, x))


def resize(img, height, width, center_crop=False):
    imgh, imgw = img.shape[0:2]

    if center_crop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    if imgh > height and imgw > width:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_LINEAR
    img = cv2.resize(img, (width, height), interpolation=inter)

    return img


def load_wireframe(selected_img_name, h, w):
    line_name = selected_img_name.split("/")
    line_name[-2] = 'wireframes/' + line_name[-2]
    line_name = "/".join(line_name).replace('.png', '.pkl').replace('.jpg', '.pkl')

    wf = pickle.load(open(line_name, 'rb'))
    lmap = np.zeros((h, w))
    for i in range(len(wf['scores'])):
        if wf['scores'][i] > 0.85:
            line = wf['lines'][i].copy()
            line[0] = line[0] * h
            line[1] = line[1] * w
            line[2] = line[2] * h
            line[3] = line[3] * w
            rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
    return lmap


ones_filter = np.ones((3, 3), dtype=np.float32)
d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)


def load_masked_position_encoding(mask):
    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = mask.copy()
    mask3 = 1. - (mask3 / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0
    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (256 / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * 128).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, 128 - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct


def to_tensor(img, norm=False):
    # img = Image.fromarray(img)
    img_t = TF.to_tensor(img).float()
    if norm:
        img_t = TF.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img_t


class PLTrainer(ptl.LightningModule):
    """
    Trainer class
    """

    def __init__(self, G, D, config, out_path, num_gpus=1, **kwargs):
        super().__init__()
        self.G = G
        self.G_ema = copy.deepcopy(self.G)
        self.D = D
        self.args = config
        self.data_class = config['data_class']
        self.config = config['trainer']
        self.num_gpus = num_gpus
        self.log_step = self.config['logging_every']
        self.D_reg_interval = self.config.get('D_reg_interval', 1)

        self.total_step = config['trainer']['total_step']
        self.sample_period = config['trainer']['sample_period']

        self.sample_path = os.path.join(out_path, 'samples')
        self.eval_path = os.path.join(out_path, 'validation')
        self.model_path = os.path.join(out_path, 'models')
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.eval_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        self.test_path = config['test_path']

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

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

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
        loss_dict = dict()
        loss_G_all = torch.tensor(0.0, device=items['img'].device)
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

        return loss_G_all, loss_dict

    def train_D(self, items, do_GP=True):
        real_img_tmp = items['img'].requires_grad_(do_GP)
        real_logits, _ = self.D.forward(real_img_tmp)
        _, gen_img = self.run_G(items['img'], items['mask'])
        gen_logits, _ = self.D.forward(gen_img.detach())
        dis_real_loss, grad_penalty = discriminator_real_loss(real_batch=real_img_tmp, discr_real_pred=real_logits,
                                                              gp_coef=self.adv_args['gp_coef'], do_GP=do_GP)
        dis_fake_loss = discriminator_fake_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_dict = {'d_real': dis_real_loss, 'd_fake': dis_fake_loss}
        if do_GP:
            grad_penalty = grad_penalty * self.D_reg_interval
            loss_dict['gp'] = grad_penalty
        loss_D_all = dis_real_loss + dis_fake_loss + grad_penalty

        return loss_D_all, loss_dict

    def train_dataloader(self):
        self.train_dataset = get_obj_by_name(self.data_class)(self.args['train_flist'], mask_path=self.args['train_mask_flist'],
                                                              batch_size=self.args['batch_size'] // self.num_gpus, augment=True,
                                                              training=True, test_mask_path=None, config=self.args['dataset'])
        return DataLoader(self.train_dataset, pin_memory=True, batch_size=self.args['batch_size'] // self.num_gpus, shuffle=True, num_workers=8)

    def val_dataloader(self):
        self.val_dataset = get_obj_by_name(self.data_class)(self.args['val_flist'], mask_path=None,
                                                            batch_size=self.args['batch_size'] // self.num_gpus, augment=False, training=False,
                                                            test_mask_path=self.args['test_mask_flist'], input_size=self.args['eval_size'])
        return DataLoader(self.val_dataset, pin_memory=True, batch_size=self.args['batch_size'] // self.num_gpus, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        opt_args = self.args['optimizer']
        g_optimizer = torch.optim.Adam(self.G.parameters(), lr=opt_args['g_opt']['lr'],
                                       betas=(opt_args['g_opt']['beta1'], opt_args['g_opt']['beta2']), eps=1e-8)
        d_optimizer = torch.optim.Adam(self.D.parameters(), lr=opt_args['d_opt']['lr'],
                                       betas=(opt_args['d_opt']['beta1'], opt_args['d_opt']['beta2']), eps=1e-8)
        g_sche = get_lr_milestone_decay_with_warmup(g_optimizer, num_warmup_steps=opt_args['warmup_steps'],
                                                    milestone_steps=opt_args['decay_steps'], gamma=opt_args['decay_rate'])
        g_sche = {'scheduler': g_sche, 'interval': 'step'}  # called after each training step
        d_sche = get_lr_milestone_decay_with_warmup(d_optimizer, num_warmup_steps=opt_args['warmup_steps'],
                                                    milestone_steps=opt_args['decay_steps'],
                                                    gamma=opt_args['decay_rate'])
        d_sche = {'scheduler': d_sche, 'interval': 'step'}  # called after each training step
        return [g_optimizer, d_optimizer], [g_sche, d_sche]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # self.print('Batch_idx', batch_idx, 'Opt_idx', optimizer_idx)
        if optimizer_idx == 0:  # step for generator
            self.G.requires_grad_(True)
            self.D.requires_grad_(False)
            loss, loss_dict = self.train_G(batch)
        elif optimizer_idx == 1:  # step for discriminator
            self.G.requires_grad_(False)
            self.D.requires_grad_(True)
            loss, loss_dict = self.train_D(batch)
        else:
            raise NotImplementedError

        if self.global_step % self.sample_period == 0 and optimizer_idx == 0:
            self._sample(batch)

        return dict(loss=loss, log_info=add_prefix_to_keys(loss_dict, 'train/'))

    def training_step_end(self, batch_parts_outputs):
        # Update G_ema.
        with torch.no_grad():
            for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                p_ema.copy_(p.lerp(p_ema, self.config['ema_beta']))
            for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                b_ema.copy_(b)

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)

        return full_loss

    def validation_step(self, batch, batch_idx):
        self.G_ema.eval()
        gen_ema_img, _ = self.run_G_ema(batch['img'], batch['mask'])
        gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
        gen_ema_img = (gen_ema_img + 1) / 2 * 255.0
        gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
        for img_num in range(batch['img'].shape[0]):
            cv2.imwrite(self.eval_path + '/' + batch['name'][img_num], gen_ema_img[img_num, :, :, ::-1])

    def validation_epoch_end(self, outputs):
        if self.trainer.is_global_zero:
            time.sleep(1)
            while True:  # 等待其他进程完成保存
                input_paths = sorted(glob.glob(self.eval_path + '/*'), key=lambda x: x.split('/')[-1])
                output_paths = sorted(glob.glob(self.test_path + '/*'), key=lambda x: x.split('/')[-1])
                self.print('Waiting all validated image saving over...')
                time.sleep(0.5)
                if len(input_paths) == len(output_paths):
                    break
        if self.trainer.is_global_zero:
            self.metric = get_inpainting_metrics(self.eval_path, self.test_path, get_fid=True if self.trainer.is_global_zero else False)

            self.print(f'Steps:{self.global_step}')
            for m in self.metric:
                if m in ['PSNR', 'SSIM', 'FID', 'LPIPS']:
                    if m in self.metric:
                        self.print(m, self.metric[m])
                self.metric[m] *= self.num_gpus  # 其他进程结果均是0，这里*ngpu来平衡metric

        else:
            self.metric = {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0, 'FID': 0}

        # 非rank0进程等待rank0验证完后同步结果，否则存储后训练会死锁（无语）
        self.log('val/PSNR', self.metric['PSNR'], sync_dist=True)
        self.log('val/SSIM', self.metric['SSIM'], sync_dist=True)
        self.log('val/LPIPS', self.metric['LPIPS'], sync_dist=True)
        self.log('val/FID', self.metric['FID'], sync_dist=True)

    def _sample(self, items):

        self.G.eval()
        self.G_ema.eval()
        with torch.no_grad():
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

            if self.get_ddp_rank() in (None, 0):
                name = os.path.join(self.sample_path, str(self.global_step).zfill(6) + ".jpg")
                self.print('saving sample ' + name)
                images.save(name)

        self.G.train()

    def get_ddp_rank(self):
        return self.trainer.global_rank if (self.trainer.num_nodes * self.trainer.num_processes) > 1 else None


class FinetunePLTrainer(ptl.LightningModule):
    """
    Trainer class
    """

    def __init__(self, structure_upsample, edgeline_tsr, grad_tsr, ftr, D, config, out_path, num_gpus=1, use_ema=False,
                 dynamic_size=False, test_only=False):
        super().__init__()
        self.structure_upsample = structure_upsample
        self.structure_upsample.requires_grad_(False).eval()
        self.edgeline_tsr = edgeline_tsr
        self.grad_tsr = grad_tsr
        self.edgeline_tsr.requires_grad_(False).eval()
        self.grad_tsr.requires_grad_(False).eval()
        self.ftr = ftr
        self.ftr_ema = None
        self.D = D
        self.num_gpus = num_gpus
        self.args = config
        self.use_ema = use_ema
        self.dynamic_size = dynamic_size
        self.config = config['trainer']
        self.data_class = config['data_class']
        self.data_args = config['dataset']
        self.data_args['pos_num'] = config['g_args']['rel_pos_num']
        self.D_reg_interval = self.config.get('D_reg_interval', 1)
        self.sample_period = config['trainer']['sample_period']
        self.use_grad = config['g_args'].get('use_gradient', True)
        print("load HAWP")
        from .lsm_hawp.detector import WireframeDetector
        self.wf = WireframeDetector(is_cuda=True)
        self.wf = self.wf
        self.wf.eval()

        self.sample_path = os.path.join(out_path, 'samples')
        self.eval_path = os.path.join(out_path, 'validation')
        self.model_path = os.path.join(out_path, 'models')
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.eval_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        self.test_path = config['test_path']
        self.test_only = test_only

        # adv loss
        self.adv_args = self.config['adversarial']

        # loss
        if not test_only and self.config.get("resnet_pl", {"weight": 0})['weight'] > 0:
            self.loss_resnet_pl = ResNetPL(**self.config['resnet_pl']).to(self.device)
        else:
            self.loss_resnet_pl = None

        self.g_opt_state = None
        self.d_opt_state = None

        # global buffers for Training
        self.img_size = 256
        self.gen_img_for_train = None

    def reset_ema(self):
        self.ftr_ema = copy.deepcopy(self.ftr)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        if hasattr(self, 'g_lr'):
            items['g_lr'] = self.g_lr
        if hasattr(self, 'd_lr'):
            items['d_lr'] = self.d_lr
        items['size'] = self.img_size
        return items

    def run_G(self, items):
        out = self.ftr.forward(items)
        combined_out = items['image'] * (1 - items['mask']) + out * items['mask']

        return combined_out, out

    def run_G_ema(self, items):
        out = self.ftr_ema.forward(items)
        combined_out = items['image'] * (1 - items['mask']) + out * items['mask']

        return combined_out, out

    def postprocess(self, img, range=[-1, 1]):
        img = (img - range[0]) / (range[1] - range[0])
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def train_G(self, items):
        loss_dict = dict()
        loss_G_all = torch.tensor(0.0, device=items['image'].device)
        _, gen_img = self.run_G(items)
        self.gen_img_for_train = gen_img
        gen_logits, gen_feats = self.D.forward(gen_img)
        adv_gen_loss = generator_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_G_all += adv_gen_loss
        loss_dict['g_fake'] = adv_gen_loss

        # perceptual loss
        if self.loss_resnet_pl is not None:
            pcp_loss = self.loss_resnet_pl(gen_img, items['image'])
            loss_G_all += pcp_loss
            loss_dict['pcp'] = pcp_loss

        # feature matching
        if self.config['feature_matching']['weight'] > 0:
            _, real_feats = self.D(items['image'])
            fm_loss = feature_matching_loss(gen_feats, real_feats, mask=None) * self.config['feature_matching']['weight']
            loss_G_all += fm_loss
            loss_dict['fm'] = fm_loss

        # L1 loss
        if self.config['l1']['use_l1']:
            per_pixel_l1 = F.l1_loss(gen_img, items['image'], reduction='none')
            l1_mask = items['mask'] * self.config['l1']['weight_missing'] + (1 - items['mask']) * self.config['l1']['weight_known']
            l1_loss = (per_pixel_l1 * l1_mask).mean()
            loss_G_all += l1_loss
            loss_dict['l1'] = l1_loss

        return loss_G_all, loss_dict

    def train_D(self, items, do_GP=True):
        real_img_tmp = items['image'].requires_grad_(do_GP)
        real_logits, _ = self.D.forward(real_img_tmp)
        # _, gen_img = self.run_G(items)
        gen_img = self.gen_img_for_train
        gen_logits, _ = self.D.forward(gen_img.detach())
        dis_real_loss, grad_penalty = discriminator_real_loss(real_batch=real_img_tmp, discr_real_pred=real_logits,
                                                              gp_coef=self.adv_args['gp_coef'], do_GP=do_GP)
        dis_fake_loss = discriminator_fake_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_dict = {'d_real': dis_real_loss, 'd_fake': dis_fake_loss}
        if do_GP:
            grad_penalty = grad_penalty * self.D_reg_interval
            loss_dict['gp'] = grad_penalty
        loss_D_all = dis_real_loss + dis_fake_loss + grad_penalty

        return loss_D_all, loss_dict

    def train_dataloader(self):
        self.train_dataset = get_obj_by_name(self.data_class)(flist=self.args['train_flist'],
                                                              mask_path=self.args['train_mask_flist'],
                                                              batch_size=self.args['batch_size'] // self.num_gpus,
                                                              augment=True, training=True, test_mask_path=None,
                                                              world_size=self.num_gpus, **self.data_args)
        if not self.dynamic_size:
            return DataLoader(self.train_dataset, pin_memory=True, batch_size=self.args['batch_size'] // self.num_gpus, shuffle=True, num_workers=self.args['num_workers'])
        else:
            rank = self.get_ddp_rank()
            if rank is None:
                rank = 0
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.num_gpus, rank=rank, shuffle=True)
            return DataLoader(self.train_dataset, pin_memory=True, batch_size=self.args['batch_size'] // self.num_gpus,
                              sampler=train_sampler, num_workers=self.args['num_workers'])

    def val_dataloader(self):
        self.val_dataset = get_obj_by_name(self.data_class)(flist=self.args['val_flist'], mask_path=None,
                                                            batch_size=self.args['batch_size'] // self.num_gpus,
                                                            augment=False, training=False,
                                                            test_mask_path=self.args['test_mask_flist'],
                                                            **self.data_args)
        return DataLoader(self.val_dataset, pin_memory=True, batch_size=self.args['batch_size'] // self.num_gpus, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        opt_args = self.args['optimizer']
        g_optimizer = torch.optim.Adam(self.ftr.parameters(), lr=opt_args['g_opt']['lr'],
                                       betas=(opt_args['g_opt']['beta1'], opt_args['g_opt']['beta2']), eps=1e-8)
        d_optimizer = torch.optim.Adam(self.D.parameters(), lr=opt_args['d_opt']['lr'],
                                       betas=(opt_args['d_opt']['beta1'], opt_args['d_opt']['beta2']), eps=1e-8)
        # load pre-trained opt params
        if self.g_opt_state is not None:
            g_optimizer.load_state_dict(self.g_opt_state)
        if self.d_opt_state is not None:
            d_optimizer.load_state_dict(self.d_opt_state)

        g_sche = get_lr_milestone_decay_with_warmup(g_optimizer, num_warmup_steps=opt_args['warmup_steps'],
                                                    milestone_steps=opt_args['decay_steps'], gamma=opt_args['decay_rate'])
        g_sche = {'scheduler': g_sche, 'interval': 'step'}  # called after each training step
        d_sche = get_lr_milestone_decay_with_warmup(d_optimizer, num_warmup_steps=opt_args['warmup_steps'],
                                                    milestone_steps=opt_args['decay_steps'], gamma=opt_args['decay_rate'])
        d_sche = {'scheduler': d_sche, 'interval': 'step'}  # called after each training step
        return [g_optimizer, d_optimizer], [g_sche, d_sche]

    def on_train_start(self) -> None:
        if self.get_ddp_rank() is not None and (self.g_opt_state is not None or self.d_opt_state is not None):
            for opt in self.trainer.optimizers:
                if 'state' in opt.state_dict():
                    for k in opt.state_dict()['state']:
                        for k_ in opt.state_dict()['state'][k]:
                            if isinstance(opt.state_dict()['state'][k][k_], torch.Tensor):
                                opt.state_dict()['state'][k][k_] = opt.state_dict()['state'][k][k_].to(device=self.get_ddp_rank())

    def on_train_epoch_start(self):
        # For each epoch, we need to reset dynamic resolutions
        if self.dynamic_size:
            if self.get_ddp_rank() is None:
                self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)
            self.trainer.train_dataloader.dataset.datasets.reset_dataset(self.trainer.train_dataloader.sampler)

    def inpaint_prior(self, items, test=False):
        with torch.no_grad():
            image_size = items['image'].shape[2]

            edge_pred, line_pred = \
                self.edgeline_tsr.forward(items['img_256'], items['line_256'], masks=items['mask_256'])

            line_pred = items['line_256'] * (1 - items['mask_256']) + line_pred * items['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            if self.config['fix_256'] is None or self.config['fix_256'] is False:
                # nms for HR
                random_threshold = random.random() * 40 + 30  # [30~70]
                random_add_v = random.random() * 1.5 + 1.5  # [1.5~3]随机缩放upsample的结果
                random_mul_v = random.random() * 1.5 + 1.5  # [1.5~3]
                line_pred = self.structure_upsample(line_pred)[0]
                if test is False:
                    edge_pred = get_torch_nms(edge_pred, binary_threshold=random_threshold)
                    edge_pred = self.structure_upsample(edge_pred)[0]
                    edge_pred = torch.sigmoid((edge_pred + random_add_v) * random_mul_v)
                    line_pred = torch.sigmoid((line_pred + random_add_v) * random_mul_v)
                else:
                    edge_pred = get_np_nms(edge_pred, binary_threshold=50)
                    edge_pred = self.structure_upsample(edge_pred)[0]
                    edge_pred = torch.sigmoid((edge_pred + 2) * 2)
                    line_pred = torch.sigmoid((line_pred + 2) * 2)

                edge_pred = F.interpolate(edge_pred, size=(image_size, image_size), mode='bilinear',
                                          align_corners=False)
                line_pred = F.interpolate(line_pred, size=(image_size, image_size), mode='bilinear',
                                          align_corners=False)

            b = items['line_256'].shape[0]
            if self.global_step < int(self.config['Turning_Point']) and test is False:
                pred_rate = self.global_step / int(self.config['Turning_Point'])
                b = np.clip(int(pred_rate * b), 2, b)

            items['edge'] = edge_pred.detach()
            items['line'][:b, ...] = line_pred[:b, ...].detach()
            if self.use_grad is True:
                edge, line = self.grad_tsr.forward(items['img_256'], items['gradientx'], items['gradienty'], masks=items['mask_256'])
                edge = items['gradientx'] * (1 - items['mask_256']) + edge * items['mask_256']
                line = items['gradienty'] * (1 - items['mask_256']) + line * items['mask_256']
                if self.config['fix_256'] is None or self.config['fix_256'] is False:
                    edge = F.interpolate(edge, size=(image_size, image_size), mode='bilinear')
                    edge = edge * items['mask'] + items['gradientx_hr'] * (1 - items['mask'])

                    line = F.interpolate(line, size=(image_size, image_size), mode='bilinear')
                    line = line * items['mask'] + items['gradienty_hr'] * (1 - items['mask'])
                items['gradientx_hr'][:b, ...] = edge[:b, ...]
                items['gradienty_hr'][:b, ...] = line[:b, ...]

                items['gradientx'] = items['gradientx_hr'].detach()
                items['gradienty'] = items['gradienty_hr'].detach()
            else:
                items['gradientx'] = 0
                items['gradienty'] = 0
        return items

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.global_step % self.sample_period == 0 and optimizer_idx == 0:
            self._sample(batch)
        self.img_size = batch['image'].shape[-1]

        if optimizer_idx == 0:  # step for generator
            self.ftr.requires_grad_(True)
            self.D.requires_grad_(False)
            batch = self.inpaint_prior(batch)
            loss, loss_dict = self.train_G(batch)

        elif optimizer_idx == 1:  # step for discriminator
            self.ftr.requires_grad_(False)
            self.D.requires_grad_(True)
            loss, loss_dict = self.train_D(batch)
        else:
            raise NotImplementedError

        return dict(loss=loss, log_info=add_prefix_to_keys(loss_dict, 'train/'))

    def training_step_end(self, batch_parts_outputs):
        if self.use_ema:
            # Update ema.
            with torch.no_grad():
                for p_ema, p in zip(self.ftr_ema.parameters(), self.ftr.parameters()):
                    p_ema.copy_(p.lerp(p_ema, self.config['ema_beta']))
                for b_ema, b in zip(self.ftr_ema.buffers(), self.ftr.buffers()):
                    b_ema.copy_(b)

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)

        # show learning rate
        sche = self.trainer.lr_schedulers
        self.g_lr = sche[0]['scheduler'].get_lr()[0]
        self.d_lr = sche[1]['scheduler'].get_lr()[0]

        return full_loss

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            self.ftr_ema.eval()
        self.ftr.eval()
        batch = self.inpaint_prior(batch, test=True)
        if self.use_ema:
            gen_ema_img, _ = self.run_G_ema(batch)
        else:
            gen_ema_img, _ = self.run_G(batch)
        gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
        gen_ema_img = (gen_ema_img + 1) / 2
        gen_ema_img = gen_ema_img * 255.0
        gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
        for img_num in range(batch['image'].shape[0]):
            cv2.imwrite(self.eval_path + '/' + batch['name'][img_num], gen_ema_img[img_num, :, :, ::-1])
        self.ftr.train()

    def validation_epoch_end(self, outputs):
        if (self.trainer.num_nodes * self.trainer.num_processes) > 1:
            dist.barrier()
        if self.trainer.is_global_zero:
            self.metric = get_inpainting_metrics(self.eval_path, self.test_path)

            self.print(f'Steps:{self.global_step}')
            for m in self.metric:
                if m in ['PSNR', 'SSIM', 'FID', 'LPIPS']:
                    if m in self.metric:
                        self.print(m, self.metric[m])
                self.metric[m] *= self.num_gpus  # results of other processes are all zero, so *ngpu to balance the metric

        else:
            self.metric = {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0, 'FID': 0}

        self.log('val/PSNR', self.metric['PSNR'], sync_dist=True)
        self.log('val/SSIM', self.metric['SSIM'], sync_dist=True)
        self.log('val/LPIPS', self.metric['LPIPS'], sync_dist=True)
        self.log('val/FID', self.metric['FID'], sync_dist=True)

    def _sample(self, items):

        self.ftr.eval()
        if self.use_ema:
            self.ftr_ema.eval()
        with torch.no_grad():
            items = self.inpaint_prior(items, test=True)
            combined_gen_img, _ = self.run_G(items)
            combined_gen_img = torch.clamp(combined_gen_img, -1, 1)
            if self.use_ema:
                combined_gen_ema_img, _ = self.run_G_ema(items)
                combined_gen_ema_img = torch.clamp(combined_gen_ema_img, -1, 1)
            else:
                combined_gen_ema_img = combined_gen_img.clone()
            if self.use_grad is True:
                gradx_output = items['gradientx'].squeeze(1).cpu()
                gradx_output = (gradx_output - torch.min(torch.min(gradx_output, dim=1)[0], dim=1)[0].view(-1, 1, 1)) / (
                        torch.max(torch.max(gradx_output, dim=1)[0], dim=1)[0].view(-1, 1, 1) -
                        torch.min(torch.min(gradx_output, dim=1)[0], dim=1)[0].view(-1, 1, 1))
                gradx_output = gradx_output.unsqueeze(1)
                grady_output = items['gradienty'].squeeze(1).cpu()
                grady_output = (grady_output - torch.min(torch.min(grady_output, dim=1)[0], dim=1)[0].view(-1, 1, 1)) / (
                        torch.max(torch.max(grady_output, dim=1)[0], dim=1)[0].view(-1, 1, 1) -
                        torch.min(torch.min(grady_output, dim=1)[0], dim=1)[0].view(-1, 1, 1))
                grady_output = grady_output.unsqueeze(1)

                images = stitch_images(
                    self.postprocess((items['image']).cpu()),
                    self.postprocess((items['image'] * (1 - items['mask'])).cpu()),
                    self.postprocess(items['edge'].cpu(), range=[0, 1]),
                    self.postprocess(items['line'].cpu(), range=[0, 1]),
                    self.postprocess(gradx_output, range=[0, 1]),
                    self.postprocess(grady_output, range=[0, 1]),
                    self.postprocess((combined_gen_img).cpu()),
                    self.postprocess((combined_gen_ema_img).cpu()),
                    img_per_row=1
                )
            else:
                images = stitch_images(
                    self.postprocess((items['image']).cpu()),
                    self.postprocess((items['image'] * (1 - items['mask'])).cpu()),
                    self.postprocess(items['edge'].cpu(), range=[0, 1]),
                    self.postprocess(items['line'].cpu(), range=[0, 1]),
                    self.postprocess((combined_gen_img).cpu()),
                    self.postprocess((combined_gen_ema_img).cpu()),
                    img_per_row=1
                )

            if self.get_ddp_rank() in (None, 0):
                name = os.path.join(self.sample_path, str(self.global_step).zfill(6) + ".jpg")
                self.print('saving sample ' + name)
                images.save(name)

        self.ftr.train()

    def get_ddp_rank(self):
        return self.trainer.global_rank if (self.trainer.num_nodes * self.trainer.num_processes) > 1 else None


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


def wf_inference_test(wf, images, h, w, masks, obj_remove=False, valid_th=0.925, mask_th=0.925):
    lcnn_mean = torch.tensor([109.730, 103.832, 98.681]).to(images.device).reshape(1, 3, 1, 1)
    lcnn_std = torch.tensor([22.275, 22.124, 23.229]).to(images.device).reshape(1, 3, 1, 1)
    with torch.no_grad():
        images = images * 255.
        origin_masks = masks
        masks = F.interpolate(masks, size=(images.shape[2], images.shape[3]), mode='nearest')
        # the mask value of lcnn is 127.5
        masked_images = images * (1 - masks) + torch.ones_like(images) * masks * 127.5
        images = (images - lcnn_mean) / lcnn_std
        masked_images = (masked_images - lcnn_mean) / lcnn_std

        def to_int(x):
            return tuple(map(int, x))

        lines_tensor = []
        target_mask = origin_masks.cpu().numpy()  # origin_masks, masks size不同
        for i in range(images.shape[0]):
            lmap = np.zeros((h, w))

            output_nomask = wf(images[i].unsqueeze(0))
            output_nomask = to_device(output_nomask, 'cpu')
            if output_nomask['num_proposals'] == 0:
                lines_nomask = []
                scores_nomask = []
            else:
                lines_nomask = output_nomask['lines_pred'].numpy()
                lines_nomask = [[line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                                for line in lines_nomask]
                scores_nomask = output_nomask['lines_score'].numpy()

            output_masked = wf(masked_images[i].unsqueeze(0))
            output_masked = to_device(output_masked, 'cpu')
            if output_masked['num_proposals'] == 0:
                lines_masked = []
                scores_masked = []
            else:
                lines_masked = output_masked['lines_pred'].numpy()
                lines_masked = [[line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                                for line in lines_masked]
                scores_masked = output_masked['lines_score'].numpy()

            target_mask_ = target_mask[i, 0]
            if obj_remove:
                for line, score in zip(lines_nomask, scores_nomask):
                    line = np.clip(line, 0, 255)
                    if score > valid_th and (
                            target_mask_[to_int(line[0:2])] == 0 or target_mask_[to_int(line[2:4])] == 0):
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
                for line, score in zip(lines_masked, scores_masked):
                    line = np.clip(line, 0, 255)
                    if score > mask_th and target_mask_[to_int(line[0:2])] == 1 and target_mask_[
                        to_int(line[2:4])] == 1:
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
            else:
                for line, score in zip(lines_masked, scores_masked):
                    if score > mask_th:
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

            lmap = np.clip(lmap * 255, 0, 255).astype(np.uint8)
            lines_tensor.append(to_tensor(lmap).unsqueeze(0))

        lines_tensor = torch.cat(lines_tensor, dim=0)
    return lines_tensor.detach().to(images.device)


class FinetunePLTrainer_nms_threshold(FinetunePLTrainer):
    """
    Trainer class
    """

    def __init__(self, structure_upsample, edgeline_tsr, grad_tsr, ftr, D, config, out_path, num_gpus=1, use_ema=False, dynamic_size=False, gpu_id=1, test_only=False):
        super().__init__(structure_upsample, edgeline_tsr, grad_tsr, ftr, D, config, out_path, num_gpus, use_ema, dynamic_size, test_only)
        self.gpu_id = gpu_id

    def sample(self, upload_img, upload_mask, output_name, lama_res):
        with torch.no_grad():
            img = cv2.imread(upload_img)
            img = img[:, :, ::-1]
            # resize/crop if needed
            imgh, imgw, _ = img.shape
            img_256 = resize(img, 256, 256)
            img_512 = resize(img, 512, 512)
            img_512 = to_tensor(img_512).unsqueeze(0)

            # load mask
            mask = cv2.imread(upload_mask, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255

            mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
            mask_256[mask_256 > 0] = 255
            mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
            mask_512[mask_512 > 0] = 255
            mask_512 = to_tensor(mask_512).unsqueeze(0)

            # load line
            line_256 = wf_inference_test(self.wf, img_512.to(self.gpu_id), h=256, w=256, masks=mask_512.to(self.gpu_id),
                                         valid_th=0.85, mask_th=0.85)

            batch = dict()
            batch['image'] = to_tensor(img.copy(), norm=True)
            batch['img_256'] = to_tensor(img_256, norm=True)
            batch['mask'] = to_tensor(mask)
            batch['mask_256'] = to_tensor(mask_256)

            batch['size_ratio'] = -1

            # load pos encoding
            rel_pos, abs_pos, direct = load_masked_position_encoding(mask)
            batch['rel_pos'] = torch.LongTensor(rel_pos)
            batch['abs_pos'] = torch.LongTensor(abs_pos)
            batch['direct'] = torch.LongTensor(direct)

            batch['H'] = -1

            for k in batch:
                if type(batch[k]) is torch.Tensor:
                    batch[k] = batch[k].to(self.gpu_id).unsqueeze(0)

            batch['line_256'] = line_256

            # inapint prior
            print('TSR shape:', batch['img_256'].shape, batch['line_256'].shape, batch['mask_256'].shape)
            edge_pred, line_pred = self.edgeline_tsr.forward(batch['img_256'], batch['line_256'], masks=batch['mask_256'])

            line_pred = batch['line_256'] * (1 - batch['mask_256']) + line_pred * batch['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            current_size = 256
            while current_size * 2 <= max(imgh, imgw):
                # nms for HR
                line_pred = self.structure_upsample(line_pred)[0]
                edge_pred_nms = get_np_nms(edge_pred, binary_threshold=50)
                edge_pred_nms = self.structure_upsample(edge_pred_nms)[0]
                edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
                line_pred = torch.sigmoid((line_pred + 2) * 2)
                current_size *= 2

            edge_pred_nms = F.interpolate(edge_pred_nms, size=(imgh, imgw), mode='bilinear', align_corners=False)
            edge_pred = F.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)

            edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]

            line_pred = F.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)

            batch['edge'] = edge_pred.detach()
            batch['line'] = line_pred.detach()

            if self.use_ema:
                gen_ema_img, _ = self.run_G_ema(batch)
            else:
                gen_ema_img, _ = self.run_G(batch)
            gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
            inputs = (batch['image'] * (1 - batch['mask']))

            print('Input shape:', inputs.shape, 'LaMa shape:', lama_res.shape, 'ZITS shape:', gen_ema_img.shape)

            image_per_row = 1
            images = stitch_images(
                self.postprocess(batch['image'].cpu()),
                self.postprocess(inputs.cpu()),
                self.postprocess(batch['mask'].cpu(), range=[0, 1]),
                self.postprocess(lama_res.cpu(), range=[0, 1]),
                self.postprocess(batch['edge'].cpu(), range=[0, 1]),
                self.postprocess(batch['line'].cpu(), range=[0, 1]),
                self.postprocess(gen_ema_img.cpu()),
                img_per_row=image_per_row
            )
            print('\nsaving sample ' + output_name)
            images.save(output_name)
            print('Saving shape:', images.size)

    def inpaint_prior(self, items, test=False):
        with torch.no_grad():
            image_size = items['image'].shape[2]

            edge_pred, line_pred = \
                self.edgeline_tsr.forward(items['img_256'], items['line_256'], masks=items['mask_256'])

            line_pred = items['line_256'] * (1 - items['mask_256']) + line_pred * items['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            if self.config['fix_256'] is None or self.config['fix_256'] is False:
                # nms for HR
                random_threshold = random.random() * 40 + 30  # [30~70]
                random_add_v = random.random() * 1.5 + 1.5  # [1.5~3]随机缩放upsample的结果
                random_mul_v = random.random() * 1.5 + 1.5  # [1.5~3]
                line_pred = self.structure_upsample(line_pred)[0]
                if test is False:
                    edge_pred_nms = get_torch_nms(edge_pred, binary_threshold=random_threshold)
                    edge_pred_nms = self.structure_upsample(edge_pred_nms)[0]
                    edge_pred_nms = torch.sigmoid((edge_pred_nms + random_add_v) * random_mul_v)
                    line_pred = torch.sigmoid((line_pred + random_add_v) * random_mul_v)
                else:
                    edge_pred_nms = get_np_nms(edge_pred, binary_threshold=50)
                    edge_pred_nms = self.structure_upsample(edge_pred_nms)[0]
                    edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
                    line_pred = torch.sigmoid((line_pred + 2) * 2)

                edge_pred_nms = F.interpolate(edge_pred_nms, size=(image_size, image_size), mode='bilinear',
                                              align_corners=False)
                edge_pred = F.interpolate(edge_pred, size=(image_size, image_size), mode='bilinear',
                                          align_corners=False)

                edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]

                line_pred = F.interpolate(line_pred, size=(image_size, image_size), mode='bilinear',
                                          align_corners=False)

            b = items['line_256'].shape[0]
            if self.global_step < int(self.config['Turning_Point']) and test is False:
                pred_rate = self.global_step / int(self.config['Turning_Point'])
                b = np.clip(int(pred_rate * b), 2, b)

            items['edge'] = edge_pred.detach()
            items['line'][:b, ...] = line_pred[:b, ...].detach()
        return items
