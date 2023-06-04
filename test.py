import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch.nn.functional as FF
import torch.nn.parallel
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from tqdm import tqdm

from base.parse_config import ConfigParser
from dnnlib.util import get_obj_by_name
from trainers.nms_temp import get_nms as get_np_nms


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
    img_t = F.to_tensor(img).float()
    if norm:
        img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img_t


def main(args, config):
    # build models architecture, then print to console
    structure_upsample = get_obj_by_name(config['structure_upsample_class'])()

    edgeline_tsr = get_obj_by_name(config['edgeline_tsr_class'])()
    grad_tsr = get_obj_by_name(config['grad_tsr_class'])()
    ftr = get_obj_by_name(config['g_class'])(config=config['g_args'])
    D = get_obj_by_name(config['d_class'])(config=config['d_args'])

    if 'PLTrainer' not in config.config or config['PLTrainer'] is None:
        config.config['PLTrainer'] = 'trainers.pl_trainers.FinetunePLTrainer'

    model = get_obj_by_name(config['PLTrainer'])(structure_upsample, edgeline_tsr, grad_tsr, ftr, D, config,
                                                 'ckpts/' + args.exp_name, use_ema=args.use_ema, dynamic_size=args.dynamic_size)

    if args.use_ema:
        model.reset_ema()

    if args.ckpt_resume:
        print("Loading checkpoint: {} ...".format(args.ckpt_resume))
        checkpoint = torch.load(args.ckpt_resume, map_location='cpu')
        import utils
        utils.torch_init_model(model, checkpoint, key='state_dict')

    if hasattr(model, "wf"):
        model.wf.load_state_dict(torch.load(args.wf_ckpt, map_location='cpu')['model'])

    model.cuda()

    if args.use_ema:
        model.ftr_ema.eval()
    else:
        model.ftr.eval()

    os.makedirs(args.save_path, exist_ok=True)

    # dataset
    data = glob(args.img_dir + '/*')
    data = sorted(data, key=lambda x: x.split('/')[-1])

    mask_list = glob(args.mask_dir + '/*')
    mask_list = sorted(mask_list, key=lambda x: x.split('/')[-1])

    with torch.no_grad():
        for index in tqdm(range(len(data))):
            # load image
            img = cv2.imread(data[index])
            img = img[:, :, ::-1]
            # resize/crop if needed
            imgh, imgw, _ = img.shape
            img_512 = resize(img, 512, 512)
            img_256 = resize(img, 256, 256)

            # load mask
            mask = cv2.imread(mask_list[index % len(mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255

            mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
            mask_256[mask_256 > 0] = 255

            selected_img_name = data[index]

            # load gradient
            img_gray = rgb2gray(img_256) * 255
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float)

            img_gray = rgb2gray(img) * 255
            sobelx_hr = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float)
            sobely_hr = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float)

            batch = dict()
            batch['image'] = to_tensor(img.copy(), norm=True)
            batch['img_256'] = to_tensor(img_256, norm=True)
            batch['mask'] = to_tensor(mask)
            batch['mask_256'] = to_tensor(mask_256)
            batch['gradientx'] = torch.from_numpy(sobelx).unsqueeze(0).float()
            batch['gradienty'] = torch.from_numpy(sobely).unsqueeze(0).float()

            batch['gradientx_hr'] = torch.from_numpy(sobelx_hr).unsqueeze(0).float()
            batch['gradienty_hr'] = torch.from_numpy(sobely_hr).unsqueeze(0).float()

            # batch['line'] = to_tensor(line)
            # batch['line_256'] = to_tensor(line_256)

            # load line
            from trainers.pl_trainers import wf_inference_test
            img_512 = to_tensor(img_512)
            mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_512 = (mask_512 > 127).astype(np.uint8) * 255
            mask_512 = to_tensor(mask_512)
            line_256 = wf_inference_test(model.wf, img_512[None].cuda(), h=256, w=256, masks=mask_512[None].cuda(),
                                         valid_th=0.85, mask_th=0.85)
            batch['line_256'] = line_256[0]
            batch['size_ratio'] = -1

            batch['name'] = os.path.basename(data[index])

            # load pos encoding
            rel_pos, abs_pos, direct = load_masked_position_encoding(mask)
            batch['rel_pos'] = torch.LongTensor(rel_pos)
            batch['abs_pos'] = torch.LongTensor(abs_pos)
            batch['direct'] = torch.LongTensor(direct)

            batch['H'] = -1

            for k in batch:
                if type(batch[k]) is torch.Tensor:
                    batch[k] = batch[k].cuda().unsqueeze(0)

            # inapint prior
            edge_pred, line_pred = model.edgeline_tsr.forward(batch['img_256'], batch['line_256'], masks=batch['mask_256'])
            line_pred = batch['line_256'] * (1 - batch['mask_256']) + line_pred * batch['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            current_size = 256
            if current_size != min(imgh, imgw):
                while current_size * 2 <= max(imgh, imgw):
                    # nms for HR
                    line_pred = model.structure_upsample(line_pred)[0]
                    edge_pred_nms = get_np_nms(edge_pred, binary_threshold=50)
                    edge_pred_nms = model.structure_upsample(edge_pred_nms)[0]
                    edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
                    line_pred = torch.sigmoid((line_pred + 2) * 2)
                    current_size *= 2

                edge_pred_nms = FF.interpolate(edge_pred_nms, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
            else:
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)

                if config['g_args']['use_gradient'] is True:
                    gradientx, gradienty = model.grad_tsr.forward(batch['img_256'], batch['gradientx'], batch['gradienty'], masks=batch['mask_256'])
                    gradientx = batch['gradientx'] * (1 - batch['mask_256']) + gradientx * batch['mask_256']
                    gradienty = batch['gradienty'] * (1 - batch['mask_256']) + gradienty * batch['mask_256']
                    gradientx = FF.interpolate(gradientx, size=(imgh, imgw), mode='bilinear')
                    gradientx = gradientx * batch['mask'] + batch['gradientx_hr'] * (1 - batch['mask'])

                    gradienty = FF.interpolate(gradienty, size=(imgh, imgw), mode='bilinear')
                    gradienty = gradienty * batch['mask'] + batch['gradienty_hr'] * (1 - batch['mask'])

                    batch['gradientx'] = gradientx.detach()
                    batch['gradienty'] = gradienty.detach()

            batch['edge'] = edge_pred.detach()
            batch['line'] = line_pred.detach()

            if args.use_ema:
                gen_ema_img, _ = model.run_G_ema(batch)
            else:
                gen_ema_img, _ = model.run_G(batch)
            gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
            gen_ema_img = (gen_ema_img + 1) / 2
            gen_ema_img = gen_ema_img * 255.0
            gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
            edge_res = (batch['edge'] * 255.0).permute(0, 2, 3, 1).int().cpu().numpy()
            edge_res = np.tile(edge_res, [1, 1, 1, 3])
            masked_img = (batch['image'] * (1 - batch['mask']) + 1) / 2 * 255
            masked_img = masked_img.permute(0, 2, 3, 1).int().cpu().numpy()
            final_res = np.concatenate([masked_img, edge_res, gen_ema_img], axis=2)
            cv2.imwrite(args.save_path + '/' + batch['name'], final_res[0, :, :, ::-1])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default="config.yml", type=str, help='config file path')
    args.add_argument('--exp_name', default=None, type=str, help='method name')
    args.add_argument('--dynamic_size', action='store_true', help='Whether to finetune in dynamic size?')
    args.add_argument('--use_ema', action='store_true', help='Whether to use ema?')
    args.add_argument('--ckpt_resume', default='last.ckpt', type=str, help='PL path to restore')
    args.add_argument('--wf_ckpt', type=str, default=None, help='Line detector weights')
    args.add_argument('--save_path', default='outputs', type=str, help='path to save')
    args.add_argument('--img_dir', type=str, default=None, help='Test image path')
    args.add_argument('--mask_dir', type=str, default=None, help='Mask path')

    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    assert args.img_dir is not None
    assert args.mask_dir is not None
    args.resume = None
    config = ConfigParser.from_args(args, mkdir=False)
    SEED = 123456
    torch.manual_seed(SEED)

    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus

    main(args, config)
