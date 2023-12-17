import cv2
import os
import numpy as np
import torch
from glob import glob
from skimage.color import rgb2gray
import torch.nn.functional as FF
import torchvision.transforms.functional as F


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


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, test_size=None, use_gradient=False):
        super(InpaintingDataset, self).__init__()
        self.test_size = test_size
        self.use_gradient = use_gradient
        if img_dir.endswith(".txt"):
            with open(img_dir, 'r') as f:
                data = f.readlines()
            self.data = [d.strip() for d in data]
        else:
            data = glob(img_dir + '/*')
            self.data = sorted(data, key=lambda x: x.split('/')[-1])

        mask_list = glob(mask_dir + '/*')
        self.mask_list = sorted(mask_list, key=lambda x: x.split('/')[-1])

        print('Image num:', len(self.data))
        print('Mask num:', len(self.mask_list))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(self.data[index])
        if self.test_size is not None:
            img = resize(img, self.test_size, self.test_size)
        img = img[:, :, ::-1]
        # resize/crop if needed
        imgh, imgw, _ = img.shape
        img_512 = resize(img, 512, 512)
        img_256 = resize(img, 256, 256)

        # load mask
        mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255

        mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask_256[mask_256 > 0] = 255
        mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_512 = (mask_512 > 127).astype(np.uint8) * 255

        batch = dict()
        batch['image'] = to_tensor(img.copy(), norm=True)
        batch['img_256'] = to_tensor(img_256, norm=True)
        batch['mask'] = to_tensor(mask)
        batch['mask_256'] = to_tensor(mask_256)
        batch['mask_512'] = to_tensor(mask_512)
        batch['img_512'] = to_tensor(img_512)
        batch['imgh'] = imgh
        batch['imgw'] = imgw

        batch['name'] = os.path.basename(self.data[index])

        # load pos encoding
        rel_pos, abs_pos, direct = load_masked_position_encoding(mask)
        batch['rel_pos'] = torch.LongTensor(rel_pos)
        batch['abs_pos'] = torch.LongTensor(abs_pos)
        batch['direct'] = torch.LongTensor(direct)

        # load gradient
        if self.use_gradient:
            img_gray = rgb2gray(img_256) * 255
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float32)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float32)

            img_gray = rgb2gray(img) * 255
            sobelx_hr = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float32)
            sobely_hr = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float32)

            batch['gradientx'] = torch.from_numpy(sobelx).unsqueeze(0).float()
            batch['gradienty'] = torch.from_numpy(sobely).unsqueeze(0).float()
            batch['gradientx_hr'] = torch.from_numpy(sobelx_hr).unsqueeze(0).float()
            batch['gradienty_hr'] = torch.from_numpy(sobely_hr).unsqueeze(0).float()

        return batch
