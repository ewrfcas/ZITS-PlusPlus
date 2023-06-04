import glob
import os
import pickle
import random

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from torch.utils.data import DataLoader


def to_int(x):
    return tuple(map(int, x))


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, flist, batch_size, mask_path=None, augment=True, training=True, test_mask_path=None,
                 input_size=None, load_path=False, default_size=256, mask_rate=[0.4, 0.8, 1.0], rect_mask_rate=0.0,
                 config=None):
        super(InpaintingDataset, self).__init__()
        self.augment = augment
        self.training = training
        self.batch_size = batch_size
        self.mask_rate = mask_rate
        self.config = config
        if config is not None:
            self.rect_mask_rate = config.get('rect_mask_rate', 0.0)
        else:
            self.rect_mask_rate = rect_mask_rate

        self.data = []
        if load_path:
            self.data = glob.glob(flist + '/*')
            self.data = sorted(self.data, key=lambda x: x.split('/')[-1])
        else:
            f = open(flist, 'r')
            for i in f.readlines():
                i = i.strip()
                self.data.append(i)
            f.close()

        if training:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            if test_mask_path.endswith('txt'):
                self.mask_list = []
                with open(test_mask_path, 'r') as f:
                    for line in f:
                        self.mask_list.append(line.strip())
            else:
                self.mask_list = glob.glob(test_mask_path + '/*')
                self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        img = img[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)

        # load mask
        mask = self.load_mask(img, index)

        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()

        batch = dict()
        batch['img'] = self.to_tensor(img, norm=True)
        batch['mask'] = self.to_tensor(mask)

        batch['name'] = self.load_name(index)

        return batch

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            if random.random() < self.rect_mask_rate:
                A = imgh * imgw * (random.random() * 0.1 + 0.3)  # 0.3~0.4
                mask = np.zeros((imgh, imgw), np.uint8)
                w = random.randint(imgh // 2, imgh)
                h = int(A / w)
                x = random.randint(0, imgw - w)
                y = random.randint(0, imgh - h)
                mask[max(y, 0): min(y + h, imgh), max(x, 0): min(x + w, imgw)] = 1
                mask = mask * 255
            else:
                rdv = random.random()
                if rdv < self.mask_rate[0]:
                    mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                    mask = cv2.imread(self.irregular_mask_list[mask_index],
                                      cv2.IMREAD_GRAYSCALE)
                elif rdv < self.mask_rate[1]:
                    mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                    mask = cv2.imread(self.segment_mask_list[mask_index],
                                      cv2.IMREAD_GRAYSCALE)
                else:
                    mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                    mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                    mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                       cv2.IMREAD_GRAYSCALE).astype(np.float)
                    mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                       cv2.IMREAD_GRAYSCALE).astype(np.float)
                    mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
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
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):

        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                # flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist = self.getfilelist(flist)
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def getfilelist(self, path):
        all_file = []
        for dir, folder, file in os.walk(path):
            for i in file:
                t = "%s/%s" % (dir, i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') or \
                        t.endswith('.PNG') or t.endswith('.JPEG'):
                    all_file.append(t)
        return all_file


class DynamicDataset_gradient_line(torch.utils.data.Dataset):
    def __init__(self, flist, batch_size, mask_path=None, pos_num=128, train_line_path=None, eval_line_path=None,
                 wireframe_th=0.85, augment=True, training=True, test_mask_path=None, input_size=None, load_path=False,
                 default_size=256, str_size=256, world_size=1, mask_rate=[0.4, 0.8, 1.0], round=1, rect_mask_rate=0.0,
                 config=None):
        super(DynamicDataset_gradient_line, self).__init__()
        self.augment = augment
        self.training = training
        self.train_line_path = train_line_path
        self.eval_line_path = eval_line_path
        self.wireframe_th = wireframe_th
        self.batch_size = batch_size
        self.mask_rate = mask_rate
        self.round = round  # for places2 round is 32

        if config is not None:
            self.rect_mask_rate = config.get('rect_mask_rate', 0.0)
        else:
            self.rect_mask_rate = rect_mask_rate

        self.data = []
        if load_path:
            self.data = glob.glob(flist + '/*')
            self.data = sorted(self.data, key=lambda x: x.split('/')[-1])
        else:
            f = open(flist, 'r')
            for i in f.readlines():
                i = i.strip()
                self.data.append(i)
            f.close()

        if training:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size
        self.str_size = str_size  # 256 for transformer
        self.world_size = world_size

        self.ones_filter = np.ones((3, 3), dtype=np.float32)
        self.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        self.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
        self.pos_num = pos_num

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        if self.training:
            barrel_num = int(len(self.data) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
                                     256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)
        else:
            self.input_size = self.default_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        if type(self.input_size) == list:
            maped_idx = self.idx_map[index]
            if maped_idx > len(self.input_size) - 1:
                size = 512
            else:
                size = self.input_size[maped_idx]
        else:
            size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        img = img[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)
        img_256 = self.resize(img, self.str_size, self.str_size)

        # load mask
        mask = self.load_mask(img, index)
        mask_256 = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask_256[mask_256 > 0] = 255

        img_gray = rgb2gray(img_256) * 255
        sobelx, sobely = self.load_gradient(img_gray)

        img_gray = rgb2gray(img) * 255
        sobelx_hr, sobely_hr = self.load_gradient(img_gray)

        # load line
        line = self.load_wireframe(index, size)
        line_256 = self.load_wireframe(index, self.str_size)

        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
            img_256 = img_256[:, ::-1, ...].copy()
            sobelx = sobelx[:, ::-1].copy()
            sobely = sobely[:, ::-1].copy()

            sobelx_hr = sobelx_hr[:, ::-1].copy()
            sobely_hr = sobely_hr[:, ::-1].copy()

            line = line[:, ::-1, ...].copy()
            line_256 = line_256[:, ::-1, ...].copy()

        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
            mask_256 = mask_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()
            mask_256 = mask_256[::-1, :, ...].copy()

        batch = dict()
        batch['image'] = self.to_tensor(img, norm=True)
        batch['img_256'] = self.to_tensor(img_256, norm=True)
        batch['mask'] = self.to_tensor(mask)
        batch['mask_256'] = self.to_tensor(mask_256)
        batch['gradientx'] = torch.from_numpy(sobelx).unsqueeze(0).float()
        batch['gradienty'] = torch.from_numpy(sobely).unsqueeze(0).float()

        batch['gradientx_hr'] = torch.from_numpy(sobelx_hr).unsqueeze(0).float()
        batch['gradienty_hr'] = torch.from_numpy(sobely_hr).unsqueeze(0).float()

        batch['line'] = self.to_tensor(line)
        batch['line_256'] = self.to_tensor(line_256)

        batch['size_ratio'] = size / self.default_size

        batch['name'] = self.load_name(index)

        # load pos encoding
        rel_pos, abs_pos, direct = self.load_masked_position_encoding(mask)
        batch['rel_pos'] = torch.LongTensor(rel_pos)
        batch['abs_pos'] = torch.LongTensor(abs_pos)
        batch['direct'] = torch.LongTensor(direct)

        batch['H'] = size // 8

        return batch

    def load_masked_position_encoding(self, mask):
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2]
        ori_mask = ori_mask / 255
        mask = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255
        h, w = mask.shape[0:2]
        mask3 = mask.copy()
        mask3 = 1. - (mask3 / 255.0)
        pos = np.zeros((h, w), dtype=np.int32)
        direct = np.zeros((h, w, 4), dtype=np.int32)
        i = 0
        while np.sum(1 - mask3) > 0:
            i += 1
            mask3_ = cv2.filter2D(mask3, -1, self.ones_filter)
            mask3_[mask3_ > 0] = 1
            sub_mask = mask3_ - mask3
            pos[sub_mask == 1] = i

            m = cv2.filter2D(mask3, -1, self.d_filter1)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 0] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter2)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 1] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter3)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 2] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter4)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 3] = 1

            mask3 = mask3_

        abs_pos = pos.copy()
        rel_pos = pos / (self.str_size / 2)  # to 0~1 maybe larger than 1
        rel_pos = (rel_pos * self.pos_num).astype(np.int32)
        rel_pos = np.clip(rel_pos, 0, self.pos_num - 1)

        if ori_w != w or ori_h != h:
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, abs_pos, direct

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            if random.random() < self.rect_mask_rate:
                A = imgh * imgw * (random.random() * 0.1 + 0.3)  # 0.3~0.4
                mask = np.zeros((imgh, imgw), np.uint8)
                w = random.randint(imgh // 2, imgh)
                h = int(A / w)
                x = random.randint(0, imgw - w)
                y = random.randint(0, imgh - h)
                mask[max(y, 0): min(y + h, imgh), max(x, 0): min(x + w, imgw)] = 1
                mask = mask * 255
            else:
                rdv = random.random()
                if rdv < self.mask_rate[0]:
                    mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                    mask = cv2.imread(self.irregular_mask_list[mask_index],
                                      cv2.IMREAD_GRAYSCALE)
                elif rdv < self.mask_rate[1]:
                    mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                    mask = cv2.imread(self.segment_mask_list[mask_index],
                                      cv2.IMREAD_GRAYSCALE)
                else:
                    mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                    mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                    mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                       cv2.IMREAD_GRAYSCALE).astype(np.float)
                    mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                       cv2.IMREAD_GRAYSCALE).astype(np.float)
                    mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
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
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):

        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                # flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist = self.getfilelist(flist)
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def getfilelist(self, path):
        all_file = []
        for dir, folder, file in os.walk(path):
            for i in file:
                t = "%s/%s" % (dir, i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') or \
                        t.endswith('.PNG') or t.endswith('.JPEG'):
                    all_file.append(t)
        return all_file


    def load_gradient(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0).astype(np.float)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1).astype(np.float)
        return sobelx, sobely

    def load_wireframe(self, idx, size):
        selected_img_name = self.data[idx]
        line_name = selected_img_name.split("/")
        ns = line_name[-1]
        if self.training is False:
            line_name[4] = self.eval_line_path
        else:
            line_name[4] = self.train_line_path
        line_name = line_name[:5] + line_name[6:]  # + [ns]
        line_name = "/".join(line_name).replace('.png', '.pkl').replace('.jpg', '.pkl')
        line_name = line_name.replace('/imgs/', '/')

        wf = pickle.load(open(line_name, 'rb'))
        lmap = np.zeros((size, size))
        for i in range(len(wf['scores'])):
            if wf['scores'][i] > self.wireframe_th:
                line = wf['lines'][i].copy()
                line[0] = line[0] * size
                line[1] = line[1] * size
                line[2] = line[2] * size
                line[3] = line[3] * size
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
        return lmap
