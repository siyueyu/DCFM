import os
from PIL import PILLOW_VERSION, Image, ImageOps, ImageFilter
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import random
import pandas as pd


class CoData(data.Dataset):
    def __init__(self, img_root, gt_root, img_size, transform, max_num, is_train):

        class_list = os.listdir(img_root)
        self.size = [img_size, img_size]
        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))
        self.gt_dirs = list(
            map(lambda x: os.path.join(gt_root, x), class_list))
        self.transform = transform
        self.max_num = max_num
        self.is_train = is_train

    def __getitem__(self, item):
        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))
        gt_paths = list(
            map(lambda x: os.path.join(self.gt_dirs[item], x[:-4]+'.png'), names))

        if self.is_train:
            final_num = min(num, self.max_num)

            sampled_list = random.sample(range(num), final_num)
            # print(sampled_list)
            new_img_paths = [img_paths[i] for i in sampled_list]
            img_paths = new_img_paths
            new_gt_paths = [gt_paths[i] for i in sampled_list]
            gt_paths = new_gt_paths

            final_num = final_num
        else:
            final_num = num

        imgs = torch.Tensor(final_num, 3, self.size[0], self.size[1])
        gts = torch.Tensor(final_num, 1, self.size[0], self.size[1])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):
            # print(idx)
            img = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')

            subpaths.append(os.path.join(img_paths[idx].split('/')[-2], img_paths[idx].split('/')[-1][:-4]+'.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            # ori_sizes += ((img.size[1], img.size[0]))

            [img, gt] = self.transform(img, gt)

            imgs[idx] = img
            gts[idx] = gt
        if self.is_train:
            cls_ls = [item] * int(final_num)
            return imgs, gts, subpaths, ori_sizes, cls_ls
        else:
            return imgs, gts, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img, gt):
        # assert img.size == gt.size

        img = img.resize(self.size, Image.BILINEAR)
        gt = gt.resize(self.size, Image.NEAREST)
        # gt = gt.resize(self.size, Image.BILINEAR)

        return img, gt


class ToTensor(object):
    def __call__(self, img, gt):

        return F.to_tensor(img), F.to_tensor(gt)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img, gt):
        img = F.normalize(img, self.mean, self.std)

        return img, gt


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        return img, gt


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img, mask):
        # random scale (short edge)
        # img = img.numpy()
        # mask = mask.numpy()
        short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.2))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, gt):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, Image.BILINEAR, self.expand, self.center), F.rotate(gt, angle, Image.NEAREST, self.expand, self.center)



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, gt_root, img_size, batch_size, max_num = float('inf'), istrain=True, shuffle=False, num_workers=0, pin=False):
    if istrain:
        transform = Compose([
            RandomScaleCrop(img_size*2, img_size*2),
            FixedResize(img_size),
            RandomHorizontalFlip(),

            RandomRotation((-90, 90)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = Compose([
            FixedResize(img_size),
            # RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = CoData(img_root, gt_root, img_size, transform, max_num, is_train=istrain)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_root = './data/testtrain/img/'
    gt_root = './data/testtrain/gt/'
    loader = get_loader(img_root, gt_root, 20, 1, 16, istrain=False)
    for batch in loader:
        b, c, h, w = batch[0][0].shape
        for i in range(b):
            img = batch[0].squeeze(0)[i].permute(1, 2, 0).cpu().numpy() * std + mean
            image = img * 255
            mask = batch[1].squeeze(0)[i].squeeze().cpu().numpy()
            plt.subplot(121)
            plt.imshow(np.uint8(image))
            plt.subplot(122)
            plt.imshow(mask)
            plt.show(block=True)
