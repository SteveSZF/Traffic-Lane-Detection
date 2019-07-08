import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_w, base_h, crop_w, crop_h, output_w, output_h, fill=0):
        self.base_w = base_w
        self.base_h = base_h
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.output_w = output_w
        self.output_h = output_h
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        #short_w = random.randint(int(self.base_w * 0.5), int(self.base_w * 2.0))
        short_h = random.randint(int(self.base_h * 0.5), int(self.base_h * 2.0))
        w, h = img.size
        # if h > w:
        #     ow = short_h
        #     oh = int(1.0 * h * ow / w)
        # else:
        oh = short_h
        ow = int(1.0 * w * oh / h)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_h < self.crop_h:
            padh = self.crop_h - oh if oh < self.crop_h else 0
            padw = self.crop_w - ow if ow < self.crop_w else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_w)
        y1 = random.randint(0, h - self.crop_h)
        img = img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        mask = mask.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        mask = mask.resize((self.output_w, self.output_h), Image.NEAREST)
        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_w, crop_h, output_w, output_h):
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.output_w = output_w
        self.output_h = output_h

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        #w, h = img.size
        # if w > h:
        #     oh = self.crop_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        ow = self.crop_w
        oh = self.crop_h
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_w) / 2.))
        y1 = int(round((h - self.crop_h) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        mask = mask.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        mask = mask.resize((self.output_w, self.output_h), Image.NEAREST)
        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, fix_w, fix_h, output_w, output_h):
        self.size = (fix_w, fix_h)  # size: (h, w)
        self.output_size = (output_w, output_h)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.output_size, Image.NEAREST)

        return {'image': img,
                'label': mask}