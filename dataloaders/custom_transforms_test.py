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
        #mask = sample['label']
        img = np.array(img).astype(np.float32)
        #mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        #mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        #mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        #mask = torch.from_numpy(mask).float()

        return {'image': img}

class FixedResize(object):
    def __init__(self, fix_w, fix_h):
        self.size = (fix_w, fix_h)  # size: (h, w)
        #self.output_size = (output_w, output_h)

    def __call__(self, sample):
        img = sample['image']
        #mask = sample['label']

        #assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        #mask = mask.resize(self.output_size, Image.NEAREST)

        return {'image': img}
        #return {'image': img,
        #        'label': mask}
