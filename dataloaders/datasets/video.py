from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms_test as tr
import cv2

class VIDEOSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES_PIXEL = 3
    NUM_CLASSES_SCENE = 4
    def __init__(self, args):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.cap = cv2.VideoCapture(args.video)
        self.args = args

    def __len__(self):
        return int(self.cap.get(7))


    def __getitem__(self, index):
        #_img, _target = self._make_img_gt_point_pair(index)
        #sample = {'image': _img, 'label': _target}
        _, _img = self.cap.read()
        _img = np.array(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
        _img = Image.fromarray(_img.astype('uint8'))
        
        sample = {'image': _img}

        return self.transform_test(sample)
    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(fix_w = self.args.crop_w, fix_h = self.args.crop_h),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return self.args.video


