from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class BDDSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES_PIXEL = 3
    NUM_CLASSES_SCENE = 4
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('bdd100k'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        if split == 'train':
            self._image_dir = os.path.join(self._base_dir, 'images', '100k', 'train')
            self._cat_dir = os.path.join(self._base_dir, 'drivable_maps', 'labels', 'train')
        else:
            self._image_dir = os.path.join(self._base_dir, 'images', '100k', 'val')
            self._cat_dir = os.path.join(self._base_dir, 'drivable_maps', 'labels', 'val')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        _splits_dir = os.path.join(self._base_dir)
        # if split == 'train':
        #     _splits_dir = os.path.join(self._base_dir)
        # else:
        #     _splits_dir = os.path.join(self._base_dir)

        self.im_ids = []
        self.images = []
        self.categories = []

        for split in self.split:
            with open(os.path.join(_splits_dir, split + '_id.txt'), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + "_drivable_id.png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _target = np.array(_target)
        _target = Image.fromarray(_target.astype('uint8'))
        #print("in t_make_img_gt_point_pair:", np.array(_target ))

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_w = self.args.base_w, base_h = self.args.base_h, crop_w = self.args.crop_w, crop_h = self.args.crop_h, 
                                output_w = self.args.output_w, output_h = self.args.output_h),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_w = self.args.crop_w, crop_h = self.args.crop_h, 
                                output_w = self.args.output_w, output_h = self.args.output_h),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'BDD100K(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


