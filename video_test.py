import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.erfnet_road import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.LossWithUncertainty import LossWithUncertainty
from dataloaders.utils import decode_segmap

class Test(object):
    def __init__(self, args):
        self.args = args
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.test_loader, self.nclass_pixel, self.nclass_scene = make_data_loader(args, **kwargs)
        
        self.saved_index = 0
        # Define network
        model = ERFNet(num_classes_pixel = self.nclass_pixel, num_classes_scene = self.nclass_scene,multitask = self.args.multitask)

        self.model = model
        
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def write_test(self):
        self.model.eval()
        #self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image = sample['image']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output, output_road = self.model(image)
            if output_road != None:
                pass

            label_masks = torch.max(output, 1)[1].detach().cpu().numpy()
            image = image.detach().cpu().numpy().transpose(0, 2, 3, 1)
            #image = image.detach().cpu().numpy()
            #targets = target.detach().cpu().numpy()
            #print(targets.shape) 
            for idx, label_mask in enumerate(label_masks):     
                decode_segmap(label_mask, dataset=self.args.dataset, saved_path = self.args.saved_path + "/%(idx)05d.png" % {'idx':self.saved_index}, image = image[idx])
                self.saved_index += 1

def main():
    parser = argparse.ArgumentParser(description="PyTorch Lane Detection")
    parser.add_argument('--dataset', type=str, default='bdd100k',
                        choices=['bdd100k'],
                        help='dataset name (default: bdd100k)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-w', type=int, default=960,
                        help='base image width')
    parser.add_argument('--base-h', type=int, default=640,
                        help='base image height')
    parser.add_argument('--crop-w', type=int, default=640,
                        help='crop image width')
    parser.add_argument('--crop-h', type=int, default=480,
                        help='crop image height')
    parser.add_argument('--output-w', type=int, default=640,
                        help='output image width')
    parser.add_argument('--output-h', type=int, default=480,
                        help='output image height')
    
    parser.add_argument('--multitask', type=bool, default=False,
                         help='whether to do multi-task (default: auto)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                test (default: auto)')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--write-val', action='store_true', default=False,
                        help='store val rgb results')
    parser.add_argument('--video', type=str, default=None,
                        help='video segmentation only for write-val')
    parser.add_argument('--saved-path', type=str, default=None,
                        help='path for saving segmentation result')


    args = parser.parse_args()
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    print(args)
    torch.manual_seed(args.seed)
    tester = Test(args)
    tester.write_test()

if __name__ == "__main__":
   main()
