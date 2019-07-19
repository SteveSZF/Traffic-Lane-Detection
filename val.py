import argparse
import os
import numpy as np
from tqdm import tqdm
import time

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.erfnet_road import *
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.LossWithUncertainty import LossWithUncertainty
from dataloaders.utils import decode_segmap

class Trainer(object):
    def __init__(self, args):
        self.args = args 

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass_pixel, self.nclass_scene = make_data_loader(args, **kwargs)

        
        self.saved_index = 0
        # Define network
        if args.checkname == 'erfnet':
            model = ERFNet(num_classes_pixel = self.nclass_pixel, num_classes_scene = self.nclass_scene,multitask = self.args.multitask)
        elif args.checkname == 'resnet':
            model = DeepLab(num_classes=self.nclass_pixel, backbone = 'resnet', output_stride=8)
        elif args.checkname == 'mobilenet':
            model = DeepLab(num_classes=self.nclass_pixel, backbone = 'mobilenet', output_stride=16)
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass_pixel)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)


        self.criterion_uncertainty = LossWithUncertainty().cuda() #########################

        self.model = model
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass_pixel)

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
            print("=> loaded checkpoint '{}' (epoch{})"
                  .format(args.resume, checkpoint['epoch']))

    def write_val(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        total_time = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                start = time.time()
                output, output_road = self.model(image)
                end = time.time()
                total_time += (end - start) * 1000
            if output_road != None:
                pass
            label_masks = torch.max(output, 1)[1].detach().cpu().numpy()
            targets = target.detach().cpu().numpy()
            #print(targets.shape) 
            for idx, label_mask in enumerate(label_masks):     
                decode_segmap(label_mask, dataset=self.args.dataset, saved_path = self.args.saved_path + "/%(idx)05d.png" % {'idx':self.saved_index}, target = targets[idx])
                self.saved_index += 1

    def time_val(self):
        self.model.eval()
        self.evaluator.reset()
        length = len(self.val_loader)
        print("length: ", length)
        tbar = tqdm(self.val_loader, desc='\r')
        total_time = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                start = time.time()
                output, output_road = self.model(image)
                end = time.time()
                total_time += (end - start) * 1000
            if output_road != None:
                pass
            if i == 500:
                break;
        print("average infer time per batch", total_time / length * 20.0, 'ms')

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
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
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
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
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--write-val', action='store_true', default=False,
                        help='store val rgb results')
    parser.add_argument('--video', type=str, default=None,
                        help='video segmentation only for write-val')
    parser.add_argument('--saved-path', type=str, default=None,
                        help='path for saving segmentation result')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'bdd100k': 80,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'bdd100k': 0.0001,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'lane-erfnet'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    if args.write_val:
        trainer.write_val()
    else:
        trainer.time_val()

if __name__ == "__main__":
   main()
