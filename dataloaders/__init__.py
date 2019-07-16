from dataloaders.datasets import bdd100k, video
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if isinstance(args.video, str):
        test_set = video.VIDEOSegmentation(args)
    
        num_classes_pixel = test_set.NUM_CLASSES_PIXEL
        num_classes_scene = test_set.NUM_CLASSES_SCENE

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return test_loader, num_classes_pixel, num_classes_scene


    elif args.dataset == 'bdd100k':
        train_set = bdd100k.BDDSegmentation(args, split='train')
        val_set = bdd100k.BDDSegmentation(args, split='val')
    
        num_classes_pixel = train_set.NUM_CLASSES_PIXEL
        num_classes_scene = train_set.NUM_CLASSES_SCENE

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_classes_pixel, num_classes_scene

    else:
        raise NotImplementedError

