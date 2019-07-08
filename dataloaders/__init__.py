from dataloaders.datasets import bdd100k
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'bdd100k':
        train_set = bdd100k.BDDSegmentation(args, split='train')
        val_set = bdd100k.BDDSegmentation(args, split='val')
    
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

