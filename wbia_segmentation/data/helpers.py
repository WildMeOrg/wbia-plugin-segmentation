from data.transforms import build_train_val_transforms
from data.dataset import SegDataset

from torch.utils.data import DataLoader

def get_data_loaders(args):
    train_transforms, test_transforms = build_train_val_transforms(args)

    train_set = SegDataset(args.data.train_dir, args, train_transforms)
    n_train = len(train_set)
    val_set = SegDataset(args.data.val_dir, args, test_transforms)

    # 2. Create data loaders
    loader_args = dict(batch_size=args.train.batch_size, num_workers=args.data.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    return train_loader, n_train, val_loader, val_set

def get_test_data_loader(args):
    _, test_transforms = build_train_val_transforms(args)

    test_set = SegDataset(args.data.test_dir, args, test_transforms)

    # 2. Create data loaders
    loader_args = dict(batch_size=args.train.batch_size, num_workers=args.data.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    return test_loader