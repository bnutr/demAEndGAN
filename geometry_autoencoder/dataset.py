"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    encoding_only = args.encoding_only
    if not encoding_only:
        ratio_train = args.ratio_train
        ratio_val = args.ratio_val

    print('image size: ', image_size)
    assert image_size == 128, 'currently only image size of 128 is supported'

    dataset_paths = {
        'celeba': 'CelebA',
        'bldgview_top': 'bldgview_top',
        'bldgview_allparallel': 'bldgview_allparallel',
        'bldgview_ne': 'bldgview_ne',
        'bldgview_se': 'bldgview_se',
        'bldgview_sw': 'bldgview_sw',
        'bldgview_nw': 'bldgview_nw',
        'traverse_nw': 'traverse_nw',
        'traverse_se': 'traverse_se',
        'traverse_top': 'traverse_top',
    }

    if name.lower() in dataset_paths:
        dataset_name = dataset_paths[name.lower()]
        root_path = os.path.join(dset_dir, dataset_name)

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        
        data_kwargs = {'root': root_path, 'transform': transform}
        
        dset = CustomImageFolder
    else:
        raise NotImplementedError

    data = dset(**data_kwargs)

    # Calculate lengths for train, val, test splits
    if encoding_only:
        print(f'****loading data for encoding now****')
        total_size = len(data)
        data = DataLoader(data,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          drop_last=True)
        return data
    else:
        print(f'****loading Train, Validation and Test data now****')
        total_size = len(data)
        train_size = int(ratio_train * total_size)
        val_size = int(ratio_val * total_size)
        test_size = total_size - train_size - val_size

        rand_generator = torch.Generator().manual_seed(26)
        train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size], rand_generator)

        train_loader = DataLoader(train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=True)

        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)

        test_loader = DataLoader(test_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=True)

        return train_loader, val_loader, test_loader
