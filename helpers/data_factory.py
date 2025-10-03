import os
import torch
import numpy as np
import torchvision
from pathlib import Path
# The key import for the new transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import urllib.request
import zipfile
import shutil

def get_data(dataset_name: str):
    """
    Creates dataloaders using the modern torchvision.transforms.v2 API.
    Batch-level augmentations (like CutMix) should be applied in the training loop.
    """
    DATA_ROOT = Path(__file__).parent.parent / "data"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    BATCH_SIZES = {'cifar10': 128, 'cifar100': 128, 'tiny-imagenet': 64}
    #DATA_ROOT = '../data'
    NUM_WORKERS = 4
    PIN_MEMORY = True

    MEANS = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'tiny-imagenet': (0.485, 0.456, 0.406)
    }
    STDS = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'tiny-imagenet': (0.229, 0.224, 0.225)
    }

    if dataset_name not in BATCH_SIZES:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    batch_size = BATCH_SIZES[dataset_name]
    mean = MEANS[dataset_name]
    std = STDS[dataset_name]
    os.makedirs(DATA_ROOT, exist_ok=True)

    # --- v2 Transforms for individual images ---
    # These are simpler now. We don't need TrivialAugment or RandomErasing here
    # because v2's CutMix/MixUp are more powerful replacements.
    transform_train = v2.Compose([
        v2.RandomCrop(32 if 'cifar' in dataset_name else 64, padding=4),
        v2.RandomHorizontalFlip(),
        # v2 replaces ToTensor with these two calls
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    transform_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    if dataset_name in ['cifar10', 'cifar100']:
        dataset_class = torchvision.datasets.CIFAR10 if dataset_name == 'cifar10' else torchvision.datasets.CIFAR100
        
        train_set = dataset_class(root=DATA_ROOT, train=True, download=True, transform=transform_train)
        test_set = dataset_class(root=DATA_ROOT, train=False, download=True, transform=transform_test)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        valid_loader = test_loader
        
        return train_loader, valid_loader, test_loader

    elif dataset_name == 'tiny-imagenet':
        # ... (Tiny ImageNet setup code remains the same)
        tiny_imagenet_path = os.path.join(DATA_ROOT, 'tiny-imagenet-200')
        if not os.path.isdir(tiny_imagenet_path):
            print("Downloading and setting up Tiny ImageNet...")
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            zip_path = os.path.join(DATA_ROOT, 'tiny-imagenet-200.zip')
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(DATA_ROOT)
            os.remove(zip_path)
            val_dir = os.path.join(tiny_imagenet_path, 'val')
            val_restructured_dir = os.path.join(tiny_imagenet_path, 'val_restructured')
            os.makedirs(val_restructured_dir, exist_ok=True)
            with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t'); img_name, class_id = parts[0], parts[1]
                    class_dir = os.path.join(val_restructured_dir, class_id)
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copyfile(os.path.join(val_dir, 'images', img_name), os.path.join(class_dir, img_name))
            print("Tiny ImageNet setup complete.")
        
        train_path = os.path.join(tiny_imagenet_path, 'train')
        val_path = os.path.join(tiny_imagenet_path, 'val_restructured')

        train_set = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        valid_set = torchvision.datasets.ImageFolder(root=val_path, transform=transform_test)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        test_loader = valid_loader

        return train_loader, valid_loader, test_loader