import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

def get_data(batch_size=128, num_workers=4):
    """
    Load CIFAR-10 dataset with CutMix and MixUp augmentations.
    
    Args:
        batch_size: Batch size for dataloaders (default: 128)
        num_workers: Number of workers for data loading (default: 4)
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    
    # Basic transforms for training (before CutMix/MixUp)
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(32, padding=4),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                     std=[0.2470, 0.2435, 0.2616])
    ])
    
    # Test transforms (no augmentation)
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                     std=[0.2470, 0.2435, 0.2616])
    ])
    
    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transforms
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transforms
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # CutMix and MixUp transforms (applied to batches)
    cutmix = v2.CutMix(num_classes=10)
    mixup = v2.MixUp(num_classes=10)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    return train_loader, test_loader, cutmix_or_mixup


def collate_fn_with_cutmix_mixup(batch, cutmix_or_mixup):
    """
    Custom collate function to apply CutMix/MixUp to batches.
    
    Args:
        batch: Batch of data from dataloader
        cutmix_or_mixup: Transform to apply
    
    Returns:
        Augmented images and labels
    """
    images, labels = torch.utils.data.default_collate(batch)
    images, labels = cutmix_or_mixup(images, labels)
    return images, labels


# Example usage
if __name__ == "__main__":
    # Get dataloaders
    train_loader, test_loader, cutmix_or_mixup = get_data(batch_size=128)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of batches (train): {len(train_loader)}")
    print(f"Number of batches (test): {len(test_loader)}")
    
    # Example: iterate through one batch with augmentation
    images, labels = next(iter(train_loader))
    print(f"\nOriginal batch - Images: {images.shape}, Labels: {labels.shape}")
    
    # Apply CutMix/MixUp
    images_aug, labels_aug = cutmix_or_mixup(images, labels)
    print(f"Augmented batch - Images: {images_aug.shape}, Labels: {labels_aug.shape}")
    print("Note: Labels are now one-hot encoded for mixed samples")