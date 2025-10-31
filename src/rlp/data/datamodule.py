import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from typing import Callable, Dict, Optional, Tuple


def _build_batch_augment(
    augment_cfg: Optional[Dict[str, object]],
    num_classes: int,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a callable that applies CutMix/MixUp according to configuration.

    Args:
        augment_cfg: Dict with optional keys 'cutmix', 'mixup', 'alpha',
            'cutmix_alpha', 'mixup_alpha'. Missing keys default to enabling both
            transforms with alpha=1.0.
        num_classes: Number of classes for label mixing.

    Returns:
        Callable that accepts (images, labels) and returns augmented batch.
    """
    if augment_cfg is None:
        augment_cfg = {}

    def _is_enabled(key: str, default: bool = True) -> bool:
        flag = augment_cfg.get(key, default)
        if isinstance(flag, str):
            return flag.lower() not in {"0", "false", "no"}
        return bool(flag)

    alpha_default = float(augment_cfg.get("alpha", 1.0))
    transforms = []

    if _is_enabled("cutmix", True):
        cutmix_alpha = float(augment_cfg.get("cutmix_alpha", alpha_default))
        transforms.append(v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha))

    if _is_enabled("mixup", True):
        mixup_alpha = float(augment_cfg.get("mixup_alpha", alpha_default))
        transforms.append(v2.MixUp(num_classes=num_classes, alpha=mixup_alpha))

    if not transforms:

        def _identity(images: torch.Tensor, labels: torch.Tensor):
            return images, labels

        return _identity

    if len(transforms) == 1:
        transform = transforms[0]

        def _single(images: torch.Tensor, labels: torch.Tensor):
            return transform(images, labels)

        return _single

    random_choice = v2.RandomChoice(transforms)

    def _random(images: torch.Tensor, labels: torch.Tensor):
        return random_choice(images, labels)

    return _random


def get_data(
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    num_classes: int = 10,
    augmentations: Optional[Dict[str, object]] = None,
):
    """
    Load CIFAR-10 dataloaders and an optional batch-level augmentation callable.

    Args:
        batch_size: Batch size for dataloaders (default: 128).
        num_workers: Number of workers for data loading (default: 4).
        num_classes: Number of classes (used for CutMix/MixUp one-hot targets).
        augmentations: Optional dict controlling CutMix/MixUp. Keys:
            - cutmix (bool, default True)
            - mixup (bool, default True)
            - alpha / cutmix_alpha / mixup_alpha (floats, default 1.0)

    Returns:
        train_loader: DataLoader for training set.
        test_loader: DataLoader for test set.
        cutmix_or_mixup: Callable applied in the training loop.
    """

    # Basic transforms for training (before CutMix/MixUp)
    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(32, padding=4),
            v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    # Test transforms (no augmentation)
    test_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transforms
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transforms
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    cutmix_or_mixup = _build_batch_augment(augmentations, num_classes)

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
