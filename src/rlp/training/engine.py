import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Tuple


# =========================
# Utilities
# =========================
def _infer_dataset_name(loader) -> str:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return "data"
    return getattr(ds, "name", None) or ds.__class__.__name__


def _resolve_ckpt_path(state_dict_path: str, model: nn.Module, dataname: str) -> str:
    """
    If `state_dict_path` ends with .pt/.pth -> treat as exact filename.
    Otherwise treat it as a directory and create `<ModelName>_<dataname>.pt` in it.
    """
    os.makedirs(
        state_dict_path
        if not state_dict_path.endswith((".pt", ".pth"))
        else os.path.dirname(state_dict_path) or ".",
        exist_ok=True,
    )

    if state_dict_path.endswith((".pt", ".pth")):
        return state_dict_path
    filename = f"{model.__class__.__name__}_{dataname}.pt"
    return os.path.join(state_dict_path, filename)


def _soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy that supports soft/one-hot targets.
    If `targets` is integer class indices, this reduces to standard CE.
    """
    if targets.dim() == 1 or targets.size(-1) == 1:  # class indices
        return nn.CrossEntropyLoss()(logits, targets.view(-1))
    # soft/one-hot
    log_probs = logits.log_softmax(dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


# =========================
# One-epoch helpers
# =========================
def train_epoch(model, train_loader, cutmix_or_mixup, optimizer, device) -> float:
    """
    Train for one epoch. Returns training accuracy (top-1).
    Handles both hard labels and one-hot/mixed labels for loss & accuracy.
    """
    model.train()
    correct, total = 0, 0
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Apply CutMix/MixUp (should be a no-op if you pass an identity fn)
        images, labels = cutmix_or_mixup(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = _soft_ce_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accuracy (argmax for soft labels)
        if labels.dim() > 1:  # one-hot / soft
            labels_idx = labels.argmax(dim=1)
        else:
            labels_idx = labels

        _, predicted = outputs.max(1)
        total += labels_idx.size(0)
        correct += predicted.eq(labels_idx).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )
    return 100.0 * correct / total


def test_epoch(model, test_loader, device) -> float:
    """Evaluate top-1 accuracy on test set."""
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({"acc": f"{100.0 * correct / total:.2f}%"})
    return 100.0 * correct / total


# =========================
# Public callables
# =========================
def _test(
    model: nn.Module,
    test_loader,
    device: str = "cuda",
    state_dict_path: Optional[str] = None,
) -> float:
    """
    Standalone test entrypoint.

    Args:
        model: PyTorch model
        test_loader: DataLoader
        device: 'cuda' or 'cpu'
        state_dict_path: optional checkpoint to load before testing

    Returns:
        test_acc (float)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if state_dict_path:
        ckpt = torch.load(state_dict_path, map_location=device)
        # Accept either a plain state_dict or a dict with 'state_dict' key
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=True)

    test_acc = test_epoch(model, test_loader, device)
    return test_acc


def _training(
    model: nn.Module,
    train_loader,
    test_loader,
    cutmix_or_mixup,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
    state_dict_path: str = "./",
    dataname: Optional[str] = None,
) -> Tuple[nn.Module, float]:
    """
    Training entrypoint that also evaluates and saves the BEST checkpoint.

    Behavior:
      - Trains for `epochs`.
      - After each epoch, evaluates on test set.
      - Saves best (highest test acc) weights to `<ModelName>_<DataName>.pt`
        (or to the exact filename if `state_dict_path` ends with .pt/.pth).
      - After training finishes, calls `_test()` internally (loading the best
        checkpoint) and returns `(model, test_acc)`.

    Args:
        model: PyTorch model
        train_loader: DataLoader
        test_loader: DataLoader
        cutmix_or_mixup: augmentation transform (can be identity)
        epochs: number of epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
        state_dict_path: directory OR full file path where to save best checkpoint
        dataname: optional dataset name to use in filename

    Returns:
        (trained_model, test_acc_from_best_checkpoint)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if dataname is None:
        dataname = _infer_dataset_name(train_loader)

    best_ckpt_path = _resolve_ckpt_path(state_dict_path, model, dataname)

    criterion = nn.CrossEntropyLoss()  # kept for compatibility; loss handled in helper
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training on {device}")
    print(f"Total epochs: {epochs}")
    print(f"Best checkpoint will be saved to: {best_ckpt_path}\n")

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, cutmix_or_mixup, optimizer, device)
        test_acc = test_epoch(model, test_loader, device)
        scheduler.step()

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_test_acc": best_acc,
                },
                best_ckpt_path,
            )

        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%"
        )

    print("\nTraining completed!")

    # Load best and run a clean test via _test (as required)
    final_test_acc = _test(
        model, test_loader, device=device, state_dict_path=best_ckpt_path
    )
    print(f"Best checkpoint test acc: {final_test_acc:.2f}%")

    return model, final_test_acc


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from rlp.models.base import TinyModel
    from rlp.data.datamodule import get_data

    train_loader, test_loader, cutmix_or_mixup = get_data(batch_size=128)
    model = TinyModel(in_channels=3, num_classes=10)

    # Train + auto-save best to ./TinyModel_<DataName>.pt
    model, test_acc = _training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cutmix_or_mixup=cutmix_or_mixup,
        epochs=10,
        lr=0.001,
        device="cuda",
        state_dict_path="./",  # can also pass a full path like "./ckpts/best.pt"
    )

    # Or: test later from a path (standalone)
    # acc = _test(model, test_loader, device="cuda", state_dict_path="./TinyModel_CIFAR10.pt")
