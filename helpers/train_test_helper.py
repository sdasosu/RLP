import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from .data_factory import get_data
from torchvision.transforms import v2


#======================== Basic Setups =========================

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#========================================================================


#======================== Set of Hyperparameters ========================
hyperparam = {
    'EPOCHS': 10,
    'LR': 0.01,
    'WEIGHT_DECAY': 5e-4,
    'MOMENTUM': 0.9
                }
#========================================================================

#----------------------- Accuracy Helper ------------------------
def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)

    if targets.ndim == 1:  # Hard targets (e.g., validation/test)
        correct = (preds == targets).sum().item()
    else:  # Soft targets (CutMix/MixUp) - use the class with the highest probability
        _, hard_targets = torch.max(targets, dim=1)
        correct = (preds == hard_targets).sum().item()

    return (correct / targets.size(0)) * 100
#-----------------------------------------------------------------


#---------------------- Append Result Helper ---------------------
def append_result(result_path: str, content: str):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'a') as f:
        f.write(content + "\n")

#----------------------------------------------------------------
# 
#        

#========================================================================

def train_test(model: nn.Module, dataset: str, checkpoint_path: str, output_path: str, epochs=hyperparam['EPOCHS']):
    print(f"--- Starting Training for {dataset} ---")
    model = model.to(device)
    num_classes = {'cifar10': 10, 'cifar100': 100, 'tiny-imagenet': 200}[dataset]
    augmentations = v2.RandomChoice([ v2.CutMix(num_classes=num_classes, alpha=1.0), v2.MixUp(num_classes=num_classes, alpha=0.8) ])


    train_loader, val_loader, test_loader = get_data(dataset)
    optimizer = optim.SGD(model.parameters(), lr=hyperparam['LR'], momentum=hyperparam['MOMENTUM'], weight_decay=hyperparam['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
   
   
    def mixup_criterion(preds, targets):
        return -(targets * torch.log_softmax(preds, dim=-1)).sum(dim=1).mean()
    
#=======================================================================
# 
    

    # 3.===================== Training Loop =========================
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(epochs):
       
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply Batch Augmentation (CutMix/MixUp)
            inputs, targets = augmentations(inputs, targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            running_correct += (calculate_accuracy(outputs, targets) * inputs.size(0))
            total_samples += inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_correct / total_samples


        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        total_val_samples = 0
        

        # Standard CrossEntropyLoss for validation (hard labels)
        val_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
             val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
             for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = val_criterion(outputs, targets)

                running_val_loss += loss.item()
                running_val_correct += (calculate_accuracy(outputs, targets) * inputs.size(0))
                total_val_samples += inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = running_val_correct / total_val_samples

        # ============= Print Epoch Summary ==============
        print(f"Epo {epoch+1:02d}: TrL={epoch_train_loss:.4f}, TrA={epoch_train_acc:.2f}% | "
              f"VL={epoch_val_loss:.4f}, VA={epoch_val_acc:.2f}%", end=" ")

        # ============== Checkpoint Saving ===============
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print("|| Checkpoint Saved!")
        else:
            print("")

        scheduler.step()

    # ============== Append Best Validation Result ====================
    append_result(output_path, f"--- Training Complete ---")
    append_result(output_path, f"Best Validation Accuracy: {best_val_acc:.2f}%")





    # ============== 4. Final Sanity Check (Test Set) ================
    print("\nRunning final sanity check on test set...")

    # ============== Load the best model ================
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    running_test_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        # =========== No tqdm for the final quick check as requested ============
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            running_test_correct += (calculate_accuracy(outputs, targets) * inputs.size(0))
            total_test_samples += inputs.size(0)

    sanity_check_acc = running_test_correct / total_test_samples

    # ==============  Print and Append Test Results ================
    print(f"Sanity Check (Test) Accuracy: {sanity_check_acc:.2f}%")

    content = (
        f"\n# -------------------------------------------- #\n"
        f"# Final Test Accuracy (Sanity Check): {sanity_check_acc:.2f}%\n"
        f"# -------------------------------------------- #\n"
    )
    append_result(output_path, content)

    # ===============  Closing Statement =======================
    model_name = model.__class__.__name__
    print(f"\n{model_name} trained on {dataset}. Checkpoint saved. Test Acc: {sanity_check_acc:.2f}%\n")

if __name__ == '__main__':
    # 
    # class SimpleNet(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.fc = nn.Linear(3072, 10)
    #     def forward(self, x):
    #         return self.fc(torch.flatten(x, 1))
    
    # model = SimpleNet()
    # train_test(model, 'cifar10', 'checkpoints/test.pth', 'results/test.txt')
    pass
#============================================================