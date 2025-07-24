import os
import copy
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC
from layer import FocalLoss 
import time



def train(model, criterion, optimizer, scheduler, train_dataset, k_folds=8, patience=5, num_epochs=15, save_dir='checkpoints_MobileNetv2_xView2', batch_size=32, device='cuda'):
    NUM_CLASSES = 4
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    all_metrics = []
    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, drop_last=True, persistent_workers=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

        accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average='macro').to(device)
        precision = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(device)
        recall = MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(device)
        f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)
        auc = MulticlassAUROC(num_classes=NUM_CLASSES, average='macro').to(device)

        fold_metrics = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        early_stopping_flag = False
        best_model_wts = None

        fold_checkpoint_dir = os.path.join(save_dir, f"fold_{fold + 1}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            if early_stopping_flag:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            fold_model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = fold_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

                accuracy.update(outputs, labels)
                precision.update(outputs, labels)
                recall.update(outputs, labels)
                f1.update(outputs, labels)
                auc.update(outputs, labels)

            avg_train_loss = running_loss / len(train_loader)
            train_acc = accuracy.compute()
            train_prec = precision.compute()
            train_recall = recall.compute()
            train_f1 = f1.compute()
            train_auc = auc.compute()

            accuracy.reset(), precision.reset(), recall.reset(), f1.reset(), auc.reset()

            fold_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = fold_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    accuracy.update(outputs, labels)
                    precision.update(outputs, labels)
                    recall.update(outputs, labels)
                    f1.update(outputs, labels)
                    auc.update(outputs, labels)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy.compute()
            val_prec = precision.compute()
            val_recall = recall.compute()
            val_f1 = f1.compute()
            val_auc = auc.compute()
            accuracy.reset(), precision.reset(), recall.reset(), f1.reset(), auc.reset()
            scheduler.step(avg_val_loss)

            print(f"\nFold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

            epoch_metrics = {
                'fold': fold + 1, 'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_acc.item(), 'train_precision': train_prec.item(),
                'train_recall': train_recall.item(), 'train_f1': train_f1.item(), 'train_auc': train_auc.item(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_acc.item(), 'val_precision': val_prec.item(),
                'val_recall': val_recall.item(), 'val_f1': val_f1.item(), 'val_auc': val_auc.item()
            }

            fold_metrics.append(epoch_metrics)
            all_metrics.append(epoch_metrics)

            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_wts = copy.deepcopy(fold_model.state_dict())
                torch.save(best_model_wts, os.path.join(fold_checkpoint_dir, 'best_model.pth'))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    early_stopping_flag = True

        pd.DataFrame(fold_metrics).to_csv(os.path.join(fold_checkpoint_dir, f"fold_{fold + 1}_metrics.csv"), index=False)
        fold_model.load_state_dict(best_model_wts)
    end = time.time()
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir, 'kfold_metrics.csv'), index=False)
    print(f"Training time: ~{(end-start_time)/3600:.2f}")
    return model, pd.DataFrame(all_metrics)
if __name__ == '__main__':
    DATA_DIR = './postdisaster-patches'
    IMG_SIZE = 128
    BATCH_SIZE = 32
    NUM_CLASSES = 4
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    class_counts = [141372, 128694, 154344, 167573]
    total = sum(class_counts)
    weights = [total / c for c in class_counts]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights * (len(class_counts) / weights.sum())

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    torch.backends.cudnn.benchmark = True
    criterion = FocalLoss(alpha=weights.to(DEVICE), gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    torch.cuda.empty_cache()

    model, all_fold_metrics_df = train(model, criterion, optimizer, scheduler, train_dataset=dataset,device = DEVICE)
    torch.save(model.state_dict(), "mobileNetV2_xview2.pth")
    print("Model trained and saved.")
    print(summary(model,(3,128,128)))
