import multiprocessing
import open_clip
import torch
import torch.nn as nn
from mmaction.models import build_backbone
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config import config
from dataset import NiiDataset, split_dataset
from model import swin_transformer_3d, resnet_3d_50, csn_r101, slow_r50, r2plus1d_r50, x3d, slowfast_r50, mvit, \
    slowfast_r101, vclip
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.manual_seed(config["seed"])

# 根据任务设置类别数量
if config["task"] == "binary":
    num_classes = 2
elif config["task"] == "multiclass":
    num_classes = 4
else:
    raise ValueError(f"Invalid task: {config['task']}")

model = swin_transformer_3d()
model.to(device)

# Create a GradScaler instance for mixed precision training
scaler = GradScaler()

# 根据任务选择数据集
folders = ["CT-0", "CT-1"]
train_folders, val_folders = split_dataset(folders, train_ratio=config["train_ratio"], seed=config["seed"],
                                           data_root="./data")
print("Train folders:", train_folders)
print("Validation folders:", val_folders)
train_dataset = NiiDataset(train_folders, transform=None, torchio_transform=True, split='train',
                           task=config["task"])
val_dataset = NiiDataset(val_folders, transform=None, split='val', torchio_transform=False, task=config["task"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                          num_workers=config["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

all_labels = train_dataset.get_all_labels()
num_positive_samples = sum(label == 1 for label in all_labels)
num_negative_samples = sum(label == 0 for label in all_labels)
print(f"Number of positive samples: {num_positive_samples}")
print(f"Number of negative samples: {num_negative_samples}")
pos_weight = torch.tensor([num_negative_samples / num_positive_samples], device=device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * 0.1)

for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []

    for data, labels in tqdm(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            logits = model(data)
            loss = loss_func(logits, labels)

        # Backward pass and weight update with automatic gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1).cpu()
        all_preds.extend(preds.tolist())
        labels_list = labels.cpu().tolist()
        all_labels.extend(labels_list)

    train_loss /= len(train_loader)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}/{config['epochs']} - Train loss: {train_loss:.4f}, F1: {f1:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            data, labels = data.to(device), labels.to(device)

            # Forward pass with mixed precision with autocast():
            with autocast():
                logits = model(data)
                loss = loss_func(logits, labels)

            val_loss += loss.item()
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1).cpu()

            all_preds.extend(preds.tolist())

            labels_list = labels.cpu().tolist()
            all_labels.extend(labels_list)

    val_loss /= len(val_loader)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}/{config['epochs']} - Val loss: {val_loss:.4f}, F1: {f1:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
    # Update the learning rate
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print("Current learning rate:", current_lr)

    # Save the previous epoch's weights
    torch.save(model.state_dict(), f"./data/modelckpt/previous_epoch_weight.pt")

    # Save the weights with the minimum validation loss
    if epoch == 0 or val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), f"./data/modelckpt/min_loss_weight.pt")

    # Save the weights with the maximum F1 score
    if epoch == 0 or f1 > max_f1_score:
        max_f1_score = f1
        torch.save(model.state_dict(), f"./data/modelckpt/max_f1_score_weight.pt")
