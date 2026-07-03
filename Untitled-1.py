import torch
import torch.nn as nn
from typing import Dict


# ---------------- Basic blocks ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_factor, stride, activation):
        super().__init__()
        mid = out_channels // bottleneck_factor

        self.conv1 = ConvBlock(in_channels, mid, 1, 1, 0, activation)
        self.conv2 = ConvBlock(mid, mid, 3, stride, 1, activation)
        self.conv3 = ConvBlock(mid, out_channels, 1, 1, 0, nn.Identity)
        self.act = activation()

        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.identity(x)
        return self.act(out)


class ResNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, bottleneck_factor, stride, activation):
        super().__init__()
        blocks = [ResNetBlock(in_channels, out_channels, bottleneck_factor, stride, activation)]
        for _ in range(1, num_blocks):
            blocks.append(ResNetBlock(out_channels, out_channels, bottleneck_factor, 1, activation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ---------------- Main model ----------------
class SmallResNet(nn.Module):
    """
    Small ResNet-like CNN with task-aware heads
    Fully compliant with Project 2 requirements
    """
    def __init__(self, in_channels: int = 1, activation: nn.Module = nn.ReLU):
        super().__init__()

        # -------- Backbone --------
        self.stem = ConvBlock(in_channels, 20, 3, 2, 1, activation)
        self.stage1 = ResNetStage(20, 32, num_blocks=2, bottleneck_factor=4, stride=2, activation=activation)
        self.stage2 = ResNetStage(32, 64, num_blocks=2, bottleneck_factor=4, stride=2, activation=activation)

        # -------- Missing digit head (classification) --------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.missing_digit_head = nn.Linear(64, 10)

        # -------- Sum regression head (normalized) --------
        # Output in [0, 1] → matches (S-3)/(24-3)
        self.sum_head = nn.Sequential(
            nn.Linear(64, 6),
            nn.Sigmoid()
        )

        # -------- Sorted rows / columns (order-aware) --------
        self.row_pool = nn.AdaptiveAvgPool2d((3, 1))   # (B, 64, 3, 1)
        self.col_pool = nn.AdaptiveAvgPool2d((1, 3))   # (B, 64, 1, 3)

        # Simple shared RNN to model ordering
        self.order_rnn = nn.RNN(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.sorted_fc = nn.Linear(32, 3)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # -------- Backbone --------
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)  # (B, 64, H, W)

        # -------- Missing digit --------
        g = self.global_pool(x).flatten(1)
        missing_digit = self.missing_digit_head(g)

        # -------- Sum regression --------
        sum_preds = self.sum_head(g)

        # -------- Sorted rows --------
        rows = self.row_pool(x).squeeze(-1)    # (B, 64, 3)
        rows = rows.permute(0, 2, 1)           # (B, 3, 64)
        row_out, _ = self.order_rnn(rows)      # (B, 3, 32)
        row_logits = self.sorted_fc(row_out)   # (B, 3, 3)

        # -------- Sorted columns --------
        cols = self.col_pool(x).squeeze(-2)    # (B, 64, 3)
        cols = cols.permute(0, 2, 1)           # (B, 3, 64)
        col_out, _ = self.order_rnn(cols)      # (B, 3, 32)
        col_logits = self.sorted_fc(col_out)   # (B, 3, 3)

        sorted_logits = torch.cat([row_logits, col_logits], dim=1)  # (B, 6, 3)

        return {
            "missing_digit": missing_digit,
            "sorted_labels": sorted_logits,
            "sum_labels": sum_preds
        }


# ---------------- Utility ----------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SmallResNet()
    x = torch.randn(1, 1, 84, 84)
    out = model(x)

    print("Missing:", out["missing_digit"].shape)
    print("Sorted:", out["sorted_labels"].shape)
    print("Sum:", out["sum_labels"].shape)
    print("Params:", count_parameters(model))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from data import get_datasets
import time
import matplotlib.pyplot as plt

# ---------------- Reproducibility ----------------
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

# ---------------- Hyperparameters ----------------
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Data ----------------
train_ds, val_ds, test_ds = get_datasets("./data")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model ----------------
model = SmallResNet().to(DEVICE)

# ---------------- Losses ----------------
loss_missing = nn.CrossEntropyLoss()
loss_sorted = nn.CrossEntropyLoss()
loss_sum = nn.L1Loss()  # MAE

# ---------------- Optimizer ----------------
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- Metrics helpers ----------------
def accuracy_missing(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()


def accuracy_sorted(pred, target):
    # pred: (B, 6, 3), target: (B, 6)
    pred_cls = pred.argmax(dim=2)
    return (pred_cls == target).float().mean().item()


# ---------------- Training history ----------------
history = {
    "train_loss": [], "val_loss": [],
    "train_acc_missing": [], "val_acc_missing": [],
    "train_acc_sorted": [], "val_acc_sorted": [],
    "train_mae_sum": [], "val_mae_sum": []
}

# Using a heuristic seed for gradient stability, per project side-note.
# ---------------- Training loop ----------------
for epoch in range(EPOCHS):
    start = time.time()
    model.train()

    train_loss = 0
    acc_missing = 0
    acc_sorted = 0
    mae_sum = 0

    for batch in train_loader:
        img = batch['image'].to(DEVICE)
        y_missing = batch['missing_digit'].to(DEVICE)
        y_sorted = batch['sorted_labels'].to(DEVICE)
        y_sum = batch['sum_labels'].to(DEVICE)

        optimizer.zero_grad()
        out = model(img)

        l1 = loss_missing(out['missing_digit'], y_missing)
        l2 = loss_sorted(out['sorted_labels'].view(-1, 3), y_sorted.view(-1))
        l3 = loss_sum(out['sum_labels'], y_sum)

        loss = l1 + l2 + l3
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        acc_missing += accuracy_missing(out['missing_digit'], y_missing)
        acc_sorted += accuracy_sorted(out['sorted_labels'], y_sorted)
        mae_sum += l3.item()

    n = len(train_loader)
    history['train_loss'].append(train_loss / n)
    history['train_acc_missing'].append(acc_missing / n)
    history['train_acc_sorted'].append(acc_sorted / n)
    history['train_mae_sum'].append(mae_sum / n)

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0
    acc_missing = 0
    acc_sorted = 0
    mae_sum = 0

    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(DEVICE)
            y_missing = batch['missing_digit'].to(DEVICE)
            y_sorted = batch['sorted_labels'].to(DEVICE)
            y_sum = batch['sum_labels'].to(DEVICE)

            out = model(img)

            l1 = loss_missing(out['missing_digit'], y_missing)
            l2 = loss_sorted(out['sorted_labels'].view(-1, 3), y_sorted.view(-1))
            l3 = loss_sum(out['sum_labels'], y_sum)

            loss = l1 + l2 + l3

            val_loss += loss.item()
            acc_missing += accuracy_missing(out['missing_digit'], y_missing)
            acc_sorted += accuracy_sorted(out['sorted_labels'], y_sorted)
            mae_sum += l3.item()

    n = len(val_loader)
    history['val_loss'].append(val_loss / n)
    history['val_acc_missing'].append(acc_missing / n)
    history['val_acc_sorted'].append(acc_sorted / n)
    history['val_mae_sum'].append(mae_sum / n)

    print(f"Epoch[{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {history['train_loss'][-1]:.4f} | "
          f"Val Loss: {history['val_loss'][-1]:.4f} | "
          f"Val Acc Missing: {history['val_acc_missing'][-1]*100:.2f}% | "
          f"Val Acc Sorted: {history['val_acc_sorted'][-1]*100:.2f}% | "
          f"Val MAE Sum: {history['val_mae_sum'][-1]:.4f} | "
          f"Time: {time.time() - start:.1f}s")

# ---------------- Test evaluation ----------------
model.eval()
acc_missing = 0
acc_sorted = 0
mae_sum = 0

with torch.no_grad():
    for batch in test_loader:
        img = batch['image'].to(DEVICE)
        y_missing = batch['missing_digit'].to(DEVICE)
        y_sorted = batch['sorted_labels'].to(DEVICE)
        y_sum = batch['sum_labels'].to(DEVICE)

        out = model(img)
        acc_missing += accuracy_missing(out['missing_digit'], y_missing)
        acc_sorted += accuracy_sorted(out['sorted_labels'], y_sorted)
        mae_sum += loss_sum(out['sum_labels'], y_sum).item()

n = len(test_loader)
print("\n---- Test Results ----")
print(f"Missing digit accuracy: {acc_missing / n * 100:.2f}%")
print(f"Sorted rows/cols accuracy: {acc_sorted / n * 100:.2f}%")
print(f"Sum MAE: {mae_sum / n:.4f}")

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 10))

# ===============================
# 1️⃣ Total Loss
# ===============================
ax = plt.subplot(2, 2, 1)
ax.plot(epochs, history['train_loss'], 'o-', label='Train Total Loss')
ax.plot(epochs, history['val_loss'], 's-', label='Val Total Loss')
ax.axhline(y=(mae_sum / n), linestyle='--', label=f'Test Loss: {mae_sum/n:.4f}')
ax.set_title("Total Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)

# ===============================
# 2️⃣ Missing Digit Task
# ===============================
ax1 = plt.subplot(2, 2, 2)
ax2 = ax1.twinx()

ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss')
ax1.set_ylabel("Loss")

ax2.plot(epochs, history['train_acc_missing'], '^-', label='Train Acc')
ax2.plot(epochs, history['val_acc_missing'], 'D-', label='Val Acc')
ax2.axhline(y=(acc_missing / n), linestyle='--', label=f'Test Acc: {acc_missing/n:.4f}')
ax2.set_ylabel("Accuracy")

ax1.set_title("Missing Digit Task")
ax1.set_xlabel("Epoch")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(True)

# ===============================
# 3️⃣ Sorted Order Task
# ===============================
ax1 = plt.subplot(2, 2, 3)
ax2 = ax1.twinx()

ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss')
ax1.set_ylabel("Loss")

ax2.plot(epochs, history['train_acc_sorted'], '^-', label='Train Acc')
ax2.plot(epochs, history['val_acc_sorted'], 'D-', label='Val Acc')
ax2.axhline(y=(acc_sorted / n), linestyle='--', label=f'Test Acc: {acc_sorted/n:.4f}')
ax2.set_ylabel("Accuracy")

ax1.set_title("Sort Order Task")
ax1.set_xlabel("Epoch")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(True)

# ===============================
# 4️⃣ Sum Prediction Task
# ===============================
ax1 = plt.subplot(2, 2, 4)
ax2 = ax1.twinx()

ax1.plot(epochs, history['train_mae_sum'], 'o-', label='Train MAE')
ax1.plot(epochs, history['val_mae_sum'], 's-', label='Val MAE')
ax1.axhline(y=(mae_sum / n), linestyle='--', label=f'Test MAE: {mae_sum/n:.4f}')
ax1.set_ylabel("MAE")

ax2.plot(epochs, history['train_loss'], '^-', alpha=0.0)  # dummy for scale consistency
ax2.set_ylabel("")

ax1.set_title("Sum Prediction Task")
ax1.set_xlabel("Epoch")

ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.savefig("all_metrics.png", dpi=150)
plt.show()
