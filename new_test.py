import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import  transforms
from tqdm import tqdm # ,datasets
from get_NN_dataset import CustomDataset, generate_dataset

# ---------- Data ----------
class ToBinaryTensor:
    def __call__(self, img):
        t = transforms.functional.to_tensor(img)      # [0,1]
        return torch.where(t > 0.5, 1.0, -1.0)        # {-1, +1}


filename = "10_Million_samples_LP_DEG_SAC.pt"
print("Loading dataset: ", filename)
dataset = torch.load(filename, weights_only=False, map_location="cpu")


dataset.data = torch.stack(dataset.data)  # Convert list of tensors to a single tensor
dataset.labels = torch.stack(dataset.labels)  # Convert list of tensors to a single tensor

#dataset.labels = np.delete(dataset.labels, 8, axis=1)  # Remove the first column if it is not needed
# dataset.labels = np.delete(dataset.labels, 7, axis=1)  # Remove the first column if it is not needed
# dataset.labels = np.delete(dataset.labels, 6, axis=1)  # Remove the first column if it is not needed
# dataset.labels = np.delete(dataset.labels, 5, axis=1)  # Remove the first column if it is not needed
# dataset.labels = np.delete(dataset.labels, 4, axis=1)  # Remove the first column if it is not needed
#dataset.labels = np.delete(dataset.labels, 3, axis=1)  # Remove the first column if it is not needed
#dataset.labels = np.delete(dataset.labels, 2, axis=1)  # Remove the first column if it is not needed
#dataset.labels = np.delete(dataset.labels, 1, axis=1)  # Remove the first column if it is not needed

transform = transforms.Compose([ToBinaryTensor()])
# Set your desired train-test ratio (e.g., 0.8 for 80% train, 20% test)
train_test_ratio = 0.8
print()
# Calculate sizes
total_size = len(dataset)  # Assuming dataset.data is a tensor of shape (N, C, H, W)
train_size = int(total_size * train_test_ratio)
test_size = total_size - train_size
print(f"Total samples: {total_size}, Train samples: {train_size}, Test samples: {test_size}")
# Split the dataset
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=256)

bits = dataset.labels.shape[1]

# ---------- Model ----------
class BinaryMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.inl = nn.Linear(32, hidden_dim) # hidden layer
        self.activation1 = nn.ReLU()
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.activation2 = nn.ReLU()
        self.outl = nn.Linear(hidden_dim, bits)   # output

    def forward(self, x):
        if self.training:
          temp = self.flatten(x)
          temp = self.inl(temp)
          temp = self.activation1(temp)
        #   temp = self.hidden1(temp)
        #   temp = self.activation2(temp)
          return self.outl(temp)
        else:
          temp = self.flatten(x)
          temp = torch.where(temp > 0, 1.0, -1.0)
          temp = self.inl(temp)
          temp = self.activation1(temp)
        #   temp = self.hidden1(temp)
        #   temp = self.activation2(temp)
          temp = torch.where(temp > 0, 1.0, -1.0)
          return self.outl(temp)

# model = BinaryMLP(hidden_dim=256).to("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model ----------
class BinaryCNN(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        # self.act1 = nn.ReLU()
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(hidden_dim * 32, bits)  # hidden_dim channels * 32 length
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32, bits)  # 32 channels * 32 length

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, 32]
        if self.training:
            temp = self.act1(self.conv1(x))
            temp = self.act2(self.conv2(temp))
            temp = self.flatten(temp)
            return self.fc(temp)
        else:
            temp = self.act1(self.conv1(x))
            temp = torch.where(temp > 0, 1.0, -1.0)
            temp = self.act2(self.conv2(temp))
            temp = torch.where(temp > 0, 1.0, -1.0)
            temp = self.flatten(temp)
            return self.fc(temp)

model = BinaryCNN(hidden_dim=256).to("cuda" if torch.cuda.is_available() else "cpu")
# model = BinaryMLP(hidden_dim=256).to("cpu")


error_distribution = torch.tensor([0]*bits)

# ---------- Training loop ----------
class BinaryLogisticLoss(torch.nn.Module):
    def forward(self, logits, targets):
        # targets: -1 or 1
        return torch.nn.functional.softplus(-logits * targets).mean()
    
loss_fn   = BinaryLogisticLoss()  # works on raw logits
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=0)
device    = next(model.parameters()).device

def accuracy(loader):
    global error_distribution
    correct = total = 0
    for x, y in loader:
        logits = model(x.to(device))
        # Predict: 1 if logits > 0, else -1
        pred = torch.where(logits.cpu() > 0, 1.0, -1.0)
        correct += (pred == y.cpu()).sum().item()
        error_distribution += (pred != y.cpu()).sum(dim=0)
        total += y.numel()
    return correct / total



epochs = 100
5
losses = []
train_accs = []
test_accs = []

for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
    error_distribution = torch.tensor([0]*bits)
    model.train()
    running_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)

    train_acc = accuracy(train_dl)
    model.eval()
    with torch.no_grad():
      test_acc  = accuracy(test_dl)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    tqdm.write(f"Epoch {epoch}: "
          f"loss={running_loss/len(train_ds):.4f}  "
          f"train_acc={train_acc:.3%}  test_acc={test_acc:.3%}")
    losses.append(running_loss / len(train_ds))

import matplotlib.pyplot as plt

# Plot loss and accuracy rates on two subplots in one figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
epochs_range = range(1, epochs + 1)

# Loss subplot
ax1.plot(epochs_range, losses, label='Training Loss', color='tab:blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss over Epochs')
ax1.legend()
ax1.grid(True)

# Accuracy subplot
ax2.plot(epochs_range, train_accs, label='Train Accuracy', color='tab:green')
ax2.plot(epochs_range, test_accs, label='Test Accuracy', color='tab:orange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy over Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# plt.figure(figsize=(8, 5))
# plt.bar(range(len(error_distribution)), error_distribution.cpu().numpy(), color='tab:red')
# plt.xlabel('Output Bit Index')
# plt.ylabel('Error Count')
# plt.title('Error Distribution per Output Bit')
# plt.xticks(range(len(error_distribution)))
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

import pandas as pd

# Initialize confusion matrix for each bit: [TP, TN, FP, FN]
confusion = np.zeros((len(error_distribution), 4), dtype=int)  # columns: TP, TN, FP, FN

for x, y in test_dl:
    logits = model(x.to(device))
    pred = torch.where(logits.cpu() > 0, 1.0, -1.0)
    y = y.cpu()
    # For each bit
    for bit in range(pred.shape[1]):
        # True Positives: pred==1 and y==1
        confusion[bit, 0] += ((pred[:, bit] == 1) & (y[:, bit] == 1)).sum().item()
        # True Negatives: pred==-1 and y==-1
        confusion[bit, 1] += ((pred[:, bit] == -1) & (y[:, bit] == -1)).sum().item()
        # False Positives: pred==1 and y==-1
        confusion[bit, 2] += ((pred[:, bit] == 1) & (y[:, bit] == -1)).sum().item()
        # False Negatives: pred==-1 and y==1
        confusion[bit, 3] += ((pred[:, bit] == -1) & (y[:, bit] == 1)).sum().item()

# Display as a table
df = pd.DataFrame(confusion, columns=['TP', 'TN', 'FP', 'FN'])
df.index.name = 'Bit'
print(df)

# Print metrics for bit 0 only
bit = 0
TP = confusion[bit, 0]
TN = confusion[bit, 1]
FP = confusion[bit, 2]
FN = confusion[bit, 3]
Total = confusion[bit].sum()
FN_percent = 100 * FN / Total if Total > 0 else 0.0
FP_percent = 100 * FP / Total if Total > 0 else 0.0
TN_percent = 100 * TN / Total if Total > 0 else 0.0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
print()
print(f"Bit 0: TN % = {TN_percent:.2f}%, FN % = {FN_percent:.2f}%, FP % = {FP_percent:.2f}%\nRecall = {100*recall:.2f}%, Precision = {100*precision:.2f}%, F1 = {f1:.4f}")

bits = confusion.shape[0]
FN = confusion[:, 3]
FP = confusion[:, 2]
bit_indices = range(bits)

# plt.figure(figsize=(8, 5))
# plt.bar(bit_indices, FN, label='False Negative', color='tab:blue')
# plt.bar(bit_indices, FP, bottom=FN, label='False Positive', color='tab:orange')
# plt.xlabel('Output Bit Index')
# plt.ylabel('Error Count')
# plt.title('Error Distribution per Output Bit (FN + FP)')
# plt.xticks(bit_indices)
# plt.legend()
# plt.tight_layout()
# plt.show()