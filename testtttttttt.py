import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import  transforms
from tqdm import tqdm # ,datasets
from get_NN_dataset import CustomDataset, generate_dataset
import os
import numpy as np
import matplotlib.pyplot as plt

# TYPE = "CNN"
TYPE = "MLP"
BIT_0_COEFF = 1.5
TN_FN_COEFF = 1.2

WD = False
# ---------- Data ----------
class ToBinaryTensor:
    def __call__(self, img):
        t = transforms.functional.to_tensor(img)      # [0,1]
        return torch.where(t > 0.5, 1.0, -1.0)        # {-1, +1}


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
        self.fc = nn.Linear(32 * 32, 9)  # 32 channels * 32 length
        self.model_type = "CNN"

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, 32]
        if self.training:
            temp = self.conv1(x)
            temp = self.act1(temp)
            temp = self.conv2(temp)
            temp = self.act2(temp)
            temp = self.flatten(temp)
            return self.fc(temp)
        else:
            temp = self.conv1(x)
            temp = self.act1(temp)
            temp = torch.where(temp > 0, 1.0, -1.0)
            temp = self.conv2(temp)
            temp = self.act2(temp)
            temp = torch.where(temp > 0, 1.0, -1.0)
            temp = self.flatten(temp)
            temp = self.fc(temp)
            return torch.where(temp > 0, 1.0, -1.0)

class BinaryMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.inl = nn.Linear(32, hidden_dim) # hidden layer
        self.activation1 = nn.ReLU()
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.activation2 = nn.ReLU()
        self.outl = nn.Linear(hidden_dim, bits)   # output
        self.model_type = "MLP"

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

if __name__ == "__main__":
    
    filename = "10_Million_samples_LP_DEG_SAC.pt"
    print("Loading dataset: ", filename)
    dataset = torch.load(filename, weights_only=False, map_location="cpu")

    dataset.data = torch.stack(dataset.data)  # Convert list of tensors to a single tensor
    dataset.labels = torch.stack(dataset.labels)  # Convert list of tensors to a single tensor

    #dataset.labels = np.delete(dataset.labels, 8, axis=1)  # Remove the first column if it is not needed
    #dataset.labels = np.delete(dataset.labels, 7, axis=1)  # Remove the first column if it is not needed
    #dataset.labels = np.delete(dataset.labels, 6, axis=1)  # Remove the first column if it is not needed
    #dataset.labels = np.delete(dataset.labels, 5, axis=1)  # Remove the first column if it is not needed
    #dataset.labels = np.delete(dataset.labels, 4, axis=1)  # Remove the first column if it is not needed
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
    test_dl  = DataLoader(test_ds,  batch_size=128)

    bits = dataset.labels.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = BinaryMLP(hidden_dim=256).to(device) if TYPE == "MLP" else BinaryCNN(hidden_dim=256).to(device)
    print("model type:", model.model_type)

    error_distribution = torch.tensor([0]*bits)

    # ---------- Training loop ----------
    class BinaryLogisticLoss(torch.nn.Module):
        """Per-element softplus logistic loss with per-bit weights.

        This is implemented in a vectorized manner: per-element loss = softplus(-logit * target),
        then each column (bit) is multiplied by its per-bit weight and averaged.
        Bit 0 is given a higher weight (4x by default).
        """
        def __init__(self):
            super().__init__()
            # create weights tensor once and register as buffer so it moves with the module
            w = torch.ones(bits, dtype=torch.float32)

        def forward(self, logits, targets):
            # logits, targets: [batch, bits]
            # per-element logistic-style softplus loss:
            # bit 0 focuses on TN and FN: TN/FN contribute +- 2.2, while TP/FP contribute +-0.2
            loss_bit_0 = torch.sigmoid(
                BIT_0_COEFF * ((-logits[:, 0] * targets[:, 0]) * (TN_FN_COEFF - logits[:, 0]))
            ).unsqueeze(1)  # -> [batch, 1]
            loss_others = torch.sigmoid(-logits[:, 1:] * targets[:, 1:])  # -> [batch, bits-1]
            per_el = torch.cat((loss_bit_0, loss_others), dim=1)  # -> [batch, bits]
            return per_el.mean()

    loss_fn = BinaryLogisticLoss()  # bit 0 worth 4x others
    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5 if WD == True else 0)
    device    = next(model.parameters()).device
    # Move loss_fn buffers to the same device as model parameters to avoid per-call copies
    loss_fn.to(device)
    
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

    epochs = 50
    reps = 20
    TNs = []
    Recalls = []

    for gamma in tqdm(np.arange(1.1, 2, 0.1), desc="training for gamma values"):
        TN_FN_COEFF = gamma
        TN_total = 0
        recall_total = 0
        for _ in range(reps):
            for epoch in range(1, epochs + 1):
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
                    test_acc = accuracy(test_dl)
                if epoch % 5 == 0 or epoch == 1:
                    tqdm.write(f"Epoch {epoch}: "
                        f"loss={running_loss/len(train_ds):.4f}  "
                        f"train_acc={train_acc:.3%}  test_acc={test_acc:.3%}")
                    

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

            # Print metrics for bit 0 only
            bit = 0
            TP = confusion[bit, 0]
            TN = confusion[bit, 1]
            FP = confusion[bit, 2]
            FN = confusion[bit, 3]
            Total = confusion[bit].sum()
            TN_percent = 100 * TN / Total if Total > 0 else 0.0

            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            TN_total += TN_percent
            recall_total += recall

        TNs.append(TN_total/reps)
        Recalls.append(recall_total/reps)

    # Save trained model using the requested naming convention

    # model_dir = "models"
    # os.makedirs(model_dir, exist_ok=True)
    # # dataset base name without extension
    # dataset_base = os.path.splitext(os.path.basename(filename))[0]
    # model_state_name = f"BNN_Model_{model.model_type}_{epochs}_{dataset_base}.pt"
    # model_state_path = os.path.join(model_dir, model_state_name)
    # # save state_dict
    # torch.save(model.state_dict(), model_state_path)
    # # also save full model (optional convenience)
    # model_full_name = f"BNN_Model_{model.model_type}_{epochs}_{dataset_base}_full.pt"
    # model_full_path = os.path.join(model_dir, model_full_name)
    # torch.save(model, model_full_path)
    # print(f"Saved model state_dict to: {model_state_path}")
    # print(f"Saved full model to: {model_full_path}")

    # prepare x axis (try to reconstruct gamma values, otherwise use indices)
    if len(TNs) > 0:
        try:
            x = np.array([1.1 + i * 0.1 for i in range(len(TNs))])
            xlabel = "gamma"
        except Exception:
            x = np.arange(len(TNs))
            xlabel = "index"
    else:
        x = np.arange(len(TNs))
        xlabel = "index"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, TNs, marker="o", linestyle="-")
    ax1.set_title("True Negative % (bit 0)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("TN (%)")
    ax1.grid(True)

    ax2.plot(x, Recalls, marker="o", color="C1", linestyle="-")
    ax2.set_title("Recall (bit 0)")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Recall")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()