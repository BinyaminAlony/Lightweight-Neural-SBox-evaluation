"""
run_cnn_training.py
-------------------
Main entry point for CNN training on S-box dataset.
Run this from the workspace root directory.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import sys
import csv
import matplotlib.pyplot as plt

# ==================== Configuration ====================
# Training settings
WEIGHT_DECAY = False
EPOCHS = 250
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 128
LEARNING_RATE = 1e-3
TRAIN_TEST_RATIO = 0.8
HIDDEN_DIM = 256

# Dataset file (change as needed)
DATASET_FILE = "10_Million_samples_LP_DEG_SAC.pt"


# ==================== CNN Model ====================
class SboxCNN(nn.Module):
    """
    1D CNN for S-box cryptographic metric prediction.
    
    Architecture:
    - Conv1d layers for feature extraction
    - ReLU activations
    - Fully connected output layer
    
    Standard (non-binarized) CNN with continuous activations.
    """
    def __init__(self, input_size: int = 32, hidden_dim: int = 256, num_outputs: int = 3):
        super().__init__()
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_size, hidden_dim)
        self.act_fc = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
        self.model_type = "CNN"

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)  # [batch, 1, input_size]
        
        # Standard CNN forward pass (no binarization)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc(x)
        return self.fc2(x)


# ==================== Evaluation ====================
def compute_mse(model, loader, device):
    """Compute MSE loss on a dataset."""
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            preds = model(x)
            total_mse += ((preds - y) ** 2).sum().item()
            total_samples += y.numel()
    return total_mse / total_samples


def compute_mae(model, loader, device):
    """Compute MAE (Mean Absolute Error) on a dataset."""
    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            preds = model(x)
            total_mae += (preds - y).abs().sum().item()
            total_samples += y.numel()
    return total_mae / total_samples


def compute_metrics_per_output(model, loader, device, num_outputs):
    """Compute regression metrics for each output."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = {}
    for i in range(num_outputs):
        pred_i = all_preds[:, i]
        target_i = all_targets[:, i]
        
        mse = ((pred_i - target_i) ** 2).mean().item()
        mae = (pred_i - target_i).abs().mean().item()
        
        # R² score
        ss_res = ((target_i - pred_i) ** 2).sum().item()
        ss_tot = ((target_i - target_i.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics[f'output_{i}'] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    return metrics


def print_metrics(metrics, output_idx=0):
    """Print metrics for a specific output."""
    m = metrics[f'output_{output_idx}']
    output_names = ['LP', 'DEG', 'SAC']
    name = output_names[output_idx] if output_idx < len(output_names) else f'Output {output_idx}'
    print(f"\n=== Metrics for {name} (output {output_idx}) ===")
    print(f"  MSE:  {m['mse']:.6f}")
    print(f"  RMSE: {m['rmse']:.6f}")
    print(f"  MAE:  {m['mae']:.6f}")
    print(f"  R²:   {m['r2']:.6f}")


# ==================== Training ====================
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss


def train_model(model, train_loader, test_loader, optimizer, loss_fn, device, 
                epochs, train_size, num_outputs):
    """Full training loop."""
    train_mses, test_mses = [], []
    
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        running_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        train_mse = compute_mse(model, train_loader, device)
        test_mse = compute_mse(model, test_loader, device)
        train_mses.append(train_mse)
        test_mses.append(test_mse)
        
        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"Epoch {epoch:3d}: loss={running_loss/train_size:.6f}  "
                      f"train_mse={train_mse:.6f}  test_mse={test_mse:.6f}")
    
    return train_mses, test_mses


# ==================== Main ====================
if __name__ == "__main__":
    print(f"Loading dataset: {DATASET_FILE}")
    dataset = torch.load(DATASET_FILE, weights_only=False, map_location="cpu")
    
    # Convert lists to tensors if needed
    if isinstance(dataset.data, list):
        dataset.data = torch.stack(dataset.data)
    if isinstance(dataset.labels, list):
        dataset.labels = torch.stack(dataset.labels)
    
    input_size = dataset.data.shape[1]
    num_outputs = dataset.labels.shape[1]
    print(f"Input size: {input_size}, Number of outputs: {num_outputs}")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_TEST_RATIO)
    test_size = total_size - train_size
    print(f"Total: {total_size}, Train: {train_size}, Test: {test_size}")
    
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_TEST)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = SboxCNN(input_size=input_size, hidden_dim=HIDDEN_DIM, num_outputs=num_outputs).to(device)
    print(f"Model: {model.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                          weight_decay=1e-5 if WEIGHT_DECAY else 0)
    
    print(f"\nTraining with MSELoss for {EPOCHS} epochs...")
    print("=" * 60)
    
    # Train
    train_mses, test_mses = train_model(
        model, train_loader, test_loader, optimizer, loss_fn, device,
        EPOCHS, train_size, num_outputs
    )
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation:")
    
    metrics = compute_metrics_per_output(model, test_loader, device, num_outputs)
    
    for i in range(num_outputs):
        print_metrics(metrics, i)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/CNN_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "mse", "rmse", "mae", "r2"])
        for i in range(num_outputs):
            m = metrics[f'output_{i}']
            writer.writerow([i, m['mse'], m['rmse'], m['mae'], m['r2']])
    print("\nSaved results to: results/CNN_metrics.csv")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/CNN_model_{EPOCHS}epochs.pt")
    print(f"Saved model to: models/CNN_model_{EPOCHS}epochs.pt")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_mses, label='Train')
    ax1.plot(test_mses, label='Test')
    ax1.set_title('MSE over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(True)
    
    output_names = ['LP', 'DEG', 'SAC']
    mses = [metrics[f'output_{i}']['mse'] for i in range(num_outputs)]
    r2s = [metrics[f'output_{i}']['r2'] for i in range(num_outputs)]
    x = np.arange(num_outputs)
    ax2.bar(x, r2s, 0.5, label='R² Score')
    ax2.set_title('R² Score per Output')
    ax2.set_xlabel('Output')
    ax2.set_ylabel('R²')
    ax2.set_xticks(x)
    ax2.set_xticklabels(output_names[:num_outputs])
    ax2.legend()
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/CNN_training_plots.png', dpi=150)
    plt.show()
