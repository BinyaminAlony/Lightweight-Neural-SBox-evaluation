"""
get_NN_dataset.py
-----------------
This script generates and analyzes datasets for neural network-based S-box evaluation. It provides a PyTorch Dataset class for creating labeled binary input samples, where labels are derived from cryptographic metrics:
- Linear Probability (LP)
- Algebraic Degree (DEG)
- Avalanche Criterion (SAC)

Features:
- CustomDataset: Generates random binary inputs and computes LP, DEG, SAC metrics, assigning threshold-based labels.
- Dataset saving/loading: Uses torch.save/torch.load for persistence.
- Visualization: Plots the distribution of label bits across the dataset.

Usage:
- Set parameters (num_bits, num_samples) and filename.
- Optionally generate and save a dataset.
- Load and analyze dataset, visualize label distribution.

Dependencies:
- torch, numpy, tqdm, matplotlib
- Local modules: utils.binary_utils, sbox_metrics.*
"""

# Import required libraries
from random import sample
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os


# Add parent directory to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom utility and metric functions
from utils.binary_utils import dec_to_bin_vec
from sbox_metrics.algebraic_degree import single_bit_algebraic_degree
from sbox_metrics.avelanch_criterion import avelanch_criterion
from sbox_metrics.linear_probability import linear_probability

def generate_unique_balanced_batch(batch_size, n_elem, seen_set):
    """Generate a batch of unique balanced samples, updating the seen set."""
    k = n_elem // 2
    batch_data = []

    while len(batch_data) < batch_size:
        ones = tuple(sorted(np.random.choice(n_elem, k, replace=False)))

        if ones in seen_set:
            continue

        seen_set.add(ones)
        sample = np.zeros(n_elem, dtype=np.uint8)
        sample[list(ones)] = True
        batch_data.append(sample)

    return np.array(batch_data, dtype=np.uint8)

# Custom PyTorch Dataset for S-box evaluation
class SboxDataset(Dataset):

    def __init__(self, n_samples=0, n_bits=5, data=None, labels=None):
        """
        Initialize dataset. Can be created empty and loaded from batches,
        or created with n_samples to generate data.
        """
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.n_elem = 1 << n_bits

        if data is not None and labels is not None:
            self.data = data
            self.labels = labels
        else:
            self.data = []
            self.labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Dequantize only when accessed
        LP = label[0].float() / 255.0
        DEG = label[1]
        SAC = label[2].float() / 255.0

        return self.data[idx], torch.tensor([LP, DEG.float(), SAC])


def generate_dataset_batched(n_samples, n_bits, filename, batch_size=100_000, save_interval=1_000_000):
    """
    Generate dataset in batches to reduce memory usage.
    Saves intermediate results every `save_interval` samples.
    
    Args:
        n_samples: Total number of samples to generate
        n_bits: Number of bits for S-box
        filename: Output filename for final dataset
        batch_size: Number of samples per batch (kept in memory)
        save_interval: Save checkpoint every this many samples
    """
    n_elem = 1 << n_bits
    seen_set = set()
    
    all_data = []
    all_labels = []
    
    checkpoint_dir = filename.replace('.pt', '_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    samples_generated = 0
    checkpoint_idx = 0
    
    with tqdm(total=n_samples, desc="Generating dataset") as pbar:
        while samples_generated < n_samples:
            # Determine batch size for this iteration
            remaining = n_samples - samples_generated
            current_batch_size = min(batch_size, remaining)
            
            # Generate batch of unique samples
            batch_inputs = generate_unique_balanced_batch(current_batch_size, n_elem, seen_set)
            
            # Process batch
            for input_sample in batch_inputs:
                LP = linear_probability(input_sample, n_bits, 1)[0]
                SAC = avelanch_criterion(input_sample, n_bits, 1)[0]
                DEG = single_bit_algebraic_degree(input_sample, n_bits)

                # Quantized storage
                LP_q = int(round(LP * 255))
                SAC_q = int(round(SAC * 255))
                DEG_q = int(DEG)

                label = torch.tensor([LP_q, DEG_q, SAC_q], dtype=torch.uint8)
                all_data.append(torch.tensor(input_sample, dtype=torch.bool))
                all_labels.append(label)
                
                samples_generated += 1
                pbar.update(1)
            
            # Save checkpoint if we've hit the interval
            if samples_generated % save_interval == 0 or samples_generated == n_samples:
                checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{checkpoint_idx}.pt')
                checkpoint_data = {
                    'data': all_data.copy(),
                    'labels': all_labels.copy(),
                    'seen_set': seen_set,
                    'samples_generated': samples_generated
                }
                torch.save(checkpoint_data, checkpoint_file)
                tqdm.write(f"\nCheckpoint saved: {samples_generated}/{n_samples} samples -> {checkpoint_file}")
                
                # Clear memory after saving checkpoint
                all_data.clear()
                all_labels.clear()
                checkpoint_idx += 1
    
    # Combine all checkpoints into final dataset
    print("Combining checkpoints into final dataset...")
    final_data = []
    final_labels = []
    
    for i in range(checkpoint_idx):
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{i}.pt')
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        final_data.extend(checkpoint['data'])
        final_labels.extend(checkpoint['labels'])
        del checkpoint  # Free memory
    
    dataset = SboxDataset(n_samples=n_samples, n_bits=n_bits, data=final_data, labels=final_labels)
    torch.save(dataset, filename)
    print(f"Final dataset saved to {filename}")
    
    # Optionally clean up checkpoints
    # import shutil
    # shutil.rmtree(checkpoint_dir)
    
    return dataset


def resume_from_checkpoint(checkpoint_dir, n_samples, n_bits, filename, batch_size=100_000, save_interval=1_000_000):
    """Resume dataset generation from the last checkpoint."""
    # Find latest checkpoint
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
    
    if not checkpoints:
        print("No checkpoints found, starting fresh...")
        return generate_dataset_batched(n_samples, n_bits, filename, batch_size, save_interval)
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"Resuming from {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, weights_only=False)
    seen_set = checkpoint['seen_set']
    samples_generated = checkpoint['samples_generated']
    checkpoint_idx = int(checkpoints[-1].split('_')[1].split('.')[0]) + 1
    
    del checkpoint  # Free memory
    
    n_elem = 1 << n_bits
    all_data = []
    all_labels = []
    
    with tqdm(total=n_samples, initial=samples_generated, desc="Generating dataset (resumed)") as pbar:
        while samples_generated < n_samples:
            remaining = n_samples - samples_generated
            current_batch_size = min(batch_size, remaining)
            
            batch_inputs = generate_unique_balanced_batch(current_batch_size, n_elem, seen_set)
            
            for input_sample in batch_inputs:
                LP = linear_probability(input_sample, n_bits, 1)[0]
                SAC = avelanch_criterion(input_sample, n_bits, 1)
                DEG = single_bit_algebraic_degree(input_sample, n_bits)

                LP_q = int(round(LP * 255))
                SAC_q = int(round(SAC * 255))
                DEG_q = int(DEG)

                label = torch.tensor([LP_q, DEG_q, SAC_q], dtype=torch.uint8)
                all_data.append(torch.tensor(input_sample, dtype=torch.bool))
                all_labels.append(label)
                
                samples_generated += 1
                pbar.update(1)
            
            if samples_generated % save_interval == 0 or samples_generated == n_samples:
                checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{checkpoint_idx}.pt')
                checkpoint_data = {
                    'data': all_data.copy(),
                    'labels': all_labels.copy(),
                    'seen_set': seen_set,
                    'samples_generated': samples_generated
                }
                torch.save(checkpoint_data, checkpoint_file)
                print(f"\nCheckpoint saved: {samples_generated}/{n_samples} samples")
                
                all_data.clear()
                all_labels.clear()
                checkpoint_idx += 1
    
    # Combine all checkpoints
    print("Combining checkpoints into final dataset...")
    final_data = []
    final_labels = []
    
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.startswith('checkpoint_'):
            checkpoint = torch.load(os.path.join(checkpoint_dir, f), weights_only=False)
            final_data.extend(checkpoint['data'])
            final_labels.extend(checkpoint['labels'])
            del checkpoint
    
    dataset = SboxDataset(n_samples=n_samples, n_bits=n_bits, data=final_data, labels=final_labels)
    torch.save(dataset, filename)
    print(f"Final dataset saved to {filename}")
    
    return dataset


# Function to generate and save dataset
# Parameters: n_samples (int), n_bits (int), filename (str)
def generate_dataset(n_samples, n_bits, filename):
    # Create the dataset
    dataset = SboxDataset(n_samples, n_bits)
    # Save the dataset to a file
    torch.save(dataset, filename)
    # To load the dataset later:
    # loaded_dataset = torch.load("dataset.pt")
    return dataset

# Utility for formatting large numbers
import math
millnames = ['', '_Thou', '_Mil']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

if __name__ == "__main__":
    # Set parameters for dataset generation
    num_bits = 5  # Number of bits
    num_samples = 100_000_000  # Number of samples
    # num_samples = 100_000  # Number of samples
    filename = f"new_{millify(num_samples)}_samples_LP_DEG_SAC.pt"
    
    # Generate with batching (memory-efficient)
    print(f"Generating dataset with {num_samples} samples, saving to {filename}")
    dataset = generate_dataset_batched(
        n_samples=num_samples,
        n_bits=num_bits,
        filename=filename,
        batch_size=100_000,       # Process 100k samples at a time
        save_interval=1_000_000   # Save checkpoint every 1M samples
    )
    
    # To resume from checkpoint if interrupted, comment above and use:
    # checkpoint_dir = filename.replace('.pt', '_checkpoints')
    # dataset = resume_from_checkpoint(checkpoint_dir, num_samples, num_bits, filename)

    print(f"Dataset created with {len(dataset)} samples")
    
    # Fix for duplicate OpenMP library issue (for some environments)
    import os
    from collections import Counter
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Extract all labels from dataset
    all_labels = torch.stack(dataset.labels)
    temp = all_labels.tolist()
    print("len temp = ", len(temp))
    # Count number of unique lists in temp (list of lists)
    unique_count = len({tuple(lst) for lst in temp})
    print(f"Number of unique label vectors: {unique_count}")
    # Count occurrences of each unique label vector in temp

    label_counts = Counter(tuple(lst) for lst in temp)
    print("Unique label counts: ", label_counts.values())
    total = 0
    # for label, count in label_counts.items():
    #     total += count
    #     print(f"Label: {label}, Count: {count}")
    print(f"Total samples counted: {total}")
    # # Count the number of times each bit is 1 in the labels
    # bit_counts = torch.sum(all_labels == 1, dim=0)

    # # Create a bar plot for label bit distribution
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(bit_counts)), bit_counts.numpy(), color='blue', alpha=0.7)
    # plt.xlabel("Label Bit Index")
    # plt.ylabel("Count of '1' Values")
    # plt.title("Distribution of Label Bits")
    # plt.xticks(range(len(bit_counts)))
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

# import torch
# LP = 0.1
# LP_thresholds = torch.tensor([0, 0.0625, 0.145, 0.25, 0.5])  # 4 intervals
# LP_result = torch.where(
#     (LP > LP_thresholds[:-1]) & (LP <= LP_thresholds[1:]),
#     1.0, -1.0
# )
# print(LP_result)