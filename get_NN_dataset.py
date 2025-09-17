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

# Custom PyTorch Dataset for S-box evaluation
class CustomDataset(Dataset):

    def __init__(self, other):
        if other is not None:
            # Copy data and labels from another CustomDataset instance
            self.n_samples = other.n_samples
            self.n_bits = other.n_bits
            self.n_elem = other.n_elem
            self.LP_thresholds = other.LP_thresholds
            self.Alg_deg_thresholds = other.Alg_deg_thresholds
            self.data = list(other.data)
            self.labels = list(other.labels)
        return

    def __init__(self, n_samples, n_bits):
        # Store parameters
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.n_elem = 1 << n_bits  # Number of elements in input vector
        # Thresholds for metrics
        self.LP_thresholds = torch.tensor([0, 0.0625, 0.145, 0.25, 0.5]) # bits 0-3
        self.Alg_deg_thresholds = torch.tensor([2, 3, 4, 5]) # bits 4-7
        self.data = []
        self.labels = []
        
        # Generate random input numbers
        inputs_as_nums = sample(range(4294967296), n_samples)  # Generate random inputs
        print('done sampling inputs')
        
        counts = dict()  # Track label counts
        
        # For each input, calculate metrics and assign label
        for input_as_num in tqdm(inputs_as_nums, desc="Generating dataset"):
            input = dec_to_bin_vec(input_as_num, self.n_elem)
            LP = torch.tensor([linear_probability(input, n_bits, 1)[0]])
            DEG = torch.tensor([single_bit_algebraic_degree(input, n_bits)])
            SAC = torch.tensor(avelanch_criterion(input, n_bits, 1))

            # Compare LP and DEG against all thresholds
            LP_result = torch.where(
                (LP > self.LP_thresholds[:-1]) & (LP <= self.LP_thresholds[1:]),
                1.0, -1.0
            )
            DEG_result = torch.where(DEG == self.Alg_deg_thresholds, 1.0, -1.0)
            SAC_result = torch.where(abs(SAC - 0.5) <= 0.05, 1.0, -1.0)
                      
            

            # Concatenate the results into a single feature vector
            # lable = torch.cat([LP_result, DEG_result], dim=0)
            lable = torch.cat([LP_result, DEG_result, SAC_result], dim=0)
            
            
            
            # Limit to 10000 samples per label
            if counts.get(tuple(lable.numpy()), 0) < 10000:
                counts[tuple(lable.numpy())] = counts.get(tuple(lable.numpy()), 0) + 1
                self.data.append(torch.tensor(input, dtype=torch.float32))
                self.labels.append(lable)
        # print(counts)

    def __len__(self):
        # Return number of samples
        return len(self.data)
    
    def shape(self):
        return self.__len__()

    def __getitem__(self, idx):
        # Return data and label for given index
        return self.data[idx], self.labels[idx]

# Function to generate and save dataset
# Parameters: n_samples (int), n_bits (int), filename (str)
def generate_dataset(n_samples, n_bits, filename):
    # Create the dataset
    dataset = CustomDataset(n_samples, n_bits)
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
    filename = f"{millify(num_samples)}_samples_LP_DEG_SAC.pt"
    # Uncomment to generate and save dataset
    # print(f"Generating dataset with {num_samples} samples and {millify(num_samples)} bits, saving to {filename}")
    # generate_dataset(num_samples, num_bits, filename)

    # Example usage: Load the dataset
    dataset = torch.load(filename, weights_only=False)
    print(dataset.__len__())
    
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