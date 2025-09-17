from random import sample
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbox_metrics.algebraic_degree import single_bit_algebraic_degree
from sbox_metrics.avelanch_criterion import avelanch_criterion
from utils.binary_utils import dec_to_bin_vec
from sbox_metrics.linear_probability import linear_probability


class CustomDataset(Dataset):
    def __init__(self, n_samples, n_bits):
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.n_elem = 1 << n_bits
        self.LP_thresholds = torch.tensor([0.0625, 0.145, 0.25, 0.5])
        self.Alg_deg_thresholds = torch.tensor([2, 3, 4, 5])
        self.data = []
        self.labels = []
        
        inputs_as_nums = sample(range(4294967296), n_samples)  # Generate random inputs
        print('done sampling inputs')
        
        counts = dict()
        
        for input_as_num in tqdm(inputs_as_nums, desc="Generating dataset"):
            input = dec_to_bin_vec(input_as_num, self.n_elem)
            LP = torch.tensor([linear_probability(input, n_bits, 1)[0]])
            DEG = torch.tensor([single_bit_algebraic_degree(input, n_bits)])
            SAC = torch.tensor(avelanch_criterion(input, n_bits, 1))

            # Compare LP and DP against all thresholds
            LP_result = torch.where(LP <= self.LP_thresholds, 1.0, -1.0)
            DEG_result = torch.where(DEG <= self.Alg_deg_thresholds, 1.0, -1.0)
            SAC_result = torch.where(SAC == 0.5, 1.0, -1.0)

            # Concatenate the results into a single feature vector
            # lable = torch.cat([LP_result, DEG_result], dim=0)
            lable = torch.cat([LP_result, DEG_result, SAC_result], dim=0)
            if counts.get(tuple(lable.numpy()), 0) < 10000:  # Limit to 1000 samples per label'
                counts[tuple(lable.numpy())] = counts.get(tuple(lable.numpy()), 0) + 1
                self.data.append(torch.tensor(input, dtype=torch.float32))
                self.labels.append(lable)
        print(counts)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    # Parameters


if __name__ == "__main__":
    dataset = torch.load("NN_project/10_Million_samples_LP_DEG_SAC.pt", weights_only=False)
    all_labels = torch.stack(dataset.labels)
    num_samples = all_labels.shape[0]
    num_features = all_labels.shape[1]
    bit_counts = torch.sum(all_labels == 1, dim=0)
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print("Number of times each label bit is 1:")
    for i, count in enumerate(bit_counts):
        print(f"Bit {i}: {count.item()}")

    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_features), bit_counts.numpy(), color='blue', alpha=0.7)
    plt.xlabel("Label Bit Index")
    plt.ylabel("Count of '1' Values")
    plt.title("Distribution of Label Bits (Number of times each bit is 1)")
    plt.xticks(range(num_features))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
