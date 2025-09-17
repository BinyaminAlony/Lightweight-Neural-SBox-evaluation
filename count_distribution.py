from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt

from get_NN_dataset import CustomDataset, generate_dataset
from utils.binary_utils import bin_vec_to_dec

filename = "100_Mil_samples_LP_DEG_SAC.pt"
dataset = torch.load(filename, weights_only=False, map_location="cpu")

labels = np.array(dataset.labels)
labels = ((labels + 1) // 2).astype(int)  # Convert {-1, +1} to {0, 1} and to int
labels_as_str = [''.join(str(bit) for bit in label) for label in labels]
ctr = Counter(labels_as_str)

labels_no_bit4 = np.delete(labels, 4, axis=1)  # Remove the first column if it is not needed
labels_no_bit4_as_str = [''.join(str(bit) for bit in label) for label in labels_no_bit4]
ctr_no_bit4 = Counter(labels_no_bit4_as_str)

# trans = np.transpose(labels)
# trans = (trans+1)//2

# bit_counts = trans.sum(axis=1).tolist()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First subplot for ctr
axs[0].bar(ctr.keys(), ctr.values())
axs[0].set_xticklabels(ctr.keys(), rotation=45, ha='right')
axs[0].set_xlabel('Label Value')
axs[0].set_ylabel('Count')
axs[0].set_title('Histogram of Label Value Counts')

# Second subplot for ctr_no_bit4
axs[1].bar(ctr_no_bit4.keys(), ctr_no_bit4.values())
axs[1].set_xticklabels(ctr_no_bit4.keys(), rotation=45, ha='right')
axs[1].set_xlabel('Label Value (bit 4 removed)')
axs[1].set_ylabel('Count')
axs[1].set_title('Histogram of Label Value Counts (bit 4 removed)')

plt.tight_layout()
plt.show()