import os
import math
import torch
# from torch.utils import data
from torch.utils.data import Dataset

from sbox_metrics.linear_probability import linear_probability
from sbox_metrics.differential_probability import differential_probability
from sbox_metrics.avelanch_criterion import avelanch_criterion
from sbox_metrics.algebraic_degree import multi_bit_algebraic_degree
from tqdm import tqdm
from datetime import datetime

class NToNDataset(Dataset):

    def __init__(self, n_bits: int, size: int, accepted_WP: float, save_dir: str = "dataset_chunks", chunk_size: int = 5000):
        """
        Initialize the dataset with chunking support.
        """
        current_date = datetime.now().strftime("%d_%m_%Y")
        self.save_dir = f"{save_dir}_{n_bits}bit_{size}size_{chunk_size}cs_{current_date}"
        self.chunk_size = chunk_size
        self.size = size
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        n_elements = 1 << n_bits
        num_chunks = math.ceil(size / chunk_size)

        for chunk_idx in tqdm(range(num_chunks), desc="Overall Progress"):
            chunk_path = os.path.join(self.save_dir, f"chunk_{chunk_idx}.pt")
            if os.path.exists(chunk_path) and chunk_idx != num_chunks - 1:
                continue
                
            data_list = []
            labels_list = []
            current_chunk_size = min(chunk_size, size - chunk_idx * chunk_size)

            for _ in tqdm(range(current_chunk_size), desc=f"Generating chunk {chunk_idx + 1}/{num_chunks}", leave=False):

                example = torch.randperm(n_elements, dtype=torch.uint8).tolist() # UP TO 8 -> 8 bits

                LP = linear_probability(example, n_bits, n_bits)[0]
                LP_flag = float(LP <= accepted_WP)

                DP = differential_probability(example, n_bits)[0]
                DP_flag = float(DP <= accepted_WP)

                avelanches = avelanch_criterion(example, n_bits)
                max_av_diff_from_half = max([abs(0.5-avalanche) for avalanche in avelanches])

                AD = multi_bit_algebraic_degree(example, n_bits)

                label = [LP, LP_flag, DP, DP_flag, max_av_diff_from_half, max(AD)] 

                data_list.append(example)
                labels_list.append(label)

            chunk_data = torch.tensor(data_list, dtype=torch.float32)
            chunk_labels = torch.tensor(labels_list, dtype=torch.float32)
            torch.save({'data': chunk_data, 'labels': chunk_labels}, chunk_path)
            print(f"Generated and saved chunk {chunk_idx + 1}/{num_chunks}")

        self._current_chunk_idx = -1
        self._current_data = None
        self._current_labels = None

        self.__verifysize__()

    def __verifysize__(self):
        """
        Verify that the total number of samples across all chunks matches the expected size.
        """
        total_samples = 0
        for chunk_idx in range(math.ceil(self.size / self.chunk_size)):
            chunk_path = os.path.join(self.save_dir, f"chunk_{chunk_idx}.pt")
            if os.path.exists(chunk_path):
                chunk_data = torch.load(chunk_path, weights_only=True)['data']
                total_samples += len(chunk_data)
        
        assert total_samples == self.size, f"Expected {self.size} samples, but found {total_samples} across all chunks."

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        PyTorch uses this to know the size of the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """
        Fetch and return a single sample and its corresponding label/target     
        at the given index `idx`.
        """
        chunk_idx = idx // self.chunk_size
        item_idx = idx % self.chunk_size
        
        # Load the chunk only if it isn't currently loaded to save RAM/disk I/O
        if chunk_idx != self._current_chunk_idx:
            chunk_path = os.path.join(self.save_dir, f"chunk_{chunk_idx}.pt")
            saved_chunk = torch.load(chunk_path, weights_only=True)
            self._current_data = saved_chunk['data']
            self._current_labels = saved_chunk['labels']
            self._current_chunk_idx = chunk_idx
            
        sample = self._current_data[item_idx]
        label = self._current_labels[item_idx]
        
        return sample, label
    

if __name__ == "__main__":
    dataset = NToNDataset(n_bits=5, size=225_000_000, accepted_WP=0.0625, chunk_size=225_000)
    print(dataset[0])