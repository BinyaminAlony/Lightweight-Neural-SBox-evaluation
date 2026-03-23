import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class SBoxDataset(Dataset):
    def __init__(self, n_bits, n_samples, data_dir='.'):
        """
        PyTorch Dataset for Large SBox Data (Memmapped).
        
        :param n_bits: The 'n' used in generation (e.g., 5). 
                       Input vector size = 2^n.
        :param n_samples: Total number of samples generated (e.g., 100_000_000).
        :param data_dir: Directory containing the .bin files.
        """
        super().__init__()
        
        # 1. Calculate Dimensions
        self.vec_len = 1 << n_bits          # e.g., 2^5 = 32 bits
        self.packed_dim = self.vec_len // 8 # e.g., 32 / 8 = 4 bytes
        self.n_samples = n_samples
        
        # 2. File Paths
        input_file = os.path.join(data_dir, f'data_n{n_bits}_inputs.bin')
        target_file = os.path.join(data_dir, f'data_n{n_bits}_targets.bin')
        
        # 3. Check if files exist
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # 4. Load via Memory Mapping (Zero-Copy)
        # This acts like an array but stays on disk, using minimal RAM.
        self.inputs = np.memmap(
            input_file, 
            dtype='uint8', 
            mode='r', 
            shape=(n_samples, self.packed_dim)
        )
        
        self.targets = np.memmap(
            target_file, 
            dtype='uint8', 
            mode='r', 
            shape=(n_samples, 3) # [LP, DEG, SAC]
        )

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # A. Read Bytes from Disk
        # This reads just 4 bytes (for n=5) from the file
        packed_sample = self.inputs[idx]
        target_sample = self.targets[idx]
        
        # B. Unpack Bits (The Decompression)
        # Converts [Byte] -> [0, 1, 1, 0, ...]
        # Output shape: (vec_len,) e.g. (32,)
        # Note: unpackbits is very fast (C-optimized)
        unpacked_bits = np.unpackbits(packed_sample)
        
        # C. Convert to PyTorch Tensors
        # Inputs: Neural networks expect Float32 (0.0 or 1.0)
        input_tensor = torch.from_numpy(unpacked_bits).float()
        
        # Targets: Float32 is standard for Regression (MSELoss). 
        # If doing Classification, change to .long()
        # .copy() is needed here because memmap arrays are read-only views
        target_tensor = torch.from_numpy(target_sample.copy()).float()
        
        return input_tensor, target_tensor

# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    from generate_NN_dataset_LP_AV_ADEG import label
    # Configuration (Must match what you generated)
    N_BITS = 5
    TOTAL_SAMPLES = 100_000_000
    
    print("Initializing Dataset...")
    try:
        dataset = SBoxDataset(n_bits=N_BITS, n_samples=TOTAL_SAMPLES)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you run the generation script first!")
        exit()

    print(f"Dataset Loaded. Size: {len(dataset):,}")
    
    # Create DataLoader
    # num_workers=4 is safe and recommended for Memmap
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    print("Fetching first batch...")
    
    # Grab one batch to demonstrate
    for inputs, targets in loader:
        print("\n--- Batch Info ---")
        print(f"Input Shape:  {inputs.shape} (Batch, 32)")
        print(f"Target Shape: {targets.shape} (Batch, 3)")
        
        print("\nSample 0 Input:")
        print(inputs[0])
        print("HW = ", inputs[0].sum().item(), ", len =" , len(inputs[0]))
        
        # print("\nSample 0 Target [LP, DEG, SAC]:")
        # print(targets[0])

        # print("correct label:")
        # print(label(N_BITS, inputs[0].numpy().astype(np.uint8)))
        

        for i in range(len(inputs)):
            if not np.array_equal(label(N_BITS, inputs[i].numpy().astype(np.uint8)), targets[i].numpy().astype(np.uint8)):
                print("Mismatch in sample ", i)
                print("Computed: ", label(N_BITS, inputs[i].numpy().astype(np.uint8)))
                print("Stored:   ", targets[i].numpy().astype(np.uint8))
        print("All samples in batch match their labels.")
        break