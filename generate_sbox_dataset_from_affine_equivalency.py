import torch
import numpy as np
from tqdm import tqdm
import os

from sbox_metrics.linear_probability import linear_probability
from sbox_metrics.differential_probability import differential_probability
from sbox_metrics.avelanch_criterion import avelanch_criterion
from sbox_metrics.algebraic_degree import multi_bit_algebraic_degree
from utils.file_management import load_from_csv
from generate_new_sboxes import generate_invertible_gf2, multiply_int_by_matrix

def precalculate_labels(base_sboxes, n, accepted_WP):
    """Calculates affine invariants once for all parent S-boxes."""
    base_labels = {}
    print(f"Pre-calculating invariants for {len(base_sboxes)} base S-boxes...")
    for i, s in enumerate(tqdm(base_sboxes)):
        lp = linear_probability(s, n, n)[0]
        dp = differential_probability(s, n)[0]
        ad = multi_bit_algebraic_degree(s, n)
        # Invariants: [LP, LP_flag, DP, DP_flag, max_AD]
        base_labels[i] = [lp, float(lp <= accepted_WP), dp, float(dp <= accepted_WP), max(ad)]
    return base_labels

def generate_affine_dataset(base_sboxes, base_labels, wanted_count, chunk_size, n=5, save_dir = "dataset_chunks"):
    seen_sboxes = set()
    num_chunks = wanted_count // chunk_size
    base_arrays = [np.array(s, dtype=np.uint8) for s in base_sboxes]

    os.makedirs(save_dir, exist_ok=True)

    for chunk_idx in range(num_chunks):
        chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx}.pt")
        if os.path.exists(chunk_path): continue
        
        data_list, labels_list = [], []
        pbar = tqdm(total=chunk_size, desc=f"Chunk {chunk_idx + 1}/{num_chunks}")
        
        while len(data_list) < chunk_size:
            # 1. Selection
            base_idx = np.random.randint(len(base_arrays))
            
            # 2. Transform params
            m1, m2 = generate_invertible_gf2(n), generate_invertible_gf2(n)
            x_mask, y_mask = np.random.randint(0, 32), np.random.randint(0, 32)
            
            # 3. LUT-based Transformation (The 32-value lookup)
            lut1 = np.array([multiply_int_by_matrix(x, m1) for x in range(32)], dtype=np.uint8)
            lut2 = np.array([multiply_int_by_matrix(x, m2) for x in range(32)], dtype=np.uint8)
            
            new_sbox = lut2[base_arrays[base_idx][lut1 ^ x_mask]] ^ y_mask
            
            # 4. Deduplicate (C-speed hash)
            sbox_bytes = new_sbox.tobytes()
            if sbox_bytes not in seen_sboxes:
                seen_sboxes.add(sbox_bytes)
                
                # 5. Labeling (Invariants + SAC)
                ave = avelanch_criterion(new_sbox.tolist(), n)
                max_av_diff = max([abs(0.5 - a) for a in ave])
                
                l = base_labels[base_idx]
                # Label: [LP, LP_flag, DP, DP_flag, max_av_diff, max_AD]
                labels_list.append([l[0], l[1], l[2], l[3], max_av_diff, l[4]])
                data_list.append(new_sbox)
                pbar.update(1)
        
        pbar.close()
        torch.save({
            'data': torch.tensor(np.array(data_list), dtype=torch.uint8),
            'labels': torch.tensor(labels_list, dtype=torch.float32)
        }, chunk_path)

if __name__ == "__main__":
    N_BITS = 5
    WANTED = 100_000_000
    CHUNK = 100_000
    THRESHOLD = 0.0625
    
    # Load parents
    parents = load_from_csv('new_nn_dataset_5x5\\good_sboxes_5_15000_0.15LPTH.csv')
    
    # Calculate invariants once
    labels_map = precalculate_labels(parents, N_BITS, THRESHOLD)
    
    # Run heavy generation
    generate_affine_dataset(parents, labels_map, WANTED, CHUNK, N_BITS, save_dir="dataset_affine_to_good_sboxes_5_0149LP")