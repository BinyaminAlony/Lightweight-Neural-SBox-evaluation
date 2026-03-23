import math
import numpy as np
import os
import signal
import sys
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def init_worker():
    """Initialize worker to ignore SIGINT, letting the main process handle it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# --- IMPORT YOUR METRICS ---
from sbox_metrics.algebraic_degree import single_bit_algebraic_degree
from sbox_metrics.avelanch_criterion import avelanch_criterion
from sbox_metrics.linear_probability import linear_probability

# --- CONFIGURATION ---
N_BITS = 5                  # N=5 (32-bit vector), N=6 (64-bit vector)
TOTAL_SAMPLES = 100_000_000 
CHUNK_SIZE = 10_000        
FILE_INPUTS = f'data_{TOTAL_SAMPLES}_n{N_BITS}_inputs.bin'
FILE_TARGETS = f'data_{TOTAL_SAMPLES}_n{N_BITS}_targets.bin'

# --- 1. METRIC CALCULATION ---
def label(n_bits, input_sample):
    """
    input_sample: numpy array of 0s and 1s
    """
    # Note: Ensure your metric functions accept numpy arrays
    LP = linear_probability(input_sample, n_bits, 1)[0]
    SAC = avelanch_criterion(input_sample, n_bits, 1)[0]
    DEG = single_bit_algebraic_degree(input_sample, n_bits)

    LP_q = int(round(LP * 255))
    SAC_q = int(round(SAC * 255))
    DEG_q = int(DEG)

    # Return numpy array to avoid PyTorch overhead inside the worker loop
    return np.array([LP_q, DEG_q, SAC_q], dtype=np.uint8)

# --- 2. WORKER FUNCTION ---
def label_worker(args):
    inputs_batch, n_bits = args
    batch_size = len(inputs_batch)
    vec_len = 1 << n_bits
    # A. Vectorized Bit Extraction
    shifts = np.arange(vec_len, dtype=np.uint32) # LSB at index 0
    # (Batch, 1) >> (n_bits,) -> (Batch, n_bits)
    raw_bits = ((inputs_batch[:, None] >> shifts) & 1).astype(np.uint8)

    # B. Calculate Labels
    targets = np.zeros((batch_size, 3), dtype=np.uint8)

    # if not all([sum(i) == (1 << (N_BITS -1)) for i in raw_bits]):
    #     print("Error: Not all raw_bits are balanced!")
    #     print("HW should be ", (1 << ( N_BITS -1)), " but HW is ", [sum(i) for i in raw_bits][:10])
    #     print(len(raw_bits)- sum([sum(i) == (1 << ( (1 << int(N_BITS)) -1)) for i in raw_bits]), " incorrect out of ", len(raw_bits))
    #     return
    # else:
    #     print("All raw_bits are balanced.")

    for i in range(batch_size):
        targets[i] = label(n_bits, raw_bits[i])

    # C. Pack Bits
    packed_inputs = np.packbits(raw_bits, axis=1)

    # if not all([(sum([i.bit_count() for i in input]) == (1 << ( N_BITS -1))) for input in packed_inputs]):
    #     print("Error in worker: Not all packed inputs are balanced!")
    #     print("HW should be ", (1 << ( N_BITS -1)), " but HW is ", [sum([i.bit_count() for i in input]) for input in packed_inputs][:10])
    #     print(len(packed_inputs)- sum([(sum([i.bit_count() for i in input]) == (1 << ( N_BITS -1))) for input in packed_inputs]), " incorrect out of ", len(packed_inputs))
    #     return
    # else:
    #     print(" in worker: All packed inputs are balanced.")


    return packed_inputs, targets

# --- 3. INPUT GENERATOR (OPTIMIZED ONE-SHOT) ---
def generate_balanced_inputs(n_samples, n_bits):
    """
    Generates n_samples of unique balanced vectors (Hamming weight = N/2).
    Strategy: Oversample -> Unique -> Shuffle.
    """
    vec_dim = 1 << n_bits
    k = vec_dim // 2
    
    # LIMITATION CHECK: 
    # n_bits=5 -> vec_dim=32 (Fits in uint32)
    # n_bits=6 -> vec_dim=64 (Fits in uint64)
    # n_bits=7 -> vec_dim=128 (CRASH: Cannot fit in numpy integer)
    if vec_dim > 64:
        raise ValueError(f"This script relies on numpy integers, which max out at 64 bits. "
                         f"Your N_BITS={n_bits} requires {vec_dim} bits. "
                         f"Please use N_BITS <= 6.")

    dtype = np.uint32 if vec_dim <= 32 else np.uint64
    total_possible = math.comb(vec_dim, k)
    
    if n_samples > total_possible:
        raise ValueError(f"Requested {n_samples:,} samples, but only {total_possible:,} exist.")

    print(f"Generating {n_samples:,} samples (Space: {total_possible:,})...")
    
    # 1. Oversample Calculation
    # We generate more candidates to handle collisions in one go.
    # If we need 100M from 600M space, we generate ~1.3x
    fill_ratio = n_samples / total_possible
    oversample_factor = 1.3 if fill_ratio < 0.5 else 3.0
    generate_count = int(n_samples * oversample_factor)
    
    # 2. Vectorized Generation (All at once)
    # This takes ~800MB - 1.5GB RAM for 120M samples (Safe)
    print(f"  - Generating {generate_count:,} candidates...")
    
    # Process in chunks to avoid spiking RAM with the float noise array
    candidates_list = []
    generated_so_far = 0
    gen_chunk_size = 10_000_000 # Generate 10M at a time
    
    with tqdm(total=generate_count, desc="Raw Generation") as pbar:
        while generated_so_far < generate_count:
            current_batch_size = min(gen_chunk_size, generate_count - generated_so_far)
            
            noise = np.random.rand(current_batch_size, vec_dim)
            top_k = np.argsort(noise, axis=1)[:, -k:]
            
            batch_ints = np.zeros(current_batch_size, dtype=dtype)
            # count = 0
            for col in range(k):
                batch_ints |= (np.array(1, dtype=dtype) << top_k[:, col]).astype(dtype)
                # count += 1
            # if count != N_BITS//2:
            #     print("Error in bit setting logic!")
            #     print("Count =", count, ", Expected =", (1<<(N_BITS-1)))
            #     exit()
            # if not all([i.bit_count() == N_BITS/2 for i in batch_ints]):
            #     print("Error: Not all generated candidates are balanced!")
            #     print(len(batch_ints)- sum([i.bit_count() ==  N_BITS/2 for i in batch_ints]), " incorrect out of ", len(batch_ints))
            #     exit()

            candidates_list.append(batch_ints)
            generated_so_far += current_batch_size
            pbar.update(current_batch_size)
            del noise, top_k # Free RAM immediately
            
    # 3. Concatenate and Unique
    print("  - Filtering duplicates...")
    all_candidates = np.concatenate(candidates_list)
    unique_candidates = np.unique(all_candidates)
    
    # 4. Check if we have enough
    if len(unique_candidates) < n_samples:
        raise ValueError(f"Collision rate was higher than expected! "
                         f"Got {len(unique_candidates):,} unique samples. "
                         f"Please re-run or increase oversample_factor.")
        
    print(f"  - Success! Got {len(unique_candidates):,} unique samples.")
    
    # 5. Shuffle and Trim
    np.random.shuffle(unique_candidates)
    return unique_candidates[:n_samples]

# --- 4. MAIN ORCHESTRATOR ---
if __name__ == '__main__':
    # A. Generate Inputs
    try:
        all_inputs_uint32 = generate_balanced_inputs(TOTAL_SAMPLES, N_BITS)
    except ValueError as e:
        print(e)
        exit()
    # B. Setup Disk Storage
    packed_dim = (1 << N_BITS) // 8  # e.g., 32 bits / 8 = 4 bytes
    
    print(f"Storage: Inputs({TOTAL_SAMPLES}, {packed_dim}) | Targets({TOTAL_SAMPLES}, 3)")
    
    fp_in = np.memmap(FILE_INPUTS, dtype='uint8', mode='w+', shape=(TOTAL_SAMPLES, packed_dim))
    fp_out = np.memmap(FILE_TARGETS, dtype='uint8', mode='w+', shape=(TOTAL_SAMPLES, 3))
    
    # C. Prepare Multiprocessing
    num_chunks = TOTAL_SAMPLES // CHUNK_SIZE
    chunks = np.array_split(all_inputs_uint32, num_chunks)
    tasks = [(chunk, (1 << N_BITS)) for chunk in chunks] # Pass vector length (e.g., 32)
    # c1 = chunks[0]
    # if not all([i.bit_count() == (1 << ( N_BITS -1)) for i in c1]):
    #     print("Error: Not all inputs in first chunk are balanced!")
    #     print("HW should be ", (1 << ( N_BITS -1)), " but HW is ", [i.bit_count() for i in c1][:10])
    #     print(len(c1)- sum([i.bit_count() ==  N_BITS/2 for i in c1]), " incorrect out of ", len(c1))
    #     exit()
    # else:
    #     print("All inputs in first chunk are balanced.")

    # Correcting n_bits passing:
    # label_worker expects 'n_bits' to be the VECTOR LENGTH (e.g. 32) 
    # or the Exponent (e.g. 5)?
    # Your `label` function uses `shifts = np.arange(n_bits)`.
    # So if you pass 5, shifts=[0,1,2,3,4]. That extracts 5 bits.
    # But your vector is 32 bits long!
    # CORRECTION: You should pass the vector length (vec_dim) to label_worker
    tasks = [(chunk, N_BITS) for chunk in chunks]
    workers_count = min(cpu_count(), len(tasks))
    print(f"Using {workers_count} workers for {len(tasks)} tasks...")
    
    pool = None
    try:
        pool = Pool(workers_count, initializer=init_worker)
        results_iter = pool.imap(label_worker, tasks)
        
        cursor = 0
        for inputs_packed, targets_batch in tqdm(results_iter, total=len(tasks)):

            # if not all([(sum([i.bit_count() for i in input]) == (1 << ( N_BITS -1))) for input in inputs_packed]):
            #     print("Error: Not all packed inputs are balanced!")
            #     print("HW should be ", (1 << ( N_BITS -1)), " but HW is ", [sum([i.bit_count() for i in input]) for input in inputs_packed][:10])
            #     print(len(inputs_packed)- sum([(sum([i.bit_count() for i in input]) == (1 << ( N_BITS -1))) for input in inputs_packed]), " incorrect out of ", len(inputs_packed))
            #     exit()
            # else:
            #     print("All packed inputs are balanced.")
            n = inputs_packed.shape[0]
            fp_in[cursor : cursor+n] = inputs_packed
            fp_out[cursor : cursor+n] = targets_batch
            cursor += n
            
            if cursor % (CHUNK_SIZE * 10) == 0:
                fp_in.flush()
                fp_out.flush()

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating workers...")
        if pool is not None:
            pool.terminate()
            pool.join()
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred: {e}. Terminating workers...")
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    finally:
        if pool is not None:
            pool.terminate()  # Ensure cleanup even on exit()

    fp_in.flush()
    fp_out.flush()
    print("Done.")