import random

import numpy as np
from tqdm import tqdm
from sbox_metrics.linear_probability import linear_probability
from sbox_metrics.differential_probability import differential_probability
from sbox_metrics.avelanch_criterion import avelanch_criterion
from utils.file_management import load_from_csv
from utils.binary_utils import bin_vec_to_dec, dec_to_bin_vec

def is_invertible_gf2(matrix):
    """Checks if a binary matrix is invertible over GF(2) using Gaussian elimination."""
    n = matrix.shape[0]
    mat = matrix.astype(np.int8).copy()
    
    rank = 0
    for col in range(n):
        if rank >= n:
            break
        # Find a pivot in the current column (from the current rank row downwards)
        pivot_row = np.where(mat[rank:, col] == 1)[0]
        
        if len(pivot_row) > 0:
            i = pivot_row[0] + rank
            # Swap current row with pivot row
            mat[[rank, i]] = mat[[i, rank]]
            
            # Eliminate other 1s in this column using XOR
            for k in range(n):
                if k != rank and mat[k, col] == 1:
                    mat[k] ^= mat[rank]
            rank += 1
            
    return rank == n

def generate_invertible_gf2(n):
    while True:
        # Generate a random n x n binary matrix
        matrix = np.random.randint(0, 2, (n, n), dtype=np.int8)
        if is_invertible_gf2(matrix):
            return matrix

def matrix_to_int(matrix):
    """Converts a binary matrix to a single integer."""
    # Flatten and convert bits to a string, then to an int
    # Or use bit-shifting for better performance
    res = 0
    for row in matrix:
        for bit in row:
            res = (res << 1) | int(bit)
    return res

def multiply_int_by_matrix(integer, matrix):
    """Multiplies an integer by a binary matrix over GF(2)."""
    # Convert integer to a binary vector
    vec = dec_to_bin_vec(integer, len(matrix))
    # Multiply the vector by the matrix
    result_vec = np.mod(np.dot(vec, matrix), 2)
    # Convert the result back to an integer
    return bin_vec_to_dec(result_vec)

def get_fast_transform(matrix):
    # Pre-compute the linear part
    return np.array([multiply_int_by_matrix(x, matrix) for x in range(32)], dtype=np.int8)

def encode_vector_32x5(vec):
    """
    Encodes a vector of 32 integers (0-31) into one 160-bit integer.
    """
    res = 0
    for val in vec:
        # Shift left by 5 bits and add the next value
        res = (res << 5) | int(val)
    return res

def generate_new_sboxes(n: int, wanted_count: int, load_sboxes_from: str):
    original_boxes = load_from_csv(load_sboxes_from)
    funcs = []
    old_size = 0
    seen_sboxes = set()
    with tqdm(total=wanted_count, desc="Generating new S-boxes") as pbar:
        while len(seen_sboxes) < wanted_count:
            matrix1 = generate_invertible_gf2(n)
            matrix2 = generate_invertible_gf2(n)
            x_mask = random.randint(0, 2**n - 1)
            y_mask = random.randint(0, 2**n - 1)
            new_Sboxes = [[multiply_int_by_matrix(Sbox[multiply_int_by_matrix(x, matrix1)^x_mask], matrix2)^y_mask for x in range(1<<n)] for Sbox in original_boxes]
            for new_Sbox in new_Sboxes:
                seen_sboxes.add(encode_vector_32x5(new_Sbox))
            if len(seen_sboxes) > old_size:
                funcs.exend(new_Sboxes)
                old_size = len(seen_sboxes)
                pbar.update(len(seen_sboxes) - old_size)
    print(f"Generated {len(funcs)} new S-boxes with {len(seen_sboxes)} unique transformation matrices.")

# metric_funcs = [
#     lambda f: linear_probability(f, n, n)[0],
#     lambda f: differential_probability(f, n)[0],
#     lambda f: max([abs(0.5-i) for i in avelanch_criterion(f, n)])
# ]
# metric_names = ["LP", "DP", "SAC"]
# for metric_func, metric_name in zip(metric_funcs, metric_names):
#     counts = dict()
#     for f in tqdm(funcs, leave=False):
#         metric = metric_func(f)
#         if metric in counts:
#             counts[metric] += 1
#         else:
#             counts[metric] = 1
#     print(f"{metric_name}: {counts}")

