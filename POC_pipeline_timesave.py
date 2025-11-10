import os
import glob
from random import sample, shuffle
import torch
from tqdm import tqdm
from testtttttttt import BinaryCNN
from utils.binary_utils import dec_to_bin_vec
SUPPORTED_EXTS = [".pt", ".pth"]
from torch.utils.data import DataLoader, TensorDataset
from sbox_metrics.linear_probability import linear_probability
import numpy as np
import time

def find_model_file(name, search_dir="."):
    if os.path.exists(name):
        return os.path.abspath(name)
    pattern = os.path.join(search_dir, f"{name}*")
    candidates = []
    for ext in SUPPORTED_EXTS:
        candidates += glob.glob(pattern + ext)
    candidates += glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No model file found for '{name}' in '{search_dir}'")
    return os.path.abspath(sorted(candidates)[0])

def generate_dataset(n_bits, dataset_size):
    n_elem = 1<<n_bits
    halfhalf = [0]*(n_elem//2) + [1]*(n_elem//2)
    res = []
    for _ in tqdm(range(dataset_size)):
        shuffle(halfhalf)
        res.append(halfhalf.copy())
    return res

def main():
    
    # load model:
    MODEL_NAME = os.environ.get("MODEL_NAME", "BNN_Model_50_10_Million_samples_LP_DEG_SAC")   # name or path (e.g. "mymodel", "mymodel.pt")
    SEARCH_DIR = "C:\\Users\\USER\\Documents\\binyamin\\studies\\סמסטר ח\\ארכיטקטורות מתקדמות\\code\\new code\\Lightweight-Neural-SBox-evaluation\\models"                                     # directory to search if name is not a path
    PATH  = find_model_file(MODEL_NAME, SEARCH_DIR)
    
    print("loading model...")
    model = BinaryCNN()
    model.load_state_dict(torch.load(PATH, weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print("model loaded using device:", device, "\n")
    
    # Run experiments for multiple sample sizes and collect timings + TP drop
    sizes = [1_000, 10_000, 50_000, 100_000, 200_000, 500_000]
    max_n = max(sizes)

    # print(f"generating dataset for up to {max_n} samples (will reuse for smaller sizes)")
    # inputs_as_nums = sample(range(4294967296), max_n)
    # samples_list = [dec_to_bin_vec(x, 32) for x in tqdm(inputs_as_nums, desc="Generating dataset")]
    # samples = torch.tensor(samples_list, dtype=torch.float32)
    # print("dataset generated\n")

    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 512))

    nn_times = []
    bf_times = []
    nn_tps = []
    bf_tps = []

    for n in sizes:
        print(f"\n=== Running experiment for n={n} ===")
        # Generate fresh random samples for this experiment size
        # inputs_as_nums_n = sample(range(4294967296), n)
        # subsamples_list = [dec_to_bin_vec(x, 32) for x in inputs_as_nums_n]
        # subsamples = torch.tensor(subsamples_list, dtype=torch.float32)
        subsamples = generate_dataset(5, n)
        print(f"uniques: {len(set(tuple(x) for x in subsamples))}, true positives: {len([1 for func in subsamples if linear_probability(func, 5, 1)[0] <= 0.0625])}")
        # NN pipeline (inference + filtering)
        NN_start_time = time.perf_counter()
        dataset = TensorDataset(torch.tensor(subsamples, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device == 'cuda'))

        positives = []
        with torch.no_grad():
            for batch in loader:
                input_batch = batch[0].to(device)
                output = model(input_batch)
                # model in eval returns +/-1; mask where model predicts bit0 == 1
                mask = output[:, 0] == 1
                if mask.any():
                    positive_batch = input_batch[mask].detach().cpu()
                    positives.extend(positive_batch)

        # filter positives to true positives
        true_positives = []
        for func in positives:
            # func is a tensor of shape (32,), values -1/1
            if linear_probability([int(y) for y in func], 5, 1)[0] <= 0.0625:
                true_positives.append(func)

        NN_finish_time = time.perf_counter()
        nn_time = NN_finish_time - NN_start_time
        nn_times.append(nn_time)
        nn_tps.append(len(true_positives))
        print(f"n={n}: NN positives after model={len(positives)}, NN true positives={len(true_positives)}, time={nn_time:.2f}s")

        # Bruteforce baseline
        BF_start_time = time.perf_counter()
        BF_positives = []
        # for func in subsamples_list:
        #     if linear_probability([int(y) for y in func], 5, 1)[0] <= 0.0625:
        for func in subsamples:
            if linear_probability(func, 5, 1)[0] <= 0.0625:
                BF_positives.append(func)
        BF_finish_time = time.perf_counter()
        bf_time = BF_finish_time - BF_start_time
        bf_times.append(bf_time)
        bf_tps.append(len(BF_positives))
        print(f"n={n}: BF true positives={len(BF_positives)}, time={bf_time:.2f}s")

    # Compute dropped percentage: fraction of BF TP missed by NN
    drop_percents = []
    for nn_tp, bf_tp in zip(nn_tps, bf_tps):
        if bf_tp == 0:
            drop_percents.append(0.0)
        else:
            drop_percents.append(100.0 * (bf_tp - nn_tp) / bf_tp)

    # Print summary table
    print("\nSummary across sizes:")
    print("n, BF_time(s), NN_time(s), BF_TP, NN_TP, TP_drop(%)")
    for n, bf_t, nn_t, bf_tp, nn_tp, drop in zip(sizes, bf_times, nn_times, bf_tps, nn_tps, drop_percents):
        print(f"{n}, {bf_t:.2f}, {nn_t:.2f}, {bf_tp}, {nn_tp}, {drop:.2f}%")

    # Plot timings and TP drop
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(sizes, bf_times, marker='o', label='Brute Force')
    ax1.plot(sizes, nn_times, marker='o', label='NN + filter')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of samples (log scale)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Processing time: BF vs NN')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.5)

    ax2.plot(sizes, drop_percents, marker='s', color='tab:red')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of samples (log scale)')
    ax2.set_ylabel('TP dropped (%)')
    ax2.set_title('True Positive dropped percentage (BF vs NN)')
    ax2.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    out_png = 'bf_nn_times_tpdrop.png'
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")
    
if __name__ == "__main__":
    main()