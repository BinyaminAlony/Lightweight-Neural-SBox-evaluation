"""
main_script_memmap.py
=====================
BNN (Binary Neural Network) training & evaluation script that works with
the **memmap .bin dataset** produced by ``generate_NN_dataset_LP_AV_ADEG.py``.

=====================================================================
Dataset difference  (old  ←→  new)
=====================================================================

Old pipeline  (main_script.py  +  get_NN_dataset.py / create_NN_dataset.py)
  ▸ Storage  : single ``.pt`` file  (serialised ``CustomDataset`` object)
  ▸ Inputs   : float32 tensor (32,)  with values {0, 1}
               – drawn from *all* 32-bit integers (not necessarily balanced).
  ▸ Labels   : 9 binary bits  {-1, +1}
               4 LP-bucket bits  +  4 DEG-bucket bits  +  1 SAC-threshold bit
  ▸ Task     : multi-label binary classification

New pipeline  (this script  +  generate_NN_dataset_LP_AV_ADEG.py)
  ▸ Storage  : two ``.bin`` memory-mapped files (inputs + targets)
  ▸ Inputs   : packed uint8 bytes  →  unpacked to float32 (32,) with {0, 1}
               – **balanced** vectors only  (Hamming weight = vec_dim / 2).
  ▸ Labels   : 3 quantised uint8 values  →  normalised to [0, 1] for training:
               LP  (0-255 → 0-1),  DEG  (0-N_BITS → 0-1),  SAC  (0-255 → 0-1)
  ▸ Task     : multi-output **regression**

=====================================================================
Model
=====================================================================
Same BinaryMLP / BinaryCNN architecture as ``main_script.py``:
  • Internal activations are binarised to {-1, +1} at inference time,
    preserving the BNN character.
  • The **final output layer is NOT binarised** — it produces 3 continuous
    regression predictions instead of 9 binary bits.

=====================================================================
Loss
=====================================================================
Smooth-L1 (Huber) loss on the normalised [0, 1] targets.
Per-output weighting is available via ``LOSS_WEIGHTS``.

=====================================================================
Evaluation metrics
=====================================================================
Per-metric **MAE** reported on original scales:
  LP  — Mean Absolute Error on [0, 1]  (the raw probability)
  DEG — Mean Absolute Error on integer scale  [0 … N_BITS]
  SAC — Mean Absolute Error on [0, 1]

=====================================================================
Outputs
=====================================================================
  results/memmap_<TYPE>_<EPOCHS>ep_<REPS>reps.csv   – per-rep MAE + summary
  results/memmap_<TYPE>_<EPOCHS>ep_<REPS>reps.png   – bar-chart of MAE per rep

=====================================================================
Usage
=====================================================================
  1. Generate the dataset first:
         python generate_NN_dataset_LP_AV_ADEG.py
     This creates  data_<TOTAL_SAMPLES>_n<N_BITS>_inputs.bin
                    data_<TOTAL_SAMPLES>_n<N_BITS>_targets.bin

  2. Run this script:
         python main_script_memmap.py

  3. Adjust the CONFIGURATION block below as needed.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURATION
# =====================================================================
N_BITS          = 5                # Must match the generation script
TOTAL_SAMPLES   = 100_000_000      # Must match the generation script
VEC_DIM         = 1 << N_BITS      # 32 for N_BITS=5
PACKED_DIM      = VEC_DIM // 8     # 4  for N_BITS=5

TYPE            = "MLP"            # "MLP" or "CNN"
HIDDEN_DIM      = 256
EPOCHS          = 25
REPS            = 5               # Independent training repetitions
TRAIN_RATIO     = 0.8
BATCH_TRAIN     = 256              # Larger than main_script (64) to cope with 100M samples
BATCH_TEST      = 512
LR              = 1e-3
WD              = False            # Weight-decay toggle

NUM_WORKERS     = 4                # DataLoader workers (0 = main-process only)

# File paths  (must match generate_NN_dataset_LP_AV_ADEG.py output names)
FILE_INPUTS     = f'data_{TOTAL_SAMPLES}_n{N_BITS}_inputs.bin'
FILE_TARGETS    = f'data_{TOTAL_SAMPLES}_n{N_BITS}_targets.bin'

# Target normalisation constants
LP_SCALE        = 255.0
DEG_SCALE       = float(N_BITS)    # Max algebraic degree equals N_BITS
SAC_SCALE       = 255.0

# Optional per-output loss weighting  [LP, DEG, SAC]
# Set all to 1.0 for equal weighting.
LOSS_WEIGHTS    = [1.0, 1.0, 1.0]


# =====================================================================
# DATASET — reads the .bin files produced by generate_NN_dataset_LP_AV_ADEG.py
# =====================================================================
class MemmapSBoxDataset(Dataset):
    """
    Memory-mapped PyTorch Dataset for the binary S-box evaluation data.

    Each sample:
        input  – float32 tensor  (VEC_DIM,)   values in {0.0, 1.0}
                 (unpacked from the packed-byte representation on disk)
        target – float32 tensor  (3,)         normalised to [0, 1]
                 [LP / 255,  DEG / N_BITS,  SAC / 255]

    Because data lives on disk via ``np.memmap``, RAM usage stays low
    even for hundreds of millions of samples.

    NOTE: The memmap arrays are opened **lazily** (on first ``__getitem__``
    call) so that the Dataset can be pickled by DataLoader workers on
    Windows.  Each worker process opens its own read-only memmap handle.
    """

    def __init__(self, inputs_path: str, targets_path: str,
                 n_samples: int, packed_dim: int,
                 lp_scale: float = 255.0,
                 deg_scale: float = 5.0,
                 sac_scale: float = 255.0):
        if not os.path.isfile(inputs_path):
            raise FileNotFoundError(
                f"Input file not found: {inputs_path}\n"
                f"Run generate_NN_dataset_LP_AV_ADEG.py first.")
        if not os.path.isfile(targets_path):
            raise FileNotFoundError(
                f"Target file not found: {targets_path}\n"
                f"Run generate_NN_dataset_LP_AV_ADEG.py first.")

        # Store paths & shape so we can re-open in each worker process
        self.inputs_path  = inputs_path
        self.targets_path = targets_path
        self.n_samples    = n_samples
        self.packed_dim   = packed_dim

        self.lp_scale  = lp_scale
        self.deg_scale = deg_scale
        self.sac_scale = sac_scale

        # Lazy handles — opened on first access (see _ensure_open)
        self._inputs  = None
        self._targets = None

    def _ensure_open(self):
        """Open memmap files if not already open (called per-worker)."""
        if self._inputs is None:
            self._inputs  = np.memmap(self.inputs_path,  dtype='uint8',
                                      mode='r',
                                      shape=(self.n_samples, self.packed_dim))
            self._targets = np.memmap(self.targets_path, dtype='uint8',
                                      mode='r',
                                      shape=(self.n_samples, 3))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        self._ensure_open()

        # ---- Unpack packed bytes → bit vector  (VEC_DIM,) ----
        packed  = self._inputs[idx]
        bits    = np.unpackbits(packed).astype(np.float32)  # {0.0, 1.0}

        # ---- Normalise targets to [0, 1] ----
        raw     = self._targets[idx].copy().astype(np.float32)
        raw[0] /= self.lp_scale    # LP   : 0-255 → 0-1
        raw[1] /= self.deg_scale   # DEG  : 0-N   → 0-1
        raw[2] /= self.sac_scale   # SAC  : 0-255 → 0-1

        return torch.from_numpy(bits), torch.from_numpy(raw)


# =====================================================================
# MODELS — same BNN architecture, adapted for 3 regression outputs
# =====================================================================
class BinaryCNN(nn.Module):
    """
    1-D convolutional BNN (mirrors ``main_script.py`` BinaryCNN).

    Architecture
    ------------
    Conv1d(1→16, k=3) → ReLU [→ binarise@eval]
    Conv1d(16→32, k=3) → ReLU [→ binarise@eval]
    Flatten → Linear(32*VEC_DIM → n_outputs)

    The final layer outputs raw regression values (no binarisation).
    """

    def __init__(self, n_outputs: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.conv1   = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.act1    = nn.ReLU()
        self.conv2   = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.act2    = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(32 * VEC_DIM, n_outputs)
        self.model_type = "CNN"

    def forward(self, x):
        x = x.unsqueeze(1)                          # (B, 1, VEC_DIM)
        if self.training:
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))
            return self.fc(self.flatten(x))
        else:
            # --- BNN-style inference: binarise intermediate activations ---
            x = self.act1(self.conv1(x))
            x = torch.where(x > 0, 1.0, -1.0)
            x = self.act2(self.conv2(x))
            x = torch.where(x > 0, 1.0, -1.0)
            return self.fc(self.flatten(x))          # raw regression output


class BinaryMLP(nn.Module):
    """
    Fully-connected BNN (mirrors ``main_script.py`` BinaryMLP).

    Architecture
    ------------
    Linear(VEC_DIM → hidden_dim) → ReLU [→ binarise@eval]
    Linear(hidden_dim → n_outputs)

    The final layer outputs raw regression values (no binarisation).
    """

    def __init__(self, n_outputs: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.flatten     = nn.Flatten()
        self.inl         = nn.Linear(VEC_DIM, hidden_dim)
        self.activation1 = nn.ReLU()
        self.outl        = nn.Linear(hidden_dim, n_outputs)
        self.model_type  = "MLP"

    def forward(self, x):
        if self.training:
            x = self.flatten(x)
            x = self.activation1(self.inl(x))
            return self.outl(x)
        else:
            # --- BNN-style inference: binarise input & hidden activations ---
            x = torch.where(self.flatten(x) > 0, 1.0, -1.0)
            x = self.activation1(self.inl(x))
            x = torch.where(x > 0, 1.0, -1.0)
            return self.outl(x)                      # raw regression output


# =====================================================================
# LOSS — weighted Smooth-L1 (Huber)
# =====================================================================
class WeightedSmoothL1Loss(nn.Module):
    """
    Per-output weighted Smooth-L1 loss.

    Parameters
    ----------
    weights : list[float]
        Multiplier for each output column.  Defaults to [1, 1, 1].
    """

    def __init__(self, weights=None):
        super().__init__()
        w = torch.tensor(weights or [1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer('weights', w)
        self.base_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds, targets):
        per_el = self.base_loss(preds, targets)       # (B, 3)
        weighted = per_el * self.weights               # broadcast (3,)
        return weighted.mean()


# =====================================================================
# EVALUATION — per-metric MAE on original scales
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, device, deg_scale):
    """
    Compute per-metric MAE on the **original** (un-normalised) scales.

    Returns
    -------
    np.ndarray of shape (3,) :  [MAE_LP, MAE_DEG, MAE_SAC]
        LP  and SAC  are in [0, 1]  (the true probability / criterion value).
        DEG is on integer scale [0 … N_BITS].
    """
    model.eval()
    abs_errors = torch.zeros(3)
    count = 0
    for x, y_norm in loader:
        preds_norm = model(x.to(device)).cpu()
        diff = (preds_norm - y_norm).abs()
        # De-normalise DEG column so its MAE is on the integer scale
        diff[:, 1] *= deg_scale
        abs_errors += diff.sum(dim=0)
        count += y_norm.size(0)
    return (abs_errors / count).numpy()                # [MAE_LP, MAE_DEG, MAE_SAC]


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1.  Load dataset
    # ------------------------------------------------------------------
    print(f"Loading memmap dataset…")
    print(f"  Inputs : {FILE_INPUTS}")
    print(f"  Targets: {FILE_TARGETS}")

    dataset = MemmapSBoxDataset(
        FILE_INPUTS, FILE_TARGETS,
        n_samples  = TOTAL_SAMPLES,
        packed_dim = PACKED_DIM,
        lp_scale   = LP_SCALE,
        deg_scale  = DEG_SCALE,
        sac_scale  = SAC_SCALE,
    )

    # ------------------------------------------------------------------
    # 2.  Train / test split
    # ------------------------------------------------------------------
    total      = len(dataset)
    train_size = int(total * TRAIN_RATIO)
    test_size  = total - train_size
    print(f"Total: {total:,}  |  Train: {train_size:,}  |  Test: {test_size:,}")

    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_TEST,
                          num_workers=NUM_WORKERS, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 3.  Training loop  (mirrors main_script.py repetition structure)
    # ------------------------------------------------------------------
    N_OUTPUTS = 3
    all_results = []

    for rep in tqdm(range(REPS), desc="Repetitions"):
        # ---- fresh model + optimiser per repetition ----
        if TYPE == "MLP":
            model = BinaryMLP(n_outputs=N_OUTPUTS, hidden_dim=HIDDEN_DIM).to(device)
        else:
            model = BinaryCNN(n_outputs=N_OUTPUTS, hidden_dim=HIDDEN_DIM).to(device)

        loss_fn   = WeightedSmoothL1Loss(weights=LOSS_WEIGHTS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR,
                               weight_decay=1e-5 if WD else 0)

        # ---- epoch loop ----
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * y.size(0)

            # ---- periodic logging ----
            if epoch % 10 == 0 or epoch == 1:
                test_mae = evaluate(model, test_dl, device, DEG_SCALE)
                tqdm.write(
                    f"[Rep {rep+1}/{REPS}] Epoch {epoch:>3d}/{EPOCHS}: "
                    f"loss={running_loss / train_size:.6f}  "
                    f"Test MAE  LP={test_mae[0]:.4f}  "
                    f"DEG={test_mae[1]:.2f}  SAC={test_mae[2]:.4f}"
                )

        # ---- final evaluation for this repetition ----
        final_mae = evaluate(model, test_dl, device, DEG_SCALE)
        all_results.append(final_mae)
        tqdm.write(
            f"★ [Rep {rep+1}] Final Test MAE:  "
            f"LP={final_mae[0]:.4f}   DEG={final_mae[1]:.2f}   SAC={final_mae[2]:.4f}"
        )

    # ------------------------------------------------------------------
    # 4.  Aggregate & print results
    # ------------------------------------------------------------------
    all_results = np.array(all_results)          # (REPS, 3)
    mean_mae    = all_results.mean(axis=0)
    std_mae     = all_results.std(axis=0)

    print(f"\n{'=' * 60}")
    print(f"Results over {REPS} repetitions  ({TYPE}, {EPOCHS} epochs)")
    print(f"  LP  MAE : {mean_mae[0]:.4f}  ±  {std_mae[0]:.4f}")
    print(f"  DEG MAE : {mean_mae[1]:.2f}  ±  {std_mae[1]:.2f}")
    print(f"  SAC MAE : {mean_mae[2]:.4f}  ±  {std_mae[2]:.4f}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 5.  Save CSV results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    csv_name = f"memmap_{TYPE}_{EPOCHS}ep_{REPS}reps.csv"
    csv_path = os.path.join("results", csv_name)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rep", "MAE_LP", "MAE_DEG", "MAE_SAC"])
        for i, row in enumerate(all_results):
            writer.writerow([i + 1, *row])
        writer.writerow(["mean", *mean_mae])
        writer.writerow(["std",  *std_mae])
    print(f"Saved CSV  → {csv_path}")

    # ------------------------------------------------------------------
    # 6.  Plot
    # ------------------------------------------------------------------
    metric_names = ["LP (MAE)", "DEG (MAE)", "SAC (MAE)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, name) in enumerate(zip(axes, metric_names)):
        ax.bar(range(1, REPS + 1), all_results[:, i], alpha=0.7)
        ax.axhline(mean_mae[i], color='r', linestyle='--',
                    label=f'mean = {mean_mae[i]:.4f}')
        ax.set_title(name)
        ax.set_xlabel("Repetition")
        ax.set_ylabel("MAE")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_name = f"memmap_{TYPE}_{EPOCHS}ep_{REPS}reps.png"
    plot_path = os.path.join("results", plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Saved plot → {plot_path}")
    print("Done.")
