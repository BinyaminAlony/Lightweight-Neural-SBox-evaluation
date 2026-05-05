from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch


CHUNK_PATTERN = re.compile(r"^chunk_(\d+)(?:_fin)?\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity checks for merged chunk datasets (.pt files)."
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=None,
        help="Directory with chunk files. Defaults to merged_chunks_YYYY-MM-DD in the current workspace.",
    )
    parser.add_argument(
        "--first-n-dist",
        type=int,
        default=50,
        help="Number of first chunks to use for global label/LP distribution statistics.",
    )
    parser.add_argument(
        "--first-n-compare",
        type=int,
        default=10,
        help="Number of first chunks to compare distribution similarity.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Decimal precision for floating labels before counting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many most common values/vectors to print.",
    )
    parser.add_argument(
        "--include-vector-distribution",
        action="store_true",
        help="Also count full label-vector frequencies for the first N chunks. This may use significant memory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N processed chunk files.",
    )
    parser.add_argument(
        "--total-count-mode",
        choices=["estimate-uniform", "exact"],
        default="estimate-uniform",
        help="How to compute total samples: quick estimate from chunk_0 size or exact scan of all chunks.",
    )
    return parser.parse_args()


def default_chunks_dir() -> Path:
    workspace_root = Path(__file__).resolve().parent
    date_suffix = datetime.now().strftime("%Y-%m-%d")
    return workspace_root / f"merged_chunks_{date_suffix}"


def chunk_index(path: Path) -> int:
    match = CHUNK_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected chunk filename format: {path.name}")
    return int(match.group(1))


def list_chunk_files(chunks_dir: Path) -> list[Path]:
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")

    files = [p for p in chunks_dir.glob("chunk_*.pt") if CHUNK_PATTERN.match(p.name)]
    if not files:
        raise FileNotFoundError(f"No chunk files found in: {chunks_dir}")

    return sorted(files, key=chunk_index)


def normalize_value(value: float, decimals: int):
    as_float = float(value)
    rounded_int = round(as_float)
    if abs(as_float - rounded_int) < 1e-10:
        return int(rounded_int)
    return round(as_float, decimals)


def labels_to_2d_tensor(raw_labels) -> torch.Tensor:
    if isinstance(raw_labels, list):
        if len(raw_labels) == 0:
            return torch.empty((0, 0), dtype=torch.float32)
        labels_tensor = torch.stack([torch.as_tensor(item) for item in raw_labels])
    else:
        labels_tensor = torch.as_tensor(raw_labels)

    labels_tensor = labels_tensor.detach().cpu()

    if labels_tensor.ndim == 1:
        labels_tensor = labels_tensor.unsqueeze(1)
    elif labels_tensor.ndim > 2:
        labels_tensor = labels_tensor.reshape(labels_tensor.shape[0], -1)

    return labels_tensor.float()


def load_labels(file_path: Path) -> torch.Tensor:
    loaded = torch.load(file_path, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "labels" in loaded:
        return labels_to_2d_tensor(loaded["labels"])
    if hasattr(loaded, "labels"):
        return labels_to_2d_tensor(loaded.labels)
    if isinstance(loaded, torch.Tensor):
        return labels_to_2d_tensor(loaded)

    raise ValueError(
        f"Unsupported format in {file_path}. Expected dict['labels'], object.labels, or tensor."
    )


def labels_to_rows(labels: torch.Tensor, decimals: int) -> list[tuple]:
    return [
        tuple(normalize_value(v, decimals) for v in row)
        for row in labels.tolist()
    ]


def tv_distance(counter_a: Counter, total_a: int, counter_b: Counter, total_b: int) -> float:
    keys = set(counter_a.keys()) | set(counter_b.keys())
    if total_a == 0 or total_b == 0 or not keys:
        return 0.0

    distance = 0.0
    for key in keys:
        p = counter_a.get(key, 0) / total_a
        q = counter_b.get(key, 0) / total_b
        distance += abs(p - q)
    return 0.5 * distance


def similarity_band(max_tv: float) -> str:
    if max_tv < 0.03:
        return "very similar"
    if max_tv < 0.07:
        return "similar"
    if max_tv < 0.12:
        return "moderately similar"
    return "noticeable shift"


def print_counter(counter: Counter, total: int, title: str, top_k: int) -> None:
    print(f"\n{title}")
    if total == 0:
        print("  No samples")
        return

    for key, count in counter.most_common(top_k):
        pct = 100.0 * count / total
        print(f"  {key} -> {count} ({pct:.2f}%)")

    remaining = len(counter) - min(top_k, len(counter))
    if remaining > 0:
        print(f"  ... {remaining} more unique values not shown")


def main() -> None:
    args = parse_args()
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else default_chunks_dir()
    chunk_files = list_chunk_files(chunks_dir)

    print(f"Using chunk directory: {chunks_dir}")
    print(f"Found {len(chunk_files)} chunk files")

    label_dim: int | None = None
    first_chunk_size: int | None = None

    first_n_dist = min(args.first_n_dist, len(chunk_files))
    first_n_compare = min(args.first_n_compare, len(chunk_files))
    first_n_needed = max(first_n_dist, first_n_compare)

    global_samples = 0
    global_per_label: list[Counter] | None = None
    global_vector_counter: Counter | None = Counter() if args.include_vector_distribution else None

    chunk_label_counters: list[list[Counter]] = []
    chunk_sizes: list[int] = []
    compare_indices: list[int] = []

    if args.total_count_mode == "exact":
        files_to_scan = chunk_files
    else:
        files_to_scan = chunk_files[:first_n_needed]

    total_samples_exact = 0

    for file_idx, file_path in enumerate(files_to_scan):
        labels = load_labels(file_path)
        n_rows, n_cols = labels.shape
        if first_chunk_size is None:
            first_chunk_size = n_rows

        if label_dim is None:
            label_dim = n_cols
            global_per_label = [Counter() for _ in range(label_dim)]
        elif n_cols != label_dim:
            raise ValueError(
                f"Label dimension mismatch in {file_path.name}: expected {label_dim}, got {n_cols}"
            )

        total_samples_exact += n_rows

        if file_idx >= first_n_needed:
            if args.progress_every > 0 and (file_idx + 1) % args.progress_every == 0:
                print(f"Processed {file_idx + 1}/{len(files_to_scan)} chunks...")
            continue

        rows = labels_to_rows(labels, args.round_decimals)

        if file_idx < first_n_dist:
            global_samples += len(rows)
            if global_vector_counter is not None:
                global_vector_counter.update(rows)
            for row in rows:
                for i, value in enumerate(row):
                    global_per_label[i][value] += 1

        if file_idx < first_n_compare:
            compare_indices.append(chunk_index(file_path))
            chunk_sizes.append(len(rows))
            counters = [Counter() for _ in range(label_dim)]
            for row in rows:
                for i, value in enumerate(row):
                    counters[i][value] += 1
            chunk_label_counters.append(counters)

        if args.progress_every > 0 and (file_idx + 1) % args.progress_every == 0:
            print(f"Processed {file_idx + 1}/{len(files_to_scan)} chunks...")

    if len(files_to_scan) % max(args.progress_every, 1) != 0:
        print(f"Processed {len(files_to_scan)}/{len(files_to_scan)} chunks...")

    if args.total_count_mode == "exact":
        total_samples = total_samples_exact
        total_mode_note = "exact"
    else:
        if first_chunk_size is None:
            raise RuntimeError("Could not infer chunk size for total sample estimation.")
        total_samples = first_chunk_size * len(chunk_files)
        total_mode_note = "estimate-uniform"

    print(f"\n[1] Total samples across all chunks: {total_samples} ({total_mode_note})")

    print(f"\n[2] Distribution in first {first_n_dist} chunks")
    print(f"Samples analyzed: {global_samples}")

    if global_vector_counter is not None:
        print_counter(
            global_vector_counter,
            global_samples,
            "Top full label-vector frequencies:",
            args.top_k,
        )
    else:
        print("\nTop full label-vector frequencies: skipped (use --include-vector-distribution to enable)")

    for i, counter in enumerate(global_per_label or []):
        name = "LP" if i == 0 else f"Label[{i}]"
        print_counter(counter, global_samples, f"Top values for {name}:", args.top_k)

    print(f"\n[3] Similarity across first {first_n_compare} chunks")
    if first_n_compare < 2:
        print("Need at least 2 chunks to compare distributions.")
        return

    for label_idx in range(label_dim or 0):
        label_name = "LP" if label_idx == 0 else f"Label[{label_idx}]"
        pairwise: list[tuple[float, tuple[int, int]]] = []

        for a, b in combinations(range(first_n_compare), 2):
            tv = tv_distance(
                chunk_label_counters[a][label_idx],
                chunk_sizes[a],
                chunk_label_counters[b][label_idx],
                chunk_sizes[b],
            )
            pairwise.append((tv, (compare_indices[a], compare_indices[b])))

        tv_values = [item[0] for item in pairwise]
        avg_tv = sum(tv_values) / len(tv_values)
        min_tv = min(tv_values)
        max_tv = max(tv_values)
        max_pair = max(pairwise, key=lambda item: item[0])[1]

        print(
            f"{label_name}: avg TV={avg_tv:.4f}, min TV={min_tv:.4f}, "
            f"max TV={max_tv:.4f} (chunks {max_pair[0]} vs {max_pair[1]}) -> {similarity_band(max_tv)}"
        )


if __name__ == "__main__":
    main()
