from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_INPUT_FOLDERS = [
    Path(r"C:\Users\USER\Documents\tommy_binyamin_proj\git-updated-version\Lightweight-Neural-SBox-evaluation\dataset_affine_to_good_sboxes_5_0149LP"),
    Path(r"C:\Users\USER\Documents\tommy_binyamin_proj\git-updated-version\Lightweight-Neural-SBox-evaluation\dataset_affine_to_good_sboxes_5_00625LP"),
    Path(r"C:\Users\USER\Documents\tommy_binyamin_proj\git-updated-version\Lightweight-Neural-SBox-evaluation\dataset_chunks_5bit_153450000size_225000cs_28_04_2026"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate matching chunk_i.pt files from multiple folders into chunk_i_fin.pt files."
    )
    parser.add_argument(
        "--input-folders",
        nargs="+",
        default=[str(folder) for folder in DEFAULT_INPUT_FOLDERS],
        help="Folders that contain matching chunk_i.pt files.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Parent folder for the dated output directory. Defaults to the current workspace.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date suffix for the output folder. Defaults to today's date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First chunk index to process.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last chunk index to process, inclusive. If omitted, the script uses the common chunk count across all folders.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        default=True,
        help="Load files with torch.load(weights_only=True). Enabled by default for safety.",
    )
    return parser.parse_args()


def load_chunk(path: Path) -> dict:
    return torch.load(path, weights_only=True)


def concatenate_chunk_dicts(chunk_dicts: list[dict]) -> dict:
    merged: dict = {}
    keys = chunk_dicts[0].keys()

    for key in keys:
        values = [chunk_dict[key] for chunk_dict in chunk_dicts]
        first_value = values[0]

        if torch.is_tensor(first_value):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first_value, list):
            merged[key] = sum(values, [])
        elif isinstance(first_value, tuple):
            merged[key] = tuple(item for value in values for item in value)
        else:
            raise TypeError(f"Unsupported value type for key '{key}': {type(first_value)!r}")

    return merged


def common_chunk_count(input_folders: Iterable[Path]) -> int:
    counts = []
    for folder in input_folders:
        chunk_files = sorted(folder.glob("chunk_*.pt"))
        counts.append(len(chunk_files))

    if not counts:
        raise ValueError("No input folders were provided.")

    return min(counts)


def main() -> None:
    args = parse_args()
    input_folders = [Path(folder) for folder in args.input_folders]

    missing_folders = [str(folder) for folder in input_folders if not folder.exists()]
    if missing_folders:
        raise FileNotFoundError(f"Missing input folders: {missing_folders}")

    date_suffix = args.date or datetime.now().strftime("%Y-%m-%d")
    output_root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parent
    output_folder = output_root / f"merged_chunks_{date_suffix}"
    output_folder.mkdir(parents=True, exist_ok=True)

    total_chunks = args.end_index - args.start_index + 1 if args.end_index is not None else common_chunk_count(input_folders) - args.start_index
    if total_chunks <= 0:
        raise ValueError("No chunks to process with the selected start/end indices.")

    for chunk_index in range(args.start_index, args.start_index + total_chunks):
        chunk_paths = [folder / f"chunk_{chunk_index}.pt" for folder in input_folders]
        missing_files = [str(path) for path in chunk_paths if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing chunk files for index {chunk_index}: {missing_files}")

        chunk_dicts = [load_chunk(path) for path in chunk_paths]
        merged_chunk = concatenate_chunk_dicts(chunk_dicts)

        output_path = output_folder / f"chunk_{chunk_index}_fin.pt"
        torch.save(merged_chunk, output_path)
        print(f"Saved {output_path}")

    print(f"Finished writing merged chunks to {output_folder}")


if __name__ == "__main__":
    main()
