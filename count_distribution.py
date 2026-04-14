import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch


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


def load_labels_from_pt(pt_file: Path) -> torch.Tensor:
	loaded = torch.load(pt_file, map_location="cpu", weights_only=False)

	if isinstance(loaded, dict) and "labels" in loaded:
		return labels_to_2d_tensor(loaded["labels"])

	if hasattr(loaded, "labels"):
		return labels_to_2d_tensor(loaded.labels)

	if isinstance(loaded, torch.Tensor):
		return labels_to_2d_tensor(loaded)

	raise ValueError(
		f"Unsupported format in {pt_file}. Expected dict['labels'], object.labels, or a tensor."
	)


def find_pt_files(input_path: Path, recursive: bool) -> list[Path]:
	if input_path.is_file():
		if input_path.suffix.lower() != ".pt":
			raise ValueError(f"File is not a .pt file: {input_path}")
		return [input_path]

	if not input_path.is_dir():
		raise ValueError(f"Path does not exist: {input_path}")

	pattern = "**/*.pt" if recursive else "*.pt"
	files = sorted(input_path.glob(pattern))
	if not files:
		raise ValueError(f"No .pt files found in: {input_path}")
	return files


def format_percent(count: int, total: int) -> str:
	if total == 0:
		return "0.00%"
	return f"{(100.0 * count / total):.2f}%"


def print_vector_distribution(
	vector_counter: Counter,
	total_samples: int,
	max_rows: int,
):
	print("\n=== Full Label Vector Distribution ===")
	if total_samples == 0:
		print("No label rows found.")
		return

	sorted_items = sorted(vector_counter.items(), key=lambda x: x[1], reverse=True)
	shown = sorted_items[:max_rows]
	hidden = len(sorted_items) - len(shown)

	for vec, count in shown:
		print(f"{vec} -> {count} ({format_percent(count, total_samples)})")

	if hidden > 0:
		print(f"... {hidden} more unique label vectors not shown")


def print_per_label_distribution(
	per_label_counters: Iterable[Counter],
	total_samples: int,
	max_values: int,
):
	print("\n=== Per Label Index Distribution ===")
	if total_samples == 0:
		print("No label rows found.")
		return

	for index, counter in enumerate(per_label_counters):
		print(f"\nLabel[{index}] value counts:")
		sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
		shown = sorted_items[:max_values]
		hidden = len(sorted_items) - len(shown)

		for value, count in shown:
			print(f"  {value} -> {count} ({format_percent(count, total_samples)})")

		if hidden > 0:
			print(f"  ... {hidden} more values not shown")


def main():
	parser = argparse.ArgumentParser(
		description="Count label distributions from a single .pt file or a directory of .pt files."
	)
	parser.add_argument(
		"path",
		help="Path to a .pt file or a directory containing .pt files.",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="When path is a directory, search .pt files recursively.",
	)
	parser.add_argument(
		"--max-vector-rows",
		type=int,
		default=30,
		help="Max number of unique full-label vectors to print.",
	)
	parser.add_argument(
		"--max-values-per-label",
		type=int,
		default=30,
		help="Max number of distinct values to print per label index.",
	)
	parser.add_argument(
		"--round-decimals",
		type=int,
		default=6,
		help="Decimal precision used for float labels before counting.",
	)

	args = parser.parse_args()
	input_path = Path(args.path)

	pt_files = find_pt_files(input_path, recursive=args.recursive)
	print(f"Found {len(pt_files)} .pt file(s).")

	vector_counter: Counter = Counter()
	per_label_counters: list[Counter] = []
	total_samples = 0
	expected_label_dim = None

	for file_path in pt_files:
		labels = load_labels_from_pt(file_path)
		if labels.numel() == 0:
			print(f"Skipping empty labels in: {file_path}")
			continue

		n_rows, n_cols = labels.shape
		print(f"Reading {file_path.name}: {n_rows} rows, label dim={n_cols}")

		if expected_label_dim is None:
			expected_label_dim = n_cols
			per_label_counters = [Counter() for _ in range(n_cols)]
		elif n_cols != expected_label_dim:
			raise ValueError(
				f"Label dimension mismatch in {file_path}. "
				f"Expected {expected_label_dim}, got {n_cols}."
			)

		rows = labels.tolist()
		for row in rows:
			normalized_row = tuple(
				normalize_value(value, args.round_decimals) for value in row
			)
			vector_counter[normalized_row] += 1

			for i, value in enumerate(normalized_row):
				per_label_counters[i][value] += 1

		total_samples += n_rows

	print(f"\nTotal samples counted: {total_samples}")
	print_vector_distribution(
		vector_counter=vector_counter,
		total_samples=total_samples,
		max_rows=args.max_vector_rows,
	)
	print_per_label_distribution(
		per_label_counters=per_label_counters,
		total_samples=total_samples,
		max_values=args.max_values_per_label,
	)


if __name__ == "__main__":
	main()