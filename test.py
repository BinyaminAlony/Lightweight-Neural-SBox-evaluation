import csv
import os

recalls = [0.85, 0.87, 0.90]
specificities = [0.80, 0.82, 0.85]
alpha_range = [0.1, 0.2, 0.3]

os.makedirs("results", exist_ok=True)
results_file = os.path.join("results", "alpha_metrics.csv")
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["alpha", "specificity", "recall"])
    for a, s, r in zip(alpha_range, specificities, recalls):
        writer.writerow([a, float(s), float(r)])
print(f"Saved results to: {results_file}")