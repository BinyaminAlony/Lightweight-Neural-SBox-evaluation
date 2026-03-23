"""Generate four plots from CSV metrics in the `results/` folder:

1) alpha (x) vs specificity (y)
2) alpha (x) vs recall (y)
3) FB (x) vs specificity (y)
4) FB (x) vs recall (y)

The script is resilient to small header differences and missing files. It saves PNGs to `results/plots`.
"""
import csv
import os
import sys
import math
import matplotlib.pyplot as plt


def load_three_cols(path):
    """Load first three numeric columns from CSV. Return (label, xs, specs, recs) or None if no data."""
    if not os.path.exists(path):
        print(f"skip missing file: {path}")
        return None
    with open(path, newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"empty file: {path}")
            return None
        label = os.path.splitext(os.path.basename(path))[0]
        xs = []
        specs = []
        recs = []
        for row in reader:
            if not row:
                continue
            # ensure at least three columns
            if len(row) < 3:
                continue
            try:
                x = float(row[0])
                s = float(row[1])
                r = float(row[2])
            except ValueError:
                # skip rows that are not numeric
                continue
            if any(math.isnan(v) for v in (x, s, r)):
                continue
            xs.append(x)
            specs.append(s)
            recs.append(r)
        if not xs:
            print(f"no numeric rows in: {path}")
            return None
        return {
            'label': label,
            'x': xs,
            'specificity': specs,
            'recall': recs,
        }


def plot_metric(datasets, xkey, ykey, xlabel, ylabel, outpath, title=None):
    plt.figure(figsize=(8,5))
    for d in datasets:
        plt.plot(d['x'], d[ykey], marker='o', label=d['label'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"wrote: {outpath}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(repo_root, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # candidate BA/alpha files (some files use header 'BA' and some 'alpha')
    ba_files = [
        os.path.join(results_dir, 'BA_metrics.csv'),
        os.path.join(results_dir, 'BA_metrics_75_epochs.csv'),
    ]
    fb_files = [
        os.path.join(results_dir, 'FB_metrics_75_epochs.csv'),
        os.path.join(results_dir, 'FB_metrics_175_epochs.csv'),
    ]

    ba_dsets = [load_three_cols(p) for p in ba_files]
    ba_dsets = [d for d in ba_dsets if d is not None]

    fb_dsets = [load_three_cols(p) for p in fb_files]
    fb_dsets = [d for d in fb_dsets if d is not None]

    if not ba_dsets:
        print("No BA/alpha datasets found. Skipping alpha plots.")
    else:
        # sort each dataset by x so lines look correct
        for d in ba_dsets:
            pairs = sorted(zip(d['x'], d['specificity'], d['recall']))
            d['x'], d['specificity'], d['recall'] = map(list, zip(*pairs))

        plot_metric(ba_dsets, 'x', 'specificity', 'alpha', 'specificity', os.path.join(plots_dir, 'alpha_specificity.png'), title='alpha vs specificity')
        plot_metric(ba_dsets, 'x', 'recall', 'alpha', 'recall', os.path.join(plots_dir, 'alpha_recall.png'), title='alpha vs recall')

    if not fb_dsets:
        print("No FB datasets found. Skipping FB plots.")
    else:
        for d in fb_dsets:
            pairs = sorted(zip(d['x'], d['specificity'], d['recall']))
            d['x'], d['specificity'], d['recall'] = map(list, zip(*pairs))

        plot_metric(fb_dsets, 'x', 'specificity', 'FB', 'specificity', os.path.join(plots_dir, 'FB_specificity.png'), title='FB vs specificity')
        plot_metric(fb_dsets, 'x', 'recall', 'FB', 'recall', os.path.join(plots_dir, 'FB_recall.png'), title='FB vs recall')

    print('\nDone. Plots (if any) are in:')
    print(plots_dir)


if __name__ == '__main__':
    main()
