#!/usr/bin/env python
"""Plot nearest-neighbour distance percentiles per target number of isolates (target_n).

Purpose
-------
This script visualizes how the distribution of nearest-neighbour (NN) distances
changes as a function of the downsample size (`target_n`).

- Each NN distance measures how close an isolate is to its closest other isolate
  in allele space (cgMLST profile), ignoring missing loci.
- By computing percentiles (e.g., 50th, 75th, 90th), we can see how the
  overall "spread" of the downsampled isolates changes as the subset size increases.

Interpretation
--------------
- Lower NN distances indicate that isolates are closer together; higher NN distances
  indicate isolates are more separated.
- The curves can help **identify an optimal `target_n`**:
    - If the 50th percentile and upper percentiles plateau, increasing target_n
      further may not increase diversity substantially.
    - Very low target_n → high NN distances may reflect sparse sampling.
- This provides a principled way to choose a representative downsample size
  that balances diversity and computational efficiency.

Input
-----
TSV file with three columns (no header or with header removed):
    isolate_id    target_n    nn_distance

Usage
-----
python plot_nn_percentiles.py nn_distances.tsv
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} nn_distances.tsv", file=sys.stderr)
    sys.exit(1)

tsv_file = sys.argv[1]
df = pd.read_csv(tsv_file, sep="\t", names=["isolate_id", "target_n", "nn_distance"])
df_summary = df.groupby(["isolate_id", "target_n"])["nn_distance"].median().reset_index()

# Compute percentiles per target_n
percentiles = [50, 75, 90]  # median, 75th, 90th
percentile_rows = []

for target_n, group in df_summary.groupby("target_n"):
    nn_values = group["nn_distance"].values
    p_vals = np.percentile(nn_values, percentiles)
    for perc, val in zip(percentiles, p_vals):
        percentile_rows.append({"target_n": target_n, "percentile": perc, "nn_distance": val})

df_plot = pd.DataFrame(percentile_rows)

# Plot
sns.set_theme(style="whitegrid", palette="deep")
fig, ax = plt.subplots(figsize=(8, 5))

sns.lineplot(data=df_plot, x="target_n", y="nn_distance", hue="percentile", marker="o", ax=ax)

ax.set_xlabel("Target number of isolates (target_n)")
ax.set_ylabel("Nearest-neighbour distance")
ax.set_title("Nearest-neighbour distance percentiles vs. target_n")
ax.legend(title="Percentile")

# Optional: log scale for wide range of target_n
# ax.set_xscale("log")

plt.tight_layout()
plt.show()
