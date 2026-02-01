#!/usr/bin/env bash
# downsample_loop.sh
# Usage:
#   bash downsample_loop.sh analysis/profiles.tsv 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 -1 > analysis/nn_distances.tsv
#
# Writes combined TSV to stdout, header only once.
# Safe to re-run after interruption.

set -euo pipefail

# -----------------------------
# Default parameters
# -----------------------------
PROJECTION_DIM=64     # Dimensionality for sparse random projection
                      # High enough to approximately preserve distances
                      # for ~44k isolates, while remaining fast.

CLUSTER_FACTOR=3      # Number of clusters ≈ CLUSTER_FACTOR * target_n
                      # Slight over-clustering improves coverage of allele space.

REPS=5                # Number of independent replicates per target_n
                      # Accounts for stochasticity in random projection and KMeans,
                      # yielding more stable NN distance summaries.

TMP_DIR=".tmp_downsample_markers"

mkdir -p "$TMP_DIR"

# -----------------------------
# Args
# -----------------------------
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 profiles.tsv target_n1 [target_n2 ...]" >&2
    exit 1
fi

DATA="$1"
shift
TARGET_NS=("$@")

first=1

# -----------------------------
# Main loop
# -----------------------------
for n in "${TARGET_NS[@]}"; do
    for seed in $(seq 1 "$REPS"); do

        marker="${TMP_DIR}/$(basename "$DATA").n${n}.seed${seed}.done"

        # Skip if already completed
        if [[ -f "$marker" ]]; then
            echo "[SKIP] target_n=$n seed=$seed already done" >&2
            continue
        fi

        echo "[RUN ] target_n=$n seed=$seed" >&2

        if [[ $first -eq 1 ]]; then
            ngroups downsample "$DATA" \
                --target-n "$n" \
                --projection-dim "$PROJECTION_DIM" \
                --cluster-factor "$CLUSTER_FACTOR" \
                --random-state "$seed"
            first=0
        else
            ngroups downsample "$DATA" \
                --target-n "$n" \
                --projection-dim "$PROJECTION_DIM" \
                --cluster-factor "$CLUSTER_FACTOR" \
                --random-state "$seed" | tail -n +2
        fi

        # Mark successful completion
        touch "$marker"
    done
done

# -----------------------------
# Only reached if *everything* succeeded
# -----------------------------
echo "[OK] All runs completed successfully, cleaning up markers" >&2
rm -rf "$TMP_DIR"