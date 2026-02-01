import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.random_projection import SparseRandomProjection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)


def downsample(
    data: Path,
    target_n: int,
    projection_dim: int,
    cluster_factor: int,
    max_missing: float = 0.05,
    random_state: int = 42,
):
    """Downsample cgMLST allele profiles to a representative subset of isolates
    and compute nearest-neighbour distances for each selected isolate.

    This function performs the following steps:
    1. Loads a cgMLST profile TSV file.
    2. Preprocesses the data, filtering isolates with a proportion of missing alleles
       above `max_missing`.
    3. Downsamples the dataset to `target_n` isolates using a random projection
       + KMeans clustering + medoid selection.
       - If target_n=-1 or larger than the available isolates after filtering, all isolates are kept.
    4. Computes nearest-neighbour distances (Hamming distance ignoring missing loci)
       between each selected isolate and its closest other isolate.
    5. Outputs a TSV to stdout with columns:
       isolate_id, target_n, nn_distance

    Parameters
    ----------
    data : Path
        Path to the input cgMLST profile TSV file.
        The first column should be isolate IDs, and the remaining columns are allele calls.
    target_n : int
        Number of isolates to retain. If -1 or greater than available isolates,
        all filtered isolates are kept.
    projection_dim : int, optional
        Dimensionality of sparse random projection used before clustering (default=64).
    cluster_factor : int, optional
        Factor controlling number of clusters relative to target_n
        (default=3 → n_clusters ≈ cluster_factor × target_n).
    max_missing : float, optional
        Maximum allowed proportion of missing alleles per isolate (0–1).
        Isolates exceeding this proportion are filtered out (default=0.05).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns:
    -------
    None
        The function writes the results directly to stdout as a TSV.
        Each row corresponds to a selected isolate, with:
        - isolate_id: the ID of the isolate
        - target_n: the number of isolates requested (or actual retained)
        - nn_distance: nearest-neighbour distance for that isolate

    Notes:
    -----
    - Missing alleles should be encoded consistently (e.g., -1) and are ignored
      in distance calculations.
    - Logging messages are printed to stdout to track progress and timing for
      loading, preprocessing, downsampling, and NN distance computation.
    """
    # Load
    logger.info("Loading data from %s", data)
    df = pd.read_csv(data, sep="\t", dtype=str, index_col=0)
    # Extract allele matrix
    logger.info("Preprocessing allele matrix (max_missing=%s)", max_missing)
    X, isolate_ids = preprocess_cgmlst(df, max_missing)
    logger.info("Preprocessing done: %d isolates, %d loci", X.shape[0], X.shape[1])
    # Downsample or keep all
    if target_n == -1 or target_n >= len(X):
        idx = np.arange(len(X))
        actual_n = len(X)
        logger.info("Target_n=-1 or exceeds available isolates, keeping all %d isolates", actual_n)
    else:
        logger.info("Downsampling to target_n=%d", target_n)
        idx = downsample_cgmlst(
            X,
            target_n=target_n,
            projection_dim=projection_dim,
            cluster_factor=cluster_factor,
            random_state=random_state,
        )
        actual_n = target_n
        logger.info("Downsampling selected %d isolates", len(idx))
    # Map back to isolate IDs
    X_sub = X[idx]
    ids_sub = isolate_ids[idx]
    logger.info("Computing nearest-neighbour distances for %d isolates", len(idx))
    nn_dist = nearest_neighbour_distances(X_sub)
    # Build output DataFrame
    df_out = pd.DataFrame({"isolate_id": ids_sub, "target_n": actual_n, "nn_distance": nn_dist})
    logger.info("Writing output to stdout")
    df_out.to_csv(sys.stdout, sep="\t", index=False, header=False)


def preprocess_cgmlst(
    df: pd.DataFrame,
    max_missing: float | None = None,
    missing: int = -1,
):
    """Preprocess cgMLST dataframe:
      - Convert allele calls to numeric
      - Filter isolates with high missing proportion (if max_missing set)
      - Fill missing values
      - Deduplicate identical profiles

    Parameters
    ----------
    df : pd.DataFrame
        Rows = isolates, columns = loci. Index = isolate IDs.
    max_missing : float, optional
        Maximum proportion of missing alleles per isolate to keep (0–1).
    missing : int, optional
        Value to fill for missing alleles (default=-1).

    Returns:
    -------
    X : np.ndarray, shape (n_isolates, n_loci)
        Allele matrix (int), missing values filled.
    isolate_ids : pd.Index
        IDs of isolates retained after filtering and deduplication.
    """
    # Convert to numeric
    X = df.apply(pd.to_numeric, errors="coerce")

    # --- Global missing stats ---
    n_missing_total = X.isna().sum().sum()
    n_calls_total = X.size
    logger.info(
        "Missing allele calls: %d / %d (%.2f%%)",
        n_missing_total,
        n_calls_total,
        100 * n_missing_total / n_calls_total,
    )

    # --- Per-isolate missing proportion ---
    missing_per_row = X.isna().mean(axis=1)

    # --- Filter isolates with too many missing ---
    if max_missing is not None:
        keep = missing_per_row <= max_missing
        logger.info(
            "Filtering isolates: %d kept, %d removed (threshold %.1f%%)",
            keep.sum(),
            (~keep).sum(),
            100 * max_missing,
        )
        X = X.loc[keep]

    # --- Deduplicate identical profiles ---
    profile_tuples = [tuple(row) for row in X.fillna(missing).to_numpy(dtype=np.int32)]
    df_profiles = pd.DataFrame({"isolate_id": X.index, "profile": profile_tuples})

    df_unique = df_profiles.drop_duplicates(subset="profile")
    n_duplicates = len(X) - len(df_unique)
    if n_duplicates > 0:
        logger.info("Removed %d exact duplicate isolates", n_duplicates)

    # --- Final allele matrix and isolate IDs ---
    unique_idx = df_unique.index
    isolate_ids = X.index[unique_idx]
    X_final = X.loc[isolate_ids].fillna(missing).to_numpy(dtype=np.int32)

    return X_final, isolate_ids


def downsample_cgmlst(
    X: np.ndarray,
    target_n: int = 5000,
    projection_dim: int = 64,
    cluster_factor: int = 3,
    random_state: int = 42,
    missing: int = -1,
) -> np.ndarray:
    """Downsample cgMLST allele matrix to a representative subset.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_loci)
        cgMLST allele matrix. Missing values must be encoded as `missing`.
    target_n : int
        Number of isolates to retain.

    Returns:
    -------
    np.ndarray
        Row indices of selected isolates.
    """
    n_samples = X.shape[0]
    if target_n >= n_samples:
        return np.arange(n_samples)

    rp = SparseRandomProjection(n_components=projection_dim, random_state=random_state)
    X_proj = rp.fit_transform(X)

    n_clusters = min(n_samples, max(target_n + 10, cluster_factor * target_n))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, random_state=random_state)
    labels = kmeans.fit_predict(X_proj)

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    alloc = np.round(cluster_sizes / n_samples * target_n).astype(int)

    diff = target_n - alloc.sum()
    if diff > 0:
        for i in np.argsort(-cluster_sizes):
            if diff == 0:
                break
            alloc[i] += 1
            diff -= 1
    elif diff < 0:
        for i in np.argsort(cluster_sizes):
            if diff == 0:
                break
            if alloc[i] > 0:
                alloc[i] -= 1
                diff += 1

    selected = []

    for c in np.where(alloc > 0)[0]:
        idx = np.where(labels == c)[0]
        n_select = alloc[c]

        # If a cluster is allocated more samples than it contains just take them all.
        if n_select >= len(idx):
            selected.extend(idx)
            continue

        subX = X[idx]
        dsum = np.zeros(len(idx), dtype=int)

        for i in range(len(idx)):
            dsum[i] = _masked_hamming_rowwise(subX, subX[i], missing).sum()

        # Sort by distance sum (ascending) and take the top n_select indices
        # This picks the 'n_select' most central isolates.
        best_candidates_indices = np.argsort(dsum)[:n_select]
        selected.extend(idx[best_candidates_indices])

    return np.array(selected[:target_n], dtype=int)


def _masked_hamming_rowwise(X, vec, missing):
    """Compute Hamming distance between vec and each row of X, ignoring loci missing in either profile."""
    mask = (X != missing) & (vec != missing)
    return np.sum((X != vec) & mask, axis=1)


def nearest_neighbour_distances(X, missing=-1):
    n = len(X)
    nn = np.zeros(n)

    for i in range(n):
        xi = X[i]
        dmin = np.inf
        for j in range(n):
            if i == j:
                continue
            xj = X[j]
            mask = (xi != missing) & (xj != missing)
            if mask.sum() == 0:
                continue
            d = (xi[mask] != xj[mask]).mean()
            dmin = min(dmin, d)
        nn[i] = dmin

    return nn
