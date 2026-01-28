import numpy as np

def k_medoids(dist_matrix, k, max_iter=100, random_seed=42):
    """Vectorised K-medoids clustering using Lloyd's heuristic."""
    rng = np.random.default_rng(random_seed)
    n_samples = dist_matrix.shape[0]
    # Initialise medoids
    medoids = rng.choice(n_samples, size=k, replace=False)
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        # Vectorised Assignment
        # dist_matrix[:, medoids] gets shape (n_samples, k)
        distances_to_medoids = dist_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        # Update Medoids
        new_medoids = medoids.copy()
        for j in range(k):
            cluster_points = np.where(labels == j)[0]
            if len(cluster_points) == 0:
                # Handle empty cluster: keep old medoid or pick random point
                continue
            # Extract sub-matrix for this cluster
            sub_matrix = dist_matrix[np.ix_(cluster_points, cluster_points)]
            # Find point with minimum sum of distances to others in cluster
            intra_distances = sub_matrix.sum(axis=1)
            best_idx_local = np.argmin(intra_distances)
            new_medoids[j] = cluster_points[best_idx_local]
        # Sort indices before comparing to handle permutation invariance if needed,
        # though strictly comparing values works if order is preserved.
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return labels, medoids
