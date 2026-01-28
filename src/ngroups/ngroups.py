#!/usr/bin/env python3

import pandas as pd
import numpy as np
import logging

from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from pathlib import Path
from .utils import k_medoids
from sklearn.metrics import pairwise_distances

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)
"""logging.Logger: logger instance for the module."""


def ngroups_full_workflow(outdir: Path, k_range: tuple[int, int, int]):
    if not outdir.is_dir():
        raise NotADirectoryError(f"{outdir} is not a directory")

    X_file = Path('/Users/stephenricher/repos/NeighbourGroups/data/C.jejuni-UKisolates.csv')
    X_df = pd.read_csv(X_file).set_index("id").astype(int)
    cat_features = list(range(X_df.shape[1]))

    core_file = X_file #"C_jejuni-coreMLST.csv"
    core_df = pd.read_csv(core_file).set_index("id").astype(int)

    # Ensure core_df and X_df have the same isolates in the same order
    common_ids = X_df.index.intersection(core_df.index)
    X_sub = X_df.loc[common_ids]

    core_sub= core_df.loc[common_ids]
    dist_matrix = pairwise_distances(core_df, metric="hamming")

    results = []
    start, stop, step = k_range
    for k in range(start, stop + 1, step):
        labels, _ = k_medoids(dist_matrix, k)
        X = X_sub.copy()
        y = pd.Series(labels, index=common_ids).to_numpy()
        f1 = evaluate_macro_f1(X_sub, y, cat_features, n_splits=5)
        results.append({"k": k, "macro_F1": f1,})
        logging.info("F1=%.4f, n_clusters=%d", f1, k)


def evaluate_macro_f1(X, y, cat_features: list[int], n_splits: int = 5, iterations: int = 10, thread_count: int = -1):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = CatBoostClassifier(
            iterations=iterations,
            depth=3,
            learning_rate=0.5,
            loss_function='MultiClass',
            thread_count=thread_count,
            verbose=False,
            random_seed=42
        )
        model.fit(X_train, y_train, cat_features=cat_features)
        y_pred = model.predict(X_val)
        fold_f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(fold_f1)
        logging.info("[CV] Fold %d: macro F1 = %.4f", fold, fold_f1)
    return np.mean(f1_scores)
