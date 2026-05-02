#!/usr/bin/env python3

import gzip
import logging
import os
import shutil

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from ete3 import ClusterTree
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)
"""logging.Logger: logger instance for the module."""


def download(url: str, outdir: str, gunzip: bool = False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    filename = url.split("/")[-1]
    file_path = os.path.join(outdir, filename)

    logger.info("Starting download | file=%s", filename)

    try:
        r = requests.get(url, stream=True)
    except Exception as e:
        logger.error("Request failed for %s | error=%s", url, str(e))
        raise

    if not r.ok:
        logger.error(
            "Download failed | file=%s | status_code=%d | response=%s",
            filename,
            r.status_code,
            r.text[:200],
        )
        raise RuntimeError(f"Failed to download {url}")

    total_bytes = 0
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 8):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    logger.info(
        "Download complete | file=%s | size=%.2f MB | path=%s",
        filename,
        total_bytes / (1024 * 1024),
        file_path,
    )

    # Optional gunzip
    if gunzip:
        if not file_path.endswith(".gz"):
            logger.warning("gunzip=True but file does not end with .gz | file=%s", filename)
            return

        out_path = file_path[:-3]

        logger.info("Decompressing gzip file | input=%s | output=%s", file_path, out_path)

        try:
            with gzip.open(file_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            logger.error("Failed to decompress %s | error=%s", file_path, str(e))
            raise

        logger.info("Decompression complete | output=%s", out_path)

        # Optionally remove original .gz
        os.remove(file_path)
        logger.info("Removed compressed file | path=%s", file_path)


def readNewick(nwk):
    with open(nwk) as fh:
        return fh.readline()


def nwk2linkage(newick: str):
    """Convert newick tree into scipy linkage matrix"""
    logger.info("Reading Newick tree from %s", newick)

    tree = ClusterTree(newick)

    logger.info("Computing cophenetic distance matrix")
    cophenetic, newick_labels = tree.cophenetic_matrix()

    logger.info(
        "Cophenetic matrix computed | size=%d x %d",
        len(newick_labels),
        len(newick_labels),
    )

    cophenetic = pd.DataFrame(
        cophenetic,
        columns=newick_labels,
        index=newick_labels,
    )

    logger.info("Converting to condensed distance matrix")
    pairDist = pdist(cophenetic)

    logger.info("Performing hierarchical linkage")
    Z = linkage(pairDist)

    logger.info("Linkage complete | shape=%s", Z.shape)

    return Z, np.array(cophenetic.columns), cophenetic


def readTree(prefix: str, mode: str):
    logger.info("Reading tree files | prefix=%s | mode=%s", prefix, mode)

    linkage_path = f"{prefix}-{mode}-linkage.npy"
    labels_path = f"{prefix}-{mode}-labels.npy"

    linkageMatrix = np.load(linkage_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    logger.info(
        "Tree files loaded | linkage_shape=%s | n_labels=%d",
        linkageMatrix.shape,
        len(labels),
    )

    return linkageMatrix, labels


def processNewick(linkageMatrix: np.array, labels: np.array, nGroup: int, name: str):
    logger.info("Clustering tree into %d groups", nGroup)

    clusters = fcluster(linkageMatrix, t=int(nGroup), criterion="maxclust")
    labelsID = pd.DataFrame(clusters, labels, columns=[name])

    logger.info("Clustering complete | n_labels=%d", len(labelsID))

    return labelsID


def train_model(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    logger.info("Training CatBoost model | samples=%d | features=%d", X.shape[0], X.shape[1])

    model = CatBoostClassifier(verbose=0, random_seed=seed, allow_writing_files=False)
    model.fit(X, y, cat_features=list(X.columns))

    logger.info("Model training complete")

    return model


def train_model_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    cv: int = 5,
    min_class_eval: int = 5,
):
    logger.info(
        "Starting cross-validation | samples=%d | classes=%d | cv=%d",
        len(X),
        y.nunique(),
        cv,
    )

    class_counts = y.value_counts()
    eval_classes = class_counts[class_counts >= min_class_eval].index

    logger.info(
        "Classes eligible for evaluation: %d / %d (min_class_eval=%d)",
        len(eval_classes),
        len(class_counts),
        min_class_eval,
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info("CV fold %d/%d", i, cv)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = train_model(X_train, y_train, seed=seed)

        mask = y_val.isin(eval_classes)
        y_val_eval = y_val[mask]
        X_val_eval = X_val[mask]

        preds = model.predict(X_val_eval)
        score = f1_score(y_val_eval, preds, average="weighted")

        logger.info("Fold %d F1=%.4f", i, score)
        scores.append(score)

    logger.info("Fitting final model on full dataset")
    final_model = CatBoostClassifier(verbose=0, random_seed=seed, allow_writing_files=False)
    final_model.fit(X, y, cat_features=list(X.columns))

    return final_model, np.array(scores)


def readFull(prefix: str):
    """Read full dataset"""
    logger.info("Loading full dataset for prefix='%s'", prefix)

    testIsolates = pd.read_csv(f"{prefix}-test.csv")["id"].astype(str).tolist()
    data = pd.read_csv(f"{prefix}-full.csv").astype(str)

    logger.info("Marking test samples | total=%d | test=%d", len(data), len(testIsolates))

    data["test"] = data["id"].apply(lambda x: x in testIsolates)
    data = data.set_index("id")

    return data


def mergeData(model_path: str, data: pd.DataFrame, linkageMatrix: np.array, labels: np.array, nGroup: int):
    logger.info("Merging predictions with tree labels | nGroup=%d", nGroup)

    model = CatBoostClassifier()
    model.load_model(model_path)

    data[f"NG{nGroup}"] = model.predict(data)

    labelsID = processNewick(linkageMatrix, labels, nGroup, name=f"NG{nGroup}-truth")
    data = pd.merge(data, labelsID, left_index=True, right_index=True, how="outer")

    logger.info("Merge complete | rows=%d", len(data))

    return data


def testNG(
    model_path: str,
    data: pd.DataFrame,
    linkageMatrix: np.array,
    labels: np.array,
    nGroup: int,
):
    logger.info("Running testNG | nGroup=%s", nGroup)

    # Merge predictions and truth
    data = mergeData(model_path, data, linkageMatrix, labels, nGroup)

    # Extract test subset
    testSubset = data.loc[data["test"]]
    logger.info("Test subset size | nGroup=%s | samples=%d", nGroup, len(testSubset))

    # Compute Adjusted Rand Index
    adjRand = adjusted_rand_score(
        testSubset[f"NG{nGroup}"],
        testSubset[f"NG{nGroup}-truth"],
    )

    logger.info("Computed Adjusted Rand Index | nGroup=%s | score=%.4f", nGroup, adjRand)

    return adjRand


def trainNG(
    prefix: str,
    data: pd.DataFrame,
    linkageMatrix: np.array,
    labels: np.array,
    nGroup: int,
    full: bool = False,
    seed: int = 42,
):
    """Train model for each Neighbour Group number"""
    logger.info(
        "Preparing data for nGroup=%d | samples=%d | features=%d",
        nGroup,
        data.shape[0],
        data.shape[1],
    )

    # Generate labels from tree
    labelsID = processNewick(linkageMatrix, labels, nGroup, name="trainNG")
    logger.info("Generated cluster labels for nGroup=%d", nGroup)

    featureCols = data.columns
    sourceCount = len(data)

    # Merge labels
    data = pd.merge(data, labelsID, left_index=True, right_index=True, how="left")

    missing = sourceCount - len(data)
    if missing > 0:
        logger.error(
            "%d of %d isolates (%.2f%%) missing from Newick labels",
            missing,
            sourceCount,
            missing / sourceCount,
        )

    # Prepare features
    logger.info("Preparing categorical features")
    X = prepare_categorical(data[featureCols])
    y = data.pop("trainNG")

    logger.info("Training data ready | X_shape=%s | y_classes=%d", X.shape, y.nunique())

    # Train model
    if full:
        logger.info("Running cross-validation for nGroup=%d", nGroup)
        model, scores = train_model_with_cv(X, y, seed=seed)

        logger.info(
            "CV completed | nGroup=%d | mean_f1=%.4f | std_f1=%.4f",
            nGroup,
            np.mean(scores),
            np.std(scores),
        )
        logger.debug("Fold F1 scores: %s", np.round(scores, 4))

    else:
        logger.info("Training model without CV for nGroup=%d", nGroup)
        model = train_model(X, y, seed=seed)

    # Save model
    suffix = "final-" if full else ""
    model_path = f"{prefix}-{nGroup}-{suffix}trained.cbm"
    model.save_model(model_path)

    logger.info("Model saved | path=%s", model_path)

    # Full retrain outputs
    if full:
        logger.info("Generating final predictions for full dataset (nGroup=%d)", nGroup)

        data = readFull(prefix)
        linkageMatrix, labels = readTree(prefix, mode="full")

        data = mergeData(model_path, data, linkageMatrix, labels, nGroup)
        data[f"NG{nGroup}"] = model.predict(data)

        data.index = data.index.rename("id")
        out_path = f"{prefix}-{nGroup}-final.csv"
        data.to_csv(out_path)

        logger.info("Final output saved | path=%s | rows=%d", out_path, len(data))


def validColumns(cols, features, IDcol):
    valid = True
    if IDcol in features:
        valid = False
        logging.error(f"ID column ({IDcol}) cannot be a feature.")
    if len(features) != len(set(features)):
        valid = False
        logging.error("Duplicates found in feature names")
    for feature in features:
        if feature not in cols:
            valid = False
            logging.error(f"{feature} not in input header.")
    return valid


def prepare_categorical(X: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing categorical encoding | shape=%s", X.shape)

    X = X.copy()
    for col in X.columns:
        numeric_col = pd.to_numeric(X[col], errors="coerce")

        if numeric_col.notna().sum() >= 0.9 * len(X[col]):
            if (numeric_col.dropna() % 1 == 0).all():
                X[col] = numeric_col.astype("Int64")
            else:
                X[col] = numeric_col.round(6)
        else:
            X[col] = X[col].astype("string")

    X = X.astype("string")
    X = X.fillna("missing")

    logger.info("Categorical preparation complete")

    return X
