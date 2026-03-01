#!/usr/bin/env python3

import logging
import os

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


def download(url: str, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = url.split("/")[-1]
    file_path = os.path.join(dir, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        logging.error(f"Download failed: status code {r.status_code}\n{r.text}")


def readNewick(nwk):
    with open(nwk) as fh:
        return fh.readline()


def nwk2linkage(newick: str):
    """Convert newick tree into scipy linkage matrix"""
    tree = ClusterTree(newick)
    cophenetic, newick_labels = tree.cophenetic_matrix()
    cophenetic = pd.DataFrame(cophenetic, columns=newick_labels, index=newick_labels)
    # reduce square distance matrix to condensed distance matrices
    pairDist = pdist(cophenetic)
    return linkage(pairDist), np.array(cophenetic.columns), cophenetic


def readTree(prefix: str, mode: str):
    linkageMatrix = np.load(f"{prefix}-{mode}-linkage.npy", allow_pickle=True)
    labels = np.load(f"{prefix}-{mode}-labels.npy", allow_pickle=True)
    return linkageMatrix, labels


def processNewick(linkageMatrix: np.array, labels: np.array, nGroup: int, name: str):
    clusters = fcluster(linkageMatrix, t=int(nGroup), criterion="maxclust")
    labelsID = pd.DataFrame(clusters, labels, columns=[name])
    return labelsID


def train_model(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    model = CatBoostClassifier(verbose=0, random_seed=seed, allow_writing_files=False)
    model.fit(X, y, cat_features=list(range(X.shape[1])))
    return model


def train_model_with_cv(X: pd.DataFrame, y: pd.Series, seed: int = 42, cv: int = 5, min_class_eval: int = 5):
    # Compute class counts
    class_counts = y.value_counts()
    eval_classes = class_counts[class_counts >= min_class_eval].index
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # Train on all classes
        model = train_model(X_train, y_train, seed=seed)
        # Evaluate only on sufficiently large classes
        mask = y_val.isin(eval_classes)
        y_val_eval = y_val[mask]
        X_val_eval = X_val[mask]
        preds = model.predict(X_val_eval)
        score = f1_score(y_val_eval, preds, average="weighted")
        scores.append(score)
    # Fit final model on the full dataset (all classes)
    final_model = CatBoostClassifier(verbose=0, random_seed=seed, allow_writing_files=False)
    final_model.fit(X, y, cat_features=list(range(X.shape[1])))
    return final_model, np.array(scores)


def readFull(prefix: str):
    """Read full dataset"""
    # Read test data to extract isolate IDs
    testIsolates = pd.read_csv(f"{prefix}-test.csv")["id"].astype(str).tolist()
    # Read full data and set a test column
    data = pd.read_csv(f"{prefix}-full.csv").astype(str)
    data["test"] = data["id"].apply(lambda x: x in testIsolates)
    data = data.set_index("id")
    return data


def mergeData(model_path: str, data: pd.DataFrame, linkageMatrix: np.array, labels: np.array, nGroup: int):
    model = CatBoostClassifier()
    model.load_model(model_path)
    data[f"NG{nGroup}"] = model.predict(data)
    labelsID = processNewick(linkageMatrix, labels, nGroup, name=f"NG{nGroup}-truth")
    data = pd.merge(data, labelsID, left_index=True, right_index=True, how="outer")
    return data


def testNG(model_path: str, data: pd.DataFrame, linkageMatrix: np.array, labels: np.array, nGroup: int):
    data = mergeData(model_path, data, linkageMatrix, labels, nGroup)
    testSubset = data.loc[data["test"]]
    adjRand = adjusted_rand_score(testSubset[f"NG{nGroup}"], testSubset[f"NG{nGroup}-truth"])
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
    labelsID = processNewick(linkageMatrix, labels, nGroup, name="trainNG")
    featureCols = data.columns
    sourceCount = len(data)
    data = pd.merge(data, labelsID, left_index=True, right_index=True, how="left")
    X = data[featureCols]
    y = data.pop("trainNG")
    missing = sourceCount - len(data)
    if missing > 0:
        logging.error(
            f"{missing} of {sourceCount} isolates ({missing / sourceCount:.2%}) are absent from the Newick tree labels."
        )
    if full:
        model, scores = train_model_with_cv(X, y, seed=seed)
        logger.info(f"CV F1 scores: {scores}")
        logger.info(f"Mean CV F1: {np.mean(scores):.4f}")
        logger.info(
            "Cross-validation completed nGroup=%s | fold_scores=%s | mean_f1=%.4f | std_f1=%.4f",
            nGroup,
            np.round(scores, 4),
            np.mean(scores),
            np.std(scores),
        )
    else:
        model = train_model(X, y, seed=seed)
    suffix = "final-" if full else ""
    model_path = f"{prefix}-{nGroup}-{suffix}trained.cbm"
    model.save_model(model_path)

    if full:
        data = readFull(prefix)
        linkageMatrix, labels = readTree(prefix, mode="full")
        data = mergeData(model_path, data, linkageMatrix, labels, nGroup)
        data[f"NG{nGroup}"] = model.predict(data)
        data.index = data.index.rename("id")
        data.to_csv(f"{prefix}-{nGroup}-final.csv")


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
