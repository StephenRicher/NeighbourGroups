#!/usr/bin/env python3

import glob
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import *


def splitTestTrain(
    prefix: str,
    data: str,
    IDcol: None,
    features: list = None,
    missingVal: str = "unknown",
    trainSize: float = 0.8,
    seed: int = 42,
):
    if not (0 < trainSize < 1):
        logging.error(f"--trainSize {trainSize} not in range (0, 1).")
        return 1
    data = pd.read_csv(data, sep="\t").astype(str)
    IDcol = data.columns[0] if IDcol is None else IDcol
    data = data.set_index(IDcol)

    if features is not None:
        isValid = validColumns(data.columns, features, IDcol)
        if not isValid:
            return 1
        else:
            data = data[features]

    data.index = data.index.rename("id")
    data = data.fillna(missingVal)

    y = list(range(len(data)))
    X_train, X_test, _, _ = train_test_split(data, list(range(len(data))), train_size=trainSize, random_state=seed)
    os.makedirs(Path(prefix).parent, exist_ok=True)
    data.to_csv(f"{prefix}-full.csv")
    X_train.to_csv(f"{prefix}-train.csv")
    X_test.to_csv(f"{prefix}-test.csv")


def prepTree(prefix: str, fullTree: str, trainTree: str):
    """Save Newick format as linkage matrix"""
    info = {
        f"{prefix}-full": fullTree,
        f"{prefix}-train": trainTree,
    }
    for name, tree in info.items():
        linkageMatrix, labels, cophenetic = nwk2linkage(tree)
        np.save(f"{name}-linkage.npy", linkageMatrix)
        np.save(f"{name}-labels.npy", labels)
        cophenetic.to_pickle(f"{name}-cophenetic.pkl")


def trainAll(prefix: str, nGroup: list, full: bool = False, seed: int = 42):
    """Wrapper to training all nGroup models"""
    mode = "full" if full else "train"
    # Read data
    data = pd.read_csv(f"{prefix}-{mode}.csv").astype(str).set_index("id")
    linkageMatrix, labels = readTree(prefix, mode)
    for ng in nGroup:
        if ng < 2:
            logging.error(f"Neighbour Group {ng} must be 2 or more - skipping.")
            continue
        else:
            logging.info(f"Training Neighbour Group {ng}.")
            trainNG(prefix, data.copy(), linkageMatrix, labels, ng, full, seed)


def testAll(prefix: str):
    """Wrapper to testing all trained nGroup models"""
    data = readFull(prefix)
    if data["test"].sum() == 0:
        logging.error("No valid test data detected - exciting")
        return 1
    # Process full tree and add fullNG labels
    linkageMatrix, labels = readTree(prefix, mode="full")
    scores = {}
    for model_path in sorted(glob.glob(f"{prefix}-*-trained.cbm")):
        nGroup = model_path.split("-")[-2]
        if nGroup == "final":
            continue
        adjRand = testNG(model_path, data.copy(), linkageMatrix, labels, nGroup)
        scores[int(nGroup)] = adjRand
    print("NeighbourGroup", "AdjRand", sep=",")
    for nGroup in sorted(scores):
        adjRand = scores[nGroup]
        print(nGroup, adjRand, sep=",")


def runNG(model_path: str, data: str, col: str = "NG"):
    """Generate NG classifications using trained model"""
    data = pd.read_csv(data, low_memory=False, sep="\t")
    model = CatBoostClassifier()
    model.load_model(model_path)
    # Dont overwrite an existing column
    assert col not in data.columns
    assert f"{col}-prob" not in data.columns
    model_data = prepare_categorical(data[model.feature_names_])
    data[col] = model.predict(model_data)
    data[f"{col}-prob"] = model.predict_proba(model_data).max(axis=1)
    data.to_csv(sys.stdout, index=False)


def downloadExample(outdir: str = "."):
    """Download example dataset from GitHub repo"""
    os.makedirs(outdir, exist_ok=True)
    prefix = "https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data"
    download(f"{prefix}/global_jejuni_coli_isolates-45k-profiles.tsv.gz", outdir)
    download(f"{prefix}/MGENPaper_Rerun_diverse_global_jejuni_coli_isolates_7MLST_only-10k-downsample.txt.gz", outdir)
    download(f"{prefix}/global_jejuni_coli_isolates-10k-downsample-full.nwk", outdir)
    download(f"{prefix}/global_jejuni_coli_isolates-10k-downsample-train.nwk", outdir)
    download(f"{prefix}/global_jejuni_coli_isolates_all_isolates.txt.gz", outdir)
