#!/usr/bin/env python3

import glob
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
    logger.info("Starting train/test split | data=%s | trainSize=%.2f | seed=%d", data, trainSize, seed)

    if not (0 < trainSize < 1):
        logger.error("--trainSize %s not in range (0, 1).", trainSize)
        return 1

    # Load data
    logger.info("Loading input data from %s", data)
    data = pd.read_csv(data, sep="\t").astype(str)
    logger.info("Loaded dataset with %d rows and %d columns", data.shape[0], data.shape[1])

    # Set ID column
    IDcol = data.columns[0] if IDcol is None else IDcol
    logger.info("Using '%s' as ID column", IDcol)
    data = data.set_index(IDcol)

    # Feature selection
    if features is not None:
        logger.info("Selecting %d feature columns", len(features))
        isValid = validColumns(data.columns, features, IDcol)
        if not isValid:
            logger.error("Feature validation failed")
            return 1
        else:
            data = data[features]
    else:
        logger.info("No feature subset provided, using all %d columns", data.shape[1])

    # Preprocessing
    data.index = data.index.rename("id")
    data = data.fillna(missingVal)
    logger.info("Missing values filled with '%s'", missingVal)

    # Train/test split
    logger.info("Performing train/test split")
    X_train, X_test, _, _ = train_test_split(data, list(range(len(data))), train_size=trainSize, random_state=seed)

    logger.info(
        "Split complete | train=%d samples | test=%d samples",
        len(X_train),
        len(X_test),
    )

    # Output
    outdir = Path(prefix).parent
    os.makedirs(outdir, exist_ok=True)
    logger.info("Saving outputs to prefix '%s'", prefix)

    data.to_csv(f"{prefix}-full.csv")
    X_train.to_csv(f"{prefix}-train.csv")
    X_test.to_csv(f"{prefix}-test.csv")

    logger.info("Files written: %s-full.csv, %s-train.csv, %s-test.csv", prefix, prefix, prefix)


def prepTree(prefix: str, fullTree: str, trainTree: str):
    """Save Newick format as linkage matrix"""
    logger.info(
        "Starting tree preprocessing | prefix=%s | fullTree=%s | trainTree=%s",
        prefix,
        fullTree,
        trainTree,
    )

    info = {
        f"{prefix}-full": fullTree,
        f"{prefix}-train": trainTree,
    }

    for name, tree in info.items():
        logger.info("Processing tree '%s' from file %s", name, tree)

        linkageMatrix, labels, cophenetic = nwk2linkage(tree)

        logger.info(
            "Tree '%s' processed | n_leaves=%d | linkage_shape=%s",
            name,
            len(labels),
            linkageMatrix.shape,
        )

        np.save(f"{name}-linkage.npy", linkageMatrix)
        np.save(f"{name}-labels.npy", labels)
        cophenetic.to_pickle(f"{name}-cophenetic.pkl")

        logger.info(
            "Saved outputs for '%s': linkage.npy, labels.npy, cophenetic.pkl",
            name,
        )

    logger.info("Tree preprocessing completed for prefix '%s'", prefix)


def trainAll(prefix: str, nGroup: list, full: bool = False, seed: int = 42):
    """Wrapper to training all nGroup models"""
    mode = "full" if full else "train"

    logger.info(
        "Starting training pipeline | prefix=%s | mode=%s | nGroups=%s | seed=%d",
        prefix,
        mode,
        nGroup,
        seed,
    )

    # Read data
    data_path = f"{prefix}-{mode}.csv"
    logger.info("Loading training data from %s", data_path)
    data = pd.read_csv(data_path).astype(str).set_index("id")
    logger.info("Loaded data | samples=%d | features=%d", data.shape[0], data.shape[1])

    logger.info("Loading tree data for mode='%s'", mode)
    linkageMatrix, labels = readTree(prefix, mode)
    logger.info("Tree loaded | n_labels=%d | linkage_shape=%s", len(labels), linkageMatrix.shape)

    for ng in nGroup:
        if ng < 2:
            logger.error("Neighbour Group %d must be >= 2 - skipping", ng)
            continue

        logger.info("---- Training Neighbour Group %d ----", ng)
        trainNG(prefix, data.copy(), linkageMatrix, labels, ng, full, seed)

    logger.info("Training pipeline completed for prefix='%s'", prefix)


def testAll(prefix: str):
    """Wrapper to testing all trained nGroup models"""
    logger.info("Starting model evaluation | prefix=%s", prefix)

    # Load full dataset
    data = readFull(prefix)
    n_total = len(data)
    n_test = data["test"].sum()

    logger.info("Loaded dataset | total_samples=%d | test_samples=%d", n_total, n_test)

    if n_test == 0:
        logger.error("No valid test data detected - exiting")
        return 1

    # Load tree
    logger.info("Loading full tree for evaluation")
    linkageMatrix, labels = readTree(prefix, mode="full")

    # Find models
    model_paths = sorted(glob.glob(f"{prefix}-*-trained.cbm"))
    logger.info("Discovered %d trained models", len(model_paths))

    scores = {}

    for model_path in model_paths:
        nGroup = model_path.split("-")[-2]

        if nGroup == "final":
            logger.debug("Skipping final model: %s", model_path)
            continue

        logger.info("Evaluating model | nGroup=%s | path=%s", nGroup, model_path)

        adjRand = testNG(model_path, data.copy(), linkageMatrix, labels, nGroup)

        logger.info("Result | nGroup=%s | AdjustedRand=%.4f", nGroup, adjRand)

        scores[int(nGroup)] = adjRand

    logger.info("Evaluation complete | evaluated_models=%d", len(scores))

    # Output results
    print("NeighbourGroup", "AdjRand", sep=",")
    for nGroup in sorted(scores):
        adjRand = scores[nGroup]
        print(nGroup, adjRand, sep=",")


def runNG(model_path: str, data: str, col: str = "NG"):
    """Generate NG classifications using trained model"""
    logger.info("Starting prediction | model=%s | data=%s | output_col=%s", model_path, data, col)

    # Load data
    logger.info("Loading input data")
    data = pd.read_csv(data, low_memory=False, sep="\t")
    logger.info("Data loaded | rows=%d | columns=%d", data.shape[0], data.shape[1])

    # Load model
    logger.info("Loading model from %s", model_path)
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Validate output columns
    if col in data.columns or f"{col}-prob" in data.columns:
        logger.error("Output columns '%s' or '%s-prob' already exist in input data", col, col)
        raise ValueError(f"Columns {col} or {col}-prob already exist")

    # Validate feature alignment
    expected_features = list(model.feature_names_)
    logger.info("Model expects %d features", len(expected_features))

    missing_features = [f for f in expected_features if f not in data.columns]
    if missing_features:
        logger.error("Missing required features: %s", missing_features[:10])
        raise ValueError(f"Missing required features (showing up to 10): {missing_features[:10]}")

    logger.info("Preparing categorical features")
    model_data = prepare_categorical(data[expected_features])

    logger.info("Running predictions")
    preds = model.predict(model_data)
    probs = model.predict_proba(model_data).max(axis=1)

    data[col] = preds
    data[f"{col}-prob"] = probs

    logger.info(
        "Prediction complete | rows=%d | unique_classes=%d",
        len(data),
        pd.Series(preds).nunique(),
    )

    # Output
    logger.info("Writing predictions to stdout")
    data.to_csv(sys.stdout, index=False)


def downloadExample(outdir: str = "."):
    """Download example dataset from GitHub repo"""
    logger.info("Starting download of example dataset | outdir=%s", outdir)

    os.makedirs(outdir, exist_ok=True)

    prefix = "https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data"

    files = [
        "global_jejuni_coli_isolates-45k-profiles.tsv.gz",
        "MGENPaper_Rerun_diverse_global_jejuni_coli_isolates_7MLST_only-10k-downsample.txt.gz",
        "global_jejuni_coli_isolates-10k-downsample-full.nwk",
        "global_jejuni_coli_isolates-10k-downsample-train.nwk",
        "global_jejuni_coli_isolates_all_isolates.txt.gz",
    ]

    for fname in files:
        url = f"{prefix}/{fname}"
        logger.info("Downloading %s", fname)
        download(url, outdir)

    logger.info("All example data downloaded successfully to %s", outdir)
