#!/usr/bin/env python3

from .utils import *

import os
import sys
import glob
import pickle
import logging
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split


def splitTestTrain(
        prefix: str, data: str, IDcol: None, features: list = None,
        missingVal: str = 'unknown', trainSize: float  = 0.8, seed: int = 42):
    if not (0 < trainSize < 1):
        logging.error(f'--trainSize {trainSize} not in range (0, 1).')
        return 1
    data = pd.read_csv(data, sep='\t').astype(str)
    IDcol = data.columns[0] if IDcol is None else IDcol
    data = data.set_index(IDcol)

    if features is not None:
        isValid = validColumns(data.columns, features, IDcol)
        if not isValid:
            return 1
        else:
            data = data[features]

    data.index = data.index.rename('id')
    data = data.fillna(missingVal)

    y = list(range(len(data)))
    X_train, X_test, _, _ = (
        train_test_split(
            data, list(range(len(data))),
            train_size=trainSize, random_state=seed)
    )
    os.makedirs(Path(prefix).parent, exist_ok=True)
    data.to_csv(f'{prefix}-full.csv')
    X_train.to_csv(f'{prefix}-train.csv')
    X_test.to_csv(f'{prefix}-test.csv')


def prepTree(prefix: str, fullTree: str, trainTree: str):
    """ Save Newick format as linkage matrix """
    info = ({
        f'{prefix}-full': fullTree,
        f'{prefix}-train': trainTree,
    })
    for name, tree in info.items():
        linkageMatrix, labels, cophenetic = nwk2linkage(tree)
        np.save(f'{name}-linkage.npy', linkageMatrix)
        np.save(f'{name}-labels.npy', labels)
        cophenetic.to_pickle(f'{name}-cophenetic.pkl')


def trainAll(prefix: str, nGroup: list, full: bool = False, seed: int = 42):
    """ Wrapper to training all nGroup models """
    mode = 'full' if full else 'train'
    # Read data
    data = pd.read_csv(f'{prefix}-{mode}.csv').astype(str).set_index('id')
    linkageMatrix, labels = readTree(prefix, mode)
    for ng in nGroup:
        if ng < 2:
            logging.error(f'Neighbour Group {ng} must be 2 or more - skipping.')
            continue
        else:
            logging.info(f'Training Neighbour Group {ng}.')
            trainNG(prefix, data.copy(), linkageMatrix, labels, ng, full, seed)


def testAll(prefix: str):
    """ Wrapper to testing all trained nGroup models """
    data = readFull(prefix)
    if data['test'].sum() == 0:
        logging.error('No valid test data detected - exciting')
        return 1
    # Process full tree and add fullNG labels
    linkageMatrix, labels = readTree(prefix, mode='full')

    scores = {}
    models = sorted(glob.glob(f'{prefix}-*-trained.pkl'))
    for model in models:
        nGroup = model.split('-')[-2]
        if nGroup == 'final':
            continue
        adjRand = testNG(prefix, data.copy(), linkageMatrix, labels, nGroup)
        scores[int(nGroup)] = adjRand
    print('NeighbourGroup', 'AdjRand', sep=',')
    for nGroup in sorted(scores):
        adjRand = scores[nGroup]
        print(nGroup, adjRand, sep=',')


def runNG(model: str, data: str, col: str = 'NG'):
    """ Generate NG classifications using trained model """
    data = pd.read_csv(data).astype(str)
    model = readPickle(model)
    # Dont overwrite an excisting column
    assert col not in data.columns
    assert f'{col}-prob' not in data.columns
    data[col] = model.predict(data)
    data[f'{col}-prob'] = model.predict_proba(data).max(axis=1)
    data.to_csv(sys.stdout, index=False)


def analyseNG(prefix: str):
    allData = glob.glob(f'{prefix}-*-final.csv')
    for data in allData:
        nGroup = data.split('-')[-2]
        data = pd.read_csv(data).astype(str).set_index('id')
        distances = pd.read_pickle(f'{prefix}-full-cophenetic.pkl')
        ngDistances = processNGdistances(data, distances, nGroup)
        ngDistances.to_csv(f'{prefix}-{nGroup}-meanNGdist.csv')
        ngDistances = (
            ngDistances.pivot(index='NG_r', columns='NG_c').droplevel(0, axis=1)
        )
        fig = sns.clustermap(ngDistances, cmap='viridis')
        ax = fig.ax_heatmap
        ax.set_xlabel(f'Neighbour Group ({nGroup})')
        ax.set_ylabel(f'Neighbour Group ({nGroup})')
        fig.cax.set_visible(False)
        fig.savefig(f'{prefix}-{nGroup}-meanNGdist.svg')


def processNGdistances(data: pd.DataFrame, distances: pd.DataFrame, nGroup: int):
    groupNames = data[f'NG{nGroup}-truth'].unique()
    ngDistances = {}
    for g1 in groupNames:
        g1_id = data.loc[data[f'NG{nGroup}-truth'] == g1].index.tolist()
        sub = distances[g1_id].copy()
        for g2 in groupNames:
            if g1 == g2:
                ngDistances[(g1, g2)] = 0
            else:
                g2_id = data.loc[data[f'NG{nGroup}-truth'] == g2].index.tolist()
                dist = sub.loc[g2_id].stack().dropna().mean()
                ngDistances[(g1, g2)] = dist
                ngDistances[(g2, g1)] = dist
    ngDistances = (
        pd.Series(ngDistances)
        .reset_index()
        .rename({'level_0': 'NG_r', 'level_1': 'NG_c', 0: 'distance'}, axis=1)
        .astype({'NG_r': int, 'NG_c': int, 'distance': float})
    )
    return ngDistances


def downloadExample(dir: str = '.'):
    """ Download example dataset from GitHub repo """
    os.makedirs(dir, exist_ok=True)
    prefix = 'https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data'
    download(f'{prefix}/C.jejuni-UKisolates.csv', dir)
    download(f'{prefix}/C.jejuni-full.nwk', dir)
    download(f'{prefix}/C.jejuni-train.nwk', dir)
    download(f'{prefix}/C.jejuni-UKisolates-CC.csv', dir)

def downloadModel(path: str = '.'):
    """ Download pre-trained model from publication. """
    prefix = 'https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data'
    logging.error('Not Implemented')
    return 1
    download(f'{prefix}/C.jejuni-UKisolates.pkl', dir)
