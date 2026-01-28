#!/usr/bin/env python3

import os
import pickle
import logging
import requests
import numpy as np
import pandas as pd
from ete3 import ClusterTree
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from scipy.spatial.distance import pdist
from sklearn.compose import ColumnTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.cluster import adjusted_rand_score

def download(url: str, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = url.split('/')[-1]
    file_path = os.path.join(dir, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else: # HTTP status code 4XX/5XX
        logging.error(f'Download failed: status code {r.status_code}\n{r.text}')


def readNewick(nwk):
    with open(nwk) as fh:
        return fh.readline()


def nwk2linkage(newick: str):
    """ Convert newick tree into scipy linkage matrix """
    tree = ClusterTree(newick)
    cophenetic, newick_labels = tree.cophenetic_matrix()
    cophenetic = pd.DataFrame(
        cophenetic, columns=newick_labels, index=newick_labels)
    # reduce square distance matrix to condensed distance matrices
    pairDist = pdist(cophenetic)
    return linkage(pairDist), np.array(cophenetic.columns), cophenetic


def readTree(prefix: str, mode: str):
    linkageMatrix = np.load(f'{prefix}-{mode}-linkage.npy',  allow_pickle=True)
    labels = np.load(f'{prefix}-{mode}-labels.npy',  allow_pickle=True)
    return linkageMatrix, labels


def processNewick(
        linkageMatrix: np.array, labels: np.array, nGroup: int, name: str):
    clusters = fcluster(linkageMatrix, t=nGroup, criterion='maxclust')
    labelsID = pd.DataFrame(clusters, labels, columns=[name])
    return labelsID


def trainModel(data: pd.DataFrame, featureCols: list, seed: int = 42):
    y = data.pop('trainNG')
    X = data[featureCols]
    featureIdx = list(range(len(X.columns)))

    transformers = ([('categories', FunctionTransformer(), featureCols)])
    featureTransformer = ColumnTransformer(
        transformers=transformers, remainder='drop')
    model = Pipeline(steps=[
        ('columnTransform', featureTransformer),
        ('estimator',       CatBoostClassifier(
            verbose=0, random_seed=seed,
            allow_writing_files=False, cat_features=featureIdx)),
    ])
    model = model.fit(X, y)
    return model


def writePickle(name: str, model):
    with open(name, 'wb') as fh:
        pickle.dump(model, fh)

def readPickle(name: str):
    with open(name, 'rb') as fh:
        return pickle.load(fh)

def readFull(prefix: str):
    """ Read full dataset """
    # Read test data to extract isolate IDs
    testIsolates = pd.read_csv(f'{prefix}-test.csv')['id'].astype(str).tolist()
    # Read full data and set a test column
    data = pd.read_csv(f'{prefix}-full.csv').astype(str)
    data['test'] = data['id'].apply(lambda x: x in testIsolates)
    data = data.set_index('id')
    return data


def mergeData(
        prefix: str, data: pd.DataFrame, linkageMatrix: np.array,
        labels: np.array, nGroup: int):
    model = readPickle(f'{prefix}-{nGroup}-trained.pkl')
    data[f'NG{nGroup}'] = model.predict(data)
    labelsID = processNewick(
        linkageMatrix, labels, nGroup, name=f'NG{nGroup}-truth')
    data = pd.merge(
        data, labelsID, left_index=True, right_index=True, how='outer')
    return data


def testNG(
        prefix: str, data: pd.DataFrame, linkageMatrix: np.array,
        labels: np.array, nGroup: int):
    data = mergeData(prefix, data, linkageMatrix, labels, nGroup)
    testSubset = data.loc[data['test']]
    adjRand = adjusted_rand_score(
        testSubset[f'NG{nGroup}'], testSubset[f'NG{nGroup}-truth'])
    return adjRand


def trainNG(
        prefix: str, data: pd.DataFrame, linkageMatrix: np.array,
        labels: np.array, nGroup: int, full: bool = False, seed: int = 42):
    """ Train model for each Neighbour Group number """

    labelsID = processNewick(linkageMatrix, labels, nGroup, name='trainNG')
    featureCols = data.columns
    sourceCount = len(data)
    data = pd.merge(
        data, labelsID, left_index=True, right_index=True, how='left')
    missing = sourceCount - len(data)
    if missing > 0:
        logging.error(
            f'{missing} of {sourceCount} isolates ({missing/sourceCount:.2%}) '
            f'are absent from the Newick tree labels.')
    model = trainModel(data, featureCols)
    suffix = 'final-' if full else ''
    writePickle(f'{prefix}-{nGroup}-{suffix}trained.pkl', model)

    if full:
        data = readFull(prefix)
        linkageMatrix, labels = readTree(prefix, mode='full')
        data = mergeData(prefix, data, linkageMatrix, labels, nGroup)
        data[f'NG{nGroup}'] = model.predict(data)
        data.index = data.index.rename('id')
        data.to_csv(f'{prefix}-{nGroup}-final.csv')


def validColumns(cols, features, IDcol):
    valid = True
    if IDcol in features:
        valid = False
        logging.error(f'ID column ({IDcol}) cannot be a feature.')
    if len(features) != len(set(features)):
        valid = False
        logging.error('Duplicates found in feature names')
    for feature in features:
        if feature not in cols:
            valid = False
            logging.error(f'{feature} not in input header.')
    return valid
