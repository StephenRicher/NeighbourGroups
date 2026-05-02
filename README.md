# NeighbourGroups

## Table of contents

  * [Installation](#installation)
  * [Usage](#usage)
    * [1. Download Publication Data](#1-download-publication-data)
    * [2. Split Training and Testing Data](#2-split-training-and-testing-data)
    * [3. Build Phylogenetic Trees](#3-build-phylogenetic-tree)
    * [4. Pre-process the Trees](#4-pre-process-the-trees)
    * [5. Training the Model](#5-training-the-model)
    * [6. Testing the Model](#6-testing-the-model)
    * [7. Re-train the Model with Full Data](#7-re-train-the-model-with-full-data)
    * [8. Using the Model](#8-using-the-model)


## Installation
```bash
pip install git+https://github.com/bgrdessislava/NeighbourGroups.git
```

## Usage
NeighbourGroups is a command-line tool. Run:
```bash
ngroups --help
```

Each subcommand also provides help:
```bash
ngroups train --help
```

The steps below reproduce the analysis from the publication.

### 1. Download Publication Data
Download the example dataset:
```bash
ngroups get-data --outdir analysis
```

Alternatively, download directly from the [GitHub repository](https://github.com/bgrdessislava/NeighbourGroups/tree/main/data).

### 2. Downsample the data
The full dataset (~45k isolates) is too large for efficient analysis. Downsampling preserves diversity while reducing size.

```bash
ngroups downsample \
  --target-n 10000 \
  --projection-dim 128 \
  --max-missing 0.05 \
  --random-state 42 \
  --cluster-factor 4 \
  analysis/global_jejuni_coli_isolates-45k-profiles.tsv \
  > analysis/global_jejuni_coli-10k-downsample.tsv
```

This outputs selected isolate IDs, which can be used to retrieve 7-MLST profiles.

For convenience, example 7-MLST data is already provided:
```bash
MGENPaper_Rerun_diverse_global_jejuni_coli_isolates_7MLST_only-10k-downsample.txt
```

### 3. Split Training and Testing Data
Split the dataset into training and test sets.
```bash
ngroups prepare analysis/example \
  analysis/MGENPaper_Rerun_diverse_global_jejuni_coli_isolates_7MLST_only-10k-downsample.txt \
  --trainSize 0.8 \
  --seed 42 \
  --features aspA glnA gltA glyA pgm tkt uncA
```

*Notes:*
- The prefix (example) defines all output files (e.g. example-train.csv)
- Use the same prefix throughout the workflow

### 4. Build Phylogenetic Trees
You must generate two Newick-format trees externally:
- *Full tree:* built from all isolates
- *Training tree:* built from the training subset

The original method uses a minimum spanning tree (MST) from PubMLST, but any phylogenetic method is acceptable.

Important: Tree labels must exactly match isolate IDs.

*Note:* If using the provided example data, precomputed trees are included—skip this step.`

#### Full Tree
The first tree is constructed from the full set of isolates - in the example these are saved to ``analysis/example-full.csv``.
The full tree will be used following model training to assess the prediction accuracy of the hold-out test set.
In addition the full tree can later be used to re-train a final model on the full data set, following validation.

#### Training Tree
The second tree is constructed from the training subset of isolates - in the example these are saved to ``analysis/example-train.csv``.
The training tree is used to extract target Neigbour Groups and train the classifier model.


### 5. Pre-process the Trees
Convert Newick trees into linkage matrices:

```bash
ngroups tree analysis/example \
  data/global_jejuni_coli_isolates-10k-downsample-full.nwk \
  data/global_jejuni_coli_isolates-10k-downsample-train.nwk
```

### 6. Training the Model
Train models across multiple Neighbour Group (NG) levels:
```bash
ngroups train analysis/example $(seq 2 50) --seed 42
```

- Each value (e.g. 2–50) corresponds to a different clustering resolution
- Multiple models are trained in one run

### 7. Testing the Model
Evaluate model performance using Adjusted Rand Index:
```bash
ngroups test analysis/example > analysis/adjustedRandScores.csv
```

### 8. Re-train the Model with Full Data
After evaluation, retrain using the full dataset:

```bash
ngroups train analysis/example 44 --full --seed 100
```

*Outputs:*
- Model: `analysis/example-44-final-trained.cbm`
- Predictions: `analysis/example-44-final.csv`

### 9. Using the Model
Apply a trained model to new data:
```bash
ngroups predict \
  coli_jejuni_MLST_isolates.csv \
  analysis/example-44-final-trained.cbm \
  > coli_jejuni_MLST_isolates-classified.csv
```

*Requirements:*
- Column names must match training feature names
