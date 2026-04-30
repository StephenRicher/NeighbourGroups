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
Neighbour Groups can be run from the command line and additional help is provided via ```ngroups --help```.
Each sub-command also has help documentation (e.g. `ngroups train --help`).
The following commands can be used to reproduce the findings of the publication.

### 1. Download Publication Data
The publication data can be downloaded as below or obtained directly from the [GitHub repository](https://github.com/bgrdessislava/NeighbourGroups/tree/main/data).
The following command will download the data and save it to the directory `./data`.

```bash
ngroups get-data --outdir data
gunzip data/*gz
```

### 2. Downsample the data
At the time of writing there were 45,773 isolates available on PubMLST. 
Many of these are duplicates and the NeighbourGroup analysis is not computationally practical on such a large number.
The downsample command can be used to downsample the isolates, according to the cgMLST profiles in such a way to preserve maximum diversity.

```bash
ngroups downsample --target-n 10000 --projection-dim 128 --max-missing 0.05 --random-state 42 --cluster-factor 4 data/global_jejuni_coli_isolates-45k-profiles.tsv > data/jejuni_coli-10k-downsample.tsv
```

The above command will output a list of 10,000 isolate IDs - these can be fed back into PubMLST to retrieve the 7-MLST profiles for the selected isolates.
This data has already been provided for you for this example see `data/MGENPaper_Rerun_diverse_global_jejuni_coli_isolates_7MLST_only-10k-downsample.txt`.

### 3. Split Training and Testing Data
Once the data is downsampled - the user must download the 7 MLST metadata  of the data the user must create **two** phylogenetic trees in newick format; this step must be performed externally.
The following command splits the example data into a training and testing data set.
The first argument of most `ngroups` commands is the `prefix` - this defines the directory and filename prefix of Neighbour Groups outputs
For example, below each output file is prefixed with `./example` (e.g. `./example-test.csv`).
The prefix should be kept the same through a given analysis workflow.

```bash
ngroups prepare example data/MGENPaper_Rerun_diverse_global_jejuni_coli_isolates_7MLST_only-10k-downsample.txt --trainSize 0.8 --seed 42 --features aspA glnA gltA glyA pgm tkt uncA
```

### 4. Build Phylogenetic Trees
Following splitting of the data the user must create **two** phylogenetic trees in newick format; this step must be performed externally.
The published methodology builds a Minnimum Spamming tree (MST) from the core MLST loci using [PubMLST](https://pubmlst.org/).
However, in principle, any phylogenetic approach can be used.
The key requirement is that the labels of the Newick trees match the corresponding isolate IDs of the full and training data set.

*Note: The example data downloaded in step 1 already included pre-computed newick trees from the 10,000 isolates selected as part of the downsample. If using the example data, skip to step 4.*

#### Full Tree
The first tree is constructed from the full set of isolates - in the example these are saved to ``output/example-full.csv``.
The full tree will be used following model training to assess the prediction accuracy of the hold-out test set.
In addition the full tree can later be used to re-train a final model on the full data set, following validation.

#### Training Tree
The second tree is constructed from the training subset of isolates - in the example these are saved to ``output/example-train.csv``.
The training tree is used to extract target Neigbour Groups and train the classifier model.


### 5. Pre-process the Trees

```bash
ngroups tree example data/global_jejuni_coli_isolates-10k-downsample-full.nwk data/global_jejuni_coli_isolates-10k-downsample-train.nwk
```

### 6. Training the Model
After completing the previous steps the mode can be trained as follows.
The number of Neighbour Groups to classify must be specified as positional arguments following the prefix and a seed can be set for reproducibility.
Multiple Neighbour Group clusters can be provided to train different models at different tree hierarchy levels.

```bash
ngroups train example $(seq 2 50) --seed 42
```

### 7. Testing the Model
Following training, the `ngroups test` command can be used to assess classifier performance.
For each Neighbour Group (e.g. 22 and 44 above) an adjusted Rand index will be computed and written to stdout.

```bash
ngroups test example > adjustedRandScores.csv
```

### 8. Re-train the Model with Full Data
Following testing, the model can be retrained using the full dataset.
To retrain the model re-run the `ngroups train` command from step 5 with an additional `--full` flag.
This will output a final trained model at the location `{prefix}-{nGroup}-final-trained.pkl`.
For example, in the following command the model will be written to `./example-20-final-trained.pkl`

In addition, a final CSV final will be written to `{prefix}-{nGroup}-final.csv` which includes the NG predictions and original tree groups for all of the input data.

```bash
ngroups train example 44 --full --seed 100
```

### 9. Using the Model
Now the classifier is trained, it can be used on other data.
The `ngroups predict` command requires a path to the data (CSV format) and the trained model.

*Note: The header names of the CSV must include the features names used when training the model.*

```bash
ngroups predict data/C.jejuni-UKisolates.csv example-20-final-trained.pkl \
  > C.jejuni-UKisolates-classified.csv
```