#!python3
"""Neighbour Groups"""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .downsample import downsample
from .ngroups import analyseNG, downloadExample, downloadModel, prepTree, runNG, splitTestTrain, testAll, trainAll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
"""logging.Logger: logger instance for the module."""


def main(argv: list[str] | None = None) -> None:
    """Main entrypoint for NeighbourGroups.

    Args:
        argv: Command line arguments.

    Returns:
        Exit status
    """
    epilog = "Dessislava Veltcheva, University of Oxford (bgrdessislava@gmail.com)"
    parser = argparse.ArgumentParser(epilog=epilog, description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s {}".format(__version__))
    subparser = parser.add_subparsers(
        title="required commands", description="", dest="command", metavar="Commands", help="Description:"
    )
    downsample_sp = subparser.add_parser(
        "downsample",
        description=(
            "Downsample cgMLST allele profiles to a representative subset of isolates "
            "that are approximately maximally separated in allele space."
        ),
        help="Downsample cgMLST profiles while preserving genetic diversity",
        epilog=parser.epilog,
    )
    downsample_sp.add_argument(
        "data", type=Path, help="TSV file containing cgMLST profiles. The first column must be the isolate ID"
    )
    downsample_sp.add_argument(
        "--target-n",
        dest="target_n",
        metavar="X",
        type=int,
        default=5000,
        help="Number of isolates to retain after downsampling (default: %(default)s)",
    )
    downsample_sp.add_argument(
        "--projection-dim",
        dest="projection_dim",
        metavar="X",
        type=int,
        default=64,
        help=(
            "Dimensionality of the sparse random projection used prior to clustering. "
            "Lower values are faster but less accurate (default: %(default)s)"
        ),
    )
    downsample_sp.add_argument(
        "--cluster-factor",
        dest="cluster_factor",
        metavar="X",
        type=int,
        default=3,
        help=(
            "Multiplier controlling the number of clusters relative to the target size "
            "(number of clusters ≈ cluster-factor × target-n). "
            "Higher values improve diversity at the cost of runtime (default: %(default)s)"
        ),
    )
    downsample_sp.add_argument(
        "--max-missing",
        dest="max_missing",
        metavar="X",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed proportion of missing loci per isolate (0–1). "
            "Isolates exceeding this threshold are excluded before downsampling (default: %(default)s)"
        ),
    )
    downsample_sp.add_argument(
        "--random-state",
        dest="random_state",
        metavar="X",
        type=int,
        default=42,
        help="Random seed for reproducible projection, clustering, and sampling (default: %(default)s)",
    )
    downsample_sp.set_defaults(func=downsample)

    prepare_sp = subparser.add_parser(
        "prepare",
        description=splitTestTrain.__doc__,
        help="Split isolates into test and training set.",
        epilog=parser.epilog,
    )
    prepare_sp.add_argument("prefix", help="File prefix to read/write data.")
    prepare_sp.add_argument("data", help="Path to data file in .csv format")
    prepare_sp.add_argument(
        "--trainSize", type=float, default=0.8, help="Proportion of data to use as training (default: %(default)s)"
    )
    prepare_sp.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducing train/test split (default: %(default)s)"
    )
    prepare_sp.add_argument(
        "--missingVal", default="unknown", help="Value to replace missing data (default: %(default)s)"
    )
    prepare_sp.add_argument(
        "--IDcol", help="Column index (zero-based) of data corresponding to isolate ID (default: %(default)s)"
    )
    prepare_sp.add_argument(
        "--features",
        nargs="+",
        help="Column indices (zero-based) of training features. If not "
        "provided, all columns except index 0 are assumed to be "
        "training features",
    )
    prepare_sp.set_defaults(func=splitTestTrain)

    tree_sp = subparser.add_parser(
        "tree", description=prepTree.__doc__, help="Pre-process the Newick trees.", epilog=parser.epilog
    )
    tree_sp.add_argument("prefix", help="File prefix to read/write data.")
    tree_sp.add_argument("fullTree", help="Path to full tree in newick format.")
    tree_sp.add_argument("trainTree", help="Path to training tree in newick format.")
    tree_sp.set_defaults(func=prepTree)

    train_sp = subparser.add_parser(
        "train", description=trainAll.__doc__, help="Train the CatBoost classifer.", epilog=parser.epilog
    )
    train_sp.add_argument("prefix", help="File prefix to read/write data.")
    train_sp.add_argument("nGroup", type=int, nargs="+", help="Number of Neighbour Groups to classify.")
    train_sp.add_argument(
        "--full",
        action="store_true",
        help="Train model using full dataset instead of training subset (default: %(default)s)",
    )
    train_sp.add_argument(
        "--seed", type=int, default=42, help="Seed for defining random state of classifer (default: %(default)s)"
    )
    train_sp.set_defaults(func=trainAll)

    test_sp = subparser.add_parser(
        "test", description=testAll.__doc__, help="Test the CatBoost classifer.", epilog=parser.epilog
    )
    test_sp.add_argument("prefix", help="File prefix to read/write data.")
    test_sp.set_defaults(func=testAll)

    predict_sp = subparser.add_parser(
        "predict", description=runNG.__doc__, help="Classify isolates using the trained model.", epilog=parser.epilog
    )
    predict_sp.add_argument("data", help="Path to data file in .csv format")
    predict_sp.add_argument("model_path", help="Path to trained NeighbourGroup model.")
    predict_sp.add_argument("--col", default="NG", help="Column name to write predictions (default: %(default)s)")
    predict_sp.set_defaults(func=runNG)

    stats_sp = subparser.add_parser(
        "stats", description=analyseNG.__doc__, help="Interrogate NG relationships.", epilog=parser.epilog
    )
    stats_sp.add_argument("prefix", help="File prefix to read/write data.")
    stats_sp.set_defaults(func=analyseNG)

    getdata_sp = subparser.add_parser(
        "getData", description=downloadExample.__doc__, help="Download example data.", epilog=parser.epilog
    )
    getdata_sp.add_argument("--dir", default=".", help="Directory to save example data (default: %(default)s)")
    getdata_sp.set_defaults(func=downloadExample)

    getmodel_sp = subparser.add_parser(
        "getModel",
        description=downloadExample.__doc__,
        help="Download pre-trained model from publication.",
        epilog=parser.epilog,
    )
    getmodel_sp.add_argument("--dir", default=".", help="Directory to save model (default: %(default)s)")
    getmodel_sp.set_defaults(func=downloadModel)

    pargs = parser.parse_args(argv)
    if "func" not in pargs:
        parser.print_help()
        sys.exit(1)
    _ = pargs.__dict__.pop("command")
    func = pargs.__dict__.pop("func")
    return func(**vars(pargs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
