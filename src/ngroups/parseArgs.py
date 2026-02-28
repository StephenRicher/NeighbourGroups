#!/usr/bin/env python3

"""Neighbour Groups - ML Classification"""

import argparse
import logging
import sys
from timeit import default_timer as timer

from ._version import __version__
from .main import analyseNG, downloadExample, downloadModel, prepTree, runNG, splitTestTrain, testAll, trainAll


def parseArgs() -> argparse.Namespace:
    epilog = "Dessislava Veltcheva, University of Oxford (bgrdessislava@gmail.com)"
    baseParser = getBaseParser(__version__)
    parser = argparse.ArgumentParser(epilog=epilog, description=__doc__, parents=[baseParser])
    subparser = parser.add_subparsers(
        title="required commands", description="", dest="command", metavar="Commands", help="Description:"
    )

    sp1 = subparser.add_parser(
        "prepare",
        description=splitTestTrain.__doc__,
        help="Split isolates into test and training set.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp1.add_argument("prefix", help="File prefix to read/write data.")
    sp1.add_argument("data", help="Path to data file in .csv format")
    sp1.add_argument(
        "--trainSize", type=float, default=0.8, help="Proportion of data to use as training (default: %(default)s)"
    )
    sp1.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducing train/test split (default: %(default)s)"
    )
    sp1.add_argument("--missingVal", default="unknown", help="Value to replace missing data (default: %(default)s)")
    sp1.add_argument(
        "--IDcol", help="Column index (zero-based) of data corresponding to isolate ID (default: %(default)s)"
    )
    sp1.add_argument(
        "--features",
        nargs="+",
        help="Column indices (zero-based) of training features. If not "
        "provided, all columns except index 0 are assumed to be "
        "training features",
    )
    sp1.set_defaults(function=splitTestTrain)

    sp7 = subparser.add_parser(
        "tree",
        description=runNG.__doc__,
        help="Pre-process the Newick trees.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp7.add_argument("prefix", help="File prefix to read/write data.")
    sp7.add_argument("fullTree", help="Path to full tree in newick format.")
    sp7.add_argument("trainTree", help="Path to training tree in newick format.")
    sp7.set_defaults(function=prepTree)

    sp2 = subparser.add_parser(
        "train",
        description=trainAll.__doc__,
        help="Train the CatBoost classifer.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp2.add_argument("prefix", help="File prefix to read/write data.")
    sp2.add_argument("nGroup", type=int, nargs="+", help="Number of Neighbour Groups to classify.")
    sp2.add_argument(
        "--full",
        action="store_true",
        help="Train model using full dataset instead of training subset (default: %(default)s)",
    )
    sp2.add_argument(
        "--seed", type=int, default=42, help="Seed for defining random state of classifer (default: %(default)s)"
    )
    sp2.set_defaults(function=trainAll)

    sp3 = subparser.add_parser(
        "test",
        description=testAll.__doc__,
        help="Test the CatBoost classifer.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp3.add_argument("prefix", help="File prefix to read/write data.")
    sp3.set_defaults(function=testAll)

    sp4 = subparser.add_parser(
        "predict",
        description=runNG.__doc__,
        help="Classify isolates using the trained model.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp4.add_argument("data", help="Path to data file in .csv format")
    sp4.add_argument("model", help="Path to trained NeighbourGroup model.")
    sp4.add_argument("--col", default="NG", help="Column name to write predictions (default: %(default)s)")
    sp4.set_defaults(function=runNG)

    sp8 = subparser.add_parser(
        "stats",
        description=analyseNG.__doc__,
        help="Interrogate NG relationships.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp8.add_argument("prefix", help="File prefix to read/write data.")
    sp8.set_defaults(function=analyseNG)

    sp5 = subparser.add_parser(
        "getData",
        description=downloadExample.__doc__,
        help="Download example data.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp5.add_argument("--dir", default=".", help="Directory to save example data (default: %(default)s)")
    sp5.set_defaults(function=downloadExample)

    sp6 = subparser.add_parser(
        "getModel",
        description=downloadExample.__doc__,
        help="Download pre-trained model from publication.",
        parents=[baseParser],
        epilog=parser.epilog,
    )
    sp6.add_argument("--dir", default=".", help="Directory to save model (default: %(default)s)")
    sp6.set_defaults(function=downloadModel)

    args = parser.parse_args()
    if "function" not in args:
        parser.print_help()
        sys.exit()

    rc = executeCommand(args)
    return rc


def executeCommand(args):
    # Initialise logging
    logFormat = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    logging.basicConfig(level=args.verbose, format=logFormat)
    del args.verbose, args.command
    # Pop main function and excute script
    function = args.__dict__.pop("function")
    start = timer()
    rc = function(**vars(args))
    end = timer()
    logging.info(f"Total execution time: {end - start:.3f} seconds.")
    logging.shutdown()
    return rc


def getBaseParser(version: str) -> argparse.Namespace:
    """Create base parser of verbose/version."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", action="version", version="%(prog)s {}".format(version))
    parser.add_argument(
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.ERROR,
        help="verbose logging for debugging",
    )
    return parser
