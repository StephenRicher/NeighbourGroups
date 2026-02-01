#!python3
"""Neighbour Groups"""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .downsample import downsample
from .ngroups import ngroups_full_workflow

logger = logging.getLogger(__name__)
"""logging.Logger: logger instance for the module."""


def main(argv: list[str] | None = None) -> None:
    """Main entrypoint for OpsQC.

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
        "data",
        type=Path,
        help=(
            "Path to a tabular file (e.g. CSV/TSV/Parquet) containing cgMLST profiles. "
            "The first column must be the isolate ID; remaining columns are allele calls."
        ),
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
    sp2 = subparser.add_parser(
        "run", description="tbc", help="Run the full Neighbour Groups workflow.", epilog=parser.epilog
    )
    sp2.add_argument(
        "--k",
        dest="k_range",
        metavar=("START", "STOP", "STEP"),
        type=int,
        nargs=3,
        default=(20, 40, 5),
        help="Range of cluster counts (k) - used to generate k-medoids targets (default: %(default)s)",
    )
    sp2.add_argument(
        "--outdir",
        metavar="PATH",
        type=Path,
        default=Path.cwd(),
        help="Output directory to write results (default: %(default)s)",
    )
    sp2.set_defaults(func=ngroups_full_workflow)

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
