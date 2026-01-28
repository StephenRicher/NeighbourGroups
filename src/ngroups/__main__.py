#!python3
"""Neighbour Groups"""

import logging
import sys
import argparse
from pathlib import Path
from . import __version__
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
    epilog = (
        'Dessislava Veltcheva, University of Oxford '
        '(bgrdessislava@gmail.com)'
    )
    parser = argparse.ArgumentParser(epilog=epilog, description=__doc__)
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    subparser = parser.add_subparsers(
        title='required commands',
        description='',
        dest='command',
        metavar='Commands',
        help='Description:'
    )
    sp2 = subparser.add_parser(
        'run',
        description='tbc',
        help='Run the full Neighbour Groups workflow.',
        epilog=parser.epilog
    )
    sp2.add_argument(
        '--k',
        dest="k_range",
        metavar=("START", "STOP", "STEP"),
        type=int,
        nargs=3,
        default=(20, 40, 5),
        help="Range of cluster counts (k) - used to generate k-medoids targets (default: %(default)s)"
    )
    sp2.add_argument(
        '--outdir',
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
