"""Command line interface for NeighbourGroups.

Attributes:
    __package_name__ (Literal["neighbourgroups"]): the package name.
    __version__ (str): the version of the package.
"""

from importlib.metadata import version

__package_name__ = "neighbourgroups"
__version__ = version(__package_name__)
