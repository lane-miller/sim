"""Boxrad: config-driven 3-D acoustic FDTD for radiation from a radiator on a rigid box.

Derives the grid, time step, CPML, and run length from JSON config; writes chunked HDF5
with Huygens-surface data for post-processed Z_rad and Kirchhoff–Helmholtz far field.
"""

__version__ = "0.1.0"

from boxrad.config import Config
from boxrad.result import Result
from boxrad.runner import run

__all__ = ["Config", "Result", "__version__", "run"]