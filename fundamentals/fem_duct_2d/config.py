# config.py
from pathlib import Path

import numpy as np

# --- Geometry ---
L = 1.0          # duct length (m)
H = 0.1          # duct height (m)

# --- Physics ---
C = 343.0        # speed of sound (m/s)
F = 500.0        # frequency (Hz)
K = 2 * np.pi * F / C   # wavenumber (rad/m)

# --- Mesh ---
NX = int(np.ceil(L / (C / F / 6)))   # = ceil(6 * L * F / C)
NY = 4           # elements along y (pressure doesn't vary along y)

# --- Output ---
_BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_SCRATCH = _BASE_DIR / "scratch" / "outputs"
OUTPUT_DIR_FENICSX = _BASE_DIR / "fenicsx" / "outputs"

# --- Assertions ---
assert F < C / (2 * H), "Frequency beyond plane wave cutoff"