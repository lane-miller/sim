"""
HRTF POC — Configuration
========================
FABIAN HATS BEM simulation using Bempp, validated against TU Berlin
measured and simulated HRTFs.

Reciprocal formulation: point monopole at blocked ear canal entrance,
rigid Neumann BC on all surfaces, evaluate total field on far-field sphere.
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FABIAN_ROOT = Path("/Volumes/LPM02 storage/Datasets/Audio/HRTF/FABIAN/FABIAN_HRTF_DATABASE_v4")

# Source mesh (original scan)
MESH_ORIGINAL = FABIAN_ROOT / "2 SurfaceMeshes" / "FABIAN_6k_HATO0.stl"
MESH_GRADED = OUTPUT_DIR / "FABIAN_6k_HATO0_graded.stl"

# Processed mesh (truncated, recentered to interaural midpoint)
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MESH_TRUNCATED = OUTPUT_DIR / "FABIAN_6k_HATO0_truncated.stl"

# Reference HRTFs (SOFA format)
HRIR_DIR = FABIAN_ROOT / "1 HRIRs"
HRIR_MEASURED = HRIR_DIR / "FABIAN_HRIR_measured_HATO_0.sofa"
HRIR_SIMULATED = HRIR_DIR / "FABIAN_HRIR_modeled_HATO_0.sofa"

# ---------------------------------------------------------------------------
# Mesh coordinate convention (native STL frame, post-recentering)
# ---------------------------------------------------------------------------
# Origin: interaural midpoint (center of line segment between ear canals)
# X: front (+) / back (-)
# Y: left (+) / right (-)
# Z: up (+) / down (-)
# Units: millimeters (convert to meters before BEM solve)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_AIR = 343.18        # m/s
RHO_AIR = 1.1839      # kg/m³

# ---------------------------------------------------------------------------
# Reciprocal source positions (mm, in recentered mesh frame)
# From mesh.py output — shared vertices of ear canal element clusters.
# These sit ON the surface; solve.py must offset ~1mm outward along
# the local surface normal to avoid Green's function singularity.
# ---------------------------------------------------------------------------
SOURCE_LEFT_MM = np.array([-2.22, 66.23, -2.00])
SOURCE_RIGHT_MM = np.array([2.22, -66.23, 2.00])

# ---------------------------------------------------------------------------
# Evaluation sphere (for reciprocal HRTF extraction)
# ---------------------------------------------------------------------------
EVAL_RADIUS_M = 1.7   # meters — FABIAN SOFA measurement radius
EVAL_THETA = [0.0]    # degrees elevation (start with horizontal plane only)
EVAL_PHI_STEP = 5.0   # degrees azimuth step

# ---------------------------------------------------------------------------
# Mesh processing
# ---------------------------------------------------------------------------
TORSO_CUT_Z = -200.0  # mm — truncation plane
SNAP_TOL = 4.5        # mm — pre-snap tolerance for sliver prevention

# Ear canal element clusters (1-indexed, original FABIAN_6k_HATO0.stl)
EAR_L_ELEMENTS = [1920, 1921, 1922, 1923, 1924]
EAR_R_ELEMENTS = [5392, 5393, 5394, 5395, 5396]