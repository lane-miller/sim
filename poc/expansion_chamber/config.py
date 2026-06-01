import numpy as np

# Fluid properties
C = 343.0          # speed of sound [m/s]
RHO = 1.225        # density [kg/m^3]

# Geometry [m] — concentric expansion chamber, symmetric about x and y midplanes
# Inlet and outlet ducts have identical cross-sections
INLET_W   = 0.020   # duct x-dimension (width)
INLET_H   = 0.030   # duct y-dimension (height)
CHAMBER_W = 0.060   # chamber x-dimension
CHAMBER_H = 0.080   # chamber y-dimension
INLET_L   = 0.040   # inlet duct length
CHAMBER_L = 0.150   # expansion chamber length
OUTLET_L  = 0.040   # outlet duct length

# Derived
TOTAL_L = INLET_L + CHAMBER_L + OUTLET_L

# Frequencies — geometrically spaced, staying below first higher-order mode cutoff
# First higher-order mode in chamber: c/(2*CHAMBER_H) ≈ 2144 Hz
F_MAX = 2000.0
N_FREQ = 25
FREQS = np.geomspace(50.0, F_MAX, N_FREQ)
