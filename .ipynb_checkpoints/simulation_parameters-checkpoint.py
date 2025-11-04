import numpy as np

# --- Grid and Geometry ---
NUM_ZONES = 10        # Number of zones in the 'i' direction (flow direction)
NX = 25                # Number of nodes in x-direction
NZ = 25                # Number of nodes in z-direction
W = 0.002              # Half-width between fins (m)
H = 0.02               # Half-height of the fin space (m)
D = 0.10               # Depth of the fin space (m)
L = 0.5                # Length of the reactor in y-direction (m)
NUM_FINS_GROUPED = 50  # Number of fins modeled by this 2D mesh
ZONE_LENGTH = 0.1      # Physical length of one zone (m)
HTF_PATH_LENGTH = NUM_ZONES * ZONE_LENGTH # Total HTF path length

# --- Time Discretization ---
T_SIMULATION = 60000    # Total simulation time (s)
DT = 600.0             # Time step (s)
NUM_CROSS_SECTION_PLOTS = 100
NUM_TIMESTEPS = int(T_SIMULATION / DT)

# --- Initial Conditions ---
T_INITIAL = 353        # Initial temperature (K)
P_INF = 1200           # Ambient vapor pressure at z=H/2 (Pa)
P_INITIAL = P_INF      # Initial pressure (Pa)

# Initial alpha fractions [a1, a2, a3, a4]
# Na2S·0.5H2O, Na2S·1.27H2O, Na2S·3.3H2O, Na2S·5H2O
ALPHA_INITIAL = np.array([0.0, 0.0, 0.0, 1.0]) # Start fully dehydrated

# --- Heat Transfer Fluid (HTF) Properties (Water) ---
T_HTF_IN = 333.15      # Inlet HTF temperature (K)
RHO_HTF = 1000         # Density of HTF (kg/m^3)
CP_HTF = 4186          # Specific heat of HTF (J/(kg*K))
HTF_MASS_FLOW = 0.1    # Mass flow rate of HTF (kg/s)

# --- Physical Properties ---
# Molar density of solid Na2S (approx)
RHO_SALT_MOLAR = 21700 # mol/m^3 (based on 1.69 g/cm^3 / 78.045 g/mol * 1e6 cm^3/m^3)
# Enthalpy of water vaporization/condensation (approx)
DELTA_H_WATER = -66.0  # kJ/mol (negative for hydration/condensation)
# Density of water vapor (approx, e.g., at 60C, 1200 Pa)
RHO_VAPOR = 0.009 # kg/m^3 (This is an approximation, ideally P*M/(R*T))


# --- Solver Settings ---
MAX_ITER_COUPLING = 500   # Max iterations for the main coupling loop
MAX_ITER_ZONE = 100      # Max iterations for inner zone convergence
MAX_ITER_PRESSURE = 15000 # Max iterations for the pressure solver (Increased)
TOLERANCE_COUPLING = 1e-3  # Convergence tolerance for main loop
TOLERANCE_ZONE = 1e-4    # Convergence tolerance for fin temperature in a zone
TOLERANCE_PRESSURE = 1e-9  # Convergence tolerance for pressure solver (Tightened)

# --- Adaptive Relaxation Parameters ---
OMEGA_MASS_SOURCE_INITIAL = 0.10   # Starting relaxation factor
OMEGA_MASS_SOURCE_MIN = 0.001      # Minimum allowed factor (for stability)
OMEGA_MASS_SOURCE_MAX = 0.25       # Maximum allowed factor (1.0 = no relaxation)
ADAPTIVE_OMEGA_INCREASE = 1.2     # Multiplier when converging
ADAPTIVE_OMEGA_DECREASE = 0.5     # Multiplier when diverging

OMEGA_STATE_RELAXATION_T = 0.05
OMEGA_STATE_RELAXATION_P = 0.25
