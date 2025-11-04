import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing
from simulation_parameters import * # Import constants

# --- (Constants from your original file) ---
R_GAS = 8.314  # J/(mol*K)
MOLAR_MASS_NA2S = 78.045 # g/mol
MOLAR_MASS_H2O = 18.015 # g/mol
REACTION_KEYS = ['a4_to_a3', 'a3_to_a2', 'a2_to_a1', 'a1_to_a2', 'a2_to_a3', 'a3_to_a4']
REACTANT_INDICES = np.array([3, 2, 1, 0, 1, 2])
IS_HYDRATION = np.array([False, False, False, True, True, True])
ONSET_A = np.array([-7577.2, -8900.2, -8900.2, -7056.1002, -7056.1002, -3565.6695])
ONSET_B = np.array([24.828, 28.196, 28.196, 23.3223, 23.3223, 13.0542])
K_R = np.array([582.2, 590.1, 586.0, 596.0, 640.2, 587.3])
E_A = np.array([26119.1, 27269.0, 27599.3, 23047.1, 26961.8, 21783.8])
Q_PARAM = np.array([0.494, 0.556, 0.553, 0.506, 0.568, 0.507])
M_PARAM = np.array([1.027, 1.093, 0.989, 1.034, 1.061, 1.079])
N_PARAM = np.array([0.629, 0.176, 0.140, 0.784, 0.998, 0.942])
WATER_MOLES_PER_REACTION = np.array([1.7, 2.03, 0.77, 0.77, 2.03, 1.7])
WATER_MOLES_PER_STATE = np.array([0.5, 1.27, 3.30, 5.0])

# --- Placeholder values for source calculation ---
# These should be defined in simulation_parameters.py
# (Keeping them here for now as they are only used in this file)
DELTA_H_WATER = -66.0 # kJ/mol of water (Example value)
N_TOTAL_MOLES_SALT = 100 # Example value for moles of salt in one grid cell

def analyze_state_optimized(T, alphas, pv):
    pv = max(pv, 1e-3) # Prevent pressure from being exactly zero
    if T <= 1e-6: return np.zeros(6)

    p_star_pa = np.exp(ONSET_A / T + ONSET_B) * 100.0
    p_star_pa[4] = max(p_star_pa[4], p_star_pa[3])
    p_star_pa[5] = max(p_star_pa[5], p_star_pa[4])

    dehyd_possible = (pv < p_star_pa) & ~IS_HYDRATION
    hyd_possible = (pv > p_star_pa) & IS_HYDRATION
    reactant_present = alphas[REACTANT_INDICES] > 1e-12
    active_mask = (dehyd_possible | hyd_possible) & reactant_present
    
    rates = np.zeros(6)
    if not np.any(active_mask): return rates

    active_p_star = p_star_pa[active_mask]
    active_m = M_PARAM[active_mask]
    pressure_ratio = np.where(IS_HYDRATION[active_mask], pv / active_p_star, active_p_star / pv)
    pressure_term_base = pressure_ratio - active_m
    
    kinetically_favorable_mask = pressure_term_base > 0
    final_active_indices = np.where(active_mask)[0][kinetically_favorable_mask]
    if final_active_indices.size == 0: return rates

    final_pressure_term_base = pressure_term_base[kinetically_favorable_mask]
    # Clamp alphas to prevent negative values in the exponent
    final_reactant_alphas = np.maximum(alphas[REACTANT_INDICES[final_active_indices]], 0.0)
    
    k = K_R[final_active_indices] * np.exp(-E_A[final_active_indices] / (R_GAS * T))
    
    calculated_rates = (k * (final_reactant_alphas ** Q_PARAM[final_active_indices]) * (final_pressure_term_base ** N_PARAM[final_active_indices]) / 60.0)
    
    calculated_rates = np.nan_to_num(calculated_rates, nan=0.0, posinf=1e10, neginf=-1e10)
    
    rates[final_active_indices] = calculated_rates
    return rates

def alpha_ode_system(t, alphas, T, P):
    rates = analyze_state_optimized(T, alphas, P)
    r_a43, r_a32, r_a21, r_a12, r_a23, r_a34 = rates

    d_alpha1_dt = r_a21 - r_a12
    d_alpha2_dt = r_a12 + r_a32 - r_a21 - r_a23
    d_alpha3_dt = r_a23 + r_a43 - r_a32 - r_a34
    d_alpha4_dt = r_a34 - r_a43
    
    # solve_ivp expects a list or 1D array
    return [d_alpha1_dt, d_alpha2_dt, d_alpha3_dt, d_alpha4_dt]

# --- Worker for parallel ODE solving ---
def ode_worker(args):
    """Solves the alpha ODE system for a single grid cell."""
    y0, T_cell, P_cell, dt = args
    t_span = [0, dt]
    sol = solve_ivp(
        fun=alpha_ode_system,
        t_span=t_span, y0=y0, args=(T_cell, P_cell),
        method='BDF', rtol=1e-5, atol=1e-8
    )
    # Get the final state
    y_final = sol.y[:, -1]
    
    # --- Clamp and re-normalize alphas ---
    # Prevent floating-point overshoot from the solver
    y_final = np.clip(y_final, 0.0, 1.0)
    
    # Re-normalize to ensure the sum is exactly 1.0
    total = np.sum(y_final)
    if total > 1e-9:
        y_final = y_final / total
    else:
        # Failsafe in case sum is zero
        y_final = np.array([1.0, 0.0, 0.0, 0.0]) # Default to lowest state
    
    return y_final

# --- OPTIMIZATION: Runs all ODEs for the entire reactor in one pool ---
def predict_final_alphas_parallel(alphas_old_zones, T_guess_zones, P_guess_zones, dt):
    """
    PREDICTOR STEP: Solves for the final alphas at ALL grid points
    in the ENTIRE reactor using a single parallel pool.
    """
    # Create one giant list of arguments for every cell in every zone
    args_list = []
    for i in range(len(alphas_old_zones)): # Use len() to iterate over the list
        Nz, Nx, _ = alphas_old_zones[i].shape
        for k in range(Nz):
            for j in range(Nx):
                args_list.append((
                    alphas_old_zones[i][k, j, :],
                    T_guess_zones[i][k, j],
                    P_guess_zones[i][k, j],
                    dt
                ))

    with multiprocessing.Pool() as pool:
        results = pool.map(ode_worker, args_list)
    
    # Reconstruct the list of 3D arrays
    final_alphas_list = []
    idx = 0
    for i in range(len(alphas_old_zones)):
        Nz, Nx, _ = alphas_old_zones[i].shape
        count = Nz * Nx
        final_alphas_list.append(np.array(results[idx:idx+count]).reshape(Nz, Nx, -1))
        idx += count
        
    return final_alphas_list

# --- Averaging function ---
def calculate_avg_sources(alphas_old_mesh, alphas_final_mesh, dt):
    """
    AVERAGING STEP: Calculates average VOLUMETRIC heat and mass sources.
    Returns:
        avg_heat_source (W/m^3)
        avg_mass_source (kg/m^3/s)
    """
    moles_water_old = np.einsum('...i,i', alphas_old_mesh, WATER_MOLES_PER_STATE)
    moles_water_final = np.einsum('...i,i', alphas_final_mesh, WATER_MOLES_PER_STATE)

    # Calculate the average rate of water production/consumption (mol_water / mol_salt / s)
    avg_rate_water_per_salt = (moles_water_final - moles_water_old) / dt

    # --- MODIFIED: Convert to VOLUMETRIC sources using RHO_SALT_MOLAR ---
    # Convert to total heat generation rate (W/m^3)
    avg_heat_source = avg_rate_water_per_salt * RHO_SALT_MOLAR * (-DELTA_H_WATER * 1000.0)
    
    # Hydration (positive rate) CONSUMES gas -> NEGATIVE mass source
    # (mol_water/mol_salt/s) * (mol_salt/m^3) * (kg_water/mol_water) = kg_water/m^3/s
    avg_mass_source = -1 * avg_rate_water_per_salt * RHO_SALT_MOLAR * (MOLAR_MASS_H2O / 1000.0)

    return avg_heat_source, avg_mass_source

# --- Conversion Calculation (Moved from utils.py) ---
def calculate_conversion(Alphas):
    """
    Calculates the conversion fraction (0.0 to 1.0) based on alpha state.
    Vectorized to handle a mesh of alphas.
    """
    w_min, w_max = WATER_MOLES_PER_STATE.min(), WATER_MOLES_PER_STATE.max()
    # Use np.einsum for efficient dot product on the last axis
    current_water_content = np.einsum('...i,i', Alphas, WATER_MOLES_PER_STATE)
    return (current_water_content - w_min) / (w_max - w_min)

