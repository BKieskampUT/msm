import numpy as np
from simulation_parameters import * # For ZONE_LENGTH, W, H, NX, NZ
from hydration_kinetics import (predict_final_alphas_parallel, calculate_avg_sources,
                                  calculate_conversion)

def get_effective_thermal_conductivity(alphas):
    """Calculates effective thermal conductivity based on local state."""
    Nz, Nx, _ = alphas.shape
    # Placeholder:
    return np.full((Nz, Nx), 0.5)  # W/(m*K)

def get_effective_density(alphas):
    """Calculates effective density based on local state."""
    Nz, Nx, _ = alphas.shape
    # Placeholder:
    return np.full((Nz, Nx), 1500.0) # kg/m^3

def get_effective_heat_capacity(alphas):
    """Calculates effective heat capacity based on local state."""
    Nz, Nx, _ = alphas.shape
    # Placeholder:
    return np.full((Nz, Nx), 1200) # J/(kg*K)

def get_effective_permeability(alphas):
    """Calculates effective permeability for Darcy flow."""
    Nz, Nx, _ = alphas.shape
    # Placeholder:
    return np.full((Nz, Nx), 3e-10) # m^2

# --- Convergence Check ---
def check_convergence(v_new, v_old, tolerance):
    """
    Checks for convergence based on relative error.
    Handles both single arrays and lists of arrays.
    """
    # Convert lists of arrays into single, flat arrays
    if isinstance(v_new, list):
        v_new = np.concatenate([arr.ravel() for arr in v_new])
    if isinstance(v_old, list):
        v_old = np.concatenate([arr.ravel() for arr in v_old])

    v_old_norm = np.linalg.norm(v_old)
    if v_old_norm < 1e-9: # Avoid division by zero
        return np.linalg.norm(v_new - v_old) < tolerance
    return np.linalg.norm(v_new - v_old) / v_old_norm < tolerance


def calculate_residual(new_val_list, old_val_list):
    """Calculates the normalized L2 norm of the residual between two lists of arrays."""
    total_error_sq = 0.0
    total_norm_sq = 0.0
    
    is_single_array = isinstance(new_val_list, np.ndarray)
    if is_single_array:
         new_val_list = [new_val_list]
         old_val_list = [old_val_list]

    for new, old in zip(new_val_list, old_val_list):
        total_error_sq += np.sum(np.square(new - old))
        total_norm_sq += np.sum(np.square(old)) + 1e-10 # Add epsilon for stability
        
    return np.sqrt(total_error_sq / total_norm_sq)

def update_sources_and_relax(alphas_old_zones, T_guess_zones, P_guess_zones, dt, 
                             mass_source_guess_zones, heat_source_guess_zones, omega):
    """
    Refactored function.
    1. Predicts final alphas for all zones.
    2. Calculates new average heat/mass sources for all zones.
    3. Applies relaxation to the new sources.
    Returns the final alphas and the relaxed mass/heat source lists.
    """
    
    # --- 1. Predict Sources (Parallel) ---
    alphas_final_zones = predict_final_alphas_parallel(
        alphas_old_zones, T_guess_zones, P_guess_zones, dt
    )
    
    avg_heat_source_zones_new_calc = [] 
    avg_mass_source_zones_new_calc = [] 
    
    for i in range(len(alphas_old_zones)):
        avg_heat, avg_mass = calculate_avg_sources(
            alphas_old_zones[i], alphas_final_zones[i], dt
        )
        avg_heat_source_zones_new_calc.append(avg_heat)
        avg_mass_source_zones_new_calc.append(avg_mass) 

    # --- Check for NaN/Inf from kinetics solver ---
    has_nan = any(np.isnan(arr).any() for arr in avg_mass_source_zones_new_calc)
    has_inf = any(np.isinf(arr).any() for arr in avg_mass_source_zones_new_calc)

    if has_nan or has_inf:
        print(f"  CRITICAL: Kinetics solver failed (NaN/Inf detected).")
        print("  This is likely due to numerical instability (oscillating T/P guesses).")
        print(f"  RECOMMENDATION: Reduce DT (currently {dt}) in simulation_parameters.py.")
        raise RuntimeError(f"Kinetics solver failed (NaN/Inf). Reduce DT from {dt}.")

    # --- 2. Apply Source Relaxation ---
    mass_source_relaxed_zones = []
    heat_source_relaxed_zones = []
    
    for i in range(len(alphas_old_zones)):
        mass_relaxed = (
            mass_source_guess_zones[i] * (1.0 - omega) +
            avg_mass_source_zones_new_calc[i] * omega
        )
        heat_relaxed = (
            heat_source_guess_zones[i] * (1.0 - omega) +
            avg_heat_source_zones_new_calc[i] * omega
        )
        mass_source_relaxed_zones.append(mass_relaxed)
        heat_source_relaxed_zones.append(heat_relaxed)
        
    return alphas_final_zones, mass_source_relaxed_zones, heat_source_relaxed_zones


def log_history(history, t_current, T_zones, P_zones, alphas_zones, 
                T_fin_zones, T_htf_zones, avg_mass_source_zones, dt):
    """
    Refactored function.
    Calculates averages and appends all current state data to the history dict.
    """
    history['time'].append(t_current)
    history['T_avg'].append([np.mean(T) for T in T_zones])
    history['P_avg'].append([np.mean(P) for P in P_zones])
    history['alpha_avg'].append([np.mean(calculate_conversion(a)) for a in alphas_zones])
    history['T_fin_avg'].append(np.mean(T_fin_zones))
    history['T_htf_out'].append(np.copy(T_htf_zones))
    
    # Calculate total mass vaporized in this step
    total_mass_vaporized = 0.0
    try:
        # Calculate cell volume once
        V_cell = (W / (NX - 1)) * (H / (NZ - 1)) * (ZONE_LENGTH)
        for mass_source_mesh in avg_mass_source_zones:
            total_mass_vaporized += np.sum(mass_source_mesh) * V_cell * dt
    except (NameError, AttributeError):
        # Fallback if params aren't loaded
        for mass_source_mesh in avg_mass_source_zones:
             total_mass_vaporized += np.sum(mass_source_mesh) # Placeholder sum
    
    history['mass_vaporized'].append(total_mass_vaporized)