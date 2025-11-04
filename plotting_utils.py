import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LogNorm # <-- Import LogNorm
import os # <-- For creating folders and saving files
from hydration_kinetics import calculate_conversion
from simulation_parameters import *
from utils import log_history, get_effective_thermal_conductivity, get_effective_permeability
from IPython.display import clear_output # Import the clear_output function

# --- Subfolder for saving plots ---
PLOT_SAVE_FOLDER = "cross_sections"

# --- Global trackers for fixed color scales ---
_T_VMIN, _T_VMAX = None, None
_P_VMIN, _P_VMAX = None, None
_M_VMIN, _M_VMAX = None, None

PLOT_CROSS_SECTION_ZONE_INDICES = [0]  # Which zone(s) to view (e.g., [0, 5, 9])

indices = np.round(np.linspace(0, NUM_TIMESTEPS - 1, NUM_CROSS_SECTION_PLOTS)).astype(int)
indices = np.unique(indices).tolist()
print(f"Plotting cross-sections for time step indices: {indices}")

PLOT_CROSS_SECTION_TIME_STEP_INDICES = indices

def plot_reactor_heatmap(ax, data_history, total_time_s, htf_path_length, title, cbar_label):
    """
    Generates a 2D heatmap of reactor data over time and location.
    """
    if not data_history:
        ax.text(0.5, 0.5, 'No data to plot.', ha='center', va='center')
        ax.set_title(title)
        return

    data_array = np.array(data_history)
    data_to_plot = data_array.T

    # --- Get explicit min/max for color scaling ---
    vmin = np.min(data_to_plot)
    vmax = np.max(data_to_plot)
    
    # Handle case where all data is the same
    if vmin == vmax:
        vmin -= 1e-6 # Add a tiny buffer
        vmax += 1e-6

    im = ax.imshow(
        data_to_plot,
        aspect='auto', 
        origin='lower', 
        cmap='viridis',
        extent=[0, total_time_s, 0, htf_path_length],
        interpolation='nearest', # Use 'nearest' for sharp blocks
        vmin=vmin,               # Set explicit vmin
        vmax=vmax                # Set explicit vmax
    )

    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Location along HTF Path (m)')


def plot_zone_cross_section(T_mesh, P_mesh, alphas_mesh, mass_source_mesh, 
                            W, H, zone_index, time_step_index, 
                            dt, nx, nz, k_eff, perm): # <-- NEW PARAMETERS
    """
    Generates and SAVES a 4x1 plot showing 2D heatmaps of T, P, Conversion, 
    and Mass Source for a single zone at a single time step.
    
    Includes a text block with simulation parameters.
    Plots are mirrored along the x=0 axis.
    """
    global _T_VMIN, _T_VMAX, _P_VMIN, _P_VMAX, _M_VMIN, _M_VMAX
    
    T_mesh_full = np.concatenate((np.fliplr(T_mesh), T_mesh), axis=1)
    P_mesh_full = np.concatenate((np.fliplr(P_mesh), P_mesh), axis=1)
    mass_source_mesh_full = np.concatenate((np.fliplr(mass_source_mesh), mass_source_mesh), axis=1)
    
    conversion_mesh = calculate_conversion(alphas_mesh)
    mol_mol_mesh = conversion_mesh * (5.0 - 0.5) + 0.5
    mol_mol_mesh_full = np.concatenate((np.fliplr(mol_mol_mesh), mol_mol_mesh), axis=1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 7)) # Increased width/height for text
    
    # --- Main Title ---
    fig.suptitle(f"Cross-Section for Zone {zone_index} at Time Step {time_step_index}", fontsize=16, y=1.05) # Increased y for spacing

    param_text = (
        f"Timestep: {dt} s\n"
        f"Domain: {W*2*1000:.1f} mm x {H*1000:.1f} mm\n"
        f"Mesh: {nx*2} x {nz}\n" # Show full mesh width
        f"Conductivity (k_eff): {k_eff:.2f} W/mK\n"
        f"Permeability (K_perm): {perm:.2e} m²"
    )
    # Place text in the top right, relative to the figure
    fig.text(0.98, 0.98, param_text, ha='right', va='top', fontsize=9, color='gray', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)) # Moved Y up

    extent = [-W, W, 0, H]

    # --- 1. Temperature Plot ---
    ax = axes[0]
    if _T_VMIN is None: # First plot, set the global limits
        _T_VMIN = T_mesh_full.min()
        _T_VMAX = T_mesh_full.max()
        if _T_VMIN == _T_VMAX:
            _T_VMIN -= 1e-6
            _T_VMAX += 1e-6
    norm_t = Normalize(vmin=_T_VMIN, vmax=_T_VMAX)
        
    im_T = ax.imshow(T_mesh_full, aspect='auto', cmap='inferno', origin='lower', extent=extent, norm=norm_t)
    cbar_T = fig.colorbar(im_T, ax=ax, label='Temperature (K)', fraction=0.046, pad=0.04, format=mticker.ScalarFormatter(useOffset=False), norm=norm_t)
    ax.set_title('Solid Temperature')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # --- 2. Pressure Plot ---
    ax = axes[1]
    if _P_VMIN is None: # First plot, set the global limits
        _P_VMIN = P_mesh_full.min()
        _P_VMAX = P_mesh_full.max()
        if _P_VMIN == _P_VMAX:
            _P_VMIN -= 1e-6
            _P_VMAX += 1e-6
    norm_p = Normalize(vmin=_P_VMIN, vmax=_P_VMAX)
        
    im_P = ax.imshow(P_mesh_full, aspect='auto', cmap='viridis', origin='lower', extent=extent, norm=norm_p)
    cbar_P = fig.colorbar(im_P, ax=ax, label='Pressure (Pa)', fraction=0.046, pad=0.04, format=mticker.ScalarFormatter(useOffset=False), norm=norm_p)
    ax.set_title('Vapor Pressure')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # --- 3. Molar Content Plot ---
    ax = axes[2]
    im_C = ax.imshow(mol_mol_mesh_full, aspect='auto', cmap='cividis', origin='lower', extent=extent, vmin=0.5, vmax=5.0)
    cbar_C = fig.colorbar(im_C, ax=ax, label='Molar Water Content (mol H₂O / mol Na₂S)', fraction=0.046, pad=0.04)
    ax.set_title('Molar Water Content')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # --- 4. Mass Source Plot ---
    ax = axes[3]
    # Use absolute value and add epsilon for log scale
    abs_mass_source = np.abs(mass_source_mesh_full) + 1e-12 
    
    if _M_VMIN is None: # First plot, set the global limits
        _M_VMAX = abs_mass_source.max()
        # Set min 5 orders of magnitude lower ---
        _M_VMIN = max(1e-12, _M_VMAX * 1e-3) 
        if _M_VMIN >= _M_VMAX: # Failsafe if max is also tiny
            _M_VMAX = _M_VMIN * 10 # Ensure max > min

    # Use LogNorm
    norm_m = LogNorm(vmin=_M_VMIN, vmax=_M_VMAX)

    im_M = ax.imshow(abs_mass_source, aspect='auto', cmap='jet', origin='lower',
                         extent=extent, norm=norm_m)
    
    cbar_M = fig.colorbar(im_M, ax=ax, label='Log Abs. Mass Source (kg_water/m³/s)', fraction=0.046, pad=0.04, 
                           norm=norm_m)
    
    ax.set_title('Log Abs. Mass Source Rate')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # Adjust layout to make room for suptitle and fig.text
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Lowered the top boundary
    
    # --- Save the figure ---
    os.makedirs(PLOT_SAVE_FOLDER, exist_ok=True)
    
    filename = f"{zone_index:04d}_{time_step_index:04d}.png"
    filepath = os.path.join(PLOT_SAVE_FOLDER, filename)
    
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    
    plt.show()
    plt.close(fig) # Close the figure to save memory

def plot_convergence_history(residual_histories, time_step_index):
    """
    Plots the convergence residuals for P, T, Fin, and HTF on a log scale.
    """
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(residual_histories['P']) + 1)
    
    plt.semilogy(iterations, residual_histories['P'], 'o-', label='Pressure Residual')
    plt.semilogy(iterations, residual_histories['T'], 's-', label='Temperature Residual')
    plt.semilogy(iterations, residual_histories['Fin'], '^-', label='Fin Residual')
    plt.semilogy(iterations, residual_histories['HTF'], 'd-', label='HTF Residual')
    
    plt.xlabel('Coupling Iteration')
    plt.ylabel('Normalized Residual (log scale)')
    plt.title(f'Convergence History for Time Step {time_step_index + 1}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=1e-9) # Set a reasonable lower limit for log plot
    plt.show()


def log_and_plot_timestep(n, T_zones, P_zones, alphas_zones, avg_mass_source_zones, T_htf_zones, T_fin_zones, history, dt, nx, nz, residual_histories):
    """Handles all output, plotting, and history saving for a completed time step."""
    
    # --- 1. Plot Convergence History (occurs after clear_output) ---
    plot_convergence_history(residual_histories, n)
    
    # --- 2. Print Status ---
    T_outlet = T_htf_zones[NUM_ZONES - 1]
    print(f"Time step {n+1} complete. HTF Outlet Temperature: {T_outlet:.2f} K")

    # --- 3. Plot Cross-Sections ---
    if n in PLOT_CROSS_SECTION_TIME_STEP_INDICES:
        print(f"Generating cross-section plots for Time Step {n}...")
        
        for zone_idx in PLOT_CROSS_SECTION_ZONE_INDICES:
            alphas = alphas_zones[zone_idx]
            # Handle potential shape mismatches if alphas isn't 3D
            try:
                k_eff = get_effective_thermal_conductivity(alphas)[0, 0]
                perm = get_effective_permeability(alphas)[0, 0]
            except (IndexError, TypeError):
                k_eff = 0.0 # Or some default
                perm = 0.0  # Or some default
            
            plot_zone_cross_section(
                T_zones[zone_idx], P_zones[zone_idx], alphas_zones[zone_idx],
                avg_mass_source_zones[zone_idx],
                W, H, zone_idx, n,
                dt, nx, nz, k_eff, perm # Pass new params
            )

    # --- 4. Log History ---
    log_history(history, (n+1) * dt, T_zones, P_zones, alphas_zones, 
                T_fin_zones, T_htf_zones, avg_mass_source_zones, dt)


def plot_final_heatmaps(history):
    """Generates the 4-panel heatmap of all simulation results."""
    print("Generating heatmaps...")
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle('Reactor Performance Over Time', fontsize=16)
    
    plot_reactor_heatmap(axes[0, 0], history['T_htf_out'], T_SIMULATION, HTF_PATH_LENGTH,
                         title='HTF Temperature Evolution', cbar_label='HTF Temperature (K)')

    plot_reactor_heatmap(axes[0, 1], history['T_avg'], T_SIMULATION, HTF_PATH_LENGTH,
                         title='Average Solid Temperature Evolution', cbar_label='Avg. Temperature (K)')

    plot_reactor_heatmap(axes[1, 0], history['P_avg'], T_SIMULATION, HTF_PATH_LENGTH,
                         title='Average Vapor Pressure Evolution', cbar_label='Avg. Pressure (Pa)')

    plot_reactor_heatmap(axes[1, 1], history['alpha_avg'], T_SIMULATION, HTF_PATH_LENGTH,
                         title='Average Conversion Evolution', cbar_label='Conversion (0.0 - 1.0)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for main title
    plt.show()

