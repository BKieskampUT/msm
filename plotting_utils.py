import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LogNorm # <-- Import LogNorm
import os # <-- For creating folders and saving files
# Import from hydration_kinetics instead of utils
from hydration_kinetics import calculate_conversion
from simulation_parameters import W, H # Import geometry

# --- Subfolder for saving plots ---
PLOT_SAVE_FOLDER = "cross_sections"

# --- Global trackers for fixed color scales ---
_T_VMIN, _T_VMAX = None, None
_P_VMIN, _P_VMAX = None, None
_M_VMIN, _M_VMAX = None, None

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
    # --- MODIFIED: Use global variables for fixed color scales ---
    global _T_VMIN, _T_VMAX, _P_VMIN, _P_VMAX, _M_VMIN, _M_VMAX
    # --- END MODIFICATION ---

    # --- MODIFICATION: Create mirrored data for full plot ---
    T_mesh_full = np.concatenate((np.fliplr(T_mesh), T_mesh), axis=1)
    P_mesh_full = np.concatenate((np.fliplr(P_mesh), P_mesh), axis=1)
    mass_source_mesh_full = np.concatenate((np.fliplr(mass_source_mesh), mass_source_mesh), axis=1)
    
    # --- MODIFICATION: Rescale conversion to mol/mol ---
    conversion_mesh = calculate_conversion(alphas_mesh)
    mol_mol_mesh = conversion_mesh * (5.0 - 0.5) + 0.5
    mol_mol_mesh_full = np.concatenate((np.fliplr(mol_mol_mesh), mol_mol_mesh), axis=1)
    # --- END MODIFICATION ---

    fig, axes = plt.subplots(1, 4, figsize=(16, 7)) # Increased width/height for text
    
    # --- Main Title ---
    fig.suptitle(f"Cross-Section for Zone {zone_index} at Time Step {time_step_index}", fontsize=16, y=1.05) # Increased y for spacing

    # --- NEW: Parameter Text Block (Moved to top right) ---
    param_text = (
        f"Timestep: {dt} s\n"
        # --- MODIFICATION: Show full domain width ---
        f"Domain: {W*2*1000:.1f} mm x {H*1000:.1f} mm\n"
        f"Mesh: {nx*2} x {nz}\n" # Show full mesh width
        f"Conductivity (k_eff): {k_eff:.2f} W/mK\n"
        f"Permeability (K_perm): {perm:.2e} m²"
    )
    # Place text in the top right, relative to the figure
    fig.text(0.98, 0.98, param_text, ha='right', va='top', fontsize=9, color='gray', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)) # Moved Y up
    # --- END NEW ---

    # --- MODIFICATION: Updated extent for mirrored plot ---
    extent = [-W, W, 0, H]

    # --- 1. Temperature Plot ---
    ax = axes[0]
    # --- MODIFIED: Set/use global fixed color limits ---
    if _T_VMIN is None: # First plot, set the global limits
        _T_VMIN = T_mesh_full.min()
        _T_VMAX = T_mesh_full.max()
        if _T_VMIN == _T_VMAX:
            _T_VMIN -= 1e-6
            _T_VMAX += 1e-6
    norm_t = Normalize(vmin=_T_VMIN, vmax=_T_VMAX)
    # --- END MODIFICATION ---
        
    im_T = ax.imshow(T_mesh_full, aspect='auto', cmap='inferno', origin='lower', extent=extent, norm=norm_t)
    cbar_T = fig.colorbar(im_T, ax=ax, label='Temperature (K)', fraction=0.046, pad=0.04, format=mticker.ScalarFormatter(useOffset=False), norm=norm_t)
    ax.set_title('Solid Temperature')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # --- 2. Pressure Plot ---
    ax = axes[1]
    # --- MODIFIED: Set/use global fixed color limits ---
    if _P_VMIN is None: # First plot, set the global limits
        _P_VMIN = P_mesh_full.min()
        _P_VMAX = P_mesh_full.max()
        if _P_VMIN == _P_VMAX:
            _P_VMIN -= 1e-6
            _P_VMAX += 1e-6
    norm_p = Normalize(vmin=_P_VMIN, vmax=_P_VMAX)
    # --- END MODIFICATION ---
        
    im_P = ax.imshow(P_mesh_full, aspect='auto', cmap='viridis', origin='lower', extent=extent, norm=norm_p)
    cbar_P = fig.colorbar(im_P, ax=ax, label='Pressure (Pa)', fraction=0.046, pad=0.04, format=mticker.ScalarFormatter(useOffset=False), norm=norm_p)
    ax.set_title('Vapor Pressure')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # --- 3. Molar Content Plot (MODIFIED) ---
    ax = axes[2]
    im_C = ax.imshow(mol_mol_mesh_full, aspect='auto', cmap='cividis', origin='lower', extent=extent, vmin=0.5, vmax=5.0)
    cbar_C = fig.colorbar(im_C, ax=ax, label='Molar Water Content (mol H₂O / mol Na₂S)', fraction=0.046, pad=0.04)
    ax.set_title('Molar Water Content')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')

    # --- 4. Mass Source Plot (MODIFIED) ---
    ax = axes[3]
    # Use absolute value and add epsilon for log scale
    abs_mass_source = np.abs(mass_source_mesh_full) + 1e-12 
    
    if _M_VMIN is None: # First plot, set the global limits
        _M_VMAX = abs_mass_source.max()
        # --- MODIFIED: Set min 5 orders of magnitude lower ---
        _M_VMIN = max(1e-12, _M_VMAX * 1e-3) 
        # --- END MODIFICATION ---
        if _M_VMIN >= _M_VMAX: # Failsafe if max is also tiny
            _M_VMAX = _M_VMIN * 10 # Ensure max > min

    # Use LogNorm
    norm_m = LogNorm(vmin=_M_VMIN, vmax=_M_VMAX)

    im_M = ax.imshow(abs_mass_source, aspect='auto', cmap='jet', origin='lower',
                         extent=extent, norm=norm_m)
    
    # --- MODIFIED: Removed 'formatter' kwarg to fix TypeError ---
    cbar_M = fig.colorbar(im_M, ax=ax, label='Log Abs. Mass Source (kg_water/m³/s)', fraction=0.046, pad=0.04, 
                          norm=norm_m)
    # --- END MODIFICATION ---
    
    ax.set_title('Log Abs. Mass Source Rate')
    # --- END MODIFICATION ---
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

