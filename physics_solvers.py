import numpy as np
from simulation_parameters import *
from utils import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- Physical Constants ---
R_VAPOR = 461.5  # J/(kg K)
MU_VAPOR = 1.81e-5 # Pa·s

def solve_pressure(P_guess_input, avg_mass_source_mesh, T_mesh, alphas, P_inf, use_warm_start=True):
    """
    Solves the 2D steady-state P-Squared pressure equation using
    a direct sparse matrix solver.
    
    Now assumes C_p2 is a function of (x,z)
    
    PDE: C_p2(x,z) * grad^2(Phi) = -m_dot_v   (where Phi = P^2)
    FDM: C_center*Phi_i - C_x*(Phi_r+Phi_l) - C_z*(Phi_u+Phi_d) = (m_dot_v * dx2*dz2) / C_p2(x,z)
    """
    Nz, Nx = avg_mass_source_mesh.shape
    dx = W / (Nx - 1) if Nx > 1 else W
    dz = H / (Nz - 1) if Nz > 1 else H
    dx2, dz2 = dx*dx, dz*dz

    # --- P-Squared Formulation Constants (now as meshes) ---
    
    # k_eff_mesh is assumed to be (Nz, Nx)
    k_eff_mesh = get_effective_permeability(alphas)
    
    # Ensure temperature is non-zero (physically safe)
    T_mesh_safe = np.maximum(T_mesh, 1.0) # Avoid T=0K
    
    # C_p2_mesh is (Nz, Nx)
    C_p2_mesh = (k_eff_mesh / MU_VAPOR) / (2.0 * R_VAPOR * T_mesh_safe)
    
    # If C_p2 is zero everywhere, pressure gradient is impossible
    if np.all(C_p2_mesh == 0):
        return np.full((Nz, Nx), P_inf)

    P_inf_sq = P_inf * P_inf
    
    C_x = dz2
    C_z = dx2
    C_center = 2.0 * (C_x + C_z)
    if C_center == 0:
        return np.full((Nz, Nx), P_inf)

    # --- Pre-calculate the scaling factor for the source term (now a mesh) ---
    # (m_dot * dx2 * dz2) / C_p2
    # Add a small epsilon to avoid division by zero where k_eff or T might be zero
    source_factor_mesh = (dx2 * dz2) / (C_p2_mesh + 1e-30)

    # --- Build the sparse matrix A and vector b for A*Phi = b ---
    N = Nx * Nz
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)

    # Helper function to map 2D grid index (k,j) to 1D vector index i
    def idx(k, j):
        return k * Nx + j

    for k in range(Nz): # z-direction (rows)
        for j in range(Nx): # x-direction (cols)
            i = idx(k, j)
            
            # --- 1. Top Boundary (k = Nz-1): Dirichlet P = P_inf ---
            if k == Nz - 1:
                A[i, i] = 1.0
                b[i] = P_inf_sq
                continue

            # --- 2. Interior and Neumann BC nodes ---
            
            # Source term - uses the spatially-varying source factor
            b[i] = avg_mass_source_mesh[k, j] * source_factor_mesh[k, j]
            
            # Diagonal term
            A[i, i] = C_center

            # --- Z-Neighbors (Up/Down) ---
            if k == 0:
                # Bottom Boundary (k=0): Neumann dP/dz=0 (Phi_down = Phi_up)
                A[i, idx(k + 1, j)] -= 2.0 * C_z
            else:
                # Interior Z-node
                A[i, idx(k + 1, j)] -= C_z # Up
                A[i, idx(k - 1, j)] -= C_z # Down

            # --- X-Neighbors (Left/Right) ---
            if j == 0:
                # Left Boundary (j=0): Neumann dP/dx=0 (Phi_left = Phi_right)
                A[i, idx(k, j + 1)] -= 2.0 * C_x
            elif j == Nx - 1:
                # Right Boundary (j=Nx-1): Neumann dP/dx=0 (Phi_right = Phi_left)
                A[i, idx(k, j - 1)] -= 2.0 * C_x
            else:
                # Interior X-node
                A[i, idx(k, j + 1)] -= C_x # Right
                A[i, idx(k, j - 1)] -= C_x # Left
                
    # --- Solve the linear system ---
    try:
        # Convert A to Compressed Sparse Row (CSR) format for fast solving
        A_csr = A.tocsr()
        
        # Use a fast direct solver
        Phi_vec = spla.spsolve(A_csr, b)
        
        # Reshape 1D solution vector back to 2D grid
        Phi = Phi_vec.reshape((Nz, Nx))
        
    except (spla.LinAlgError, RuntimeError) as e:
        print(f"Warning: Sparse solve failed ({e}). Returning P_inf.")
        return np.full((Nz, Nx), P_inf)
    
    # Ensure no negative P^2 values (physically impossible)
    Phi[Phi < 0] = 0.0
    
    # Return P (square root of Phi)
    return np.sqrt(Phi)

def solve_heat_equation(T_old, avg_heat_source_mesh, alphas, fin_temp, dt):
    """
    Solves the 2D transient heat equation using an 
    IMPLICIT sparse matrix solver (Backward Euler).
    
    PDE: rho*cp*dT/dt = nabla.(k_eff*nabla T) + Q_source
    
    Rearranged to Ax=b form:
    [ (rho*cp/dt) - k_eff*nabla^2 ] * T_new = (rho*cp/dt)*T_old + Q_source
    """
    Nz, Nx = T_old.shape
    dx = W / (Nx - 1) if Nx > 1 else W
    dz = H / (Nz - 1) if Nz > 1 else H
    dx2, dz2 = dx*dx, dz*dz

    # --- Get Effective Properties (as 2D meshes) ---
    rho_eff_mesh = get_effective_density(alphas)
    cp_eff_mesh = get_effective_heat_capacity(alphas)
    k_eff_mesh = get_effective_thermal_conductivity(alphas)

    # --- Pre-calculate terms ---
    accum_term_mesh = (rho_eff_mesh * cp_eff_mesh) / dt
    K_x_mesh = (k_eff_mesh / dx2)
    K_z_mesh = (k_eff_mesh / dz2)
    
    N = Nx * Nz
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)

    def idx(k, j):
        return k * Nx + j

    for k in range(Nz): # z-direction (rows)
        for j in range(Nx): # x-direction (cols)
            i = idx(k, j)
            
            # Get local properties
            K_x = K_x_mesh[k, j]
            K_z = K_z_mesh[k, j]
            accum_term = accum_term_mesh[k, j]
            
            # --- Right Hand Side (b vector) ---
            # (rho*cp/dt)*T_old + Q_source
            b[i] = accum_term * T_old[k, j] + avg_heat_source_mesh[k, j]
            
            # --- Left Hand Side (A matrix) ---
            
            # --- Right Boundary (j = Nx-1): Dirichlet (Fin Temp) ---
            if j == Nx - 1:
                A[i, i] = 1.0
                b[i] = fin_temp
                continue # This node is done

            # --- Start with the diagonal term ---
            # (rho*cp/dt) + diffusion terms
            # We add all potential diffusion neighbors, then subtract
            # for Neumann BCs where neighbors don't exist.
            A[i, i] = accum_term + 2.0 * K_x + 2.0 * K_z
            
            # --- Z-Neighbors ---
            if k == 0:
                # Bottom Boundary (Neumann): T_down = T_up
                A[i, idx(k + 1, j)] -= 2.0 * K_z # Add connection to T_up twice
            elif k == Nz - 1:
                # Top Boundary (Neumann): T_up = T_down
                A[i, idx(k - 1, j)] -= 2.0 * K_z # Add connection to T_down twice
            else:
                # Interior Z-node
                A[i, idx(k + 1, j)] -= K_z # Up
                A[i, idx(k - 1, j)] -= K_z # Down
                
            # --- X-Neighbors (excluding the Dirichlet boundary) ---
            if j == 0:
                # Left Boundary (Neumann): T_left = T_right
                A[i, idx(k, j + 1)] -= 2.0 * K_x # Add connection to T_right twice
            else:
                # Interior X-node (j < Nx - 1 is guaranteed by 'if j==Nx-1' check)
                A[i, idx(k, j + 1)] -= K_x # Right
                A[i, idx(k, j - 1)] -= K_x # Left

    # --- Solve the linear system ---
    try:
        A_csr = A.tocsr()
        T_new_vec = spla.spsolve(A_csr, b)
        T_new = T_new_vec.reshape((Nz, Nx))
        
    except (spla.LinAlgError, RuntimeError) as e:
        print(f"Warning: Sparse solve for HEAT failed ({e}). Returning T_old.")
        # --- MODIFICATION: Return 0.0 for flux on failure ---
        return np.copy(T_old), 0.0
    
    # This is the total heat (in Watts) flowing from the solid mesh 
    # into the fin for this entire zone.
    
    # 1. Get properties at the boundary (j=NX-2)
    # We use T_new to get the most up-to-date solid temperatures
    T_solid_neighbors = T_new[:, NX - 2] 
    T_fin_boundary = fin_temp # This is the Dirichlet BC
    
    # Approx. conductivity at the boundary face
    k_eff_boundary = k_eff_mesh[:, NX - 2] 
    
    # 2. Calculate average flux (W/m^2) at the boundary
    # q = -k * (T_fin - T_solid) / dx
    q_flux_per_cell_k = -k_eff_boundary * (T_fin_boundary - T_solid_neighbors) / dx
    q_flux_avg = np.mean(q_flux_per_cell_k) # W/m^2
    
    # 3. Calculate total surface area for this zone
    # (These params must be in simulation_parameters.py)
    A_boundary_per_cell = (2.0 * H) * D         # Area of one cell boundary (m^2)
    Num_cells_per_zone = ZONE_LENGTH / W        # Number of cells in this zone
    A_total_surface = A_boundary_per_cell * Num_cells_per_zone
    
    # 4. Total heat (Watts)
    Q_from_solid_total = q_flux_avg * A_total_surface
    
    return T_new, Q_from_solid_total

def solve_pressure_empty(P_guess_input, avg_mass_source_mesh, T_mesh, alphas, P_inf, use_warm_start=True):
    return P_guess_input

def solve_heat_equation_empty(T_old, avg_heat_source_mesh, alphas, fin_temp, dt):
    return T_old, 0.0


def solve_fin_htf_network(Q_from_solid_zones, T_fin_guess, T_htf_guess, T_htf_inlet, 
                          htf_mass_flow, cp_htf, num_zones):
    """
    Solves the coupled steady-state energy balance for all fins and HTF zones
    simultaneously in a single sparse matrix.

    Equation for Fin_i: 
        Q_from_solid_i + Q_to_htf_i = 0
        
    Equation for HTF_i: 
        -Q_to_htf_i = m_dot_zone * cp * (T_htf_out_i - T_htf_in_i)
        
    where Q_from_solid_i is an array of known constants.
    """
    
    # --- NEW IMPLEMENTATION: Sparse Solver ---
    
    # System parameters (must be in simulation_parameters.py)
    # H_CONV_FIN: Convective heat transfer coefficient (W/m^2/K)
    # A_FIN_HTF_PER_ZONE: Convective area between fin and HTF *per zone* (m^2)
    hA = H_CONV_FIN * A_FIN_HTF_PER_ZONE 
    m_dot_zone = htf_mass_flow / num_zones
    m_dot_cp = m_dot_zone * cp_htf

    if m_dot_cp == 0 or hA == 0:
        print("Warning: Fin/HTF solver has zero m_dot_cp or hA. Returning guesses.")
        return T_fin_guess, T_htf_guess

    N_unknowns = 2 * num_zones
    A = sp.lil_matrix((N_unknowns, N_unknowns))
    b = np.zeros(N_unknowns)

    for i in range(num_zones):
        fin_row = 2 * i
        htf_row = 2 * i + 1
        fin_col = 2 * i
        htf_col = 2 * i + 1
        
        Q_solid_i = Q_from_solid_zones[i]

        # --- Fin Equation ---
        # Q_solid_i + hA * (T_htf_i - T_fin_i) = 0
        # Rearranged: (-hA) * T_fin_i + (hA) * T_htf_i = -Q_solid_i
        A[fin_row, fin_col] = -hA
        A[fin_row, htf_col] = hA
        b[fin_row] = -Q_solid_i

        # --- HTF Equation ---
        # m_dot_cp * (T_htf_out_i - T_htf_in_i) = hA * (T_fin_i - T_htf_i)
        # T_htf_out_i = T_htf_i
        # T_htf_in_i = T_htf_inlet (for i=0) or T_htf_i-1 (for i>0)
        #
        # Rearranged:
        # (-hA) * T_fin_i + (hA + m_dot_cp) * T_htf_i - (m_dot_cp) * T_htf_in_i = 0
        A[htf_row, fin_col] = -hA
        A[htf_row, htf_col] = hA + m_dot_cp
        
        if i == 0:
            # Inlet BC
            b[htf_row] = m_dot_cp * T_htf_inlet
        else:
            # Connect to previous HTF zone
            htf_prev_col = 2 * (i - 1) + 1
            A[htf_row, htf_prev_col] = -m_dot_cp
            # b[htf_row] is 0.0 (already set)

    # --- Solve the linear system ---
    try:
        A_csr = A.tocsr()
        x_vec = spla.spsolve(A_csr, b)
        
        # Unpack the solution vector
        T_fin_new_zones = x_vec[0::2] # [T_fin_0, T_fin_1, ...]
        T_htf_new_zones = x_vec[1::2] # [T_htf_0, T_htf_1, ...]
        
        return T_fin_new_zones, T_htf_new_zones
        
    except (spla.LinAlgError, RuntimeError) as e:
        print(f"Warning: Sparse solve for FIN/HTF failed ({e}). Returning old guesses.")
        return T_fin_guess, T_htf_guess
    # --- END NEW IMPLEMENTATION ---