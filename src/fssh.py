#!/usr/bin/python3
import numpy as np
#import matplotlib.pyplot as plt

# ==============================================================================
# 1. POTENTIAL ENERGY SURFACE AND COUPLING DEFINITION
# ==============================================================================

omega1 = 7.743E-3
omega2 = 6.680E-3
a = 31.05

def create_tanh_pes(x, y):
    A = 0.008
    k = 0.5
    C = 0.001
    sigma = 10.0
    
    r = np.sqrt(x*x + y*y)
    if r > 1e-8:
        prefac = A * np.tanh(k * r) / r
    else:
        prefac = A * k
        
    H11 = prefac * x * y
    H22 = -prefac * x * y
    H12 = C * r * r * np.exp(-r / sigma)
    
    return np.array([[H11, H12], [H12, H22]])


def create_lvc_pes(x, y):
    c = 8.092E-5
    sigma = 0

    H11  = 0.5*(omega1*omega1*(x +a/2)*(x +a/2) + omega2*omega2*y*y + sigma)
    H22 =  0.5*(omega1*omega1*(x -a/2)*(x -a/2) + omega2*omega2*y*y - sigma)
    H12 = 2*c*y
        
    return np.array([[H11, H12], [H12, H22]])


def get_diabatic_H(x, y):
    return create_tanh_pes(x,y)

def get_diabatic_gradients(x, y, eps=1e-5):
    """Calculates numerical gradients of the diabatic matrix elements."""
    dH_dx = (get_diabatic_H(x + eps, y) - get_diabatic_H(x - eps, y)) / (2 * eps)
    dH_dy = (get_diabatic_H(x, y + eps) - get_diabatic_H(x, y - eps)) / (2 * eps)
    return dH_dx, dH_dy

def get_adiabatic_properties(x, y, U_old=None):
    """Returns sorted eigenvalues, phase-tracked eigenvectors, forces, and NACV."""
    H = get_diabatic_H(x, y)
    E, U = np.linalg.eigh(H)  # E[0] = Lower (S0), E[1] = Upper (S1)
    
    # Enforce electronic phase tracking to avoid trivial signs/phase flips
    if U_old is not None:
        for s in [0, 1]:
            if np.dot(U[:, s], U_old[:, s]) < 0:
                U[:, s] = -U[:, s]
                
    dH_dx, dH_dy = get_diabatic_gradients(x, y)
    
    # Adiabatic Hellmann-Feynman Forces: F_a = - <v_a | dH/dR | v_a>
    F0_x = -np.dot(U[:, 0], np.dot(dH_dx, U[:, 0]))
    F0_y = -np.dot(U[:, 0], np.dot(dH_dy, U[:, 0]))
    F1_x = -np.dot(U[:, 1], np.dot(dH_dx, U[:, 1]))
    F1_y = -np.dot(U[:, 1], np.dot(dH_dy, U[:, 1]))
    forces = np.array([[F0_x, F0_y], [F1_x, F1_y]])
    
    # Nonadiabatic Coupling Vector (NACV): d_01 = <v_0 | dH/dR | v_1> / (E1 - E0)
    gap = E[1] - E[0]
    if gap > 1e-6:
        d01_x = np.dot(U[:, 0], np.dot(dH_dx, U[:, 1])) / gap
        d01_y = np.dot(U[:, 0], np.dot(dH_dy, U[:, 1])) / gap
    else:
        d01_x, d01_y = 0.0, 0.0
    d01 = np.array([d01_x, d01_y])
    
    return E, U, forces, d01

# ==============================================================================
# 2. RUNTIME SIMULATION PARAMETERS
# ==============================================================================
mass = 2000.0
hbar = 1.0
dt = 0.05          
tmax = 3200.0
n_steps = int(tmax / dt)

# Granucci-Persico Parameters
C_edc = 1.0        

# --- FIX B: Sample initial conditions from a normal distribution (\sigma = 1.0) ---
r_mean = np.array([-4.0, -4.0])
p_mean = np.array([5.0, 5.0])

r_coord = np.random.normal(loc=r_mean, scale=0.1)
p_moment = np.random.normal(loc=p_mean, scale=0.1)

# Initialize Electronic states (Evolving in Adiabatic basis)
# Start 100% populated on the Upper Adiabatic state S1 (Index 1)
active_state = 1   
c_coeffs = np.array([0.0, 1.0], dtype=complex)

# Data storage arrays
history_time = []
history_pos = []
history_mom = []    # Added to store momentum components
history_state = []
history_pop = []
history_energy = []
history_D01 = []    # Added to track non-adiabatic coupling scalar over time

def tdse_deriv(c_vec, D01_val):
    dc0 = -1j/hbar * E[0] * c_vec[0] - D01_val * c_vec[1]
    dc1 = -1j/hbar * E[1] * c_vec[1] + D01_val * c_vec[0]
    return np.array([dc0, dc1])

# Initial setup
E, U, forces, d01 = get_adiabatic_properties(r_coord[0], r_coord[1])

# ==============================================================================
# 3. MAIN SURFACE HOPPING LOOP
# ==============================================================================
print("Running FSSH single trajectory simulation...")
for step in range(n_steps):
    t = step * dt
    v_velocity = p_moment / mass
    E_kin = np.dot(p_moment, p_moment) / (2.0 * mass)
    E_tot = E_kin + E[active_state]
    
    # Calculate current non-adiabatic coupling scalar projection
    D01 = np.dot(v_velocity, d01)
    
    # Save properties at current time step
    history_time.append(t)
    history_pos.append(r_coord.copy())
    history_mom.append(p_moment.copy())
    history_state.append(active_state)
    history_pop.append([np.abs(c_coeffs[0])**2, np.abs(c_coeffs[1])**2])
    history_energy.append([E_kin, E[active_state], E_tot])
    history_D01.append(D01)
    
    # 3a. Nuclear Step: Half-Step Velocity Verlet
    p_moment += 0.5 * forces[active_state] * dt
    r_coord += (p_moment / mass) * dt
    
    # Recompute potential properties at new coordinates
    E_new, U_new, forces_new, d01_new = get_adiabatic_properties(r_coord[0], r_coord[1], U_old=U)
    p_moment += 0.5 * forces_new[active_state] * dt
    
    # Update properties
    E, U, forces, d01 = E_new, U_new, forces_new, d01_new
    v_velocity = p_moment / mass
    
    # 3b. Electronic Step: Time-Dependent Schrödinger Equation Integration (RK4 Substepping)
    D01_next = np.dot(v_velocity, d01)  
    
    # Internal high-frequency electronic integration
    substeps = 16
    dt_sub = dt / substeps
    for _ in range(substeps):
        k1 = tdse_deriv(c_coeffs, D01_next)
        k2 = tdse_deriv(c_coeffs + 0.5 * dt_sub * k1, D01_next)
        k3 = tdse_deriv(c_coeffs + 0.5 * dt_sub * k2, D01_next)
        k4 = tdse_deriv(c_coeffs + dt_sub * k3, D01_next)
        c_coeffs += (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 3d. Evaluate Tully's Surface Hopping Probability
        rho = np.outer(c_coeffs, np.conj(c_coeffs))
        target_state = 1 - active_state
        if active_state == 0:
            g_prob = (2.0 * dt_sub * np.max([0.0, np.real(rho[1, 0] * D01_next)])) / np.real(rho[0, 0])
        else:
            g_prob = (2.0 * dt_sub * np.max([0.0, np.real(rho[0, 1] * (-D01_next))])) / np.real(rho[1, 1])
            
        # 3e. Stochastic Hop Test & 2D Momentum Rescaling Along the NACV Field Vector
        if np.random.rand() < g_prob:
            if np.dot(d01, d01) > 1e-6:
                a_quad = np.dot(d01, d01) / (2.0 * mass)
                b_quad = -np.dot(p_moment, d01) / mass if active_state == 0 else np.dot(p_moment, d01) / mass
                c_quad = E[target_state] - E[active_state]
                
                discriminant = b_quad**2 - 4.0 * a_quad * c_quad
                
                if discriminant >= 0:
                    root_sign = 1.0 if b_quad >= 0 else -1.0
                    gamma = (-b_quad + root_sign * np.sqrt(discriminant)) / (2.0 * a_quad)
                    
                    if active_state == 0:
                        p_moment -= gamma * d01
                    else:
                        p_moment += gamma * d01
                        
                    active_state = target_state  
                    c_coeffs = np.zeros(2, dtype=complex)
                    c_coeffs[active_state] = 1.0
                else:
                    pass  # Frustrated hop





    # 3c. Apply Granucci and Persico Decoherence Correction (EDC)
    if E_kin > 0:
        non_active = 1 - active_state
        tau_deco = (hbar / np.abs(E[1] - E[0])) * (1.0 + C_edc / E_kin)
        
        # Exponential dampening of the unassigned coherent state
        c_coeffs[non_active] *= np.exp(-dt / tau_deco)
        # Rescale active state coefficient to conserve total probability
        if np.abs(c_coeffs[active_state]) > 1e-10:
            phase_factor = c_coeffs[active_state] / np.abs(c_coeffs[active_state])
            c_coeffs[active_state] = phase_factor * np.sqrt(1.0 - np.abs(c_coeffs[non_active])**2)
        else:
            c_coeffs[active_state] = np.sqrt(1.0 - np.abs(c_coeffs[non_active])**2)

#    # 3d. Evaluate Tully's Surface Hopping Probability
#    rho = np.outer(c_coeffs, np.conj(c_coeffs))
#    target_state = 1 - active_state
#    if active_state == 0:
#        g_prob = (2.0 * dt * np.max([0.0, np.real(rho[1, 0] * D01_next)])) / np.real(rho[0, 0])
#    else:
#        g_prob = (2.0 * dt * np.max([0.0, np.real(rho[0, 1] * (-D01_next))])) / np.real(rho[1, 1])
#        
#    # 3e. Stochastic Hop Test & 2D Momentum Rescaling Along the NACV Field Vector
#    if np.random.rand() < g_prob:
#        if np.dot(d01, d01) > 1e-6:
#            a_quad = np.dot(d01, d01) / (2.0 * mass)
#            b_quad = -np.dot(p_moment, d01) / mass if active_state == 0 else np.dot(p_moment, d01) / mass
#            c_quad = E[target_state] - E[active_state]
#            
#            discriminant = b_quad**2 - 4.0 * a_quad * c_quad
#            
#            if discriminant >= 0:
#                root_sign = 1.0 if b_quad >= 0 else -1.0
#                gamma = (-b_quad + root_sign * np.sqrt(discriminant)) / (2.0 * a_quad)
#                
#                if active_state == 0:
#                    p_moment -= gamma * d01
#                else:
#                    p_moment += gamma * d01
#                    
#                active_state = target_state  
#                c_coeffs = np.zeros(2, dtype=complex)
#                c_coeffs[active_state] = 1.0
#            else:
#                pass  # Frustrated hop
#
print("Simulation finished.")

# ==============================================================================
# 4. DATA EXPORT FOR GNUPLOT (FIX C)
# ==============================================================================
data_filename = "fssh_trajectory_data.dat"
print(f"Saving trajectory logs to '{data_filename}'...")
printrate = 100
with open(data_filename, 'w') as f_out:
    # Clean space/tab separated file for easy plotting in gnuplot
    f_out.write("# 1:Time\t2:X\t3:Y\t4:Px\t5:Py\t6:Pop_S0\t7:Pop_S1\t8:D01\t9:ActiveState\n")
    for s in range(n_steps):
        if (s%printrate == 0):
            f_out.write(f"{history_time[s]:.4f}\t"
                    f"{history_pos[s][0]:.6f}\t{history_pos[s][1]:.6f}\t"
                    f"{history_mom[s][0]:.6f}\t{history_mom[s][1]:.6f}\t"
                    f"{history_pop[s][0]:.6f}\t{history_pop[s][1]:.6f}\t"
                    f"{history_D01[s]:.6f}\t"
                    f"{history_state[s]}\n")

# ==============================================================================
# 5. DATA VISUALIZATION AND IMAGE SAVE (FIX A)
# ==============================================================================
#history_pos = np.array(history_pos)
#history_pop = np.array(history_pop)
#history_energy = np.array(history_energy)

#fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#
## Plot 1: Trajectory Motion Map
#axes[0,0].plot(history_pos[:,0], history_pos[:,1], label="Nuclear Trajectory Path", color='black', lw=2)
#axes[0,0].scatter(history_pos[0,0], history_pos[0,1], color='green', marker='o', s=100, label="Start")
#axes[0,0].scatter(history_pos[-1,0], history_pos[-1,1], color='red', marker='x', s=100, label="End")
#axes[0,0].set_title("2D Coordinate Trajectory Evolution")
#axes[0,0].set_xlabel("X coordinate"); axes[0,0].set_ylabel("Y coordinate")
#axes[0,0].grid(True, linestyle=":"); axes[0,0].legend()
#
## Plot 2: Adiabatic Population Comparison
#axes[0,1].plot(history_time, history_pop[:,0], label="State S0 Pop (Lower)", color='blue', lw=2)
#axes[0,1].plot(history_time, history_pop[:,1], label="State S1 Pop (Upper)", color='orange', lw=2)
#axes[0,1].set_title("Adiabatic Electronic Populations (with EDC)")
#axes[0,1].set_xlabel("Time (fs)"); axes[0,1].set_ylabel("Population Density")
#axes[0,1].grid(True, linestyle=":"); axes[0,1].legend()
#
## Plot 3: Energy Trace
#axes[1,0].plot(history_time, history_energy[:,0], label="Kinetic Energy", color='purple', linestyle='--')
#axes[1,0].plot(history_time, history_energy[:,1], label="Potential Energy", color='teal', linestyle='--')
#axes[1,0].plot(history_time, history_energy[:,2], label="Total Energy", color='black', lw=2.5)
#axes[1,0].set_title("Energy Tracking Profile")
#axes[1,0].set_xlabel("Time (fs)"); axes[1,0].set_ylabel("Energy (a.u.)")
#axes[1,0].grid(True, linestyle=":"); axes[1,0].legend()
#
## Plot 4: Active Surface Indicator
#axes[1,1].plot(history_time, history_state, color='crimson', lw=2)
#axes[1,1].set_title("Active Trajectory Surface ID State")
#axes[1,1].set_xlabel("Time (fs)"); axes[1,1].set_ylabel("Active State Index")
#axes[1,1].set_yticks([0, 1])
#axes[1,1].grid(True, linestyle=":")
#
#plt.tight_layout()
## Save to image instead of blocking thread with show()
#image_filename = "fssh_trajectory_plots.png"
#plt.savefig(image_filename, dpi=20)
#plt.close(fig)
#print(f"Plots successfully exported to '{image_filename}'.")
