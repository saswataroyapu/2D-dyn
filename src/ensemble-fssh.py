#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. POTENTIAL ENERGY SURFACE AND COUPLING DEFINITION
# ==============================================================================
omega1 = 7.743E-3
omega2 = 6.680E-3
a = 31.05

def create_lvc_pes(x, y):
    c = 8.092E-5
    sigma = 0

    H11 = 0.5 * (omega1 * omega1 * (x + a / 2) * (x + a / 2) + omega2 * omega2 * y * y + sigma)
    H22 = 0.5 * (omega1 * omega1 * (x - a / 2) * (x - a / 2) + omega2 * omega2 * y * y - sigma)
    H12 = 2 * c * y
        
    return np.array([[H11, H12], [H12, H22]])

def get_diabatic_H(x, y):
    return create_lvc_pes(x, y)

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
mass = 1.0
hbar = 1.0
dt = 0.05          
tmax = 1000.0
n_steps = int(tmax / dt)

# Granucci-Persico Parameters
C_edc = 1.0        

# Ensemble Configurations
num_trajectories = 100
r_mean = np.array([a / 2.0, 0.0])
p_mean = np.array([0.0, 0.0])
sigma_x = 1.0
sigma_p = 1.0

# Global accumulation buffers for calculating averages
accum_pos = np.zeros((n_steps, 2))
accum_mom = np.zeros((n_steps, 2))
accum_quantum_pop = np.zeros((n_steps, 2))
accum_active_pop = np.zeros((n_steps, 2))
accum_energy = np.zeros((n_steps, 3))

def tdse_deriv(c_vec, E_arr, D01_val):
    dc0 = -1j/hbar * E_arr[0] * c_vec[0] - D01_val * c_vec[1]
    dc1 = -1j/hbar * E_arr[1] * c_vec[1] + D01_val * c_vec[0]
    return np.array([dc0, dc1])

# ==============================================================================
# 3. MAIN ENSEMBLE EXECUTION LOOP
# ==============================================================================
print(f"Starting FSSH ensemble simulation with {num_trajectories} trajectories...")

for traj in range(num_trajectories):
    if (traj + 1) % 10 == 0 or traj == 0:
        print(f"  Simulating trajectory {traj + 1}/{num_trajectories}...")
        
    # Sample independent initial coordinate & momentum variables for each unique run
    r_coord = np.array([np.random.normal(r_mean[0], sigma_x), np.random.normal(r_mean[1], sigma_x)])
    p_moment = np.array([np.random.normal(p_mean[0], sigma_p), np.random.normal(p_mean[1], sigma_p)])
    
    # Initialize state coefficients (Start 100% on S1)
    active_state = 1   
    c_coeffs = np.array([0.0, 1.0], dtype=complex)
    
    E, U, forces, d01 = get_adiabatic_properties(r_coord[0], r_coord[1])
    
    # Time evolution for the individual trajectory
    for step in range(n_steps):
        v_velocity = p_moment / mass
        E_kin = np.dot(p_moment, p_moment) / (2.0 * mass)
        E_tot = E_kin + E[active_state]
        
        # Log properties to global accumulator step-wise
        accum_pos[step] += r_coord
        accum_mom[step] += p_moment
        accum_quantum_pop[step, 0] += np.abs(c_coeffs[0])**2
        accum_quantum_pop[step, 1] += np.abs(c_coeffs[1])**2
        
        if active_state == 0:
            accum_active_pop[step, 0] += 1.0
        else:
            accum_active_pop[step, 1] += 1.0
            
        accum_energy[step] += [E_kin, E[active_state], E_tot]
        
        # 3a. Nuclear Step: Half-Step Velocity Verlet
        p_moment += 0.5 * forces[active_state] * dt
        r_coord += (p_moment / mass) * dt
        
        # Recompute potential properties at new positions
        E_new, U_new, forces_new, d01_new = get_adiabatic_properties(r_coord[0], r_coord[1], U_old=U)
        p_moment += 0.5 * forces_new[active_state] * dt
        
        E, U, forces, d01 = E_new, U_new, forces_new, d01_new
        v_velocity = p_moment / mass
        
        # 3b. Electronic Step: Integration with RK4 Substepping
        D01_next = np.dot(v_velocity, d01)  
        
        substeps = 16
        dt_sub = dt / substeps
        for _ in range(substeps):
            k1 = tdse_deriv(c_coeffs, E, D01_next)
            k2 = tdse_deriv(c_coeffs + 0.5 * dt_sub * k1, E, D01_next)
            k3 = tdse_deriv(c_coeffs + 0.5 * dt_sub * k2, E, D01_next)
            k4 = tdse_deriv(c_coeffs + dt_sub * k3, E, D01_next)
            c_coeffs += (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # 3d. Evaluate Tully's Surface Hopping Probability
            rho = np.outer(c_coeffs, np.conj(c_coeffs))
            target_state = 1 - active_state
            if active_state == 0:
                g_prob = (2.0 * dt_sub * np.max([0.0, np.real(rho[1, 0] * D01_next)])) / np.real(rho[0, 0])
            else:
                g_prob = (2.0 * dt_sub * np.max([0.0, np.real(rho[0, 1] * (-D01_next))])) / np.real(rho[1, 1])
                
            # 3e. Stochastic Hop Test & 2D Momentum Rescaling
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

        # 3c. Apply Granucci and Persico Decoherence Correction (EDC)
        if E_kin > 0:
            non_active = 1 - active_state
            tau_deco = (hbar / np.abs(E[1] - E[0])) * (1.0 + C_edc / E_kin)
            
            c_coeffs[non_active] *= np.exp(-dt / tau_deco)
            if np.abs(c_coeffs[active_state]) > 1e-10:
                phase_factor = c_coeffs[active_state] / np.abs(c_coeffs[active_state])
                c_coeffs[active_state] = phase_factor * np.sqrt(1.0 - np.abs(c_coeffs[non_active])**2)
            else:
                c_coeffs[active_state] = np.sqrt(1.0 - np.abs(c_coeffs[non_active])**2)

print("Ensemble simulation completed.")

# ==============================================================================
# 4. COMPUTE ENSEMBLE AVERAGES
# ==============================================================================
avg_pos = accum_pos / num_trajectories
avg_mom = accum_mom / num_trajectories
avg_quantum_pop = accum_quantum_pop / num_trajectories
avg_active_pop = accum_active_pop / num_trajectories
avg_energy = accum_energy / num_trajectories

# ==============================================================================
# 5. PRINT SUMMARY REPORT TO TERMINAL
# ==============================================================================
print("\n" + "="*50)
print("             FINAL ENSEMBLE AVERAGES              ")
print("="*50)
print(f"Total Trajectories Simulated : {num_trajectories}")
print(f"Total Simulation Time        : {tmax} fs")
print(f"Average Final Position (X,Y) : ({avg_pos[-1, 0]:.4f}, {avg_pos[-1, 1]:.4f})")
print(f"Average Final Momentum (Px,Py): ({avg_mom[-1, 0]:.4f}, {avg_mom[-1, 1]:.4f})")
print(f"Final Average Quantum Pops   : S0 = {avg_quantum_pop[-1, 0]:.4f}, S1 = {avg_quantum_pop[-1, 1]:.4f}")
print(f"Final Active State Fractions : S0 = {avg_active_pop[-1, 0]:.4f}, S1 = {avg_active_pop[-1, 1]:.4f}")
print(f"Average Final Total Energy   : {avg_energy[-1, 2]:.6f} a.u.")
print("="*50)

# ==============================================================================
# 6. DATA EXPORT FOR GNUPLOT
# ==============================================================================
data_filename = "fssh_ensemble_averages.dat"
print(f"\nSaving averaged trajectory logs to '{data_filename}'...")

with open(data_filename, 'w') as f_out:
    f_out.write("# 1:Time\t2:Avg_X\t3:Avg_Y\t4:Avg_Px\t5:Avg_Py\t6:Avg_Quant_S0\t7:Avg_Quant_S1\t8:Fraction_Active_S0\t9:Fraction_Active_S1\t10:Avg_Ekin\t11:Avg_Epot\t12:Avg_Etot\n")
    for s in range(n_steps):
        t = s * dt
        f_out.write(f"{t:.4f}\t"
                    f"{avg_pos[s][0]:.6f}\t{avg_pos[s][1]:.6f}\t"
                    f"{avg_mom[s][0]:.6f}\t{avg_mom[s][1]:.6f}\t"
                    f"{avg_quantum_pop[s][0]:.6f}\t{avg_quantum_pop[s][1]:.6f}\t"
                    f"{avg_active_pop[s][0]:.6f}\t{avg_active_pop[s][1]:.6f}\t"
                    f"{avg_energy[s][0]:.6f}\t{avg_energy[s][1]:.6f}\t"
                    f"{avg_energy[s][2]:.6f}\n")

# ==============================================================================
# 7. VISUALIZE AND EXPORT PLOT PANEL
# ==============================================================================
history_time = np.arange(n_steps) * dt
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Average Trajectory Spatial Path
axes[0,0].plot(avg_pos[:,0], avg_pos[:,1], label="Ensemble Mean Path", color='black', lw=2)
axes[0,0].scatter(avg_pos[0,0], avg_pos[0,1], color='green', marker='o', s=100, label="Mean Start")
axes[0,0].scatter(avg_pos[-1,0], avg_pos[-1,1], color='red', marker='x', s=100, label="Mean End")
axes[0,0].set_title("Average 2D Coordinate Evolution")
axes[0,0].set_xlabel("Mean X coordinate"); axes[0,0].set_ylabel("Mean Y coordinate")
axes[0,0].grid(True, linestyle=":"); axes[0,0].legend()

# Plot 2: Population Balance (Quantum Amplitudes vs Hopping Proportions)
axes[0,1].plot(history_time, avg_quantum_pop[:,0], label="Quantum Pop S0", color='blue', lw=2)
axes[0,1].plot(history_time, avg_quantum_pop[:,1], label="Quantum Pop S1", color='orange', lw=2)
axes[0,1].plot(history_time, avg_active_pop[:,0], label="Active Fraction S0", color='navy', linestyle='--')
axes[0,1].plot(history_time, avg_active_pop[:,1], label="Active Fraction S1", color='darkorange', linestyle='--')
axes[0,1].set_title("Ensemble Populations: Quantum vs Active Fractions")
axes[0,1].set_xlabel("Time (fs)"); axes[0,1].set_ylabel("Population / Fraction")
axes[0,1].grid(True, linestyle=":"); axes[0,1].legend()

# Plot 3: Averaged Energy Channels
axes[1,0].plot(history_time, avg_energy[:,0], label="Avg Kinetic Energy", color='purple', linestyle='--')
axes[1,0].plot(history_time, avg_energy[:,1], label="Avg Potential Energy", color='teal', linestyle='--')
axes[1,0].plot(history_time, avg_energy[:,2], label="Avg Total Energy", color='black', lw=2.5)
axes[1,0].set_title("Mean Energy Profile Tracking")
axes[1,0].set_xlabel("Time (fs)"); axes[1,0].set_ylabel("Energy (a.u.)")
axes[1,0].grid(True, linestyle=":"); axes[1,0].legend()

# Plot 4: Active Fraction Composition Map
axes[1,1].stackplot(history_time, avg_active_pop[:,0], avg_active_pop[:,1], labels=['State S0', 'State S1'], colors=['skyblue', 'peachpuff'], alpha=0.7)
axes[1,1].set_title("Active Trajectory State Population Share")
axes[1,1].set_xlabel("Time (fs)"); axes[1,1].set_ylabel("Ensemble Share")
axes[1,1].set_ylim(0, 1.0)
axes[1,1].grid(True, linestyle=":")
axes[1,1].legend(loc='lower left')

plt.tight_layout()
image_filename = "fssh_ensemble_averages.png"
plt.savefig(image_filename, dpi=300)
plt.close(fig)
print(f"Ensemble plots successfully exported to '{image_filename}'.")
