#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def create_nonadiabatic_pes(vx, vy):
    pos = np.array([[-1,-1],[-1,1],[1,-1],[1,1],[0,0]])
    signgs = np.array([-1,-1,-1,-1,1])
    signex = np.array([-0.5,-0.5,-0.5,-0.5,-1])
    sigma = 0.5 
    potentialGS = np.zeros((len(vx),len(vy)))
    potentialEX = np.zeros((len(vx),len(vy)))
    Coupling = np.zeros((len(vx),len(vy)))

    lam = 0.1
    gamma = 0.1

    for i in range(len(vx)):
        for j in range(len(vy)):
            x = vx[i]
            y = vy[j]
            for k in range(len(signgs)):
                potentialGS[i,j] += signgs[k]* np.exp(-((x-pos[k][0])**2 + (y-pos[k][1])**2)/(2*sigma**2))
                potentialEX[i,j] += signex[k]* np.exp(-((x-pos[k][0])**2 + (y-pos[k][1])**2)/(2*sigma**2)) +0.4
            Coupling[i,j] = lam/np.sqrt(x**2 + y**2 + gamma**2)

    
    return potentialGS, potentialEX, Coupling





#def create_nonadiabatic_pes(X, Y):
#    # --- Parameters to match user energy levels ---
#    # Target S1 ~ 0.17, S0 ~ -0.01. CI at 0.08
#    E_mid = 0.08      # Energy of the Conical Intersection
#    Gap = 0.09        # Vertical half-gap to wells
#    A = 0.8           # Steepness of the Mexican Hat
#    R = 0.5           # Radius of the well ring
#    delta = 0.02      # Depth of the four wells on the ring
#    
#    # Coupling parameters
#    lam = 0.05        # Coupling strength (lambda)
#    gamma = 0.1       # Broadness of Lorentzian (gamma)
#    
#    # --- Geometry Calculations ---
#    R_sq = X**2 + Y**2
#    phi = np.arctan2(Y, X)
#    
#    # 1. Distorted Mexican Hat Base
#    V_hat = A * (R_sq - R**2)**2 + delta * (1 - np.cos(4 * phi))
#    
#    # 2. Diabatic Surfaces (V1 and V2)
#    # We set them up such that at the origin, V1 = V2 = E_mid
#    # Away from origin, they split to create the upper and lower surfaces
#    V1 = V_hat + E_mid + Gap * (R_sq / (R_sq + 0.1))
#    V2 = V_hat + E_mid - Gap * (R_sq / (R_sq + 0.1))
#    
#    # 3. Lorentzian Coupling (W)
#    W = lam / np.sqrt(R_sq + gamma**2)
#    
#    return V1, V2, W
#

# For your Adiabatic Check:
# E_excited = 0.5 * (V1 + V2) + np.sqrt(0.25 * (V1 - V2)**2 + W**2)
# E_ground  = 0.5 * (V1 + V2) - np.sqrt(0.25 * (V1 - V2)**2 + W**2)




# 1. Simulation & Physics Parameters
# ==========================================
N = 128           # Grid resolution
L = 4.0           # Box size (Bohr)
dt = 0.05         # Time step
num_frames = 500  # Total animation frames
steps_per_frame = 5
m, hbar = 2000.0, 1.0 

# Initial Wavepacket Parameters
x0, y0 = -0.5, 0.5   # Center of Well 2
px0, py0 = 30.0, -30.0 # Momentum kick (adjust to "aim" the packet)
sigma = 0.2

# Potential Energy Surface Data
GS_wells = [-0.01410, -0.01298, -0.01600, -0.01462]
GS_peak = 0.2135
EX_wells = [0.17337, 0.17625, 0.16914, 0.17118]
EX_peak = 0.2158
coupling_const = 0.025
epsilon = 0.05 

# Directory Setup
folders = ["3D_Mesh_frames", "Heatmap_frames"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ==========================================
# 2. Grid and Potential Construction
# ==========================================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
V_GS, V_EX, W = create_nonadiabatic_pes(x, y) 
X, Y = np.meshgrid(x, y)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k)

def get_V_surface(target_energies, peak_val):
    A, b = 2.0, 0.5
    V = A * ((X**2 - b**2)**2 + (Y**2 - b**2)**2)
    w1, w2, w3, w4 = target_energies
    alpha = (w4 + w3 - w2 - w1) / 2
    beta  = (w4 + w2 - w3 - w1) / 2
    gamma = (w4 + w1 - w2 - w3) / 2
    avg   = np.mean(target_energies)
    V += alpha*X + beta*Y + gamma*(X*Y/0.25) + (avg - 0.25)
    V += peak_val * np.exp(-(X**2 + (Y-0.5)**2)/0.1)
    V += peak_val * np.exp(-((X-0.5)**2 + Y**2)/0.1)
    return V

#V_GS = get_V_surface(GS_wells, GS_peak)
#V_EX = get_V_surface(EX_wells, EX_peak)
#W = coupling_const / (np.abs(V_EX - V_GS) + epsilon)


# ==========================================
# 3. Time Evolution Operators
# ==========================================
# Unitary Potential Mixing Matrix (2x2 Matrix Exponentiation)
avg = (V_GS + V_EX) / 2
diff = (V_GS - V_EX) / 2
R = np.sqrt(diff**2 + W**2)
phase = np.exp(-1j * avg * dt / hbar)
cos_R = np.cos(R * dt / hbar)
sin_R = np.sin(R * dt / hbar)

U11 = phase * (cos_R - 1j * diff * sin_R / R)
U22 = phase * (cos_R + 1j * diff * sin_R / R)
U12 = phase * (-1j * W * sin_R / R)

# Kinetic Operator (Diagonal in K-space)
op_K_half = np.exp(-1j * hbar * (KX**2 + KY**2) * (dt/2) / (2 * m))

# ==========================================
# 4. Initialize Wavepacket
# ==========================================
psi_init = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2)) * \
           np.exp(1j * (px0*X + py0*Y) / hbar)
psi_init /= np.sqrt(np.sum(np.abs(psi_init)**2) * dx**2)

psi_GS = np.zeros_like(psi_init, dtype=complex) 
psi_EX = psi_init.copy() 

# Quadrant masks for box populations
mask_w1 = (X < 0) & (Y < 0); mask_w2 = (X < 0) & (Y >= 0)
mask_w3 = (X >= 0) & (Y < 0); mask_w4 = (X >= 0) & (Y >= 0)

# ==========================================
# 5. Dynamics Loop
# ==========================================
print("Running dynamics and saving frames...")
with open('coupled_well_populations.txt', 'w') as f:
    f.write("#Time\tGS_W1\tGS_W2\tGS_W3\tGS_W4\tEX_W1\tEX_W2\tEX_W3\tEX_W4\n")

    for i in range(num_frames):
        for _ in range(steps_per_frame):
            # Split-operator steps
            psi_GS = np.fft.ifft2(np.fft.fft2(psi_GS) * op_K_half)
            psi_EX = np.fft.ifft2(np.fft.fft2(psi_EX) * op_K_half)
            
            new_GS = U11 * psi_GS + U12 * psi_EX
            new_EX = U12 * psi_GS + U22 * psi_EX
            psi_GS, psi_EX = new_GS, new_EX
            
            psi_GS = np.fft.ifft2(np.fft.fft2(psi_GS) * op_K_half)
            psi_EX = np.fft.ifft2(np.fft.fft2(psi_EX) * op_K_half)

        # Densities and Timing
        t_curr = i * steps_per_frame * dt
        dens_GS = np.abs(psi_GS)**2
        dens_EX = np.abs(psi_EX)**2
        
        # Save box populations
        pop_GS = [np.sum(dens_GS[m]*dx**2) for m in [mask_w1, mask_w2, mask_w3, mask_w4]]
        pop_EX = [np.sum(dens_EX[m]*dx**2) for m in [mask_w1, mask_w2, mask_w3, mask_w4]]
        f.write(f"{t_curr:.4f}\t" + "\t".join([f"{p:.6f}" for p in pop_GS+pop_EX]) + "\n")

        # --- FIG A: 3D Mesh Plot ---
        fig3d = plt.figure(figsize=(14, 6))
        ax1 = fig3d.add_subplot(121, projection='3d')
        ax1.plot_wireframe(X, Y, V_GS, color='black', alpha=0.1, rstride=8, cstride=8)
        ax1.plot_wireframe(X, Y, dens_GS*3, color='blue', linewidth=0.5)
        ax1.set_title(f"S0 (Ground) 3D | T={t_curr:.2f}"); ax1.set_zlim(-0.1, 1.2)
        
        ax2 = fig3d.add_subplot(122, projection='3d')
        ax2.plot_wireframe(X, Y, V_EX, color='black', alpha=0.1, rstride=8, cstride=8)
        ax2.plot_wireframe(X, Y, dens_EX*3, color='red', linewidth=0.5)
        ax2.set_title(f"S1 (Excited) 3D | T={t_curr:.2f}"); ax2.set_zlim(-0.1, 1.2)
        
        for ax in [ax1, ax2]: ax.view_init(30, -45)
        fig3d.savefig(f"3D_Mesh_frames/mesh_{i:03d}.png")
        plt.close(fig3d)

        # --- FIG B: 2D Heatmaps ---
        figHM, (h1, h2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap S0
        h1.contourf(X, Y, dens_GS, levels=40, cmap='viridis')
        h1.contour(X, Y, V_GS, levels=8, colors='white', alpha=0.3, linewidths=0.5)
        h1.set_title("S0 Density (Heatmap)")
        
        # Heatmap S1
        h2.contourf(X, Y, dens_EX, levels=40, cmap='magma')
        h2.contour(X, Y, V_EX, levels=8, colors='white', alpha=0.3, linewidths=0.5)
        h2.set_title("S1 Density (Heatmap)")

        for ax in [h1, h2]:
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_aspect('equal')
        
        figHM.savefig(f"Heatmap_frames/heat_{i:03d}.png")
        plt.close(figHM)

print("Process Complete. Data in text file and images in folders.")
# --- Calculate NACV ---
# 1. Define the difference and mixing angle
dV = V_EX - V_GS
theta = 0.5 * np.arctan2(2 * W, dV)

# 2. Calculate the components of the NACV (gradient of theta)
# dx and dy are the grid spacings defined in your code
dx_val = x[1] - x[0]
dy_val = y[1] - y[0]

# np.gradient returns [d/dy, d/dx] for 2D arrays
grad_theta_y, grad_theta_x = np.gradient(theta, dy_val, dx_val)

# The NACV vector field
d12_x = grad_theta_x
d12_y = grad_theta_y

# 3. Calculate the Magnitude of NACV for visualization
nacv_magnitude = np.sqrt(d12_x**2 + d12_y**2)

# --- Visualization (Heatmap of NACV magnitude) ---
plt.figure(figsize=(7, 6))
plt.contourf(X, Y, nacv_magnitude, levels=50, cmap='inferno')
plt.colorbar(label='|d12| (Bohr^-1)')
plt.quiver(X[::8, ::8], Y[::8, ::8], d12_x[::8, ::8], d12_y[::8, ::8], color='white', alpha=0.5)
plt.title("Non-Adiabatic Coupling Vector (NACV) Field")
plt.xlabel("X (Bohr)"); plt.ylabel("Y (Bohr)")
plt.savefig("nacv_field.png")
