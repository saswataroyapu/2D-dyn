import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Simulation & Physics Parameters
# ==========================================
N = 256           # Grid resolution (optimized for speed/mesh clarity)
L = 20.0         # Box size (extended to see the wells clearly)
dt = 0.005        # Time step
num_frames = 400 
steps_per_frame = 5
m, hbar = 2.0, 1.0




# Data from User
GS_wells = [-0.01410, -0.01298, -0.01600, -0.01462]
GS_peak = 0.2135 # peaky
EX_wells = [0.17337, 0.17625, 0.16914, 0.17118]
EX_peak = 0.2358 # peaky








# User-provided potential parameters
a, b = 0.25, 2.0
alpha, beta, gamma = 0.2, 0.5, 0.3

# Well 3 (Global Minimum) and Well 2 (Target)
well_1 = np.array([2.05, -2.13])
well_2 = np.array([-2.09, 2.02])
well_4 = np.array([-1.9484, -1.9893])
well_3 = np.array([1.8980, 1.8498])


# Initial Wavepacket Setup
x0, y0 = well_4[0], well_4[1]
# Momentum kick vector pointing from Well 3 to Well 2
kick_dir = (well_2 - well_4) / np.linalg.norm(well_2 - well_4)
px0, py0 = kick_dir * 3  # Magnitude of 1.5
sigma = 0.8

# Setup Directories
os.makedirs("mesh_frames", exist_ok=True)
os.makedirs("contour_frames", exist_ok=True)

# ==========================================
# 2. Grid and Potential
# ==========================================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k)

# Asymmetric Potential Function
V_sym = a * ((X**2 - b**2)**2 + (Y**2 - b**2)**2)
V_bias = alpha * X + beta * Y + gamma * (X * Y)
V = V_sym + V_bias
V_display = np.clip(V, -5, 15) # For visualization stability




# Operators
op_V = np.exp(-1j * V * (dt/2) / hbar)
op_K = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))

# Initialize Wavepacket
psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2)) * \
      np.exp(1j * (px0*X + py0*Y) / hbar)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

# ==========================================
# 3. Main Loop: Real-time Plot & File Export
# ==========================================
plt.ion() # Interactive mode ON for real-time plotting
fig_realtime, ax_realtime = plt.subplots(figsize=(7, 6))

# Prepare 3D Figure (kept separate for file saving to avoid window conflicts)
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

print("Starting Simulation...")
for i in range(num_frames):
    # Split-Operator Time Evolution
    for _ in range(steps_per_frame):
        psi *= op_V
        psi_k = np.fft.fft2(psi)
        psi_k *= op_K
        psi = np.fft.ifft2(psi_k)
        psi *= op_V
    
    prob = np.abs(psi)**2
    t_curr = i * steps_per_frame * dt


    gammad = 2
# 2. Dissipation step (Damping)
        # This reduces the amplitude of the high-energy 'sloshing'
    psi *= np.exp(-gammad * dt)
        
        # 3. Re-normalize to keep the particle from disappearing
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**2)
    psi /= norm







    # --- Real-Time and Contour Save ---
    ax_realtime.clear()
    ct = ax_realtime.contourf(X, Y, prob, levels=30, cmap='magma')
    ax_realtime.contour(X, Y, V, levels=10, colors='white', alpha=0.2)
    ax_realtime.set_title(f"Heatmap | Time: {t_curr:.2f}")
    ax_realtime.set_xlabel("X (Bohr)")
    ax_realtime.set_ylabel("Y (Bohr)")
    
    # Save Heatmap Frame
    plt.figure(fig_realtime.number)
    plt.savefig(f"contour_frames/heat_{i:03d}.png")
    
    # Real-time Display Update
    plt.draw()
    plt.pause(0.01)

    # --- 3D Mesh Save ---
    ax_3d.clear()
    ax_3d.plot_wireframe(X, Y, V_display, rstride=4, cstride=4, color='black', alpha=0.3)
    ax_3d.plot_wireframe(X, Y, prob * 30, rstride=2, cstride=2, color='teal', linewidths=1)
    ax_3d.set_zlim(-2, 12)
    ax_3d.view_init(elev=35, azim=45)
    ax_3d.set_title(f"3D Mesh | Time: {t_curr:.2f}")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    
    plt.figure(fig_3d.number)
    plt.savefig(f"mesh_frames/mesh_{i:03d}.png")

plt.ioff() # Interactive mode OFF
plt.show()
print("Simulation Complete. Check 'mesh_frames' and 'contour_frames' folders.")
