import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Setup Simulation Parameters
# ==========================================
N = 256          # Grid points (power of 2)
L = 10.0         # Box size
dt = 0.001        # Time step
num_frames = 500 # Total PNGs to generate
steps_per_frame = 5
m = 5.0          
hbar = 1.0       

# Potential Parameters: V = a * ((X^2 - b^2)^2 + (Y^2 - b^2)^2)
a_coeff = 1.0
b_coeff = 1.5

# Initial Gaussian: Position (x0, y0) and Momentum (px0, py0)
x0, y0 = -2, -2
px0, py0 = 0, 0
sigma = 1

# Create directory for PNG sequence
output_dir = "simulation_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. Grid and Operator Setup
# ==========================================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

# Momentum grid
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k)

# Potential and Evolution Operators
V = a_coeff * ((X**2 - b_coeff**2)**2 + (Y**2 - b_coeff**2)**2)
op_V = np.exp(-1j * V * (dt/2) / hbar) 
op_K = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))

# Initialize and Normalize Wavepacket
psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2)) * \
      np.exp(1j * (px0*X + py0*Y) / hbar)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

# ==========================================
# 3. Execution and File Writing
# ==========================================
time_data = []
trajectory = []

print(f"Starting simulation. Saving frames to /{output_dir}...")

fig, ax = plt.subplots(figsize=(8, 7))

for i in range(num_frames):
    # Time evolution
    for _ in range(steps_per_frame):
        psi *= op_V
        psi_k = np.fft.fft2(psi)
        psi_k *= op_K
        psi = np.fft.ifft2(psi_k)
        psi *= op_V
    
    current_t = i * steps_per_frame * dt
    
    # Calculate expectation values <x> and <y>
    prob = np.abs(psi)**2
    total_p = np.sum(prob) * dx**2
    ex = np.sum(prob * X) * dx**2 / total_p
    ey = np.sum(prob * Y) * dx**2 / total_p
    
    time_data.append(current_t)
    trajectory.append([ex, ey])
    
    # Visualization and PNG export
    ax.clear()
    ax.contourf(X, Y, prob, levels=40, cmap='magma')
    ax.contour(X, Y, V, levels=10, colors='white', alpha=0.1, linestyles='solid')
    ax.set_title(f"Quantum Dynamics | Time: {current_t:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Save frame with leading zeros for ffmpeg sorting
    plt.savefig(f"{output_dir}/frame_{i:03d}.png")
    
    if i % 10 == 0:
        print(f"Frame {i}/{num_frames} completed.")

plt.close()

# Save trajectory data to text file
data_to_save = np.column_stack((time_data, trajectory))
np.savetxt('trajectory_data.txt', data_to_save, 
           header="Time\tExpectation_X\tExpectation_Y", 
           fmt='%.6f', delimiter='\t')

print("Simulation finished.")
print(f"To create video, use: ffmpeg -i {output_dir}/frame_%03d.png -pix_fmt yuv420p output.mp4")
