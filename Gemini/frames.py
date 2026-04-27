import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# 1. Setup Parameters
# ==========================================
N = 80           # Resolution (good balance for mesh looks)
L = 10.0         
dt = 0.02        
num_frames = 150 # Number of PNGs to save
steps_per_frame = 5
m, hbar = 1.0, 1.0

# Potential: a*((x^2-b^2)^2 + (y^2-b^2)^2)
a_coeff, b_coeff = 0.25, 2

# Create directory
output_dir = "mesh_frames_3d"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. Grids and Operators
# ==========================================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k)

V = a_coeff * ((X**2 - b_coeff**2)**2 + (Y**2 - b_coeff**2)**2)
V_display = np.clip(V, 0, 8) # Clipping helps "see" into the wells

op_V = np.exp(-1j * V * (dt/2) / hbar)
op_K = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))

# Initial Wavepacket
#psi = np.exp(-((X+1.5)**2 + (Y+1.5)**2) / (2 * 0.5**2)) * np.exp(1j * (2.5*X + 1.5*Y))


# Initial Gaussian: Position (x0, y0) and Momentum (px0, py0)
x0, y0 = -2, -2
px0, py0 = 0.5, 0
sigma = 1



# 4. Initial Wavepacket

psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2)) * \
      np.exp(1j * (px0*X + py0*Y) / hbar)







psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

# ==========================================
# 3. Frame Generation Loop
# ==========================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Scale for the density height
scale = 20.0

print(f"Generating {num_frames} mesh frames in /{output_dir}...")

for i in range(num_frames):
    # Physics Evolution
    for _ in range(steps_per_frame):
        psi *= op_V
        psi_k = np.fft.fft2(psi)
        psi_k *= op_K
        psi = np.fft.ifft2(psi_k)
        psi *= op_V
    
    prob = np.abs(psi)**2
    
    # Rendering
    ax.clear()
    
    # 1. The Potential Mesh (Fixed)
    # Using a higher stride for the background potential to keep it subtle
    ax.plot_wireframe(X, Y, V_display, rstride=4, cstride=4, 
                      color='black', alpha=0.1, linewidths=0.5)
    
    # 2. The Wavepacket Mesh (Dynamic)
    # Teal color with a denser stride to highlight the quantum state
    ax.plot_wireframe(X, Y, prob * scale, rstride=2, cstride=2, 
                      color='#008080', alpha=0.8, linewidths=1.0)
    
    # Aesthetics
    ax.set_zlim(0, 10)
    ax.view_init(elev=35, azim=45)
    ax.set_title(f"Quantum Wavepacket Mesh | T = {i*steps_per_frame*dt:.2f}", fontsize=15)
    
    # Cleaning up the plot
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Save the frame
    plt.savefig(f"{output_dir}/mesh_{i:03d}.png", dpi=150)
    
    if i % 25 == 0:
        print(f"Progress: {i}/{num_frames} frames saved.")

plt.close()
print(f"\nDone! Use this ffmpeg command to build the movie:")
print(f"ffmpeg -framerate 30 -i {output_dir}/mesh_%03d.png -c:v libx264 -pix_fmt yuv420p mesh_movie.mp4")
