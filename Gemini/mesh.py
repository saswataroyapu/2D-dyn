import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 1. Parameters
N = 256           # Grid resolution
L = 10.0         # Box size
dt = 0.02        # Time step
m = 1.0          
hbar = 1.0       
a_coeff, b_coeff = 1.0, 1.5

# 2. Grid and Potential
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k)

V = a_coeff * ((X**2 - b_coeff**2)**2 + (Y**2 - b_coeff**2)**2)
V_display = np.clip(V, 0, 10)  # Clip for visual clarity

# 3. Evolution Operators
op_V = np.exp(-1j * V * (dt/2) / hbar) 
op_K = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))


# Initial Gaussian: Position (x0, y0) and Momentum (px0, py0)
x0, y0 = -2, -2
px0, py0 = 0, 0
sigma = .5



# 4. Initial Wavepacket

psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2)) * \
      np.exp(1j * (px0*X + py0*Y) / hbar)


#psi = np.exp(-((X+1.5)**2 + (Y+1.5)**2) / (2 * 0.5**2)) * np.exp(1j * (2.0*X + 1.5*Y))
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

# 5. Animation Setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(155, projection='3d')

def update(frame):
    global psi
    # Evolve 5 steps per frame
    for _ in range(5):
        psi *= op_V
        psi_k = np.fft.fft2(psi)
        psi_k *= op_K
        psi = np.fft.ifft2(psi_k)
        psi *= op_V
    
    prob = np.abs(psi)**2
    ax.clear()
    
    # Static Potential Mesh (Light Gray)
    ax.plot_wireframe(X, Y, V_display, color='black', alpha=0.3, linewidths=0.75)
    
    # Dynamic Wavepacket Mesh (Colored)
    # Rstride/Cstride controls the density of the mesh lines
    mesh = ax.plot_wireframe(X, Y, prob * 25, color='teal', rstride=2, cstride=2, linewidths=1)
    
    ax.set_zlim(0, 10)
    ax.view_init(elev=35, azim=45)
    ax.set_title(f"3D Mesh Dynamics | Frame: {frame}")
    ax.set_axis_off() # Optional: hides the grid box for a cleaner "floating" mesh look
    return mesh,

print("Creating 3D mesh animation...")
ani = FuncAnimation(fig, update, frames=100, interval=50)
ani.save('3d_quantum_mesh.gif', writer='pillow')
plt.show()
