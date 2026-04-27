import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Simulation Parameters
N = 128          # Grid points (power of 2 for FFT)
L = 10.0         # Box size
dt = 0.01        # Time step
steps = 200      # Total steps
m = 1.0          # Mass
hbar = 1.0       # Reduced Planck's constant

# Constants for the Potential
a, b = 1.0, 1.5

# 2. Define Grids
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

# Momentum grid (for Split-Operator)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k)

# 3. Define the Potential V(X, Y)
V = a * ((X**2 - b**2)**2 + (Y**2 - b**2)**2)

# 4. Initial Gaussian Wavepacket
x0, y0 = -1.5, -1.5   # Start in the bottom-left well
px0, py0 = 2.0, 2.0   # Kick it toward the center
sigma = 0.4           # Width

psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2)) * \
      np.exp(1j * (px0*X + py0*Y) / hbar)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2) # Normalize

# 5. Operators for Split-Step
# Potential operator in position space
op_V = np.exp(-1j * V * dt / (2 * hbar)) 
# Kinetic operator in momentum space
op_K = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))

# 6. Time Evolution Loop
def evolve(psi, steps):
    for _ in range(steps):
        # Step 1: Half-step in position space
        psi *= op_V
        # Step 2: Full-step in momentum space
        psi_k = np.fft.fft2(psi)
        psi_k *= op_K
        psi = np.fft.ifft2(psi_k)
        # Step 3: Half-step in position space
        psi *= op_V
    return psi

# Evolve and Plot
psi_final = evolve(psi.copy(), steps)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("Initial Probability Density $|\psi|^2$")
plt.contourf(X, Y, np.abs(psi)**2, 30, cmap='hot')
plt.contour(X, Y, V, levels=5, colors='white', alpha=0.3) # Overlay potential

plt.subplot(122)
plt.title(f"Final Probability Density (t = {steps*dt})")
plt.contourf(X, Y, np.abs(psi_final)**2, 30, cmap='hot')
plt.contour(X, Y, V, levels=5, colors='white', alpha=0.3)
plt.show()
