import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------
# Grid
# ----------------------
Nx, Ny = 256, 256
Lx, Ly = 10.0, 10.0
dx, dy = Lx/Nx, Ly/Ny

x = np.linspace(-Lx/2, Lx/2 - dx, Nx)
y = np.linspace(-Ly/2, Ly/2 - dy, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# ----------------------
# Momentum grid
# ----------------------
kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')

# ----------------------
# Physical parameters
# ----------------------
m = 5.0
omega_x = 1.0
omega_y = 1.0
dt = 0.01

# ----------------------
# Potential
# ----------------------
V = 0.5 * m * (omega_x**2 * X**2 + omega_y**2 * Y**2)

# ----------------------
# Initial wavepacket (complex!)
# ----------------------
x0, y0 = -2.0, 0.0
px0, py0 = 10.0, 0.0
sigma = 0.2

psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)).astype(np.complex128)
psi *= np.exp(1j * (px0*X + py0*Y))

# Normalize
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)

# ----------------------
# Propagators
# ----------------------
expV_half = np.exp(-1j * V * dt / 2)
T = (KX**2 + KY**2) / (2*m)
expT = np.exp(-1j * T * dt)

def step(psi):
    psi = expV_half * psi
    psi_k = np.fft.fftn(psi)
    psi_k *= expT
    psi = np.fft.ifftn(psi_k)
    psi = expV_half * psi
    return psi

# ----------------------
# Output file
# ----------------------
outfile = open("wavepacket_output.dat", "w")

outfile.write("# Columns:\n")
outfile.write("# Re(psi_x) Im(psi_x) Re(psi_y) Im(psi_y) rho_x rho_y <x> <y> <px> <py>\n")

write_every = 20

# ----------------------
# 3D plotting setup
# ----------------------
ds = 2
X_plot = X[::ds, ::ds]
Y_plot = Y[::ds, ::ds]
V_plot = V[::ds, ::ds]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)

# ----------------------
# Time loop
# ----------------------
nsteps = 4000



zmax = np.max(np.abs(psi)**2) * 1.5 + 0.2*np.max(V_plot)

for step_idx in range(nsteps):
    psi = step(psi)

    # Density
    rho = np.abs(psi)**2

    # Projections
    psi_x = np.sum(psi, axis=1) * dy
    psi_y = np.sum(psi, axis=0) * dx

    rho_x = np.sum(rho, axis=1) * dy
    rho_y = np.sum(rho, axis=0) * dx

    # Expectation values
    x_exp = np.sum(rho * X) * dx * dy
    y_exp = np.sum(rho * Y) * dx * dy

    # Momentum expectation (spectral)
    psi_k = np.fft.fftn(psi)
    px_exp = np.sum(np.conj(psi_k) * KX * psi_k).real * (dx*dy)/(Nx*Ny)
    py_exp = np.sum(np.conj(psi_k) * KY * psi_k).real * (dx*dy)/(Nx*Ny)

    # ----------------------
    # Write to file
    # ----------------------
    if step_idx % write_every == 0:
        outfile.write(f"# t = {step_idx*dt:.4f}\n")

        for i in range(min(Nx, Ny)):
            outfile.write(f"{psi_x[i].real:.6e} {psi_x[i].imag:.6e} "
                          f"{psi_y[i].real:.6e} {psi_y[i].imag:.6e} "
                          f"{rho_x[i]:.6e} {rho_y[i]:.6e} "
                          f"{x_exp:.6e} {y_exp:.6e} "
                          f"{px_exp:.6e} {py_exp:.6e}\n")
        outfile.write("\n")

    # ----------------------
    # 3D visualization
    # ----------------------
    if step_idx % 20 == 0:
        ax.clear()

        Z = rho[::ds, ::ds]
        Z_plot = Z + 0.2*np.max(V_plot)
        zmax = np.max(np.abs(psi)**2) * 1.5 + 0.2*np.max(V_plot)

        ax.set_autoscale_on(False)
        ax.set_box_aspect((1, 1, 1))

        # Bowl
        ax.plot_surface(X_plot, Y_plot, V_plot, alpha=0.01, linewidth=0)

        # Wavepacket
        ax.plot_surface(X_plot, Y_plot, Z_plot, linewidth=0.1)

        # Expectation marker
        ax.scatter(x_exp, y_exp, np.max(Z_plot), s=40)

        ax.set_xlim(-Lx/2, Lx/2)
        ax.set_ylim(-Ly/2, Ly/2)
        ax.set_zlim(22, 28)
        #ax.set_zlim(0, zmax)

        ax.set_box_aspect((Lx, Ly, zmax))
        ax.set_title(f"t = {step_idx*dt:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.pause(0.005)

outfile.close()
plt.show()
