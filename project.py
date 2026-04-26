import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------
# Grid
# ----------------------
Nx, Ny = 512, 512
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
m = 2.0
omega_x = 1.0
omega_y = 1.0
dt = 0.01
nsteps = 3000

# ----------------------
# Potential
# ----------------------
V = 0.5 * m * (omega_x**2 * X**2 + omega_y**2 * Y**2)
#V = 0.5 * m * (-omega_x**2 * X**2 + omega_y**2 * Y**2) + 0.25 * m *(omega_x**4 * (x**4 + y**4))

# ----------------------
# Initial wavepacket
# ----------------------
x0, y0 =  0.0, -1.0
px0, py0 =5.0, 1.0
sigma = 0.5

psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)).astype(np.complex128)
psi *= np.exp(1j * (px0*X + py0*Y))
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
# Output setup
# ----------------------
write_every = 20

# Expectation values file
with open("expectation_values.dat", "w") as f:
    f.write("# t  <x>  <y>  <px>  <py>\n")

# ----------------------
# 3D plot setup
# ----------------------
ds = 2
X_plot = X[::ds, ::ds]
Y_plot = Y[::ds, ::ds]
V_plot = V[::ds, ::ds]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

# Fix axes (prevents distortion)
zmax = np.max(np.abs(psi)**2) * 1.5 + 0.2*np.max(V_plot)
ax.set_xlim(-Lx/2, Lx/2)
ax.set_ylim(-Ly/2, Ly/2)
ax.set_zlim(0, zmax)
ax.set_box_aspect((Lx, Ly, zmax))
ax.set_autoscale_on(False)
ax.view_init(elev=30, azim=45)

# ----------------------
# Time evolution
# ----------------------
surf_wave = None
surf_bowl = None
dot = None

for step_idx in range(nsteps):

    psi = step(psi)
    rho = np.abs(psi)**2

    # ----------------------
    # Expectation values
    # ----------------------
    x_exp = np.sum(rho * X) * dx * dy
    y_exp = np.sum(rho * Y) * dx * dy

    psi_k = np.fft.fftn(psi)
    px_exp = np.sum(np.conj(psi_k) * KX * psi_k).real * (dx*dy)/(Nx*Ny)
    py_exp = np.sum(np.conj(psi_k) * KY * psi_k).real * (dx*dy)/(Nx*Ny)

    # ----------------------
    # Write output
    # ----------------------
    if step_idx % write_every == 0:
        t = step_idx * dt
        tag = f"{step_idx:04d}"

        # ---- Projections ----
        psi_x = np.sum(psi, axis=1) * dy
        psi_y = np.sum(psi, axis=0) * dx

        # psi_x file
        with open(f"psi_x_t{tag}.dat", "w") as fx:
            fx.write(f"# t = {t:.6f}\n")
            fx.write("# x  Re(psi_x)  Im(psi_x)\n")
            for i in range(Nx):
                fx.write(f"{x[i]:.6e} {psi_x[i].real:.6e} {psi_x[i].imag:.6e}\n")

        # psi_y file
        with open(f"psi_y_t{tag}.dat", "w") as fy:
            fy.write(f"# t = {t:.6f}\n")
            fy.write("# y  Re(psi_y)  Im(psi_y)\n")
            for i in range(Ny):
                fy.write(f"{y[i]:.6e} {psi_y[i].real:.6e} {psi_y[i].imag:.6e}\n")

        # expectation values (append)
        with open("expectation_values.dat", "a") as f:
            f.write(f"{t:.6e} {x_exp:.6e} {y_exp:.6e} {px_exp:.6e} {py_exp:.6e}\n")

    # ----------------------
    # Visualization (no axis reset!)
    # ----------------------

    vmin = 0.0
    vmax = np.max(np.abs(psi)**2) + 0.2*np.max(V_plot)

    if step_idx % 20 == 0:

        # Remove old surfaces
        if surf_wave is not None:
            surf_wave.remove()
        if surf_bowl is not None:
            surf_bowl.remove()
        if dot is not None:
            dot.remove()


        ax.set_box_aspect((2, 2, 0.4))

        Z = rho[::ds, ::ds] + 0.2*np.max(V_plot)


        
#        surf_bowl = ax.plot_surface(
#            X_plot, Y_plot, V_plot,
#            vmin=vmin, vmax=vmax,
#            cmap='blue',alpha-0.1,   # or any colormap you like
#            linewidth=0
#        )




        surf_bowl = ax.plot_surface(X_plot, Y_plot, V_plot,
                                   alpha=0.2, linewidth=0)






#        surf_wave = ax.plot_surface(X_plot, Y_plot, Z,
#                                   linewidth=0)

        surf_wave = ax.plot_surface(
            X_plot, Y_plot, Z,
            vmin=vmin, vmax=vmax,
            cmap='magma',   # or any colormap you like
            linewidth=0
        )

        dot = ax.scatter(x_exp, y_exp, np.max(Z), s=40)

        ax.set_title(f"t = {step_idx*dt:.2f}")
        plt.pause(0.0005)

plt.show()
