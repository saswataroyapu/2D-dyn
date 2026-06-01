#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


#
def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

omega1 = 7.743E-3
omega2 = 6.680E-3
a = 31.05

def create_lvc_pes(vx, vy):
    c = 8.092E-5
    sigma = 0
    potentialGS = np.zeros((len(vx),len(vy)))
    potentialEX = np.zeros((len(vx),len(vy)))
    Coupling = np.zeros((len(vx),len(vy)))
    for i in range(len(vx)):
        for j in range(len(vy)):
            x = vx[i]
            y = vy[j]
            potentialGS[i][j]  = 0.5*(omega1*omega1*(x +a/2)*(x+sigma/2) + omega2*omega2*y*y + sigma)
            potentialEX[i][j] =  0.5*(omega1*omega1*(x -a/2)*(x-sigma/2) + omega2*omega2*y*y - sigma)
            H12 = 2*c*y
            Coupling[i][j] = H12
    return potentialGS, potentialEX, Coupling




def create_tanh_pes(vx, vy):
    A = 0.05
    k = 0.5
    C = 0.01
    D = 1E-2
    rcrit = 1.0
    sigma  = 10.0
    tolerance = 1E-5
    potentialGS = np.zeros((len(vx),len(vy)))
    potentialEX = np.zeros((len(vx),len(vy)))
    Coupling = np.zeros((len(vx),len(vy)))

    for i in range(len(vx)):
        for j in range(len(vy)):
            H12 = 0.0
            x = vx[i]
            y = vy[j]
            r = np.sqrt(x*x + y*y)
            prefac = A*np.tanh(k*r)/r
            potentialGS[i][j]  = prefac*x*y
            potentialEX[i][j] = -prefac*x*y
            H12 = C*r*r*np.exp(-r/sigma)
            Coupling[i][j] = H12
    return potentialGS, potentialEX, Coupling
            
# Parameters
N = 256
L = 160.0
dt = 0.01
num_frames = 10 # Reduced for speed in this environment
steps_per_frame = 200
mass, hbar = 1.0, 1.0 

#Initial position and momentum
x0, y0 = -a/2, 0
px0, py0 = 0, 0
sigma_init_x = 0.5*omega1
sigma_init_y = 0.5*omega2

# Setup Folders
folders = ["3D_Mesh_frames", "Heatmap_frames"]
for folder in folders:
    if not os.path.exists(folder): os.makedirs(folder)

# Grid
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y, indexing='ij')
V_GS, V_EX, W = create_lvc_pes(x, y)

fwriteS0 = open(f"S0diabat","w")
fwriteS1 = open(f"S1diabat","w")
for kk in range(len(x)):
    for jj in range(len(y)):
            fwriteS0.write(f"{x[kk]}  {y[jj]}  {V_GS[kk][jj]}\n")
            fwriteS1.write(f"{x[kk]}  {y[jj]}  {V_EX[kk][jj]}\n")
    fwriteS0.write("\n")
    fwriteS1.write("\n")
fwriteS1.close()
fwriteS0.close()
# Construct H_pot (N, N, 2, 2)
H_pot = np.zeros((N, N, 2, 2))
H_pot[:,:,0,0] = V_GS
H_pot[:,:,1,1] = V_EX
H_pot[:,:,0,1] = H_pot[:,:,1,0] = W

# Matrix exponentiation for U_V: exp(-1j * H_pot * dt / hbar)
# For a 2x2 matrix, we can use the analytic form
avg = (V_GS + V_EX) / 2
diff = (V_GS - V_EX) / 2
R = np.sqrt(diff**2 + W**2)
phase = np.exp(-1j * avg * dt / hbar)

U_V = np.zeros((N, N, 2, 2), dtype=complex)
U_V[:,:,0,0] = phase * (np.cos(R*dt/hbar) - 1j*diff*np.sin(R*dt/hbar)/R)
U_V[:,:,1,1] = phase * (np.cos(R*dt/hbar) + 1j*diff*np.sin(R*dt/hbar)/R)
U_V[:,:,0,1] = phase * (-1j*W*np.sin(R*dt/hbar)/R)
U_V[:,:,1,0] = U_V[:,:,0,1]

# Kinetic Operator
k_freq = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k_freq, k_freq, indexing='ij')
K_val = (KX**2 + KY**2) / (2 * mass)
op_K_half = np.exp(-1j * hbar * K_val * (dt/2))



E_vals, E_vecs = np.linalg.eigh(H_pot)
V_low = E_vals[:,:,0]
V_high = E_vals[:,:,1]
fwriteS0 = open(f"S0","w")
fwriteS1 = open(f"S1","w")
for kk in range(len(x)):
    for jj in range(len(y)):
            fwriteS0.write(f"{x[kk]}  {y[jj]}  {V_low[kk][jj]}\n")
            fwriteS1.write(f"{x[kk]}  {y[jj]}  {V_high[kk][jj]}\n")
    fwriteS0.write("\n")
    fwriteS1.write("\n")
fwriteS1.close()
fwriteS0.close()



# ==============================================================================
# WAVEFUNCTION INITIALIZATION (PURE ADIABATIC S0)
# ==============================================================================
# 1. Create the spatial Gaussian envelope

temp1 = (X-x0)**2/(2*sigma_init_x**2)
temp2 = (Y-y0)**2/(2*sigma_init_y**2)
psi_gauss = np.exp(-(temp1 + temp2)) * np.exp(1j * (px0*X + py0*Y) / hbar)
psi_gauss /= np.sqrt(np.sum(np.abs(psi_gauss)**2) * dx**2)

# 2. Define the pure adiabatic vector: [psi_gauss, 0]^T at every point (N, N, 2, 1)
psi_a_init = np.zeros((N, N, 2, 1), dtype=complex)
psi_a_init[:, :, 1, 0] = psi_gauss  # All initial population explicitly on S0 (index 0)

# 3. Transform to Diabatic basis: Psi_diabatic = C * Psi_adiabatic
# E_vecs shape from linalg.eigh is (N, N, 2, 2) where columns are the eigenvectors
psi_d_init = np.matmul(E_vecs, psi_a_init)

# 4. Squeeze back and transpose to match your (2, N, N) diabatic tracking array
psi_diabatic = psi_d_init.squeeze(-1).transpose(2, 0, 1)

# ==============================================================================

# Wavefunction matrix (2, N, N)
#psi_diabatic = np.zeros((2, N, N), dtype=complex)
#psi_gauss = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma_init**2)) * np.exp(1j * (px0*X + py0*Y) / hbar)
#psi_gauss /= np.sqrt(np.sum(np.abs(psi_gauss)**2) * dx**2)
#psi_diabatic[0, :, :] = psi_gauss

# Quadrant masks for box populations
mask_w1 = (X < 0) & (Y < 0); mask_w2 = (X < 0) & (X >= 0)
mask_w3 = (X >= 0) & (Y < 0); mask_w4 = (X >= 0) & (X >= 0)
# Simulation




f2 = open("popstate","w")
with open('coupled_well_populations.txt', 'w') as f:
    f.write("#Time\tGS_W1\tGS_W2\tGS_W3\tGS_W4\tEX_W1\tEX_W2\tEX_W3\tEX_W4\n")
    for i in range(num_frames):
        for _ in range(steps_per_frame):
            # Kinetic step (diabatic is fine as K is diagonal across levels here)
            psi_diabatic[0] = np.fft.ifft2(np.fft.fft2(psi_diabatic[0]) * op_K_half)
            psi_diabatic[1] = np.fft.ifft2(np.fft.fft2(psi_diabatic[1]) * op_K_half)
            
            # Potential step: Matrix Multiplication
            # Reshape psi_diabatic to (N, N, 2, 1)
            psi_vec = psi_diabatic.transpose(1, 2, 0)[..., np.newaxis]
            psi_vec = np.matmul(U_V, psi_vec)
            psi_diabatic = psi_vec.squeeze(-1).transpose(2, 0, 1)
            
            # Kinetic step
            psi_diabatic[0] = np.fft.ifft2(np.fft.fft2(psi_diabatic[0]) * op_K_half)
            psi_diabatic[1] = np.fft.ifft2(np.fft.fft2(psi_diabatic[1]) * op_K_half)

# WITH THIS OPTIMIZED VECTORIZED LINE:
#            var1 = (np.abs(X) - L/2) + (np.abs(Y) - L/2)
#            mask = 0.5 - np.tanh(sharpness * var1) / 2.0

#            psi_diabatic[0] *= mask
#            psi_diabatic[1] *= mask


#            for xindex in range(N):
#                for yindex in range(N):
#                    xval = x[xindex]
#                    yval = y[yindex]
#                    var1 = (abs(xval) - L/2) + ((abs(yval) - L/2))
#                    psi_diabatic[0,xindex,yindex] = psi_diabatic[0,xindex,yindex]* (0.5 - np.tanh(sharpness*var1)/2.0)
#                    psi_diabatic[1,xindex,yindex] = psi_diabatic[1,xindex,yindex]* (0.5 - np.tanh(sharpness*var1)/2.0)
    
        # Diagonalize H_pot to get adiabatic states
#        E_vals, E_vecs = np.linalg.eigh(H_pot)
        V_low = E_vals[:,:,0]
        V_high = E_vals[:,:,1]
        
        print("Frame")
        print(i)
    
        # Transform to Adiabatic: Psi_adiab = C^H * Psi_diab
        psi_d_vec = psi_diabatic.transpose(1, 2, 0)[..., np.newaxis]
        # np.linalg.eigh returns eigenvectors in columns: H = C E C^H. 
        # C_H is the conjugate transpose of E_vecs
        C_H = E_vecs.transpose(0, 1, 3, 2).conj()
        psi_a_vec = np.matmul(C_H, psi_d_vec)
        psi_adiabatic = psi_a_vec.squeeze(-1).transpose(2, 0, 1)
    
        t_curr = i * steps_per_frame * dt
        dens_low = np.abs(psi_adiabatic[0])**2
        dens_high = np.abs(psi_adiabatic[1])**2
        # Save box populations
        pop_GS = [np.sum(dens_low[m]*dx**2) for m in [mask_w1, mask_w2, mask_w3, mask_w4]]
        pop_EX = [np.sum(dens_high[m]*dx**2) for m in [mask_w1, mask_w2, mask_w3, mask_w4]]
        f.write(f"{t_curr:.4f}\t" + "\t".join([f"{p:.6f}" for p in pop_GS+pop_EX]) + "\n")


        # Plot 3D Mesh
        fig3d = plt.figure(figsize=(14, 12))
        ax1 = fig3d.add_subplot(121, projection='3d')
        ax1.plot_wireframe(X, Y, V_low, color='black', alpha=0.1, rstride=4, cstride=4)
        ax1.plot_surface(X, Y, dens_low*5 + V_low, cmap='viridis', alpha=0.8)
        ax1.set_title(f"Lower Adiabatic (S0) | T={i*dt*steps_per_frame:.2f}"); ax1.set_zlim(-0.1, 0.6)
        
        ax2 = fig3d.add_subplot(122, projection='3d')
        ax2.plot_wireframe(X, Y, V_high, color='black', alpha=0.1, rstride=4, cstride=4)
        ax2.plot_surface(X, Y, dens_high*5 + V_high, cmap='magma', alpha=0.8)
        ax2.set_title(f"Upper Adiabatic (S1) | T={i*dt*steps_per_frame:.2f}"); ax2.set_zlim(-0.1, 0.6)
        




        fig3d.savefig(f"3D_Mesh_frames/mesh_{i:03d}.png")
        
        plt.close(fig3d)
    
        # Plot Heatmap (requested)
        figHM, (h1, h2) = plt.subplots(1, 2, figsize=(12, 10))
        # S0 Heatmap
        im1 = h1.contourf(X, Y, dens_low, levels=40, cmap='viridis')
        h1.contour(X, Y, V_low, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        h1.set_title("S0 Density (Adiabatic)")
        plt.colorbar(im1, ax=h1)
        
        # S1 Heatmap
        im2 = h2.contourf(X, Y, dens_high, levels=40, cmap='magma')
        h2.contour(X, Y, V_high, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        h2.set_title("S1 Density (Adiabatic)")
        plt.colorbar(im2, ax=h2)
       
        for ax in [h1, h2]:
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_aspect('equal')
        figHM.savefig(f"Heatmap_frames/heat_{i:03d}.png")
        plt.close(figHM)

print("Frames generated.")
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
f2.close()
