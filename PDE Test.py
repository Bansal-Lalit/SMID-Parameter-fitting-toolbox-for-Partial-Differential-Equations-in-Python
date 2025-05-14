import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force save to files
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import os

# Import PDEmodel from PartialDEs
from PartialDEs import PDEmodel
print("Successfully imported PDEmodel from PartialDEs.py")

# Create output directory
output_dir = 'tumor_results_with_curves'
os.makedirs(output_dir, exist_ok=True)
print(f"All plots will be saved in: {os.path.abspath(output_dir)}")

# Model definition
def TumourGrowth(z, t, grid, alpha, gamma):
    dN = 1e-3
    dM = 1e-3
    eta = 10.
    
    N, F, M = z.reshape(3, -1)
    N = N.reshape(grid.shape[:-1])
    F = F.reshape(grid.shape[:-1])
    M = M.reshape(grid.shape[:-1])
    
    dx = grid[1,0,0] - grid[0,0,0]
    dy = grid[0,1,1] - grid[0,0,1]
    
    # Initialize derivatives
    dNdx = np.empty_like(N)
    dNdy = np.empty_like(N)
    dNdxx = np.empty_like(N)
    dNdyy = np.empty_like(N)
    dMdxx = np.empty_like(M)
    dMdyy = np.empty_like(M)
    
    # First-order derivatives
    dFdx = np.gradient(F, axis=0)/dx
    dFdy = np.gradient(F, axis=1)/dy
    dNdx[0,:] = gamma*N[0,:]*dFdx[0,:]/dN
    dNdx[1:-1,:] = np.gradient(N, axis=0)[1:-1,:]/dx
    dNdx[-1,:] = gamma*N[-1,:]*dFdx[-1,:]/dN
    
    dNdy[:,0] = gamma*N[:,0]*dFdy[:,0]/dN
    dNdy[:,1:-1] = np.gradient(N, axis=1)[:,1:-1]/dy
    dNdy[:,-1] = gamma*N[:,-1]*dFdy[:,-1]/dN
    
    # Second-order derivatives
    dFdxx = np.gradient(dFdx, axis=0)/dx
    dFdyy = np.gradient(dFdy, axis=1)/dy
    
    dNdxx[0,:] = (2.0*N[1,:] - dx*gamma*N[0,:]*dFdx[0,:]/dN - 2.0*N[0,:])/dx**2
    dNdxx[1:-1,:] = np.diff(N,2,axis=0)/dx**2
    dNdxx[-1,:] = (2.0*N[-2,:] - dx*gamma*N[-1,:]*dFdx[-1,:]/dN - 2.0*N[-1,:])/dx**2
    
    dNdyy[:,0] = (2.0*N[:,1] - dy*gamma*N[:,0]*dFdy[:,0]/dN - 2.0*N[:,0])/dy**2
    dNdyy[:,1:-1] = np.diff(N,2,axis=1)/dy**2
    dNdyy[:,-1] = (2.0*N[:,-2] - dy*gamma*N[:,-1]*dFdy[:,-1]/dN - 2.0*N[:,-1])/dy**2
    
    # M derivatives
    dMdxx[0,:] = (2.0*M[1,:] - 2.0*M[0,:])/dx**2
    dMdxx[1:-1,:] = np.diff(M,2,axis=0)/dx**2
    dMdxx[-1,:] = (2.0*M[-2,:] - 2.0*M[-1,:])/dx**2
    
    dMdyy[:,0] = (2.0*M[:,1] - 2.0*M[:,0])/dy**2
    dMdyy[:,1:-1] = np.diff(M,2,axis=1)/dy**2
    dMdyy[:,-1] = (2.0*M[:,-2] - 2.0*M[:,-1])/dy**2
    
    # PDEs
    dNdt = dN*(dNdxx + dNdyy) - gamma*(dNdx*dFdx + dNdy*dFdy + N*(dFdxx + dFdyy))
    dFdt = -eta*M*F
    dMdt = dM*(dMdxx + dMdyy) + alpha*N
    
    dzdt = np.array([dNdt, dFdt, dMdt])
    return dzdt.reshape(-1)

# Initial conditions
def initial_N(z, centre=[0.5,0.5], rad=0.1):
    epsilon = 2.5e-3
    x, y = z
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    if r <= rad:
        return np.exp(-r**2/epsilon)
    return 0.

def initial_F2(z):
    return (1. - initial_N(z, rad=0.5) - 0.5*initial_N(z, centre=[0.8,0.8], rad=0.2)
            - 0.8*initial_N(z, centre=[0.2,0.2], rad=0.1) - 0.6*initial_N(z, centre=[0.05,0.6], rad=0.3)
            - 0.3*initial_N(z, centre=[0.6,0.1], rad=0.05) - initial_N(z, centre=[0.6,0.6], rad=0.5)
            - initial_N(z, centre=[0.8,0.4], rad=0.5) - initial_N(z, centre=[0.9,0.2], rad=0.5)
            - initial_N(z, centre=[0.3,0.8], rad=0.7))

def initial_M(z):
    return 0.5*initial_N(z)

# Load data
print("\nLoading TumourGrowthData.csv...")
df = pd.read_csv('C:\load\TumourGrowthData.csv')
print(f"Data shape: {df.shape}")

# Check if columns need to be renamed
if isinstance(df.columns[0], int) or df.columns[0] in [0, '0']:
    df.columns = ['time', 'x', 'y', 'n', 'f', 'm']
    print("Renamed columns to: time, x, y, n, f, m")

# Create model
print("\nCreating PDE model...")
my_model = PDEmodel(df, TumourGrowth, [initial_N, initial_F2, initial_M], 
                    bounds=[(0.05, 0.15), (0.002, 0.01)], 
                    param_names=['alpha', 'gamma'],
                    nvars=3, ndims=2, nreplicates=3)

# Plot initial conditions
print("\nPlotting initial conditions...")
X = my_model.space[:,:,0]
Y = my_model.space[:,:,1]

# Combined plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im1 = axes[0].contourf(X, Y, my_model.initial_condition[0], cmap='magma', levels=20)
axes[0].set_title('n(x,y,0)', fontsize=14)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].contourf(X, Y, my_model.initial_condition[1], cmap='magma', levels=20)
axes[1].set_title('f(x,y,0)', fontsize=14)
axes[1].set_xlabel('x')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].contourf(X, Y, my_model.initial_condition[2], cmap='magma', levels=20)
axes[2].set_title('m(x,y,0)', fontsize=14)
axes[2].set_xlabel('x')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'initial_conditions.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/initial_conditions.png")

# Fit the model
print("\nFitting the model (this may take a minute)...")
my_model.fit()

# Save results with encoding specified
with open(os.path.join(output_dir, 'fit_results.txt'), 'w', encoding='utf-8') as f:
    f.write("Best fit parameters:\n")
    f.write(str(my_model.best_params))
    f.write(f"\n\nBest error: {my_model.best_error}")
    f.write("\n\nExpected values: alpha = 0.103, gamma = 0.005")
print(f"Fit complete. Results saved to: {output_dir}/fit_results.txt")

# Likelihood profiles
print("\nComputing likelihood profiles...")
my_model.likelihood_profiles(npoints=25)

# Plot profiles with curves
print("\nPlotting likelihood profiles...")
colors = plt.cm.tab10.colors

for i, pname in enumerate(my_model.param_names):
    data = my_model.result_profiles[my_model.result_profiles.parameter == pname]
    
    plt.figure(figsize=(10, 8))
    
    # Plot the curve
    plt.plot(data.value.values, data.error.values, 'b-', linewidth=3, label='Likelihood profile')
    
    # Add best fit point
    plt.scatter([my_model.best_params[pname][0]], [my_model.best_error], 
                color='red', s=200, zorder=5, marker='*', edgecolor='black', 
                linewidth=2, label='Best fit')
    
    # Add vertical line at best fit
    plt.axvline(x=my_model.best_params[pname][0], color='red', linestyle='--', 
                alpha=0.5, linewidth=2)
    
    # Set y-axis limits
    min_error = np.min(data.error.values)
    max_error = np.max(data.error.values)
    
    if max_error > 250 * min_error:
        plt.ylim(-10 * min_error, 250 * min_error)
    else:
        plt.ylim(bottom=-3 * min_error)
    
    plt.xlabel(pname, fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.title(f'Likelihood Profile: {pname}', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    
    # Add text with best value
    plt.text(0.05, 0.95, f'Best {pname}: {my_model.best_params[pname][0]:.6f}', 
             transform=plt.gca().transAxes, fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'profile_{pname}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/profile_{pname}.png")

# Bootstrap analysis
print("\nPerforming bootstrap analysis...")
my_model.bootstrap(nruns=50)

# Save bootstrap results with encoding
with open(os.path.join(output_dir, 'bootstrap_results.txt'), 'w', encoding='utf-8') as f:
    f.write("Bootstrap Summary:\n")
    f.write(str(my_model.bootstrap_summary))
print(f"Bootstrap complete. Results saved to: {output_dir}/bootstrap_results.txt")

# Plot bootstrap results with curves
print("\nPlotting bootstrap results...")

# Create pairplot-style visualization
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)

# Alpha histogram with KDE curve
ax1 = fig.add_subplot(gs[0, 0])
counts, bins, patches = ax1.hist(my_model.bootstrap_raw['alpha'], bins=20, 
                                density=True, alpha=0.7, color='skyblue', 
                                edgecolor='black')

# Add KDE curve
from scipy import stats
kde = stats.gaussian_kde(my_model.bootstrap_raw['alpha'])
x_range = np.linspace(bins[0], bins[-1], 200)
ax1.plot(x_range, kde(x_range), 'b-', linewidth=3, label='KDE')
ax1.axvline(my_model.best_params.iloc[0, 0], color='red', linestyle='--', 
            linewidth=3, label='Best fit')
ax1.set_xlabel('alpha', fontsize=14)
ax1.set_ylabel('Density', fontsize=14)
ax1.set_title('Alpha Distribution', fontsize=16)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot with confidence ellipse
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(my_model.bootstrap_raw['alpha'], my_model.bootstrap_raw['gamma'], 
            alpha=0.6, s=60, color='blue', label='Bootstrap samples', edgecolor='black')
ax2.scatter(my_model.best_params.iloc[0, 0], my_model.best_params.iloc[0, 1],
            color='red', s=300, marker='*', label='Best fit', edgecolor='black', linewidth=2)

# Add confidence ellipse
from matplotlib.patches import Ellipse
mean_alpha = my_model.bootstrap_raw['alpha'].mean()
mean_gamma = my_model.bootstrap_raw['gamma'].mean()
cov = np.cov(my_model.bootstrap_raw['alpha'], my_model.bootstrap_raw['gamma'])
eigvals, eigvecs = np.linalg.eig(cov)
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
ellipse = Ellipse((mean_alpha, mean_gamma), 
                  width=2*np.sqrt(eigvals[0]), 
                  height=2*np.sqrt(eigvals[1]),
                  angle=angle, facecolor='none', edgecolor='red', 
                  linewidth=2, alpha=0.8, label='95% confidence')
ax2.add_patch(ellipse)

ax2.set_xlabel('alpha', fontsize=14)
ax2.set_ylabel('gamma', fontsize=14)
ax2.legend(fontsize=12)
ax2.set_title('Bootstrap Results', fontsize=16)
ax2.grid(True, alpha=0.3)

# Empty plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

# Gamma histogram with KDE curve
ax4 = fig.add_subplot(gs[1, 1])
counts, bins, patches = ax4.hist(my_model.bootstrap_raw['gamma'], bins=20, 
                                density=True, alpha=0.7, color='lightgreen', 
                                edgecolor='black')
kde = stats.gaussian_kde(my_model.bootstrap_raw['gamma'])
x_range = np.linspace(bins[0], bins[-1], 200)
ax4.plot(x_range, kde(x_range), 'g-', linewidth=3, label='KDE')
ax4.axvline(my_model.best_params.iloc[0, 1], color='red', linestyle='--', 
            linewidth=3, label='Best fit')
ax4.set_xlabel('gamma', fontsize=14)
ax4.set_ylabel('Density', fontsize=14)
ax4.set_title('Gamma Distribution', fontsize=16)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Bootstrap Analysis', fontsize=20)
plt.savefig(os.path.join(output_dir, 'bootstrap_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/bootstrap_analysis.png")

# Create a profile comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Alpha profile
data_alpha = my_model.result_profiles[my_model.result_profiles.parameter == 'alpha']
ax1.plot(data_alpha.value.values, data_alpha.error.values, 'b-', linewidth=3)
ax1.scatter([my_model.best_params['alpha'][0]], [my_model.best_error], 
            color='red', s=200, zorder=5, marker='*', edgecolor='black')
ax1.set_xlabel('alpha', fontsize=16)
ax1.set_ylabel('Error', fontsize=16)
ax1.set_title('Likelihood Profile: alpha', fontsize=18)
ax1.grid(True, alpha=0.3)

# Gamma profile
data_gamma = my_model.result_profiles[my_model.result_profiles.parameter == 'gamma']
ax2.plot(data_gamma.value.values, data_gamma.error.values, 'g-', linewidth=3)
ax2.scatter([my_model.best_params['gamma'][0]], [my_model.best_error], 
            color='red', s=200, zorder=5, marker='*', edgecolor='black')
ax2.set_xlabel('gamma', fontsize=16)
ax2.set_ylabel('Error', fontsize=16)
ax2.set_title('Likelihood Profile: gamma', fontsize=18)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profiles_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/profiles_comparison.png")

print("\n=== Analysis Complete ===")
print(f"All results saved in: {os.path.abspath(output_dir)}")
print("\nFiles created:")
print("- initial_conditions.png: All three initial conditions")
print("- profile_alpha.png, profile_gamma.png: Individual likelihood profiles with curves") 
print("- bootstrap_analysis.png: Bootstrap results with histograms and KDE curves")
print("- profiles_comparison.png: Both profiles side by side")
print("- fit_results.txt, bootstrap_results.txt: Numerical results")

# Open the folder
try:
    os.startfile(output_dir)
    print(f"\nOpening {output_dir} folder...")
except:
    print(f"\nPlease manually open the folder: {os.path.abspath(output_dir)}")