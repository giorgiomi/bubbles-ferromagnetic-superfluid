import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

def V_MF(Z, param):
    omega = param[0]
    delta = param[1]
    k = param[2]
    n = param[3]
    return -(np.abs(k)*n*Z**2 + 2*omega*np.sqrt(1-Z**2) + 2*delta*Z)

def V_MF_prime(Z, param):
    omega = param[0]
    delta = param[1]
    k = param[2]
    n = param[3]
    return np.abs(k)*n*Z - omega*Z*np.sqrt(1-Z**2) + delta

def find_minima(param):
    result1 = minimize_scalar(V_MF, bounds=(-1, 0), args=(param,), method='bounded')
    result2 = minimize_scalar(V_MF, bounds=(0, 1), args=(param,), method='bounded')
    # Check if the results are zero, if so, find another minimum
    if result1.x < 1e-5:
        result1 = minimize_scalar(V_MF, bounds=(-1, -0.5), args=(param,), method='bounded')
    return (result1.x, result1.fun), (result2.x, result2.fun), np.abs(result2.fun - result1.fun)

par = [2, 1, -1150/2700, 1]
delta_values = np.linspace(-1, 1, 100)
omega_values = np.linspace(0.5, 1.5, 100)
diff_min_values_delta = []
diff_min_values_omega = []

# Calculate difference in minima as a function of delta
for delta in delta_values:
    par[1] = delta
    min1, min2, diff_min = find_minima(par)
    diff_min_values_delta.append(diff_min)

# Calculate difference in minima as a function of omega
for omega in omega_values:
    par[0] = omega
    min1, min2, diff_min = find_minima(par)
    diff_min_values_omega.append(diff_min)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot delta dependence
axs[0].plot(delta_values, diff_min_values_delta)
axs[0].set_xlabel(r'$\delta$')
axs[0].set_ylabel(r'$|V_{MF}(Z_2) - V_{MF}(Z_1)|$')
axs[0].set_title('Difference in Minima as a Function of $\delta$')

# Plot omega dependence
axs[1].plot(omega_values, diff_min_values_omega)
axs[1].set_xlabel(r'$\omega$')
axs[1].set_ylabel(r'$|V_{MF}(Z_2) - V_{MF}(Z_1)|$')
axs[1].set_title('Difference in Minima as a Function of $\omega$')

plt.tight_layout()
# plt.show()


fig, axs = plt.subplots(3, 4, figsize=(15, 8))
axs = axs.flatten()

for i, om in enumerate(range(200, 1400, 100)):
    for det in range(0, 700, 100):
        kn = 1150
        par = [om, det, -1, kn]
        min1, min2, _ = find_minima(par)
        
        Z = np.linspace(-1, 1, 1000)
        axs[i].plot(Z, V_MF(Z, par) - min2[1], color='tab:gray')

        param_text = f"$\Omega_R/2\pi$ = {par[0]}\n$\delta$ = {par[1]}\n|k|n = {np.abs(par[2])*par[3]}"
        axs[i].annotate(param_text, xy=(0.75, 0.95), xycoords='axes fraction', fontsize=12,
                        horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

        # axs[i].plot(min1[0], min1[1] - min2[1], 'o', color='tab:blue', markersize=8)  # Plot first minimum as a blue point
        # axs[i].plot(min2[0], 0, 'o', color='tab:red', markersize=8)  # Plot second minimum as a red point

        # Annotate the minima points
        # axs[i].annotate(r'$\mathbf{FV}$', xy=(min1[0], min1[1] - min2[1]), xytext=(min1[0] - 0.05, min1[1] - min2[1] + 100), fontsize=12, color='tab:blue')
        # axs[i].annotate(r'$\mathbf{TV}$', xy=(min2[0], 0), xytext=(min2[0] - 0.05, 100), fontsize=12, color='tab:red')

    axs[i].set_xlabel("Z")
    axs[i].set_ylabel("$V_{MF}$")
    axs[i].set_title(f"Mean-field energy landscape\n$\Omega_R/2\pi$ = {om}")

plt.tight_layout()
plt.show()
