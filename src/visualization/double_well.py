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
plt.show()

par = [300, 200, -1, 1150]
print(np.sqrt(300*(1150-300)))
min1, min2, _ = find_minima(par)
plt.figure()
Z = np.linspace(-1, 1, 1000)
plt.plot(Z, V_MF(Z, par) - min2[1], color='tab:gray')

param_text = fr"$\Omega$={par[0]}, $\delta$={par[1]}, k={par[2]}, n={par[3]}"
plt.annotate(param_text, xy=(0.55, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

plt.plot(min1[0], min1[1] - min2[1], 'o', color='tab:blue', markersize=8)  # Plot first minimum as a blue point
plt.plot(min2[0], 0, 'o', color='tab:red', markersize=8)  # Plot second minimum as a red point

# Plot vertical line indicating the difference in minima
plt.vlines(min1[0] + 0.1, min1[1] - min2[1], 0, colors='tab:orange')
plt.hlines(0, min1[0], min1[0] + 0.2, colors='tab:orange')
plt.hlines(min1[1] - min2[1], min1[0], min1[0] + 0.2, colors='tab:orange')
plt.annotate(f"$\Delta = ${(min1[1] - min2[1]):.2f}", xy=(0.2, 0.45), xycoords='axes fraction', fontsize=12,
             horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


plt.xlabel("Z")
plt.ylabel("$V_{MF}$")
plt.title("Mean-field energy")
plt.show()