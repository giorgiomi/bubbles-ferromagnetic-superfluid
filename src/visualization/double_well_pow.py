import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp

# Define the function V_MF symbolically
Z = sp.Symbol('Z')
omega, delta, k, n = sp.symbols('omega delta k n')

V_MF = -(sp.Abs(k) * n * Z**2 + 2 * omega * sp.sqrt(1 - Z**2) + 2 * delta * Z)

# Compute first and second derivatives
V_MF_diff = sp.diff(V_MF, Z)  # First derivative
V_MF_ddiff = sp.diff(V_MF_diff, Z)  # Second derivative

# Function to solve for critical points
def find_critical_points(params):
    omega_val, delta_val, k_val, n_val = params
    V_MF_diff_func = sp.lambdify(Z, V_MF_diff.subs({omega: omega_val, delta: delta_val, k: k_val, n: n_val}), 'numpy')
    
    # Solve V_MF'(Z) = 0 in the range Z ∈ [-1, 1]
    Z_vals = np.linspace(-0.99, 0.99, 1000)  # Avoid exact ±1 for numerical stability
    roots = fsolve(V_MF_diff_func, Z_vals)

    # Filter unique roots in valid range
    roots = np.unique(np.round(roots, 6))  # Round to avoid duplicate roots

    # Classify critical points
    classified_points = []
    V_MF_ddiff_func = sp.lambdify(Z, V_MF_ddiff.subs({omega: omega_val, delta: delta_val, k: k_val, n: n_val}), 'numpy')

    for root in roots:
        if -1 <= root <= 1:
            second_derivative = V_MF_ddiff_func(root)
            if second_derivative > 0:
                classification = "Minimum"
            elif second_derivative < 0:
                classification = "Maximum"
            else:
                classification = "Saddle Point"
            classified_points.append((root, classification))
    
    return classified_points

# Function to scan one parameter and find extrema
def scan_parameter(param_name, param_range, fixed_params):
    results = []
    for param_val in param_range:
        params = fixed_params.copy()
        params[param_name] = param_val
        extrema = find_critical_points(list(params.values()))
        results.append((param_val, extrema))
    return results

# Example: Scan delta while keeping omega, k, n fixed
fixed_params = {'omega': 1.0, 'delta': 0.0, 'k': 1.0, 'n': 1.0}
param_name = 'delta'
param_range = np.linspace(-2, 2, 50)

extrema_results = scan_parameter(param_name, param_range, fixed_params)

# Plot results
plt.figure(figsize=(8, 6))
for param_val, extrema in extrema_results:
    for z_val, classification in extrema:
        color = 'b' if classification == "Minimum" else 'r'
        plt.scatter(param_val, z_val, color=color, label=classification if param_val == param_range[0] else "")

plt.xlabel(param_name)
plt.ylabel("Z values of extrema")
plt.title(f"Extrema as a function of {param_name}")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.legend()
plt.show()
