# This script is used to fit the ACF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq
from util.parameters import importParameters
import sys
from util.methods import quadPlot, computeFFT_ACF, doublePlot, AZcorr
from util.functions import corrGauss
from scipy.optimize import curve_fit

# Data
# f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 20
padding = 10

# Print script purpose
print("\nFIT ACF\n")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: ")

# Ask for gather flag
# gather_flag = input("Enter gather method [omega/time/size]: ")

# Find max length for Common Frequency Grid (CFG)
max_length = 0
size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
max_size = int(max(size) + 1)
if max_size > max_length and max_size < 2*w:
    max_length = max_size

# CFG and CLG
CFG = rfftfreq(max_length, d=sampling_rate)
CLG = np.arange(0, window_len+1)

# Import bubble data
center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

## GATHER BY SIZE
# Define size blocks (e.g., every 10 units of size)
size_block_size = 20
size_blocks = np.arange(min(size), max(size) + size_block_size, size_block_size)

# Initialize lists to store FFT and ACF results for each size block
fft_magnitudes_size = []
acf_values_size = []
size_new = []

# Group shots by size block and compute FFT and ACF
for start_size in size_blocks:
    end_size = start_size + size_block_size
    shots_in_block = Z[(size >= start_size) & (size < end_size)]
    if len(shots_in_block) > 0:
        for shot in shots_in_block:
            s = size[(size >= start_size) & (size < end_size)][0]
            c = center[(size >= start_size) & (size < end_size)][0]
            inside = shot[int(c-s/2+10):int(c+s/2-10)]
            if len(inside) > 4*window_len:
                fft_magnitudes_size, acf_values_size, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes_size, acf_values_size, window_len)
                size_new.append(start_size)

size_fft = np.array(size_new)

fft_magnitudes_size = np.array(fft_magnitudes_size)
acf_values_size = np.array(acf_values_size)

# Compute mean FFT and ACF for each size block
fft_size_means = {start_size: np.mean(fft_magnitudes_size[size_fft == start_size], axis=0) for start_size in size_blocks if start_size in size_fft}
acf_size_means = {start_size: np.mean(acf_values_size[size_fft == start_size], axis=0) for start_size in size_blocks if start_size in size_fft}

# Fit the ACF means to the Gaussian correlation function
trunc_index = 20
fit_params = {}
fit_errors = {}
for start_size, acf_mean in acf_size_means.items():
    try:
        popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
        fit_params[start_size] = popt
        fit_errors[start_size] = np.sqrt(np.diag(pcorr))
    except RuntimeError:
        print(f"Fit failed for size block {start_size}")

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot ACF means and fits
colors = plt.cm.viridis(np.linspace(0, 1, len(acf_size_means)))

for color, (start_size, acf_mean) in zip(colors, acf_size_means.items()):
    ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'Size {start_size:.1f}', alpha=0.5)
    if start_size in fit_params:
        fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[start_size])
        ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
ax[0].set_title('Mean ACF and Fits by Size Block')
ax[0].set_xlabel('Lag')
ax[0].set_ylabel('ACF')
ax[0].legend()
# ax[0].legend()

# Plot the first fit parameter (l1) vs size
sizes = list(fit_params.keys())
l1_values = [params[0] for params in fit_params.values()]
dl1_values = [err[0] for err in fit_errors.values()]

ax[1].errorbar(sizes[:-1], l1_values[:-1], yerr=dl1_values[:-1], marker='o', linestyle='none')
ax[1].set_title('First Fit Parameter ($l_1$) vs Size')
ax[1].set_xlabel('Size')
ax[1].set_ylabel('l1')
# ax[1].set_yscale('log')

plt.tight_layout()
plt.show()