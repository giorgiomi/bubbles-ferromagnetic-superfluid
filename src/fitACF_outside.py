# This script is used to analyze the data outside the bubble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from util.parameters import importParameters
import sys
from util.methods import quadPlot, computeFFT_ACF, doublePlot
from util.functions import corrGauss
from scipy.optimize import curve_fit

# Data
# f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 20

# Print script purpose
print("\nAnalyzing FFT and ACF on OUTSIDE REGION with gathered data\n")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: ")

# Ask for gather flag
gather_flag = input("Enter gather method [omega/time/size/detuning/dE]: ")

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
# dE = find_minima(omega, detuning, -1150, 1)
dE = np.sqrt(omega*(1150-omega))
# print(dE)
# exit()

if gather_flag == 'omega':
    ## GATHER BY OMEGA
    omega_blocks = np.unique(omega)

    # Initialize lists to store FFT and ACF results for each omega block
    fft_magnitudes_omega = []
    acf_values_omega = []
    omega_new = []

    # Group shots by omega block and compute FFT and ACF
    for omega_val in omega_blocks:
        shots_in_block = Z[omega == omega_val]
        if len(shots_in_block) > 0:
            for i, shot in enumerate(shots_in_block):
                s = size[omega == omega_val][i]
                c = center[omega == omega_val][i]
                left = shot[:int(c-s/2-10)]
                right = shot[int(c+s/2+10):]
                if len(right) > 4*window_len:
                    fft_magnitudes_omega, acf_values_omega, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, right, CFG, CLG, fft_magnitudes_omega, acf_values_omega, window_len)
                    omega_new.append(omega_val)

    omega_fft = np.array(omega_new)

    fft_magnitudes_omega = np.array(fft_magnitudes_omega)
    acf_values_omega = np.array(acf_values_omega)

    # Compute mean FFT and ACF for each omega block
    fft_omega_means = {omega_val: np.mean(fft_magnitudes_omega[omega_fft == omega_val], axis=0) for omega_val in omega_blocks if omega_val in omega_fft}
    acf_omega_means = {omega_val: np.mean(acf_values_omega[omega_fft == omega_val], axis=0) for omega_val in omega_blocks if omega_val in omega_fft}
    # Compute standard deviation of FFT and ACF for each omega block
    acf_omega_err = {omega_val: np.std(acf_values_omega[omega_fft == omega_val], axis=0)/len(acf_values_omega[omega_fft == omega_val]) for omega_val in omega_blocks if omega_val in omega_fft}

    # Plot FFT and ACF results for each omega block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, omega_val in enumerate(fft_omega_means.keys()):
        ax[0].plot(CFG, fft_omega_means[omega_val], label=f'$\Omega = {omega_val}$ Hz', color=plt.cm.viridis(i / len(fft_omega_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Frequencies')

    # Plot ACF results using colormaps
    for i, omega_val in enumerate(acf_omega_means.keys()):
        ax[1].errorbar(CLG, acf_omega_means[omega_val], yerr=acf_omega_err[omega_val], label=f'$\Omega = {omega_val}$ Hz', color=plt.cm.viridis(i / len(acf_omega_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Frequencies')

    plt.tight_layout()
    plt.show()

    # Fit the ACF means to the Gaussian correlation function
    trunc_index = window_len
    fit_params = {}
    fit_errors = {}
    for omega_val, acf_mean in acf_omega_means.items():
        try:
            popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
            fit_params[omega_val] = popt
            fit_errors[omega_val] = np.sqrt(np.diag(pcorr))
        except RuntimeError:
            print(f"Fit failed for omega block {omega_val}")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ACF means and fits
    colors = plt.cm.viridis(np.linspace(0, 1, len(acf_omega_means)))

    for color, (omega_val, acf_mean) in zip(colors, acf_omega_means.items()):
        ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'$\Omega = {omega_val}$ Hz', alpha=0.5)
        if omega_val in fit_params:
            fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[omega_val])
            ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
    ax[0].set_title('Mean ACF and Fits by Omega Block')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()

    # Plot the first fit parameter (l1) vs omega
    omegas = list(fit_params.keys())
    l1_values = [params[0] for params in fit_params.values()]
    dl1_values = [err[0] for err in fit_errors.values()]

    ax[1].errorbar(omegas, l1_values, yerr=dl1_values, marker='o', linestyle='none', capsize=2)
    ax[1].set_title('First Fit Parameter ($l_1$) vs Omega')
    ax[1].set_xlabel('Omega (Hz)')
    ax[1].set_ylabel('l1')

    plt.tight_layout()
    plt.show()

elif gather_flag == 'time':
    ## GATHER BY TIME
    n_blocks = 10
    # Define time blocks (e.g., every 10 units of time)
    time_block_size = (np.max(time) - np.min(time))/n_blocks
    time_blocks = np.arange(min(time), max(time) + time_block_size, time_block_size)

    # Initialize lists to store FFT and ACF results for each time block
    fft_magnitudes_time = []
    acf_values_time = []
    time_new = []

    # Group shots by time block and compute FFT and ACF
    for start_time in time_blocks:
        end_time = start_time + time_block_size
        shots_in_block = Z[(time >= start_time) & (time < end_time)]
        if len(shots_in_block) > 0:
            for i, shot in enumerate(shots_in_block):
                s = size[(time >= start_time) & (time < end_time)][i]
                c = center[(time >= start_time) & (time < end_time)][i]
                inside = shot[int(c-s/2+10):int(c+s/2-10)]
                if len(inside) > 4*window_len:
                    fft_magnitudes_time, acf_values_time, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes_time, acf_values_time, window_len)
                    time_new.append(start_time)

    time_fft = np.array(time_new)

    fft_magnitudes_time = np.array(fft_magnitudes_time)
    acf_values_time = np.array(acf_values_time)

    # Compute mean FFT and ACF for each time block
    fft_time_means = {start_time: np.mean(fft_magnitudes_time[time_fft == start_time], axis=0) for start_time in time_blocks if start_time in time_fft}
    acf_time_means = {start_time: np.mean(acf_values_time[time_fft == start_time], axis=0) for start_time in time_blocks if start_time in time_fft}

    # Plot FFT and ACF results for each time block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, start_time in enumerate(fft_time_means.keys()):
        ax[0].plot(CFG, fft_time_means[start_time], label=f'Time {start_time:.1f}-{start_time + time_block_size:.1f}', color=plt.cm.viridis(i / len(fft_time_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Time Blocks')

    # Plot ACF results using colormaps
    for i, start_time in enumerate(acf_time_means.keys()):
        ax[1].plot(CLG, acf_time_means[start_time], label=f'Time {start_time:.1f}-{start_time + time_block_size:.1f}', color=plt.cm.viridis(i / len(acf_time_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Time Blocks')

    plt.tight_layout()
    plt.show()

    # Fit the ACF means to the Gaussian correlation function
    # Note: tried with cosine, work to be done
    trunc_index = window_len
    fit_params = {}
    fit_errors = {}
    for start_time, acf_mean in acf_time_means.items():
        try:
            popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
            fit_params[start_time] = popt
            fit_errors[start_time] = np.sqrt(np.diag(pcorr))
        except RuntimeError:
            print(f"Fit failed for time block {start_time}")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ACF means and fits
    colors = plt.cm.viridis(np.linspace(0, 1, len(acf_time_means)))

    for color, (start_time, acf_mean) in zip(colors, acf_time_means.items()):
        ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'time {start_time:.1f}', alpha=0.5)
        if start_time in fit_params:
            fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[start_time])
            ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
    ax[0].set_title('Mean ACF and Fits by time Block')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()
    # ax[0].legend()

    # Plot the first fit parameter (l1) vs time
    times = list(fit_params.keys())
    l1_values = [params[0] for params in fit_params.values()]
    dl1_values = [err[0] for err in fit_errors.values()]

    ax[1].errorbar(times, l1_values, yerr=dl1_values, marker='o', linestyle='none', capsize=2)
    ax[1].set_title('First Fit Parameter ($l_1$) vs time')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('l1')
    # ax[1].set_yscale('log')

    plt.tight_layout()
    plt.show()

elif gather_flag == 'size':
    ## GATHER BY SIZE
    # Define size blocks (e.g., every 50 units of size)
    size_block_size = 30
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
            for i, shot in enumerate(shots_in_block):
                s = size[(size >= start_size) & (size < end_size)][i]
                c = center[(size >= start_size) & (size < end_size)][i]
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

    # Plot FFT and ACF results for each size block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, start_size in enumerate(fft_size_means.keys()):
        ax[0].plot(CFG, fft_size_means[start_size], label=f'Size {start_size:.1f}-{start_size + size_block_size:.1f}', color=plt.cm.viridis(i / len(fft_size_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Size Blocks')

    # Plot ACF results using colormaps
    for i, start_size in enumerate(acf_size_means.keys()):
        ax[1].plot(CLG, acf_size_means[start_size], label=f'Size {start_size:.1f}-{start_size + size_block_size:.1f}', color=plt.cm.viridis(i / len(acf_size_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Size Blocks')

    plt.tight_layout()
    plt.show()

    # Fit the ACF means to the Gaussian correlation function
    # Note: tried with cosine, work to be done
    trunc_index = window_len
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

    ax[1].errorbar(sizes[:-1], l1_values[:-1], yerr=dl1_values[:-1], marker='o', linestyle='none', capsize=2)
    ax[1].set_title('First Fit Parameter ($l_1$) vs Size')
    ax[1].set_xlabel('Size')
    ax[1].set_ylabel('l1')
    # ax[1].set_yscale('log')

    plt.tight_layout()
    plt.show()

elif gather_flag == 'detuning':
    ## GATHER BY DETUNING
    detuning_block_size = 50
    detuning_blocks = np.arange(min(detuning), max(detuning) + detuning_block_size, detuning_block_size)

    # Initialize lists to store FFT and ACF results for each detuning block
    fft_magnitudes_detuning = []
    acf_values_detuning = []
    detuning_new = []

    # Group shots by detuning block and compute FFT and ACF
    for start_size in detuning_blocks:
        end_size = start_size + detuning_block_size
        shots_in_block = Z[(detuning >= start_size) & (detuning < end_size)]
        # print(start_size, len(shots_in_block))
        if len(shots_in_block) > 0:
            for i, shot in enumerate(shots_in_block):
                s = size[(detuning >= start_size) & (detuning < end_size)][i]
                c = center[(detuning >= start_size) & (detuning < end_size)][i]
                inside = shot[int(c-s/2+10):int(c+s/2-10)]
                if len(inside) > 4*window_len:
                    fft_magnitudes_detuning, acf_values_detuning, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes_detuning, acf_values_detuning, window_len)
                    detuning_new.append(start_size)

    detuning_fft = np.array(detuning_new)

    fft_magnitudes_detuning = np.array(fft_magnitudes_detuning)
    acf_values_detuning = np.array(acf_values_detuning)

    # Compute mean FFT and ACF for each detuning block
    fft_detuning_means = {detuning_val: np.mean(fft_magnitudes_detuning[detuning_fft == detuning_val], axis=0) for detuning_val in detuning_blocks if detuning_val in detuning_fft}
    acf_detuning_means = {detuning_val: np.mean(acf_values_detuning[detuning_fft == detuning_val], axis=0) for detuning_val in detuning_blocks if detuning_val in detuning_fft}

    # Plot FFT and ACF results for each detuning block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, detuning_val in enumerate(fft_detuning_means.keys()):
        ax[0].plot(CFG, fft_detuning_means[detuning_val], label=f'Detuning {detuning_val:.1f}', color=plt.cm.viridis(i / len(fft_detuning_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Detuning Blocks')

    # Plot ACF results using colormaps
    for i, detuning_val in enumerate(acf_detuning_means.keys()):
        ax[1].plot(CLG, acf_detuning_means[detuning_val], label=f'Detuning {detuning_val:.1f}', color=plt.cm.viridis(i / len(acf_detuning_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Detuning Blocks')

    plt.tight_layout()
    plt.show()

    # Fit the ACF means to the Gaussian correlation function
    trunc_index = window_len
    fit_params = {}
    fit_errors = {}
    for detuning_val, acf_mean in acf_detuning_means.items():
        try:
            popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
            fit_params[detuning_val] = popt
            fit_errors[detuning_val] = np.sqrt(np.diag(pcorr))
        except RuntimeError:
            print(f"Fit failed for detuning block {detuning_val}")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ACF means and fits
    colors = plt.cm.viridis(np.linspace(0, 1, len(acf_detuning_means)))

    for color, (detuning_val, acf_mean) in zip(colors, acf_detuning_means.items()):
        ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'Detuning {detuning_val:.1f}', alpha=0.5)
        if detuning_val in fit_params:
            fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[detuning_val])
            ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
    ax[0].set_title('Mean ACF and Fits by Detuning Block')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()

    # Plot the first fit parameter (l1) vs detuning
    detunings = list(fit_params.keys())
    l1_values = [params[0] for params in fit_params.values()]
    dl1_values = [err[0] for err in fit_errors.values()]

    ax[1].errorbar(detunings, l1_values, yerr=dl1_values, marker='o', linestyle='none', capsize=2)
    ax[1].set_title('First Fit Parameter ($l_1$) vs Detuning')
    ax[1].set_xlabel('Detuning')
    ax[1].set_ylabel('l1')

    plt.tight_layout()
    plt.show()

elif gather_flag == 'dE':
    ## GATHER BY dE
    dE_block_size = 20
    dE_blocks = np.arange(min(dE), max(dE) + dE_block_size, dE_block_size)

    # Initialize lists to store FFT and ACF results for each dE block
    fft_magnitudes_dE = []
    acf_values_dE = []
    dE_new = []

    # Group shots by dE block and compute FFT and ACF
    for start_dE in dE_blocks:
        end_dE = start_dE + dE_block_size
        shots_in_block = Z[(dE >= start_dE) & (dE < end_dE)]
        if len(shots_in_block) > 0:
            for i, shot in enumerate(shots_in_block):
                s = size[(dE >= start_dE) & (dE < end_dE)][i]
                c = center[(dE >= start_dE) & (dE < end_dE)][i]
                inside = shot[int(c-s/2+10):int(c+s/2-10)]
                if len(inside) > 4*window_len:
                    fft_magnitudes_dE, acf_values_dE, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes_dE, acf_values_dE, window_len)
                    dE_new.append(start_dE)

    dE_fft = np.array(dE_new)

    fft_magnitudes_dE = np.array(fft_magnitudes_dE)
    acf_values_dE = np.array(acf_values_dE)

    # Compute mean FFT and ACF for each dE block
    fft_dE_means = {dE_val: np.mean(fft_magnitudes_dE[dE_fft == dE_val], axis=0) for dE_val in dE_blocks if dE_val in dE_fft}
    acf_dE_means = {dE_val: np.mean(acf_values_dE[dE_fft == dE_val], axis=0) for dE_val in dE_blocks if dE_val in dE_fft}

    # Plot FFT and ACF results for each dE block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, dE_val in enumerate(fft_dE_means.keys()):
        ax[0].plot(CFG, fft_dE_means[dE_val], label=f'dE {dE_val:.1f}-{dE_val + dE_block_size:.1f}', color=plt.cm.viridis(i / len(fft_dE_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different dE Blocks')

    # Plot ACF results using colormaps
    for i, dE_val in enumerate(acf_dE_means.keys()):
        ax[1].plot(CLG, acf_dE_means[dE_val], label=f'dE {dE_val:.1f}-{dE_val + dE_block_size:.1f}', color=plt.cm.viridis(i / len(acf_dE_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different dE Blocks')

    plt.tight_layout()
    plt.show()

    # Fit the ACF means to the Gaussian correlation function
    trunc_index = window_len
    fit_params = {}
    fit_errors = {}
    for dE_val, acf_mean in acf_dE_means.items():
        try:
            popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
            fit_params[dE_val] = popt
            fit_errors[dE_val] = np.sqrt(np.diag(pcorr))
        except RuntimeError:
            print(f"Fit failed for dE block {dE_val}")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ACF means and fits
    colors = plt.cm.viridis(np.linspace(0, 1, len(acf_dE_means)))

    for color, (dE_val, acf_mean) in zip(colors, acf_dE_means.items()):
        ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'dE {dE_val:.1f}', alpha=0.5)
        if dE_val in fit_params:
            fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[dE_val])
            ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
    ax[0].set_title('Mean ACF and Fits by dE Block')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()

    # Plot the first fit parameter (l1) vs dE
    dEs = list(fit_params.keys())
    l1_values = [params[0] for params in fit_params.values()]
    dl1_values = [err[0] for err in fit_errors.values()]

    ax[1].errorbar(dEs, l1_values, yerr=dl1_values, marker='o', linestyle='none', capsize=2)
    ax[1].set_title('First Fit Parameter ($l_1$) vs dE')
    ax[1].set_xlabel('dE')
    ax[1].set_ylabel('l1')

    plt.tight_layout()
    plt.show()

else:
    print("Gather method has to be [omega/time/size]")