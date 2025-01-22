# This script is used to analyze the data inside the bubble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
from util.parameters import importParameters
import sys
from util.methods import scriptUsage, quadPlot, computeFFT_ACF

# Data
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 40

chosen_days = scriptUsage()

# Print script purpose
print("\nAnalyzing FFT and ACF on INSIDE REGION with density signal\n")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = int(input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: "))

omega_fft_dict = {}
omega_acf_dict = {}
detuning_fft_dict = {}
detuning_acf_dict = {}

max_length = 0
for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_size_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV_sorted.csv", header=None)
        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        max_size = int(max(b_sizeADV_sorted) + 1)
        if max_size > max_length and max_size < 2*w:
            max_length = max_size

# Common frequency grid for FFT
CFG = rfftfreq(max_length, d=sampling_rate)

# Common lag grid for ACF
CLG = np.arange(-window_len, window_len)

for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_size_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV_sorted.csv", header=None)
        df_center_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center_sorted.csv", header=None)
        df_in_left_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/in_left_sorted.csv", header=None)
        df_in_right_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/in_right_sorted.csv", header=None)
        df_D_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/density_sorted.csv", header=None)
        
        omega = Omega[day][seq]
        detuning = Detuning[day][seq]

        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        b_center_sorted = df_center_sorted.to_numpy().flatten()
        in_left_sorted = df_in_left_sorted.to_numpy().flatten()
        in_right_sorted = df_in_right_sorted.to_numpy().flatten()
        D = df_D_sorted.to_numpy()

        # FFT on bubble (inside)
        if max_length >= 2*w: continue

        inside_fft_magnitudes = []
        inside_acf_values = []
        # cycle through ordered shots from beginning to end
        for i in range(len(D)):
            y = D[i] 
            ## NEW METHOD, with inside estimation
            left = in_left_sorted[i]
            right = in_right_sorted[i]
            start = max(0, int(left))
            end = min(len(y), int(right))
            inside = y[start:end]

            # plt.plot(y)
            # plt.plot(np.arange(start, end), inside)
            # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
            # plt.show()

            # compute FFT of the inside
            N = len(inside)
            # print(day, seq, i, N)
            if N > 0:
                inside_fft_magnitudes, inside_acf_values = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, inside_fft_magnitudes, inside_acf_values, window_len)

        inside_fft_magnitudes = np.array(inside_fft_magnitudes)
        inside_fft_mean = np.mean(inside_fft_magnitudes, axis=0)
        
        inside_acf_values = np.array(inside_acf_values)
        inside_acf_mean = np.mean(inside_acf_values, axis=0)

        # Store the FFT and ACF magnitudes by omega
        if omega not in omega_fft_dict:
            omega_fft_dict[omega] = []
        omega_fft_dict[omega].append(inside_fft_mean)
        if omega not in omega_acf_dict:
            omega_acf_dict[omega] = []
        omega_acf_dict[omega].append(inside_acf_mean)

        # Store the FFT and ACF magnitudes by detuning
        if detuning not in detuning_fft_dict:
            detuning_fft_dict[detuning] = []
        detuning_fft_dict[detuning].append(inside_fft_mean)
        if detuning not in detuning_acf_dict:
            detuning_acf_dict[detuning] = []
        detuning_acf_dict[detuning].append(inside_acf_mean)

        if int(sys.argv[1]) != -1:
            fig = quadPlot(day, seq, D, "inside", CFG, CLG, 
                     inside_fft_magnitudes, inside_fft_mean, inside_acf_values, inside_acf_mean, 0)
            fig.canvas.manager.set_window_title('Density data')
            plt.show()

# FFTs and ACFs as a function of omega
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sorted_omegas = sorted(omega_fft_dict.keys())

# Average FFTs and ACFs with the same omega
for omega in sorted_omegas: 
    fft_list = omega_fft_dict[omega]
    avg_fft = np.mean(fft_list, axis=0)
    axs[0].plot(CFG[1:], avg_fft[1:], '-', label=fr'$\Omega = {omega}$ Hz')

    acf_list = omega_acf_dict[omega]
    avg_acf = np.mean(acf_list, axis=0)
    axs[1].plot(CLG, avg_acf, '-', label=fr'$\Omega = {omega}$ Hz')

# Plot FFTs
axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
axs[0].set_yscale('log')
axs[0].set_xlim(-0.02, 0.52)
axs[0].legend()
axs[0].set_title("Average inside FFTs (density)")

# Plot ACFs    
axs[1].set_xlabel("lag")
axs[1].legend()
axs[1].set_title("Average inside ACFs (density)")

plt.tight_layout()
# plt.savefig(f"thesis/figures/chap2/inside_fft_avg.png", dpi=500)
plt.show()

# FFTs and ACFs as a function of detuning
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sorted_detunings = sorted(detuning_fft_dict.keys())

# Average FFTs and ACFs with the same detuning
# Prepare data for colormap
fft_matrix = []
acf_matrix = []

for detuning in sorted_detunings: 
    fft_list = detuning_fft_dict[detuning]
    avg_fft = np.mean(fft_list, axis=0)
    fft_matrix.append(avg_fft[1:])  # Exclude the zero frequency component

    acf_list = detuning_acf_dict[detuning]
    avg_acf = np.mean(acf_list, axis=0)
    acf_matrix.append(avg_acf)

fft_matrix = np.array(fft_matrix)
acf_matrix = np.array(acf_matrix)

# Plot FFT colormap
im1 = axs[0].imshow(np.log(fft_matrix), aspect='auto', extent=[CFG[1], CFG[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im1, ax=axs[0], label='Log Magnitude')
axs[0].set_title("Average inside FFTs (density)")
axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
axs[0].set_ylabel("$\delta$")

# Plot ACF colormap
im2 = axs[1].imshow(acf_matrix, aspect='auto', extent=[CLG[0], CLG[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im2, ax=axs[1], label='ACF')
axs[1].set_title("Average inside ACFs (density)")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("$\delta$")

plt.tight_layout()
plt.show()