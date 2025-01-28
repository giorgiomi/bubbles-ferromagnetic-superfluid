# This script is used to analyze the data inside the bubble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
from util.parameters import importParameters
import sys
from util.methods import scriptUsage, quadPlot, computeFFT_ACF, doublePlot

# Data
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 30

chosen_days = scriptUsage()

# Print script purpose
print("\nAnalyzing FFT and ACF on INSIDE REGION with magnetization signal\n")

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
CLG = np.arange(0, window_len+1)

for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_size_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV_sorted.csv", header=None)
        df_center_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center_sorted.csv", header=None)
        df_time_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/time_sorted.csv", header=None)
        df_in_left_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/in_left_sorted.csv", header=None)
        df_in_right_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/in_right_sorted.csv", header=None)
        df_Z_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/Z_sorted.csv", header=None)
        
        omega = Omega[day][seq]
        detuning = Detuning[day][seq]

        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        b_center_sorted = df_center_sorted.to_numpy().flatten()
        time_sorted = df_time_sorted.to_numpy().flatten()
        in_left_sorted = df_in_left_sorted.to_numpy().flatten()
        in_right_sorted = df_in_right_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # # filter Z based on shot order
        # Z = Z[int(len(Z)/4):int(len(Z)*3/4)]

        # filter Z based on size between 150 and 200
        filtered_Z = []
        for i in range(len(b_sizeADV_sorted)):
            if b_sizeADV_sorted[i] > 3 * window_len:
                filtered_Z.append(Z[i])
        Z = np.array(filtered_Z)

        # FFT on bubble (inside)
        if max_length >= 2*w: continue

        inside_fft_magnitudes = []
        inside_acf_values = []
        max_freqs = []
        # cycle through ordered shots from beginning to end
        for i in range(len(Z)):
            y = Z[i] 
            ## NEW METHOD, with inside estimation
            left = in_left_sorted[i]
            right = in_right_sorted[i]
            start = max(0, int(left))
            end = min(len(y), int(right))
            inside = y[start:end]

            # plt.plot(y)
            # plt.plot(np.arange(start, end), inside - np.mean(inside))
            # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
            # plt.show()

            # compute FFT of the inside
            N = len(inside)
            if N > 0:
                inside_fft_magnitudes, inside_acf_values, max_freq = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, inside_fft_magnitudes, inside_acf_values, window_len)
                max_freqs.append(max_freq)

        inside_fft_magnitudes = np.array(inside_fft_magnitudes)
        inside_fft_mean = np.mean(inside_fft_magnitudes, axis=0)
        max_freqs = np.array(max_freqs)
        
        inside_acf_values = np.array(inside_acf_values)
        inside_acf_mean = np.mean(inside_acf_values, axis=0)

        # plt.figure()
        # plt.plot(time_sorted, max_freqs, '.')
        # plt.show()

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
            fig = quadPlot(day, seq, Z, "inside", CFG, CLG, 
                     inside_fft_magnitudes, inside_fft_mean, inside_acf_values, inside_acf_mean, 0)
            fig.canvas.manager.set_window_title('Magnetization data')
            # plt.savefig(f"thesis/figures/chap2/inside_day_{day}_seq_{seq}.png", dpi=500)
            plt.show()

fig = doublePlot(omega_fft_dict, omega_acf_dict, CFG, CLG, "inside")
fig.canvas.manager.set_window_title('Magnetization data')
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
im1 = axs[0].imshow(fft_matrix, aspect='auto', extent=[CFG[0], CFG[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im1, ax=axs[0], label='Magnitude')
axs[0].set_title("Average inside FFTs")
axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
axs[0].set_ylabel("$\delta$")

# Plot ACF colormap
im2 = axs[1].imshow(acf_matrix, aspect='auto', extent=[CLG[0], CLG[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im2, ax=axs[1], label='ACF')
axs[1].set_title("Average inside ACFs")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("$\delta$")

plt.tight_layout()
plt.show()

# Plot simple line plots to visualize the frequency dependence on the detuning
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

import matplotlib.cm as cm

# Prepare data for line plots
colors = cm.viridis(np.linspace(0, 1, len(sorted_detunings)))

for idx, detuning in enumerate(sorted_detunings):
    color = colors[idx]
    fft_list = detuning_fft_dict[detuning]
    avg_fft = np.mean(fft_list, axis=0)
    axs[0].plot(CFG, avg_fft, label=f'Detuning {detuning}', color=color)

    acf_list = detuning_acf_dict[detuning]
    avg_acf = np.mean(acf_list, axis=0)
    axs[1].plot(CLG, avg_acf, label=f'Detuning {detuning}', color=color)

# Plot FFTs
axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
# axs[0].set_yscale('log')
axs[0].set_xlim(-0.02, 0.52)
axs[0].set_title("FFT Magnitude vs Detuning")

# Plot ACFs
axs[1].set_xlabel("Lag")
axs[1].set_title("ACF vs Detuning")

# Add color bar
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=min(sorted_detunings), vmax=max(sorted_detunings)))
sm.set_array([])
fig.colorbar(sm, ax=axs, orientation='vertical', label='Detuning')

plt.show()
