# This script is used to analyze the data outside the bubble
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from util.methods import scriptUsage, quadPlot, computeFFT_ACF
from scipy.fft import rfft, rfftfreq
from util.parameters import importParameters

# Data
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 40

chosen_days = scriptUsage()

# Print script purpose
print("\nAnalyzing FFT and ACF on OUTSIDE REGION with magnetization signal\n")

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
        df_out_left_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/out_left_sorted.csv", header=None)
        df_out_right_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/out_right_sorted.csv", header=None)

        out_left_sorted = df_out_left_sorted.to_numpy().flatten()
        out_right_sorted = df_out_right_sorted.to_numpy().flatten()
        max_size = max(int(max(out_left_sorted) + 1), int(max(2*w - out_right_sorted) + 1))
        if max_size > max_length and max_size < 2*w:
            max_length = max_size

# Common frequency grid for FFT
CFG = rfftfreq(max_length, d=sampling_rate)

# Common lag grid for ACF
CLG = np.arange(-window_len, window_len+1)

for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_size_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV_sorted.csv", header=None)
        df_center_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center_sorted.csv", header=None)
        df_out_left_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/out_left_sorted.csv", header=None)
        df_out_right_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/out_right_sorted.csv", header=None)
        df_Z_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/Z_sorted.csv", header=None)
        
        omega = Omega[day][seq]
        detuning = Detuning[day][seq]

        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        b_center_sorted = df_center_sorted.to_numpy().flatten()
        out_left_sorted = df_out_left_sorted.to_numpy().flatten()
        out_right_sorted = df_out_right_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # FFT on outside region
        # if max_length >= w: continue

        outside_left_fft_magnitudes = []
        outside_left_acf_values = []
        outside_right_fft_magnitudes = []
        outside_right_acf_values = []

        # cycle through ordered shots from beginning to end
        for i in range(len(Z)):
            y = Z[i] 
            left = int(out_left_sorted[i])
            right = int(out_right_sorted[i])
            outside_left = y[:left]
            outside_right = y[right:]
            
            # plt.figure()
            # plt.plot(y)
            # plt.plot(np.arange(0, left), outside_left)
            # plt.plot(np.arange(right, 2*w), outside_right)
            # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
            # plt.show()

            N_left = len(outside_left)
            N_right= len(outside_right)

            if N_left > 0 and N_right > 0:
                # compute FFT and ACF of the left
                outside_left_fft_magnitudes, outside_left_acf_values = computeFFT_ACF(zero_mean_flag, outside_left, CFG, CLG, outside_left_fft_magnitudes, outside_left_acf_values)

                # compute FFT and ACF of the right
                outside_right_fft_magnitudes, outside_right_acf_values = computeFFT_ACF(zero_mean_flag, outside_right, CFG, CLG, outside_right_fft_magnitudes, outside_right_acf_values)

        outside_left_fft_magnitudes = np.array(outside_left_fft_magnitudes)
        outside_right_fft_magnitudes = np.array(outside_right_fft_magnitudes)
        outside_fft_magnitudes = (outside_left_fft_magnitudes + outside_right_fft_magnitudes)/2
        outside_fft_mean = np.mean(outside_fft_magnitudes, axis=0)
       
        outside_left_acf_values = np.array(outside_left_acf_values)
        outside_right_acf_values = np.array(outside_right_acf_values)
        outside_acf_values = (outside_left_acf_values + outside_right_acf_values)/2
        outside_acf_mean = np.mean(outside_acf_values, axis=0)

        # Store the FFT and ACF magnitudes by omega
        if omega not in omega_fft_dict:
            omega_fft_dict[omega] = []
        omega_fft_dict[omega].append(outside_fft_mean)
        if omega not in omega_acf_dict:
            omega_acf_dict[omega] = []
        omega_acf_dict[omega].append(outside_acf_mean)

        # Store the FFT and ACF magnitudes by detuning
        if detuning not in detuning_fft_dict:
            detuning_fft_dict[detuning] = []
        detuning_fft_dict[detuning].append(outside_fft_mean)
        if detuning not in detuning_acf_dict:
            detuning_acf_dict[detuning] = []
        detuning_acf_dict[detuning].append(outside_acf_mean)

        if int(sys.argv[1]) != -1:
            fig = quadPlot(day, seq, Z, "outside", CFG, CLG, 
                     outside_fft_magnitudes, outside_fft_mean, outside_acf_values, outside_acf_mean, 0)
            fig.canvas.manager.set_window_title('Magnetization data')
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
axs[0].set_title("Average outside FFTs")

# Plot ACFs    
axs[1].set_xlabel("lag")
axs[1].legend()
axs[1].set_title("Average outside ACFs")

plt.tight_layout()
# plt.savefig("thesis/figures/chap2/outside_fft_acf_avg.png", dpi=500)
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
axs[0].set_title("Average outside FFTs")
axs[0].set_xlabel(r"$k/(2\pi) [1/\mu m]$")
axs[0].set_ylabel("$\delta$")

# Plot ACF colormap
im2 = axs[1].imshow(acf_matrix, aspect='auto', extent=[CLG[0], CLG[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im2, ax=axs[1], label='ACF')
axs[1].set_title("Average outside ACFs")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("$\delta$")

plt.tight_layout()
plt.show()