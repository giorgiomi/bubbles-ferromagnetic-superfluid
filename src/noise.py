# This script is used to analyze the noise data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
from util.parameters import importParameters
import sys
from util.methods import scriptUsage, quadPlot

# Data
f, seqs, Omega, knT, Detuning = importParameters()
w = 200

chosen_days = scriptUsage()

# Print script purpose
print("\nAnalyzing FFT and ACF on NOISE with magnetization signal\n")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = int(input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: "))

omega_fft_dict = {}
omega_acf_dict = {}

detuning_fft_dict = {}
detuning_acf_dict = {}

for day in chosen_days:
    for seq, seqi in enumerate((seqs[day])):
        # data
        df_size = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/magnetization.csv")
        b_sizeADV = df_size.to_numpy().flatten()
        M = df_M.to_numpy()
        omega = Omega[day][seq]
        detuning = Detuning[day][seq]
        # print(f"Omega = {omega}")

        # FFT on background noise
        M_noise = M[np.where(b_sizeADV == 0)]

        # Common frequency grid for noise
        sampling_rate = 1.0
        common_noise_freq_grid = rfftfreq(len(M_noise[0]), d=sampling_rate)

        noise_fft_magnitudes = []
        noise_acf_values = []
        for shot in M_noise:
            if zero_mean_flag:
                noise_fft = rfft(shot - np.mean(shot)) ## doing FFT on zero-mean signal
                noise_acf = correlate(shot - np.mean(shot), shot - np.mean(shot), mode='full')
            else:
                noise_fft = rfft(shot) ## doing FFT on true signal
                noise_acf = correlate(shot, shot, mode='full')
            
            noise_spectrum = np.abs(noise_fft)
            noise_acf /= np.max(noise_acf)
            noise_freq_grid = rfftfreq(len(shot), d=1.0)

            # Interpolate onto the common frequency grid
            interpolated_noise_magnitude = np.interp(common_noise_freq_grid, noise_freq_grid, noise_spectrum)
            noise_fft_magnitudes.append(interpolated_noise_magnitude)
            
            # Interpolate onto the common lag grid
            noise_acf_values.append(noise_acf)
            lag_grid = np.arange(-len(shot) + 1, len(shot))

        noise_fft_magnitudes = np.array(noise_fft_magnitudes)
        noise_fft_mean = np.mean(noise_fft_magnitudes, axis=0)
        noise_freq = common_noise_freq_grid

        noise_acf_values = np.array(noise_acf_values)
        noise_acf_mean = np.mean(noise_acf_values, axis=0)

        # Store FFT results by omega
        if omega not in omega_fft_dict:
            omega_fft_dict[omega] = []
        omega_fft_dict[omega].append(noise_fft_mean)

        # Store ACF results by omega
        if omega not in omega_acf_dict:
            omega_acf_dict[omega] = []
        omega_acf_dict[omega].append(noise_acf_mean)

        # Store FFT results by detuning
        if detuning not in detuning_fft_dict:
            detuning_fft_dict[detuning] = []
        detuning_fft_dict[detuning].append(noise_fft_mean)

        # Store ACF results by detuning
        if detuning not in detuning_acf_dict:
            detuning_acf_dict[detuning] = []
        detuning_acf_dict[detuning].append(noise_acf_mean)

        if int(sys.argv[1]) != -1:
            fig = quadPlot(day, seq, M_noise, "noise", noise_freq, lag_grid, 
                     noise_fft_magnitudes, noise_fft_mean, noise_acf_values, noise_acf_mean, 0)
            fig.canvas.manager.set_window_title('Magnetization data')
            plt.show()

# Average all FFTs with the same omega
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Sort the omega keys
sorted_omegas = sorted(omega_fft_dict.keys())

# Plot FFTs
for omega in sorted_omegas:
    fft_list = omega_fft_dict[omega]
    avg_fft = np.mean(fft_list, axis=0)
    axs[0].plot(noise_freq[1:], avg_fft[1:], '-', label=fr'$\Omega = {omega}$')

axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
axs[0].set_yscale('log')
axs[0].set_xlim(-0.02, 0.52)
axs[0].legend()
axs[0].set_title("Average noise FFTs")

# Plot ACFs
for omega in sorted_omegas:
    acf_list = omega_acf_dict[omega]
    avg_acf = np.mean(acf_list, axis=0)
    axs[1].plot(lag_grid, avg_acf, '-', label=fr'$\Omega = {omega}$ Hz')

axs[1].set_xlabel("Lag")
axs[1].legend()
axs[1].set_title("Average noise ACFs")

plt.tight_layout()
# plt.savefig(f"thesis/figures/chap2/noise_fft_acf_avg.png", dpi=500)
plt.show()



# Average all FFTs with the same detuning
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Sort the detuning keys
sorted_detunings = sorted(detuning_fft_dict.keys())

# Prepare data for colormap FFTs
fft_matrix = np.array([np.mean(detuning_fft_dict[detuning], axis=0) for detuning in sorted_detunings])
im1 = axs[0].imshow(np.log(fft_matrix[:, 1:] + 1e-10), aspect='auto', extent=[noise_freq[1], noise_freq[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='viridis')
fig.colorbar(im1, ax=axs[0], label='Log Magnitude')
axs[0].set_title("Average noise FFTs")
axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
axs[0].set_ylabel("$\delta$")

# Prepare data for colormap ACFs
acf_matrix = np.array([np.mean(detuning_acf_dict[detuning], axis=0) for detuning in sorted_detunings])
im2 = axs[1].imshow(acf_matrix, aspect='auto', extent=[lag_grid[0], lag_grid[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='viridis')
fig.colorbar(im2, ax=axs[1], label='Magnitude')
axs[1].set_title("Average noise ACFs")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("$\delta$")

plt.tight_layout()
plt.show()
