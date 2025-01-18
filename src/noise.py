# This script is used to analyze the noise data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

from util.parameters import import_parameters
import sys

# Data
f, seqs, Omega, knT, detuning = import_parameters()
w = 200

if len(sys.argv) > 1:
    if int(sys.argv[1]) == -1:
        chosen_days = np.arange(len(seqs))
    else:
        chosen_days = [int(sys.argv[1])]
else:
    print(f"Usage: {sys.argv[0]} <chosen_days>\t use chosen_days = -1 for all")
    exit()

omega_fft_dict = {}

for day in chosen_days:
    for seq, seqi in enumerate((seqs[day])):
        # data
        df_size = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/magnetization.csv")
        b_sizeADV = df_size.to_numpy().flatten()
        M = df_M.to_numpy()
        omega = Omega[day][seq]
        # print(f"Omega = {omega}")

        # FFT on background noise
        M_noise = M[np.where(b_sizeADV == 0)]

        # Common frequency grid for noise
        sampling_rate = 1.0
        common_noise_freq_grid = rfftfreq(len(M_noise[0]), d=sampling_rate)

        noise_fft_magnitudes = []
        for shot in M_noise:
            noise_fft = rfft(shot)
            noise_spectrum = np.abs(noise_fft)
            noise_freq_grid = rfftfreq(len(shot), d=1.0)
            
            # Interpolate onto the common frequency grid
            interpolated_noise_magnitude = np.interp(common_noise_freq_grid, noise_freq_grid, noise_spectrum)
            noise_fft_magnitudes.append(interpolated_noise_magnitude)

        noise_fft_magnitudes = np.array(noise_fft_magnitudes)
        noise_fft_mean = np.mean(noise_fft_magnitudes, axis=0)
        noise_freq = common_noise_freq_grid

        # Store FFT results by omega
        if omega not in omega_fft_dict:
            omega_fft_dict[omega] = []
        omega_fft_dict[omega].append(noise_fft_mean)

        if int(sys.argv[1]) != -1:
            # Colormap
            plt.figure()
            plt.imshow(np.log(noise_fft_magnitudes + 1e-10), aspect='auto', extent=[noise_freq[0], noise_freq[-1], 0, len(M_noise)], origin='lower', cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.title(f"Noise FFT of day {day}, sequence {seq}")
            plt.xlabel("Frequency")
            plt.ylabel("Shot number")
            plt.show()
            
            # Average
            plt.figure()
            plt.plot(noise_freq, noise_fft_mean, '-', label='FFT on background noise')
            plt.annotate(f"# of noise shots = {len(M_noise)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            plt.title(f"FFT analysis of day {day}, sequence {seq}")
            plt.xlabel("f")
            plt.yscale('log')
            plt.xlim(-0.02, 0.52)
            plt.legend()
            plt.show()

# Average all FFTs with the same omega
plt.figure()

# Sort the omega keys
sorted_omegas = sorted(omega_fft_dict.keys())

for omega in sorted_omegas:
    fft_list = omega_fft_dict[omega]
    avg_fft = np.mean(fft_list, axis=0)
    plt.plot(noise_freq, avg_fft, '-', label=fr'$\Omega = {omega}$')
    
plt.xlabel("f")
plt.yscale('log')
plt.xlim(-0.02, 0.52)
plt.legend()
plt.title("Average noise FFTs")
# plt.savefig("thesis/figures/chap2/noiseFFT.png", dpi=500)
plt.show()

# Colormap for average FFTs, doesn't look to good
# plt.figure()
# all_ffts = np.array([np.mean(fft_list, axis=0) for fft_list in omega_fft_dict.values()])
# omega_values = list(omega_fft_dict.keys())
# plt.imshow(np.log(all_ffts + 1e-10), aspect='auto', extent=[noise_freq[0], noise_freq[-1], omega_values[0], omega_values[-1]], origin='lower', cmap='viridis')
# plt.colorbar(label='Log Magnitude')
# plt.title("Colormap of Average FFTs")
# plt.xlabel("Frequency")
# plt.ylabel("Omega")
# # plt.yticks(ticks=np.arange(len(omega_values)), labels=[f'{omega:.2f}' for omega in omega_values])
# plt.show()