# This script is used to analyze the noise data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

from parameters import import_parameters
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

for day in chosen_days:
#for day in np.arange(len(seqs)):
    for seq, seqi in enumerate((seqs[day])):
        df_size = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/magnetization.csv")
        b_sizeADV = df_size.to_numpy().flatten()
        M = df_M.to_numpy()

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

            # plt.plot(noise_freq_grid, noise_spectrum, alpha=0.03)

        noise_fft_magnitudes = np.array(noise_fft_magnitudes)
        noise_fft_mean = np.mean(noise_fft_magnitudes, axis=0)
        noise_freq = common_noise_freq_grid

        # Colormap
        plt.figure()
        plt.imshow(np.log(noise_fft_magnitudes + 1e-10), aspect='auto', extent=[noise_freq[0], noise_freq[-1], 0, len(M_noise)], origin='lower', cmap='viridis')
        plt.colorbar(label='Log Magnitude')
        plt.title(f"Noise FFT Magnitudes Colormap for day {day}, sequence {seq}")
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