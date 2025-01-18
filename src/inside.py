# This script is used to analyze the data inside the bubble
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

for day in chosen_days:
    for seq, seqi in enumerate((seqs[day])):
        df_size_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV_sorted.csv")
        df_center_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/center_sorted.csv")
        df_Z_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/Z_sorted.csv")

        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        b_center_sorted = df_center_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # FFT on bubble (inside)
        sampling_rate = 1.0
        max_length = int(max(b_sizeADV_sorted) + 1)

        # Common frequency grid
        common_freq_grid = rfftfreq(max_length, d=sampling_rate)

        inside_fft_magnitudes = []
        # cycle through ordered shots from half to end
        # for i in range(int(len(Z)/2), len(Z)):

        # cycle through ordered shots from beginning to end
        for i in range(len(Z)):
            y = Z[i] 
            center = b_center_sorted[i]
            extra_width = -20 # to change where FFT is done
            width = b_sizeADV_sorted[i] + extra_width
            inside = y[int(center - width/2):int(center + width/2)]

            # plt.plot(range(len(inside)), inside)
            # plt.show()

            # compute FFT of the inside
            N = len(inside)
            freq_grid = rfftfreq(N, d=sampling_rate)
            inside_fft = rfft(inside)
            inside_spectrum = np.abs(inside_fft)
            
            # Interpolate onto the common frequency grid
            interpolated_magnitude = np.interp(common_freq_grid, freq_grid, inside_spectrum)
            inside_fft_magnitudes.append(interpolated_magnitude)

            ## plot each individual FFT to debug
            # plt.figure()
            # plt.plot(freq_grid, inside_spectrum, alpha=0.03)
            # plt.yscale('log')
            # plt.show()

        inside_fft_magnitudes = np.array(inside_fft_magnitudes)
        inside_fft_mean = np.mean(inside_fft_magnitudes, axis=0)

        # Colormap
        plt.figure()
        plt.imshow(np.log(inside_fft_magnitudes + 1e-10), aspect='auto', extent=[common_freq_grid[0], common_freq_grid[-1], 0, len(Z)], origin='lower', cmap='viridis')
        plt.colorbar(label='Log Magnitude')
        plt.title(f"Inside FFT for day {day}, sequence {seq}")
        plt.xlabel("Frequency")
        plt.ylabel("Shot number")
        plt.show()

        # Average
        plt.figure()
        plt.plot(common_freq_grid, inside_fft_mean, '-', label='FFT on inside region')
        plt.title(f"FFT analysis of day {day}, sequence {seq}")
        plt.xlabel("f")
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlim(-0.02, 0.52)
        plt.legend()
        # change label depending on how many shots are analyzed
        plt.annotate(f"# of inside shots = {int(len(Z))}", xy=(0.8, 0.65), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        plt.show()