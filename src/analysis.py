# This script is used to analyze the data saved by save_data.py (from hdf to csv)
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
        df_center = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/center.csv")
        df_size = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/magnetization.csv")

        b_center = df_center.to_numpy().flatten()
        b_sizeADV = df_size.to_numpy().flatten()
        M = df_M.to_numpy()

        # Plotting the bubble (unsorted)
        fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})
        ax[0].pcolormesh(M, vmin = -1, vmax = +1, cmap = 'RdBu')
        ax[0].set_title('Unsorted shots')
        ax[0].set_xlabel('$x\ [\mu m]$')
        ax[0].set_ylabel('shots')

        # Sorting the bubble
        Zlist = np.argsort(b_sizeADV)
        Z = (M[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        b_sizeADV_sorted = (b_sizeADV[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        b_center_sorted = (b_center[Zlist])[np.where(b_sizeADV[Zlist] > 0)]

        # Shifting
        length = 2 * w
        shift = - b_center_sorted + length/2
        max_shift = np.max(np.abs(shift))
        Z_shifted = np.zeros((len(Z), length + 2 * int(max_shift)))
        for i in np.arange(len(Z_shifted)):
            Z_shifted[i, (int(max_shift + int(shift[i]))) : (int(max_shift) + int(shift[i]) + length)] = Z[i]

        # Plotting the bubble (sorted)
        im = ax[1].pcolormesh(Z_shifted, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('M', rotation=270)
        ax[1].set_title('Sorted shots')
        ax[1].set_xlabel('x')
        fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
        # plt.savefig("thesis/figures/chap2/shot_sorting.png", dpi=500)
        plt.show()

        ## FFT

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

        plt.plot(noise_freq, noise_fft_mean, '-', label='FFT on background noise')
        plt.annotate(f"# of noise shots = {len(M_noise)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        # plt.show()

        # FFT on bubble (inside)
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
