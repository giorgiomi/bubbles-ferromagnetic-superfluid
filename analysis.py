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

if int(sys.argv[1]) == -1:
    chosen_days = np.arange(len(seqs))
else:
    chosen_days = [int(sys.argv[1])]

for day in chosen_days:
#for day in np.arange(len(seqs)):
    for seq, seqi in enumerate((seqs[day])):
        df_center = pd.read_csv(f"data/day_{day}/seq_{seq}/center.csv")
        df_size = pd.read_csv(f"data/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/day_{day}/seq_{seq}/magnetization.csv")

        b_center = df_center.to_numpy().flatten()
        b_sizeADV = df_size.to_numpy().flatten()
        M = df_M.to_numpy()

        # Plotting the bubble (unsorted)
        fig, ax = plt.subplots(figsize = (10, 5), ncols = 2)
        ax[0].pcolormesh(M, vmin = -1, vmax = +1, cmap = 'RdBu')
        ax[0].set_title('Unsorted bubble')
        ax[0].set_xlabel('x')
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

        #Plotting the bubble (sorted)
        im = ax[1].pcolormesh(Z_shifted, vmin=-1, vmax=1, cmap='RdBu')
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        ax[1].set_title('Sorted bubble')
        ax[1].set_xlabel('x')
        fig.suptitle(f"Day {day}, sequence {seq}")
        plt.show()

        ## FFT
        # FFT on background noise
        M_noise = M[np.where(b_sizeADV == 0)]

        noise_fft_mean = np.zeros(201)
        for shot in M_noise:
            noise_fft = rfft(shot)
            noise_freq = rfftfreq(len(shot), d = 1)
            #plt.plot(noise_freq, np.abs(noise_fft))
            noise_fft_mean += np.abs(noise_fft)
        noise_fft_mean /= len(M_noise)

        plt.plot(noise_freq, noise_fft_mean, label='FFT on background noise')
        plt.grid(True, which='both')
        plt.title(f"Day {day}, sequence {seq}")
        plt.yscale('log')
        plt.legend()
        plt.annotate(f"# of noise shots = {len(M_noise)}", xy=(0.8, 0.85), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='square', facecolor='white', edgecolor='black'))
        plt.show()

        