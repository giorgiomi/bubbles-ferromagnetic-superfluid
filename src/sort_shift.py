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
        
        # save Z, Z_shifted on file
        # np.savetxt(f"data/processed/day_{day}/seq_{seq}/Z_shifted.csv", Z_shifted, delimiter=',')
        # np.savetxt(f"data/processed/day_{day}/seq_{seq}/Z_sorted.csv", Z, delimiter=',')
        np.savetxt(f"data/processed/day_{day}/seq_{seq}/center_sorted.csv", b_center_sorted, delimiter=',')
        np.savetxt(f"data/processed/day_{day}/seq_{seq}/sizeADV_sorted.csv", b_sizeADV_sorted, delimiter=',')

        # Plotting the bubble (sorted)
        # im = ax[1].pcolormesh(Z_shifted, vmin=-1, vmax=1, cmap='RdBu')
        # cbar = fig.colorbar(im, cax=ax[2])
        # cbar.set_label('M', rotation=270)
        # ax[1].set_title('Sorted shots')
        # ax[1].set_xlabel('x')
        # fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
        # # plt.savefig("thesis/figures/chap2/shot_sorting.png", dpi=500)
        # plt.show()

