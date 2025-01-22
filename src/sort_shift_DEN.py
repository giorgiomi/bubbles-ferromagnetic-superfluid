# This script is used to analyze the data saved by save_data.py (from hdf to csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import scriptUsage

# Data
f, seqs, Omega, knT, detuning, sel_days, sel_seq = importParameters()
w = 200
chosen_days = scriptUsage()

for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_center = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center.csv")
        df_size = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV.csv")
        df_D = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/density.csv")


        b_center = df_center.to_numpy().flatten()
        b_sizeADV = df_size.to_numpy().flatten()
        D = df_D.to_numpy()

        # Plotting the bubble (unsorted)
        # fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})
        # ax[0].pcolormesh(D, vmin = 0, vmax = +800, cmap = 'Purples')
        # ax[0].set_title('Unsorted shots')
        # ax[0].set_xlabel('$x\ [\mu m]$')
        # ax[0].set_ylabel('shots')

        # Sorting the bubble
        Zlist = np.argsort(b_sizeADV)
        D = (D[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        b_sizeADV_sorted = (b_sizeADV[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        b_center_sorted = (b_center[Zlist])[np.where(b_sizeADV[Zlist] > 0)]

        # Shifting
        length = 2 * w
        shift = - b_center_sorted + length/2
        max_shift = np.max(np.abs(shift))
        D_shifted = np.zeros((len(D), length + 2 * int(max_shift)))
        for i in np.arange(len(D_shifted)):
            D_shifted[i, (int(max_shift + int(shift[i]))) : (int(max_shift) + int(shift[i]) + length)] = D[i]
        
        # save Z, Z_shifted on file
        # np.savetxt(f"data/selected/day_{day}/seq_{seq}/density_sorted.csv", D, delimiter=',')

        # Plotting the bubble (sorted)
        # im = ax[1].pcolormesh(D_shifted, vmin=0, vmax=800, cmap='Purples')
        # cbar = fig.colorbar(im, cax=ax[2])
        # cbar.set_label('D')
        # ax[1].set_title('Sorted shots')
        # ax[1].set_xlabel('x')
        # fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
        # # plt.savefig("thesis/figures/chap2/shot_sorting.png", dpi=500)
        # plt.show()

