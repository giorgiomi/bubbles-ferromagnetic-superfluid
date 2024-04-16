# This script is used to analyze the data saved by save_data.py (from hdf to csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import import_parameters

# Data
f, seqs, Omega, knT, detuning = import_parameters()
w = 200

for day in np.arange(len(seqs)):
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
        plt.show()

        