# This script is used to shift the bubbles on their border
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
        df_center_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center_sorted.csv")
        df_size_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV_sorted.csv")
        df_Z_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/Z_sorted.csv")

        b_center = df_center_sorted.to_numpy().flatten()
        b_size = df_size_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # Shifting to RIGHT BORDER
        length = 2 * w
        shift = - b_center + w - b_size/2
        max_shift = np.max(np.abs(shift))
        x0 = max(w + b_size/2)
        Z_shifted = np.zeros((len(Z), length + 2 * int(max_shift)))
        for i in np.arange(len(Z_shifted)):
            Z_shifted[i, (int(max_shift + int(shift[i]))) : (int(max_shift) + int(shift[i]) + length)] = Z[i]

        # Plotting the bubble (shifted RIGHT BORDER)
        fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})
        im = ax[0].pcolormesh(np.arange(Z_shifted.shape[1]) - x0, np.arange(Z_shifted.shape[0]), Z_shifted, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z')
        ax[0].set_title('Shifted shots to right border')
        ax[0].set_xlabel(r'$\tilde{x}\ [\mu m]$')

        # Shifting to LEFT BORDER
        length = 2 * w
        shift = - b_center + w + b_size/2
        max_shift = np.max(np.abs(shift))
        x0 = max(w + b_size/2)
        Z_shifted = np.zeros((len(Z), length + 2 * int(max_shift)))
        for i in np.arange(len(Z_shifted)):
            Z_shifted[i, (int(max_shift + int(shift[i]))) : (int(max_shift) + int(shift[i]) + length)] = Z[i]

        # Plotting the bubble (shifted LEFT BORDER)
        im = ax[1].pcolormesh(np.arange(Z_shifted.shape[1]) - x0, np.arange(Z_shifted.shape[0]), Z_shifted, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z')
        ax[1].set_title('Shifted shots to left border')
        ax[1].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
        plt.savefig("thesis/figures/chap2/shot_shifting_border.png", dpi=500)
        plt.show()

        # Plotting all magnetization profiles shifted
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(Z)):
            ax.plot(np.arange(Z_shifted.shape[1]) - x0, Z_shifted[i], alpha=0.05)
        ax.plot(np.arange(Z_shifted.shape[1]) - x0, np.mean(Z_shifted, axis=0), label='mean')
        
        ax.set_title('All magnetization profiles shifted')
        ax.set_xlabel(r'$\tilde{x}\ [\mu m]$')
        ax.set_ylabel('Magnetization')
        ax.legend()
        # plt.savefig(f"thesis/figures/chap2/all_shifted_profiles_day_{day}_seq_{seq}.png", dpi=500)
        plt.show()

