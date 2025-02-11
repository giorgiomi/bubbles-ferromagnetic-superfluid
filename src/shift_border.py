# This script is used to shift the bubbles on their border
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import scriptUsage

selected_flag = int(input("Enter 1 for selected, 0 for all: "))
if selected_flag:
    str = 'selected'
else:
    str = 'processed'
# Data
f, seqs, Omega, knT, detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200
chosen_days = scriptUsage(sel_days)
wind = 50

for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_center_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center_sorted.csv")
        df_Z_sorted = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/Z_sorted.csv")
        df_in_left_sorted = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/in_left_sorted.csv")
        df_in_right_sorted = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/in_right_sorted.csv")
        df_off_sorted = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/off_sorted.csv")

        b_center = df_center_sorted.to_numpy().flatten()
        in_left_sorted = df_in_left_sorted.to_numpy().flatten()
        in_right_sorted = df_in_right_sorted.to_numpy().flatten()
        off_sorted = df_off_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy() - off_sorted[:, np.newaxis]

        # Shifting to RIGHT BORDER
        shift = in_right_sorted - b_center
        Z_shifted_right = np.array([np.roll(Z[i], int(shift[i])) for i in range(len(Z))])
        # Z_shifted_right = Z_shifted_right[:,175:225]

        # Plotting the bubble (shifted RIGHT BORDER)
        fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})
        im = ax[0].pcolormesh(np.arange(Z_shifted_right.shape[1]), np.arange(Z_shifted_right.shape[0]), Z_shifted_right, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z')
        ax[0].set_title('Shifted shots to right border')
        ax[0].set_xlabel(r'$\tilde{x}\ [\mu m]$')

        # Shifting to LEFT BORDER
        shift = in_left_sorted - b_center
        Z_shifted_left = np.array([np.roll(Z[i], int(shift[i])) for i in range(len(Z))])
        # Z_shifted_left = Z_shifted_left[:,175:225]

        # Plotting the bubble (shifted LEFT BORDER)
        im = ax[1].pcolormesh(np.arange(Z_shifted_left.shape[1]), np.arange(Z_shifted_left.shape[0]), Z_shifted_left, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z')
        ax[1].set_title('Shifted shots to left border')
        ax[1].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
        # plt.savefig("thesis/figures/chap2/shot_shifting_border.png", dpi=500)
        plt.show()

        # Plotting all magnetization profiles shifted
        fig, ax = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)
        
        # Plotting all magnetization profiles shifted to the left
        for i in range(len(Z_shifted_left)):
            ax[0].plot(np.arange(Z_shifted_left.shape[1]), Z_shifted_left[i], alpha=0.05)
        ax[0].plot(np.arange(Z_shifted_left.shape[1]), np.mean(Z_shifted_left, axis=0), label='mean', color='black')
        ax[0].set_title('All magnetization profiles shifted left')
        ax[0].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        ax[0].set_ylabel('Magnetization')
        ax[0].set_xlim(0, 250)
        ax[0].legend()

        # Plotting all magnetization profiles shifted to the right
        for i in range(len(Z_shifted_right)):
            ax[1].plot(np.arange(Z_shifted_right.shape[1]), Z_shifted_right[i], alpha=0.05)
        ax[1].plot(np.arange(Z_shifted_right.shape[1]), np.mean(Z_shifted_right, axis=0), label='mean', color='black')
        ax[1].set_title('All magnetization profiles shifted right')
        ax[1].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        ax[1].set_xlim(150, 400)
        ax[1].legend()

        fig.suptitle(f"All magnetization profiles shifted for day {day}, sequence {seq}")
        plt.show()
