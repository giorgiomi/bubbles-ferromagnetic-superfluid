# This script is used to visualize some magnetization profiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.parameters import importParameters
from util.methods import scriptUsage


# Data
f, seqs, Omega, knT, detuning, sel_days, sel_seq = importParameters()
w = 200

chosen_days = scriptUsage()

for day in sel_days:
#for day in np.arange(len(seqs)):
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_center = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/center.csv")
        df_size = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/magnetization.csv")
        df_D = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/density.csv")
        df_in_l = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/in_left.csv")
        df_in_r = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/in_right.csv")
        df_out_l = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/out_left.csv")
        df_out_r = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/out_right.csv")

        b_center = df_center.to_numpy().flatten()
        b_sizeADV = df_size.to_numpy().flatten()
        in_l = df_in_l.to_numpy().flatten()
        in_r = df_in_r.to_numpy().flatten()
        out_l = df_out_l.to_numpy().flatten()
        out_r = df_out_r.to_numpy().flatten()
        M = df_M.to_numpy()
        D = df_D.to_numpy()
        n_shots = len(b_center)

        for i in range(0, n_shots):
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))

            # Magnetization profile
            axs[0].plot(M[i], label="Profile", color="tab:grey")
            if b_sizeADV[i] != 0:
                axs[0].fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) >= in_l[i]) & (np.arange(len(M[i])) <= in_r[i]), color='tab:red', alpha=0.2, label="Inside bubble")
                axs[0].fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) < out_l[i]) | (np.arange(len(M[i])) > out_r[i]), color='tab:blue', alpha=0.2, label="Outside bubble")
                axs[0].fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) >= in_r[i]) & (np.arange(len(M[i])) <= out_r[i]), color='gray', alpha=0.2, label="No-man zone")
                axs[0].fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) >= out_l[i]) & (np.arange(len(M[i])) <= in_l[i]), color='gray', alpha=0.2)

                axs[1].fill_between(np.arange(len(D[i])), D[i], where=(np.arange(len(M[i])) >= in_l[i]) & (np.arange(len(M[i])) <= in_r[i]), color='tab:red', alpha=0.2, label="Inside bubble")
                axs[1].fill_between(np.arange(len(D[i])), D[i], where=(np.arange(len(M[i])) < out_l[i]) | (np.arange(len(M[i])) > out_r[i]), color='tab:blue', alpha=0.2, label="Outside bubble")
                axs[1].fill_between(np.arange(len(D[i])), D[i], where=(np.arange(len(M[i])) >= in_r[i]) & (np.arange(len(M[i])) <= out_r[i]), color='gray', alpha=0.2, label="No-man zone")
                axs[1].fill_between(np.arange(len(D[i])), D[i], where=(np.arange(len(M[i])) >= out_l[i]) & (np.arange(len(M[i])) <= in_l[i]), color='gray', alpha=0.2)

                axs[0].text(0.5, 0.9, "Bubble detected", horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            axs[0].legend(loc="lower left")
            axs[0].set_title(f"Magnetization profile of day {day}, sequence {seq}, shot {i}")
            axs[0].set_xlabel("$x\ [\mu$m]")
            axs[0].set_ylabel("$M(x)$")
            axs[0].set_ylim(-1, 1)

            # Density profile
            axs[1].plot(D[i], label="Density", color="gray")
            axs[1].legend(loc="lower left")
            axs[1].set_title(f"Density profile of day {day}, sequence {seq}, shot {i}")
            axs[1].set_xlabel("$x\ [\mu$m]")
            axs[1].set_ylabel("$D(x)$")

            plt.tight_layout()
            plt.show()
            
        