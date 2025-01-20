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
f, seqs, Omega, knT, detuning = importParameters()
w = 200

chosen_days = scriptUsage()

for day in chosen_days:
#for day in np.arange(len(seqs)):
    for seq, seqi in enumerate((seqs[day])):
        df_center = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/center.csv")
        df_size = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/magnetization.csv")
        df_D = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/density.csv")

        b_center = df_center.to_numpy().flatten()
        b_sizeADV = df_size.to_numpy().flatten()
        M = df_M.to_numpy()
        D = df_D.to_numpy()
        n_shots = len(b_center)

        for i in range(19, n_shots):
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))

            # Magnetization profile
            axs[0].plot(M[i], label="Profile", color="tab:grey")
            if b_sizeADV[i] != 0:
                axs[0].fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) >= b_center[i] - b_sizeADV[i]/2) & (np.arange(len(M[i])) <= b_center[i] + b_sizeADV[i]/2), color='tab:red', alpha=0.3, label="Inside bubble")
                axs[0].fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) < b_center[i] - b_sizeADV[i]/2) | (np.arange(len(M[i])) > b_center[i] + b_sizeADV[i]/2), color='tab:blue', alpha=0.1, label="Outside bubble")
                axs[0].text(0.5, 0.9, "Bubble detected", horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            axs[0].legend(loc="lower left")
            axs[0].set_title(f"Magnetization profile of day {day}, sequence {seq}, shot {i}")
            axs[0].set_xlabel("$x\ [\mu$m]")
            axs[0].set_ylabel("$M(x)$")
            axs[0].set_ylim(-1, 1)

            # Density profile
            axs[1].plot(D[i], label="Density", color="tab:green")
            axs[1].legend(loc="lower left")
            axs[1].set_title(f"Density profile of day {day}, sequence {seq}, shot {i}")
            axs[1].set_xlabel("$x\ [\mu$m]")
            axs[1].set_ylabel("$D(x)$")

            plt.tight_layout()
            plt.show()
            
        