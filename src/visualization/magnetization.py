# This script is used to visualize some magnetization profiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        n_shots = len(b_center)

        for i in range(19, n_shots):
            plt.plot(M[i], label="Profile", color="tab:grey")
            
            if b_sizeADV[i] != 0:
                plt.fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) >= b_center[i] - b_sizeADV[i]/2) & (np.arange(len(M[i])) <= b_center[i] + b_sizeADV[i]/2), color='tab:red', alpha=0.3, label="Inside bubble")
                plt.fill_between(np.arange(len(M[i])), M[i], where=(np.arange(len(M[i])) < b_center[i] - b_sizeADV[i]/2) | (np.arange(len(M[i])) > b_center[i] + b_sizeADV[i]/2), color='tab:blue', alpha=0.1, label="Outside bubble")

                plt.text(0.5, 0.9, "Bubble detected", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            plt.legend(loc="lower left")
            plt.title(f"Magnetization profile of day {day}, sequence {seq}, shot {i}")
            plt.xlabel("$x\ [\mu$m]")
            plt.ylabel("$M(x)$")
            plt.ylim(-1, 1)
            plt.show()
        