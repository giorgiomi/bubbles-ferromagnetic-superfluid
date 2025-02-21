# Gather the bubbles (size > 0) by omega and delta
import pandas as pd
import numpy as np
from util.parameters import importParameters

# Data
w = 200 # Thomas-Fermi radius, always the same

# Print script purpose
print("\nGather all bubbles (size > 0) with same Omega and delta\n")

selected_flag = int(input("Enter 1 for selected, 0 for all: "))
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
if selected_flag: 
    str = "selected"
else: 
    str = "processed"

Zs = []
sizes = []
centers = []
slopes = []
exp_lefts = []
exp_rights = []
in_lefts = []
in_rights = []
times = []
omegas = []
dets = []

# Cycle through days and seqs
for day in sel_days:
    for seq in sel_seq[day]:
        size = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/sizeADV.csv", header=None).to_numpy().flatten()
        center = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/center.csv", header=None).to_numpy().flatten()
        slope = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/slope.csv", header=None).to_numpy().flatten()
        time = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/time.csv", header=None).to_numpy().flatten()
        Z = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/magnetization.csv", header=None).to_numpy()
        exp_left = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/exp_left.csv", header=None).to_numpy().flatten()
        exp_right = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/exp_right.csv", header=None).to_numpy().flatten()
        in_left = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/in_left.csv", header=None).to_numpy().flatten()
        in_right = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/in_right.csv", header=None).to_numpy().flatten()

        for i, shot in enumerate(Z):
            if size[i] >= 0 and size[i] < 2*w and in_left[i] < w and in_right[i] > w:
                Zs.append(shot)
                centers.append(center[i])
                slopes.append(slope[i])
                sizes.append(size[i])
                times.append(time[i])
                omegas.append(Omega[day][seq])
                exp_lefts.append(exp_left[i])
                exp_rights.append(exp_right[i])
                in_lefts.append(in_left[i])
                in_rights.append(in_right[i])
                if selected_flag:
                    dets.append(Detuning[day][seq])

print("Saving gathered data on data/gathered/\n", end="")
np.savetxt(f"data/gathered/center.csv", centers, delimiter=',')
np.savetxt(f"data/gathered/size.csv", sizes, delimiter=',')
np.savetxt(f"data/gathered/slope.csv", slopes, delimiter=',')
np.savetxt(f"data/gathered/exp_left.csv", exp_lefts, delimiter=',')
np.savetxt(f"data/gathered/exp_right.csv", exp_rights, delimiter=',')
np.savetxt(f"data/gathered/in_left.csv", in_lefts, delimiter=',')
np.savetxt(f"data/gathered/in_right.csv", in_rights, delimiter=',')
np.savetxt(f"data/gathered/time.csv", times, delimiter=',')
np.savetxt(f"data/gathered/omega.csv", omegas, delimiter=',')
np.savetxt(f"data/gathered/Z.csv", Zs, delimiter=',')
if selected_flag:
    np.savetxt(f"data/gathered/detuning.csv", dets, delimiter=',')
    np.savetxt(f"data/gathered/selected.csv", [1])
else:
    np.savetxt(f"data/gathered/selected.csv", [0])