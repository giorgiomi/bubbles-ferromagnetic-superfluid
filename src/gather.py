# Gather the bubbles (size > 0) by omega and delta
import pandas as pd
import numpy as np
from util.parameters import importParameters

# Data
w = 200 # Thomas-Fermi radius, always the same

# Print script purpose
print("\nGather all bubbles (size > 0) with same Omega and delta\n")

selected_flag = int(input("Enter 1 for selected, 0 for processed: "))
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
if selected_flag: 
    str = "selected"
else: 
    str = "processed"

Zs = []
sizes = []
centers = []
times = []
omegas = []
dets = []

# Cycle through days and seqs
for day in sel_days:
    for seq in sel_seq[day]:
        size = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/sizeADV.csv", header=None).to_numpy().flatten()
        center = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/center.csv", header=None).to_numpy().flatten()
        time = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/time.csv", header=None).to_numpy().flatten()
        Z = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/magnetization.csv", header=None).to_numpy()

        for i, shot in enumerate(Z):
            if size[i] > 0 and size[i] < 2*w:
                Zs.append(shot)
                centers.append(center[i])
                sizes.append(size[i])
                times.append(time[i])
                omegas.append(Omega[day][seq])
                # dets.append(Detuning[day][seq])

print("Saving gathered data on data/gathered/\n", end="")
np.savetxt(f"data/gathered/center.csv", centers, delimiter=',')
np.savetxt(f"data/gathered/size.csv", sizes, delimiter=',')
np.savetxt(f"data/gathered/time.csv", times, delimiter=',')
np.savetxt(f"data/gathered/omega.csv", omegas, delimiter=',')
# np.savetxt(f"data/gathered/detuning.csv", dets, delimiter=',')
np.savetxt(f"data/gathered/Z.csv", Zs, delimiter=',')