# This script is used to look at some bubble features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same

# Print script purpose
print("Look at some bubble features")

# Import bubble data
size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
slope = pd.read_csv(f"data/gathered/slope.csv", header=None).to_numpy().flatten()
time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

# plot bubbles
fig, ax = plt.subplots(figsize=(10, 5), ncols=2)
ax[0].pcolormesh(Z, vmin=-1, vmax=+1, cmap='RdBu')
ax[0].set_xlabel('$x\ [\mu m]$')
ax[0].set_ylabel('shots')
ax[0].set_title('Unsorted')

# Sort Z by cat
sorted_indices = np.argsort(size)
Z_sorted = Z[sorted_indices]

# Display the ordered shots
ax[1].pcolormesh(Z_sorted, vmin=-1, vmax=1, cmap='RdBu')
ax[1].set_xlabel('$x\ [\mu m]$')
ax[1].set_ylabel('shots')
ax[1].set_title('Sorted')
plt.show()

# Look at size vs time
plt.figure()
plt.plot(size, slope, '.')
plt.yscale('log')
plt.show()