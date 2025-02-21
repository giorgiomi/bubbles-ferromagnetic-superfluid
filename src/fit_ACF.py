# This script is used to fit the autocorrelation function (ACF) of the gathered data
import pandas as pd
import numpy as np
from util.methods import groupFitACF
from util.parameters import importParameters

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 20

# Print script purpose
if selected_flag: 
    print("\nFitting ACF on INSIDE/OUTSIDE with SELECTED gathered data (run gather.py first to perform on ALL)")
else:
    print("\nFitting ACF on INSIDE/OUTSIDE with ALL gathered data (run gather.py first to perform on SELECTED)")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = int(input("\nEnter 1 for zero-mean signal ACF, 0 for true signal: "))

# Ask for gather flag
gather_flag = input("\nEnter gather method [omega/time/size/detuning/dE/slope]: ")

# Ask for region flag
region_flag = input("\nEnter region [inside/outside]: ")

# Import bubble data
size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
slope = pd.read_csv(f"data/gathered/slope.csv", header=None).to_numpy().flatten()
time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
in_left = pd.read_csv(f"data/gathered/in_left.csv", header=None).to_numpy().flatten()
in_right = pd.read_csv(f"data/gathered/in_right.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()
# dE = find_minima(omega, detuning, -1150, 1)
dE = np.sqrt(omega*(1150-omega))
# print(dE)
# exit()

if gather_flag == 'omega':
    groupFitACF('omega', omega, omega, 1, Z, window_len, zero_mean_flag, region_flag, in_left, in_right, 1)
elif gather_flag == 'time':
    groupFitACF('time', time, omega, 10, Z, window_len, zero_mean_flag, region_flag, in_left, in_right, 10)
elif gather_flag == 'size':
    groupFitACF('size', size, omega, 10, Z, window_len, zero_mean_flag, region_flag, in_left, in_right, 10)
elif gather_flag == 'detuning':
    print("Detuning not available")
    # groupFitACF('detuning', detuning, 10, Z, window_len, zero_mean_flag, region_flag)
elif gather_flag == 'dE':
    print("dE not available")
    # groupFitACF('dE', dE, omega, 10, Z, window_len, zero_mean_flag, region_flag, in_left, in_right)
elif gather_flag == 'slope':
    print("Detuning not available")
    # groupFitACF('slope', slope, omega, 10, Z, window_len, zero_mean_flag, region_flag, in_left, in_right)
else:
    print("Gather method has to be [omega/time/size/detuning/dE]")
    