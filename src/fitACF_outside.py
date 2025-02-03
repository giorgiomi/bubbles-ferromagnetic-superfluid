# This script is used to analyze the data outside the bubble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.methods import groupFitACF_outside

# Data
# f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 20

# Print script purpose
print("\nAnalyzing FFT and ACF on OUTSIDE REGION with gathered data\n")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: ")

# Ask for gather flag
gather_flag = input("Enter gather method [omega/time/size/detuning/dE]: ")

# Import bubble data
size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()
# dE = find_minima(omega, detuning, -1150, 1)
dE = np.sqrt(omega*(1150-omega))
# print(dE)
# exit()

if gather_flag == 'omega':
     groupFitACF_outside('omega', omega, 1, Z, size, center, window_len, zero_mean_flag)
elif gather_flag == 'time':
    groupFitACF_outside('time', time, 10, Z, size, center, window_len, zero_mean_flag)
elif gather_flag == 'size':
    groupFitACF_outside('size', size, 10, Z, size, center, window_len, zero_mean_flag)
elif gather_flag == 'detuning':
    groupFitACF_outside('detuning', detuning, 50, Z, size, center, window_len, zero_mean_flag)
elif gather_flag == 'dE':
     groupFitACF_outside('dE', dE, 20, Z, size, center, window_len, zero_mean_flag)
else:
    print("Gather method has to be [omega/time/size/detuning/dE]")