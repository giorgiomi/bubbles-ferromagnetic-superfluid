# This script is used to look at some bubble features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import AZcorr
from util.functions import corrGauss
from scipy.optimize import curve_fit

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
window_len = 30
n_blocks = 10
CLG = np.arange(0, window_len+1)
tr_idx = window_len+1

# Print script purpose
print("FIT single ACF profiles and then taking the mean values")
ZMF = int(input("1 for zero-mean, 0 for true: "))

# Import bubble data
raw_size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
raw_center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
raw_slope = pd.read_csv(f"data/gathered/slope.csv", header=None).to_numpy().flatten()
raw_exp_left = pd.read_csv(f"data/gathered/exp_left.csv", header=None).to_numpy().flatten()
raw_exp_right = pd.read_csv(f"data/gathered/exp_right.csv", header=None).to_numpy().flatten()
raw_in_left = pd.read_csv(f"data/gathered/in_left.csv", header=None).to_numpy().flatten()
raw_in_right = pd.read_csv(f"data/gathered/in_right.csv", header=None).to_numpy().flatten()
raw_time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
raw_omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
raw_Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

# omega_vals = np.unique(omega)
omega_vals = [400, 600, 800]
omega_fix = {300: 300, 400: 600, 600: 400, 800:800} #?? what is going on

# create figures
# fig = plt.figure(figsize=(12,8))
# ax = [plt.subplot(321), plt.subplot(323), plt.subplot(325), plt.subplot(122)]

for om in omega_vals:
    # filter shots with omega = om
    indices = np.where(raw_omega == om)[0]
    Z = raw_Z[indices]
    size = raw_size[indices]
    center = raw_center[indices]
    time = raw_time[indices]
    exp_left = raw_exp_left[indices]
    exp_right = raw_exp_right[indices]
    in_left = raw_in_left[indices]
    in_right = raw_in_right[indices]
    exp_width = (np.array(exp_left) + np.array(exp_right))/2

    for i, shot in enumerate(Z):
        inside = shot[int(in_left[i]):int(in_right[i])]
        if len(inside) > 4*window_len:
            acf = AZcorr(inside, int(window_len), ZMF)
            [l1, off, l2], pcorr = curve_fit(corrGauss, CLG[:tr_idx], acf[:tr_idx], p0=[2, -0.1, 2], bounds=((0, -1, 0), (20, 1, 20)))
            [dl1, doff, dl2] = np.sqrt(np.diag(pcorr))
            chi_sq = np.sum((acf - corrGauss(CLG, l1, off, l2))**2)
            print(l1, off, l2, chi_sq)

            plt.plot(CLG, acf, '-', label='data')
            plt.plot(CLG, corrGauss(CLG, l1, off, l2), '-', label='fit')
            plt.plot(CLG, (1-off)*np.exp(-(CLG/l1)**2)+off, '-', label='gauss')

            plt.axvline(l1, color='tab:red')
            plt.axvline(l2, color='tab:green')
            plt.legend()
            plt.show()
            # plt.errorbar(time[i], l1, yerr=dl1, color='tab:blue', fmt='o', capsize=2)
    plt.title(f"$\Omega_R/2\pi = {om}$")
    plt.xscale('log')
    plt.show()
