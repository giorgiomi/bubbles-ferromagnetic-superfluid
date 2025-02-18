# This script is used to look at some bubble features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
n_blocks = 10

# Print script purpose
print("Look at some bubble features")

# Import bubble data
raw_size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
raw_center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
raw_slope = pd.read_csv(f"data/gathered/slope.csv", header=None).to_numpy().flatten()
raw_exp_left = pd.read_csv(f"data/gathered/exp_left.csv", header=None).to_numpy().flatten()
raw_exp_right = pd.read_csv(f"data/gathered/exp_right.csv", header=None).to_numpy().flatten()
raw_time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
raw_omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
raw_Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

# omega_vals = np.unique(omega)
omega_vals = [300, 400, 600, 800]
omega_fix = {300: 300, 400: 600, 600: 400, 800:800} #?? what is going on

# create figures
fig = plt.figure(figsize=(12,8))
ax = [plt.subplot(321), plt.subplot(323), plt.subplot(325), plt.subplot(122)]

for om in omega_vals:
    # filter shots with omega = om
    indices = np.where(raw_omega == om)[0]
    size = raw_size[indices]
    center = raw_center[indices]
    slope = raw_slope[indices]
    exp_left = raw_exp_left[indices]
    exp_right = raw_exp_right[indices]
    time = raw_time[indices]
    Z = raw_Z[indices]
    exp_width = (1/np.array(exp_left) + 1/np.array(exp_right))/2

    # plt.plot(time, size, '.')
    # plt.xlabel("t [ms]")
    # plt.ylabel("$\sigma_B [\mu m]$")
    # plt.xscale('log')

    ax[0].plot(size, exp_width, '.', label=f'$\Omega_R/2\pi = {om}$ Hz', markersize=4, alpha=1)
    ax[1].plot(time, exp_width, '.', label=f'$\Omega_R/2\pi = {om}$ Hz', markersize=4, alpha=1)
    ax[2].plot(time, size, '.', label=f'$\Omega_R/2\pi = {om}$ Hz', markersize=4, alpha=1)
    ax[3].errorbar(om, np.mean(exp_width), yerr=np.std(exp_width)/np.sqrt(len(exp_width)), fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz', color='grey')

ax[0].set_xlabel("$\sigma_B\ [\mu m]$")
ax[0].set_ylabel("w $[\mu m]$")
ax[0].set_yscale("log")
ax[0].legend(fontsize='small', loc='lower left')

ax[1].set_xlabel("t [ms]")
ax[1].set_ylabel("w $[\mu m]$")
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].legend(fontsize='small', loc='lower left')

ax[2].set_xlabel("t [ms]")
ax[2].set_ylabel("$\sigma_B\ [\mu m]$")
ax[2].set_xscale("log")
ax[2].legend(fontsize='small', loc='lower left')

ax[3].set_xlabel("$\Omega_R/2\pi$ [Hz]")
ax[3].set_ylabel(r"$\langle w \rangle\ [\mu m]$")
# ax[1].set_yscale("log")

plt.suptitle("Shoulder width")
plt.tight_layout()
# plt.savefig("thesis/figures/chap2/b_param_omega.png", dpi=500)
plt.show()


# ## DETUNING
# # create figures
# fig = plt.figure(figsize=(10,6))
# ax = [plt.subplot(221), plt.subplot(223), plt.subplot(122)]
# cmap = plt.get_cmap("inferno")
# norm = plt.Normalize(vmin=min(detuning), vmax=max(detuning))

# for det in np.unique(detuning):
#     # filter shots with omega = om
#     indices = np.where(detuning == det)[0]
#     size = raw_size[indices]
#     center = raw_center[indices]
#     slope = raw_slope[indices]
#     exp_left = raw_exp_left[indices]
#     exp_right = raw_exp_right[indices]
#     time = raw_time[indices]
#     Z = raw_Z[indices]
#     exp_width = (1/np.array(exp_left) + 1/np.array(exp_right))/2

#     # plt.plot(time, size, '.')
#     # plt.xlabel("t [ms]")
#     # plt.ylabel("$\sigma_B [\mu m]$")
#     # plt.xscale('log')

#     color = cmap(norm(det))
#     ax[0].scatter(size, exp_width, c=[color], label=f'$\delta = {det}$ Hz', s=4, alpha=1)
#     ax[1].scatter(time, exp_width, c=[color], label=f'$\delta = {det}$ Hz', s=4, alpha=1)
#     ax[3].errorbar(det, np.mean(exp_width), yerr=np.std(exp_width)/np.sqrt(len(exp_width)), fmt='o', capsize=2, label=f'$\delta = {det}$ Hz', color='grey')

# ax[0].set_xlabel("$\sigma_B\ [\mu m]$")
# ax[0].set_ylabel("w $[\mu m]$")
# ax[0].set_yscale("log")
# # ax[0].legend(fontsize='small', loc='lower left')

# ax[1].set_xlabel("t [ms]")
# ax[1].set_ylabel("w $[\mu m]$")
# ax[1].set_yscale("log")
# ax[1].set_xscale("log")
# # ax[1].legend(fontsize='small', loc='lower left')

# ax[3].set_xlabel("$\delta$ [Hz]")
# ax[3].set_ylabel(r"$\langle w \rangle\ [\mu m]$")
# # ax[1].set_yscale("log")

# plt.suptitle("Shoulder width")
# plt.tight_layout()
# plt.show()