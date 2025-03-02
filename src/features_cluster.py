# This script is used to look at some bubble features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
n_clusters = 20
wlim = 50

# Print script purpose
print(f"We are clustering guys, n_cl = {n_clusters}")

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
fig1 = plt.figure(figsize=(12,8))
# ax = [plt.subplot(321), plt.subplot(323), plt.subplot(325), plt.subplot(322), plt.subplot(326)]
gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1]) 

# Assigning subplots
ax1 = fig1.add_subplot(gs[0, 0])  # Upper left
ax2 = fig1.add_subplot(gs[0, 1])  # Middle left
ax3 = fig1.add_subplot(gs[0, 2])  # Lower left
ax4 = fig1.add_subplot(gs[1, 0:2])  # Expands full height (right side)
ax5 = fig1.add_subplot(gs[1, 2])  # Lower right


fig2, ax_cl = plt.subplots(len(omega_vals), 3, figsize=(15, 8), sharex='col')

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
    exp_width = (np.array(exp_left) + np.array(exp_right))/2

    # plt.figure()
    # plt.plot(sorted_time, sorted_size, '.')
    # plt.xlabel("t [ms]")
    # plt.ylabel("$\sigma_B [\mu m]$")
    # # plt.xscale('log')
    # plt.show()
    
    # Reshape time for KMeans
    time_reshaped = time.reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(time_reshaped)
    labels = kmeans.labels_

    # Calculate average and error for each cluster
    avg_size = np.array([np.mean(size[labels == i]) for i in range(n_clusters)])
    err_size = np.array([np.std(size[labels == i])/np.sqrt(len(size[labels == i])) for i in range(n_clusters)])
    avg_exp_width = np.array([np.mean(exp_width[labels == i]) for i in range(n_clusters)])
    err_exp_width = np.array([np.std(exp_width[labels == i])/np.sqrt(len(exp_width[labels == i])) for i in range(n_clusters)])
    clustered_t = np.array([np.mean(time[labels == i]) for i in range(n_clusters)])
    err_t = np.array([np.std(time[labels == i])/np.sqrt(len(time[labels == i])) for i in range(n_clusters)])

    mask_exp_width = avg_exp_width < 40
    ax2.errorbar(clustered_t[mask_exp_width], avg_exp_width[mask_exp_width], xerr=err_t[mask_exp_width], yerr=err_exp_width[mask_exp_width], fmt='.', label=f'$\Omega_R/2\pi = {om}$ Hz', markersize=12, capsize=2)
    ax3.errorbar(clustered_t[mask_exp_width], avg_size[mask_exp_width], xerr=err_t[mask_exp_width], yerr=err_size[mask_exp_width], fmt='.', label=f'$\Omega_R/2\pi = {om}$ Hz', markersize=12, capsize=2)

    # Perform linear fit on avg_size vs clustered_t
    sorted_indices = np.argsort(clustered_t)
    sorted_clustered_t = clustered_t[sorted_indices]
    sorted_avg_size = avg_size[sorted_indices]
    sorted_err_size = err_size[sorted_indices]

    if om == 800: tr = 11
    else: tr = 7

    log_clustered_t = np.log(sorted_clustered_t[:tr])
    log_avg_size = np.log(sorted_avg_size[:tr])
    [m, b], cov = np.polyfit(log_clustered_t, log_avg_size, w=sorted_avg_size[:tr]/sorted_err_size[:tr], deg=1, cov=True)
    dm, db = np.sqrt(np.diag(cov))
    fit_line = lambda x: np.exp(b) * x**m
    print(om, m, dm)

    # Plot the linear fit
    ax3.plot(sorted_clustered_t[:tr], fit_line(sorted_clustered_t[:tr]), '--', color=ax3.lines[-1].get_color())
    
    # Reshape size for KMeans
    size_reshaped = size.reshape(-1, 1)

    # Apply KMeans clustering by size
    kmeans_size = KMeans(n_clusters=n_clusters, random_state=0).fit(size_reshaped)
    labels_size = kmeans_size.labels_

    # Calculate average and error for each cluster by size
    avg_time_size = np.array([np.mean(time[labels_size == i]) for i in range(n_clusters)])
    err_time_size = np.array([np.std(time[labels_size == i])/np.sqrt(len(time[labels_size == i])) for i in range(n_clusters)])
    avg_exp_width_size = np.array([np.mean(exp_width[labels_size == i]) for i in range(n_clusters)])
    err_exp_width_size = np.array([np.std(exp_width[labels_size == i])/np.sqrt(len(exp_width[labels_size == i])) for i in range(n_clusters)])
    clustered_s = np.array([np.mean(size[labels_size == i]) for i in range(n_clusters)])
    err_s = np.array([np.std(size[labels_size == i])/np.sqrt(len(size[labels_size == i])) for i in range(n_clusters)])

    # Plot clustered data on ax_cl
    for i in range(n_clusters):
        # print(om, len(time[labels == i]), len(time[labels_size == i]))
        ax_cl[omega_vals.index(om), 0].plot(time[labels == i], size[labels == i], 'o', markersize=4, alpha=1)
        ax_cl[omega_vals.index(om), 1].plot(time[(labels == i) & (exp_width < wlim)], exp_width[(labels == i) & (exp_width < wlim)], 'o', markersize=4, alpha=1)
        ax_cl[omega_vals.index(om), 2].plot(size[(labels_size == i) & (exp_width < wlim)], exp_width[(labels_size == i) & (exp_width < wlim)], 'o', markersize=4, alpha=1)

    str = rf"$\Omega_R/2\pi = {om}$ Hz" + "\n\n" + r"$\sigma_B\ [\mu m]$"
    ax_cl[omega_vals.index(om), 0].set_ylabel(str)
    ax_cl[omega_vals.index(om), 0].set_xscale("log")
    ax_cl[omega_vals.index(om), 1].set_ylabel("$w_D\ [\mu m]$")
    ax_cl[omega_vals.index(om), 1].set_xscale("log")
    # ax_cl[omega_vals.index(om), 1].set_ylim([-2, 52])
    ax_cl[omega_vals.index(om), 2].set_ylabel("$w_D\ [\mu m]$")
    # ax_cl[omega_vals.index(om), 2].set_ylim([-2, 52])
    ax_cl[omega_vals.index(om), 2].set_xlim([20, 300])

    mask = (clustered_s > 50) & (clustered_s < 300) & (avg_exp_width_size < 35)
    ax1.errorbar(clustered_s[mask], avg_exp_width_size[mask], xerr=err_s[mask], yerr=err_exp_width_size[mask], fmt='.', label=f'$\Omega_R/2\pi = {om}$ Hz', markersize=12, capsize=2)

    ax4.errorbar(om, np.mean(exp_width), yerr=np.std(exp_width)/np.sqrt(len(exp_width)), fmt='o', capsize=4, label=f'$\Omega_R/2\pi = {om}$ Hz', color='tab:purple', markersize=8, elinewidth=2)

    ax5.errorbar(om, m, yerr=dm, fmt='o', capsize=4, label=f'$\Omega_R/2\pi = {om}$ Hz', color='tab:purple', markersize=8, elinewidth=2)

ax1.set_xlabel("$\sigma_B\ [\mu m]$")
ax1.set_ylabel("$w_D\ [\mu m]$")
# ax1.set_yscale("log")
ax1.legend(fontsize='small', loc='upper left')

ax2.set_xlabel("t [ms]")
ax2.set_ylabel("$w_D\ [\mu m]$")
# ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.legend(fontsize='small', loc='upper left')

ax3.set_xlabel("t [ms]")
ax3.set_ylabel("$\sigma_B\ [\mu m]$")
ax3.set_xscale("log")
ax3.set_yscale('log')
ax3.legend(fontsize='small', loc='upper left')

ax4.set_xlabel("$\Omega_R/2\pi$ [Hz]")
ax4.set_ylabel(r"$\langle w_D \rangle\ [\mu m]$")
ax4.set_xlim([0, 820])
ax4.set_ylim([0, 30])

ax5.set_xlabel("$\Omega_R/2\pi$ [Hz]")
ax5.set_ylabel("$B$")

ax_cl[3, 0].set_xlabel("t [ms]")
ax_cl[3, 1].set_xlabel("t [ms]")
ax_cl[3, 2].set_xlabel("$\sigma_B\ [\mu m]$")

fig1.suptitle("Bubble clustered parameters")
fig2.suptitle("Clustering")
fig1.tight_layout()
fig2.tight_layout()
# fig1.savefig("thesis/figures/chap2/b_param_cluster.png", dpi=500)
# fig2.savefig("thesis/figures/chap2/clustering.png", dpi=500)
plt.show()
