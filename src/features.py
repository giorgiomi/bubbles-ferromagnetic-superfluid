# This script is used to look at some bubble features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
n_blocks = 20

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
# detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
raw_Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

# omega_vals = np.unique(omega)
omega_vals = [300, 400, 600, 800]

# create figures
# fig, [ax[0], ax[1], ax_c] = plt.subplots(1, 3, figsize=(15, 6))
fig = plt.figure(figsize=(12,8))
ax = [plt.subplot(323), plt.subplot(325), plt.subplot(321), plt.subplot(122)]

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

    # print(om, len(time))

    # plot bubbles
    # fig, ax = plt.subplots(figsize=(10, 5), ncols=2)
    # ax[0].pcolormesh(Z, vmin=-1, vmax=+1, cmap='RdBu')
    # ax[0].set_xlabel('$x\ [\mu m]$')
    # ax[0].set_ylabel('shots')
    # ax[0].set_title('Sorted by time')

    # # Sort Z by cat
    # sorted_indices = np.argsort(time)
    # Z_sorted = Z[sorted_indices]

    # # Display the ordered shots
    # ax[1].pcolormesh(Z_sorted, vmin=-1, vmax=1, cmap='RdBu')
    # ax[1].set_xlabel('$x\ [\mu m]$')
    # ax[1].set_ylabel('shots')
    # ax[1].set_title(f'Sorted by {cat_str}')
    # plt.show()

    ## Group by TIME
    # Define cat blocks with a constant number of n_blocks
    sorted_cat_data = np.sort(time)
    cat_blocks = [sorted_cat_data[int(i * len(sorted_cat_data) / n_blocks)] for i in range(n_blocks)]
    cat_block_sizes = np.diff(cat_blocks)
    cat_block_sizes = np.append(cat_block_sizes, sorted_cat_data[-1] - cat_blocks[-1])

    el_values = []
    er_values = []
    s_values = []
    c_values = []
    sl_values = []
    cat_new = []
    for i, start_cat in enumerate(cat_blocks):
        end_cat = start_cat + cat_block_sizes[i]
        shots_in_block = Z[(time >= start_cat) & (time < end_cat)]
        # print(i, start_cat, end_cat, len(shots_in_block))
        if len(shots_in_block) > 0:
            for i, shot in enumerate(shots_in_block):
                s = size[(time >= start_cat) & (time < end_cat)][i]
                c = center[(time >= start_cat) & (time < end_cat)][i]
                e_l = exp_left[(time >= start_cat) & (time < end_cat)][i]
                e_r = exp_right[(time >= start_cat) & (time < end_cat)][i]
                sl = slope[(time >= start_cat) & (time < end_cat)][i]
                el_values.append(e_l) 
                er_values.append(e_r)   
                s_values.append(s)
                c_values.append(c)
                sl_values.append(sl)
                cat_new.append(start_cat)

    el_values = np.array(el_values)
    er_values = np.array(er_values)
    s_values = np.array(s_values)
    c_values = np.array(c_values)
    sl_values = np.array(sl_values)
    cat_new = np.array(cat_new)

    # Compute mean ACF for each cat block
    el_means = {start_cat: np.mean(el_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    er_means = {start_cat: np.mean(er_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    s_means = {start_cat: np.mean(s_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    c_means = {start_cat: np.mean(c_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    sl_means = {start_cat: np.mean(sl_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    
    el_stds = {start_cat: np.std(el_values[cat_new == start_cat])/np.sqrt(len(el_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    er_stds = {start_cat: np.std(er_values[cat_new == start_cat])/np.sqrt(len(er_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    s_stds = {start_cat: np.std(s_values[cat_new == start_cat])/np.sqrt(len(s_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    c_stds = {start_cat: np.std(c_values[cat_new == start_cat])/np.sqrt(len(c_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    sl_stds = {start_cat: np.std(sl_values[cat_new == start_cat])/np.sqrt(len(sl_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    
    el_plot = [1/el_means[cat] for cat in cat_blocks if cat in el_means]
    er_plot = [1/er_means[cat] for cat in cat_blocks if cat in er_means]
    s_plot = [s_means[cat] for cat in cat_blocks if cat in s_means]
    c_plot = [c_means[cat] for cat in cat_blocks if cat in c_means]
    sl_plot = [sl_means[cat] for cat in cat_blocks if cat in sl_means]
    
    el_yerr = [el_stds[cat]/(el_means[cat]**2) for cat in cat_blocks if cat in el_stds]
    er_yerr = [er_stds[cat]/(el_means[cat]**2) for cat in cat_blocks if cat in er_stds]
    s_yerr = [s_stds[cat] for cat in cat_blocks if cat in s_stds]
    c_yerr = [c_stds[cat] for cat in cat_blocks if cat in c_stds]
    sl_yerr = [sl_stds[cat] for cat in cat_blocks if cat in sl_stds]

    # Create bins from start_cat to end_cat for each block
    bin_semiwidths = [cat_block_sizes[i]/2 for i in range(len(cat_block_sizes))]
    bin_centers = [cat_blocks[i] + bin_semiwidths[i] for i in range(len(cat_blocks))]
    # print(om, len(el_plot), len(er_plot), len(bin_centers))

    ax[0].errorbar(bin_centers, 0.5*(np.array(el_plot)+np.array(er_plot))/2, yerr=0.5*(np.array(el_yerr)+np.array(er_yerr))/2, xerr=bin_semiwidths, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
    ax[0].set_xlabel('t [ms]')
    # ax[0].set_title(f'Exp width vs time')
    ax[0].set_ylabel('w [$\mu m$]')
    ax[0].set_xscale('log')
    ax[0].legend()

    ax[1].errorbar(bin_centers, s_plot, yerr=s_yerr, xerr=bin_semiwidths, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
    ax[1].set_xlabel('t [ms]')
    # ax[1].set_title(f'Size vs time')
    ax[1].set_ylabel(r'$\langle\sigma_B\rangle\ [\mu m]$')
    ax[1].set_xscale('log')
    ax[1].legend()

    ## Group by SIZE
    # Define cat blocks with a constant number of n_blocks
    sorted_cat_data = np.sort(size)
    cat_blocks = [sorted_cat_data[int(i * len(sorted_cat_data) / n_blocks)] for i in range(n_blocks)]
    cat_block_sizes = np.diff(cat_blocks)
    cat_block_sizes = np.append(cat_block_sizes, sorted_cat_data[-1] - cat_blocks[-1])

    el_values = []
    er_values = []
    s_values = []
    c_values = []
    sl_values = []
    cat_new = []
    for i, start_cat in enumerate(cat_blocks):
        end_cat = start_cat + cat_block_sizes[i]
        shots_in_block = Z[(size >= start_cat) & (size < end_cat)]
        # print(i, start_cat, end_cat, len(shots_in_block))
        if len(shots_in_block) > 0:
            for i, shot in enumerate(shots_in_block):
                s = size[(size >= start_cat) & (size < end_cat)][i]
                c = center[(size >= start_cat) & (size < end_cat)][i]
                e_l = exp_left[(size >= start_cat) & (size < end_cat)][i]
                e_r = exp_right[(size >= start_cat) & (size < end_cat)][i]
                sl = slope[(size >= start_cat) & (size < end_cat)][i]
                el_values.append(e_l) 
                er_values.append(e_r)   
                s_values.append(s)
                c_values.append(c)
                sl_values.append(sl)
                cat_new.append(start_cat)

    el_values = np.array(el_values)
    er_values = np.array(er_values)
    s_values = np.array(s_values)
    c_values = np.array(c_values)
    sl_values = np.array(sl_values)
    cat_new = np.array(cat_new)

    # Compute mean ACF for each cat block
    el_means = {start_cat: np.mean(el_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    er_means = {start_cat: np.mean(er_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    s_means = {start_cat: np.mean(s_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    c_means = {start_cat: np.mean(c_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    sl_means = {start_cat: np.mean(sl_values[cat_new == start_cat]) for start_cat in cat_blocks if start_cat in cat_new}
    
    el_stds = {start_cat: np.std(el_values[cat_new == start_cat])/np.sqrt(len(el_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    er_stds = {start_cat: np.std(er_values[cat_new == start_cat])/np.sqrt(len(er_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    s_stds = {start_cat: np.std(s_values[cat_new == start_cat])/np.sqrt(len(s_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    c_stds = {start_cat: np.std(c_values[cat_new == start_cat])/np.sqrt(len(c_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    sl_stds = {start_cat: np.std(sl_values[cat_new == start_cat])/np.sqrt(len(sl_values[cat_new == start_cat])) for start_cat in cat_blocks if start_cat in cat_new}
    
    el_plot = [1/el_means[cat] for cat in cat_blocks if cat in el_means]
    er_plot = [1/er_means[cat] for cat in cat_blocks if cat in er_means]
    s_plot = [s_means[cat] for cat in cat_blocks if cat in s_means]
    c_plot = [c_means[cat] for cat in cat_blocks if cat in c_means]
    sl_plot = [sl_means[cat] for cat in cat_blocks if cat in sl_means]
    
    el_yerr = [el_stds[cat]/(el_means[cat]**2) for cat in cat_blocks if cat in el_stds]
    er_yerr = [er_stds[cat]/(el_means[cat]**2) for cat in cat_blocks if cat in er_stds]
    s_yerr = [s_stds[cat] for cat in cat_blocks if cat in s_stds]
    c_yerr = [c_stds[cat] for cat in cat_blocks if cat in c_stds]
    sl_yerr = [sl_stds[cat] for cat in cat_blocks if cat in sl_stds]

    # Create bins from start_cat to end_cat for each block
    bin_semiwidths = [cat_block_sizes[i]/2 for i in range(len(cat_block_sizes))]
    bin_centers = [cat_blocks[i] + bin_semiwidths[i] for i in range(len(cat_blocks))]
    # print(om, len(el_plot), len(er_plot), len(bin_centers))

    ax[2].errorbar(bin_centers, 0.5*(np.array(el_plot)+np.array(er_plot))/2, yerr=0.5*(np.array(el_yerr)+np.array(er_yerr))/2, xerr=bin_semiwidths, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
    ax[2].set_xlabel('$\sigma_B\ [\mu m]$')
    # ax[2].set_title(f'Exp width vs size')
    ax[2].set_ylabel('w [$\mu m$]')
    ax[2].legend()

    ax[3].errorbar(om, np.mean(exp_width), yerr=np.std(exp_width)/np.sqrt(len(exp_width)), fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz', color='grey')

ax[3].set_xlabel("$\Omega_R/2\pi$ [Hz]")
ax[3].set_ylabel(r"$\langle w \rangle\ [\mu m]$")
plt.tight_layout()
# plt.savefig("thesis/figures/chap2/b_param_cluster.png", dpi=500)
plt.show()