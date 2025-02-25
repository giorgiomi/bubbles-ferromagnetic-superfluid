import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import quadPlot, computeFFT_ACF
from scipy.fft import rfftfreq

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
window_len = 20

# Print script purpose
print("Sort shift gathered")
zero_mean_flag = int(input("Zero mean flag: "))

CFG = rfftfreq(2*w, d=1.0)
CLG = np.arange(0, window_len+1)

# Import bubble data
raw_size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
raw_time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
raw_center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
raw_omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
raw_detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
raw_in_left = pd.read_csv(f"data/gathered/in_left.csv", header=None).to_numpy().flatten()
raw_in_right = pd.read_csv(f"data/gathered/in_right.csv", header=None).to_numpy().flatten()
raw_Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

# omega_vals = [300, 400, 600, 800]
# det_vals = np.unique(raw_detuning)
omega_vals = [400]
det_vals = [596.5]

for om in omega_vals:
    for det in det_vals:
        indices = np.where((raw_omega == om) & (raw_detuning == det))[0]
        size = raw_size[indices]
        time = raw_time[indices]
        center = raw_center[indices]
        in_left = raw_in_left[indices]
        in_right = raw_in_right[indices]
        Z = raw_Z[indices]

        # plot bubbles
        fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})

        # Sort Z by time
        sorted_indices = np.argsort(time)
        Z_sorted = Z[sorted_indices]

        ax[0].pcolormesh(Z_sorted, vmin=-1, vmax=+1, cmap='RdBu')
        ax[0].set_xlabel('$x\ [\mu m]$')
        ax[0].set_ylabel('Time index')
        ax[0].set_title('Shots sorted by time')

        # Sort Z by size
        sorted_indices = np.argsort(size)
        non_zero_indices = size > 0
        sorted_indices = np.argsort(size[non_zero_indices])
        Z_sorted = Z[non_zero_indices][sorted_indices]

        # Display the ordered shots
        im = ax[1].pcolormesh(Z_sorted, vmin=-1, vmax=1, cmap='RdBu')
        ax[1].set_xlabel('$x\ [\mu m]$')
        ax[1].set_ylabel('Size index')
        ax[1].set_title('Shots sorted by size')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z', rotation=180)


        plt.suptitle(f"Bubble shots with $\Omega_R/2\pi = {om}$ Hz and $\delta/2\pi = {det}$ Hz")
        plt.tight_layout()
        # plt.savefig("thesis/figures/chap2/shot_sorting.png", dpi=500)
        plt.show()

        # artistic plot
        # Align shots based on center
        aligned_Z = []
        for i in range(len(Z_sorted)):
            shift = int(center[non_zero_indices][sorted_indices][i])
            aligned_shot = np.roll(Z_sorted[i], -shift + w)
            aligned_Z.append(aligned_shot)
        aligned_Z = np.array(aligned_Z)
        aligned_Z = aligned_Z[:, 20:-20]
        print(max(center[non_zero_indices][sorted_indices]))

        fig, ax = plt.subplots(figsize=(6, 8), ncols=1)
        ax.pcolormesh(aligned_Z, vmin=-1, vmax=1, cmap='RdBu')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        # plt.savefig("thesis/figures/chap1/artistic.png", dpi=500)
        plt.show()

        in_left_sorted = in_left[non_zero_indices][sorted_indices]
        in_right_sorted = in_right[non_zero_indices][sorted_indices]
        inside_fft_magnitudes_TRUE = []
        inside_fft_magnitudes = []
        inside_acf_values = []
        inside_acf_values_TRUE = []

        print(len(Z_sorted))
        for i in range(len(Z_sorted)):
            y = Z_sorted[i] 
            ## NEW METHOD, with inside estimation
            left = in_left_sorted[i]
            right = in_right_sorted[i]
            start = max(0, int(left))
            end = min(len(y), int(right))
            inside = y[start:end]

            # plt.plot(y)
            # plt.plot(np.arange(start, end), inside - np.mean(inside))
            # plt.title(f"Day {om}, Seq {det}, Shot {i}")
            # plt.show()

            # compute FFT of the inside
            N = len(inside)
            if N > 4*window_len:
                inside_fft_magnitudes, inside_acf_values, _, _ = computeFFT_ACF(1, inside, CFG, CLG, inside_fft_magnitudes, inside_acf_values, window_len)
                _, inside_acf_values_TRUE, _, _ = computeFFT_ACF(0, inside, CFG, CLG, inside_fft_magnitudes_TRUE, inside_acf_values_TRUE, window_len)

        inside_fft_magnitudes = np.array(inside_fft_magnitudes)
        inside_fft_mean = np.mean(inside_fft_magnitudes, axis=0)
        
        inside_acf_values = np.array(inside_acf_values)
        inside_acf_mean = np.mean(inside_acf_values, axis=0)

        inside_acf_values_TRUE = np.array(inside_acf_values_TRUE)
        inside_acf_mean_TRUE = np.mean(inside_acf_values_TRUE, axis=0)

        region = "Inside"
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Colormap for FFT
        im1 = axs[0, 0].imshow(inside_fft_magnitudes, aspect='auto', extent=[CFG[0], CFG[-1], 0, len(Z_sorted)], origin='lower', cmap='plasma')
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].set_title(region + f" FFT (all shots)")
        axs[0, 0].set_xlabel(r"$k\ [1/\mu m]$")
        axs[0, 0].set_ylabel("Size index")

        # Average FFT
        axs[0, 1].plot(CFG, inside_fft_mean, '-', label='FFT on background inside')
        axs[0, 1].set_title(region + f" FFT (average)")
        axs[0, 1].set_xlabel(r"$k\ [1/\mu m]$")
        axs[0, 1].set_xlim(-0.02, 0.52)
        axs[0, 1].set_ylabel("FFT")

        # Colormap for autocorrelation
        im2 = axs[1, 0].imshow(inside_acf_values, aspect='auto', extent=[CLG[0], CLG[-1], 0, len(Z_sorted)-1], origin='lower', cmap='plasma', vmin=-1, vmax=1)
        fig.colorbar(im2, ax=axs[1, 0])
        axs[1, 0].set_title(region + f" ACF (all shots)")
        axs[1, 0].set_xlabel("$\Delta x\ [\mu m]$")
        axs[1, 0].set_ylabel("Size index")
        axs[1, 0].set_xticks(np.arange(0, 21, 2))

        # Average Autocorrelation
        axs[1, 1].plot(CLG, inside_acf_mean, '-', color='tab:blue', label='Zero-mean data ACF')
        ax2 = axs[1, 1].twinx()
        ax2.plot(CLG, inside_acf_mean_TRUE, '-', color='tab:red', label='True data ACF')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Combine legends
        lines, labels = axs[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        axs[1, 1].set_title(region + f" ACF (average)")
        axs[1, 1].set_xlabel("$\Delta x\ [\mu m]$")
        axs[1, 1].set_ylabel("ACF")
        axs[1, 1].set_xticks(np.arange(0, 21, 2))

        plt.suptitle(f"FFT and ACF on shots with $\Omega_R/2\pi = {om}$ Hz and $\delta/2\pi = {det}$ Hz")
        plt.tight_layout()
        # plt.savefig("thesis/figures/chap2/inside_omdet.png", dpi=500)
        plt.show()