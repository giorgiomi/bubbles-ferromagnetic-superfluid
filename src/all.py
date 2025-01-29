# This script is used to analyze the data inside the bubble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from util.parameters import importParameters
import sys
from util.methods import quadPlot, computeFFT_ACF, doublePlot

# Data
# f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)
window_len = 20

# Print script purpose
print("\nAnalyzing FFT and ACF on INSIDE REGION with gathered data\n")

# Ask the user for FFT and ACF on true or zero mean signal
zero_mean_flag = input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: ")

# Ask for gather flag
gather_flag = input("Enter gather method [omega/time/size]: ")

# Find max length for Common Frequency Grid (CFG)
max_length = 0
size = pd.read_csv(f"data/gathered/size.csv", header=None).to_numpy().flatten()
max_size = int(max(size) + 1)
if max_size > max_length and max_size < 2*w:
    max_length = max_size

# CFG and CLG
CFG = rfftfreq(max_length, d=sampling_rate)
CLG = np.arange(0, window_len+1)

# Import bubble data
center = pd.read_csv(f"data/gathered/center.csv", header=None).to_numpy().flatten()
time = pd.read_csv(f"data/gathered/time.csv", header=None).to_numpy().flatten()
omega = pd.read_csv(f"data/gathered/omega.csv", header=None).to_numpy().flatten()
detuning = pd.read_csv(f"data/gathered/detuning.csv", header=None).to_numpy().flatten()
Z = pd.read_csv(f"data/gathered/Z.csv", header=None).to_numpy()

## GATHER BY OMEGA
if gather_flag == 'omega':
    # gather by omega and order by size
    Z_dict = {omega_val: Z[(omega == omega_val)][np.argsort(size[(omega == omega_val)])] for omega_val in [200, 300, 400, 600, 800]}
    Z_200, Z_300, Z_400, Z_600, Z_800 = Z_dict[200], Z_dict[300], Z_dict[400], Z_dict[600], Z_dict[800]

    # Plot all shots, with different dets
    fig, axs = plt.subplots(1, 5, figsize=(14, 6))

    # Plot Z_200
    axs[0].pcolormesh(Z_200, vmin=-1, vmax=1, cmap='RdBu')
    axs[0].set_title('$\Omega = 2\pi$ 200 Hz')
    axs[0].set_xlabel('$x\ [\mu m]$')
    axs[0].set_ylabel('shots')

    # Plot Z_300
    axs[1].pcolormesh(Z_300, vmin=-1, vmax=1, cmap='RdBu')
    axs[1].set_title('$\Omega = 2\pi$ 300 Hz')
    axs[1].set_xlabel('$x\ [\mu m]$')
    axs[1].set_ylabel('shots')

    # Plot Z_400
    axs[2].pcolormesh(Z_400, vmin=-1, vmax=1, cmap='RdBu')
    axs[2].set_title('$\Omega = 2\pi$ 400 Hz')
    axs[2].set_xlabel('$x\ [\mu m]$')
    axs[2].set_ylabel('shots')

    # Plot Z_600
    axs[3].pcolormesh(Z_600, vmin=-1, vmax=1, cmap='RdBu')
    axs[3].set_title('$\Omega = 2\pi$ 600 Hz')
    axs[3].set_xlabel('$x\ [\mu m]$')
    axs[3].set_ylabel('shots')

    # Plot Z_800
    axs[4].pcolormesh(Z_800, vmin=-1, vmax=1, cmap='RdBu')
    axs[4].set_title('$\Omega = 2\pi$ 800 Hz')
    axs[4].set_xlabel('$x\ [\mu m]$')
    axs[4].set_ylabel('shots')

    plt.tight_layout()
    # plt.suptitle("Gathered shots")
    plt.show()

    # Compute FFT and ACF
    fft_magnitudes = []
    acf_values = []
    omega_new = []
    for i, shot in enumerate(Z):
        s = size[i]
        c = center[i]
        inside = shot[int(c-s/2+10):int(c+s/2-10)]
        if len(inside) > 4*window_len:
            # print(len(inside))
            fft_magnitudes, acf_values, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes, acf_values, window_len)
            omega_new.append(omega[i])
            
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].plot(rfftfreq(len(inside), d=1.0), np.abs(rfft(inside - np.mean(inside))))
            # ax[1].plot(CFG, int_mag)
            # plt.yscale('log')
            # plt.xlim(0, 0.1)
            # plt.show()
            # plt.plot(np.arange(int(c-s/2+10), int(c+s/2-10)), inside)
            # plt.show()

    omega_fft = np.array(omega_new)

    fft_magnitudes = np.array(fft_magnitudes)
    acf_values = np.array(acf_values)

    fft_200 = np.mean(fft_magnitudes[omega_fft == 200], axis=0)
    acf_200 = np.mean(acf_values[omega_fft == 200], axis=0)

    fft_300 = np.mean(fft_magnitudes[omega_fft == 300], axis=0)
    acf_300 = np.mean(acf_values[omega_fft == 300], axis=0)

    fft_400 = np.mean(fft_magnitudes[omega_fft == 400], axis=0)
    acf_400 = np.mean(acf_values[omega_fft == 400], axis=0)

    fft_600 = np.mean(fft_magnitudes[omega_fft == 600], axis=0)
    acf_600 = np.mean(acf_values[omega_fft == 600], axis=0)

    fft_800 = np.mean(fft_magnitudes[omega_fft == 800], axis=0)
    acf_800 = np.mean(acf_values[omega_fft == 800], axis=0)

    # quadPlot(0, 0, Z_200, "200", CFG, CLG, fft_magnitudes[omega_fft == 200], fft_200, acf_values[omega_fft == 200], acf_200, 0)
    # quadPlot(0, 0, Z_300, "300", CFG, CLG, fft_magnitudes[omega_fft == 300], fft_300, acf_values[omega_fft == 300], acf_300, 0)
    # quadPlot(0, 0, Z_400, "400", CFG, CLG, fft_magnitudes[omega_fft == 400], fft_400, acf_values[omega_fft == 400], acf_400, 0)
    # quadPlot(0, 0, Z_600, "600", CFG, CLG, fft_magnitudes[omega_fft == 600], fft_600, acf_values[omega_fft == 600], acf_600, 0)
    # quadPlot(0, 0, Z_800, "800", CFG, CLG, fft_magnitudes[omega_fft == 800], fft_800, acf_values[omega_fft == 800], acf_800, 0)
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Plot FFT results
    ax[0].plot(CFG, fft_200, label='200 Hz')
    ax[0].plot(CFG, fft_300, label='300 Hz')
    ax[0].plot(CFG, fft_400, label='400 Hz')
    ax[0].plot(CFG, fft_600, label='600 Hz')
    ax[0].plot(CFG, fft_800, label='800 Hz')
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Frequencies')

    # Plot ACF results
    ax[1].plot(CLG, acf_200, label='200 Hz')
    ax[1].plot(CLG, acf_300, label='300 Hz')
    ax[1].plot(CLG, acf_400, label='400 Hz')
    ax[1].plot(CLG, acf_600, label='600 Hz')
    ax[1].plot(CLG, acf_800, label='800 Hz')
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Frequencies')

    plt.tight_layout()
    plt.show()

elif gather_flag == 'time':
    ## GATHER BY TIME
    n_blocks = 12
    # Define time blocks (e.g., every 10 units of time)
    time_block_size = (np.max(time) - np.min(time))/n_blocks
    time_blocks = np.arange(min(time), max(time) + time_block_size, time_block_size)

    # Initialize lists to store FFT and ACF results for each time block
    fft_magnitudes_time = []
    acf_values_time = []
    time_new = []

    # Group shots by time block and compute FFT and ACF
    for start_time in time_blocks:
        end_time = start_time + time_block_size
        shots_in_block = Z[(time >= start_time) & (time < end_time)]
        if len(shots_in_block) > 0:
            for shot in shots_in_block:
                s = size[(time >= start_time) & (time < end_time)][0]
                c = center[(time >= start_time) & (time < end_time)][0]
                inside = shot[int(c-s/2+10):int(c+s/2-10)]
                if len(inside) > 4*window_len:
                    fft_magnitudes_time, acf_values_time, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes_time, acf_values_time, window_len)
                    time_new.append(start_time)

    time_fft = np.array(time_new)

    fft_magnitudes_time = np.array(fft_magnitudes_time)
    acf_values_time = np.array(acf_values_time)

    # Compute mean FFT and ACF for each time block
    fft_time_means = {start_time: np.mean(fft_magnitudes_time[time_fft == start_time], axis=0) for start_time in time_blocks if start_time in time_fft}
    acf_time_means = {start_time: np.mean(acf_values_time[time_fft == start_time], axis=0) for start_time in time_blocks if start_time in time_fft}

    # Plot FFT and ACF results for each time block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, start_time in enumerate(fft_time_means.keys()):
        ax[0].plot(CFG, fft_time_means[start_time], label=f'Time {start_time:.1f}-{start_time + time_block_size:.1f}', color=plt.cm.viridis(i / len(fft_time_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Time Blocks')

    # Plot ACF results using colormaps
    for i, start_time in enumerate(acf_time_means.keys()):
        ax[1].plot(CLG, acf_time_means[start_time], label=f'Time {start_time:.1f}-{start_time + time_block_size:.1f}', color=plt.cm.viridis(i / len(acf_time_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Time Blocks')

    plt.tight_layout()
    plt.show()

elif gather_flag == 'size':
    ## GATHER BY SIZE
    # Define size blocks (e.g., every 50 units of size)
    size_block_size = 10
    size_blocks = np.arange(min(size), max(size) + size_block_size, size_block_size)

    # Initialize lists to store FFT and ACF results for each size block
    fft_magnitudes_size = []
    acf_values_size = []
    size_new = []

    # Group shots by size block and compute FFT and ACF
    for start_size in size_blocks:
        end_size = start_size + size_block_size
        shots_in_block = Z[(size >= start_size) & (size < end_size)]
        if len(shots_in_block) > 0:
            for shot in shots_in_block:
                s = size[(size >= start_size) & (size < end_size)][0]
                c = center[(size >= start_size) & (size < end_size)][0]
                inside = shot[int(c-s/2+10):int(c+s/2-10)]
                if len(inside) > 4*window_len:
                    fft_magnitudes_size, acf_values_size, int_mag, int_acf = computeFFT_ACF(zero_mean_flag, inside, CFG, CLG, fft_magnitudes_size, acf_values_size, window_len)
                    size_new.append(start_size)

    size_fft = np.array(size_new)

    fft_magnitudes_size = np.array(fft_magnitudes_size)
    acf_values_size = np.array(acf_values_size)

    # Compute mean FFT and ACF for each size block
    fft_size_means = {start_size: np.mean(fft_magnitudes_size[size_fft == start_size], axis=0) for start_size in size_blocks if start_size in size_fft}
    acf_size_means = {start_size: np.mean(acf_values_size[size_fft == start_size], axis=0) for start_size in size_blocks if start_size in size_fft}

    # Plot FFT and ACF results for each size block
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot FFT results using colormaps
    for i, start_size in enumerate(fft_size_means.keys()):
        ax[0].plot(CFG, fft_size_means[start_size], label=f'Size {start_size:.1f}-{start_size + size_block_size:.1f}', color=plt.cm.viridis(i / len(fft_size_means)))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('FFT Magnitude')
    ax[0].legend()
    ax[0].set_title('FFT Magnitudes for Different Size Blocks')

    # Plot ACF results using colormaps
    for i, start_size in enumerate(acf_size_means.keys()):
        ax[1].plot(CLG, acf_size_means[start_size], label=f'Size {start_size:.1f}-{start_size + size_block_size:.1f}', color=plt.cm.viridis(i / len(acf_size_means)))
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF Value')
    ax[1].legend()
    ax[1].set_title('ACF Values for Different Size Blocks')

    plt.tight_layout()
    plt.show()

else:
    print("Gather method has to be [omega/time/size]")