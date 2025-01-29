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
zero_mean_flag = int(input("Enter 1 for zero-mean signal FFT and ACF, 0 for true signal: "))

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

# gather by omega and order by size
Z_dict = {omega_val: Z[(omega == omega_val)][np.argsort(size[(omega == omega_val)])] for omega_val in [200, 300, 400, 600, 800]}
Z_200, Z_300, Z_400, Z_600, Z_800 = Z_dict[200], Z_dict[300], Z_dict[400], Z_dict[600], Z_dict[800]

# Plot all shots, with different dets
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# # Plot Z_300
# axs[0, 0].pcolormesh(Z_300, vmin=-1, vmax=1, cmap='RdBu')
# axs[0, 0].set_title('$\Omega = 2\pi$ 300 Hz')
# axs[0, 0].set_xlabel('$x\ [\mu m]$')
# axs[0, 0].set_ylabel('shots')

# # Plot Z_400
# axs[0, 1].pcolormesh(Z_400, vmin=-1, vmax=1, cmap='RdBu')
# axs[0, 1].set_title('$\Omega = 2\pi$ 400 Hz')
# axs[0, 1].set_xlabel('$x\ [\mu m]$')
# axs[0, 1].set_ylabel('shots')

# # Plot Z_600
# axs[1, 0].pcolormesh(Z_600, vmin=-1, vmax=1, cmap='RdBu')
# axs[1, 0].set_title('$\Omega = 2\pi$ 600 Hz')
# axs[1, 0].set_xlabel('$x\ [\mu m]$')
# axs[1, 0].set_ylabel('shots')

# # Plot Z_800
# axs[1, 1].pcolormesh(Z_800, vmin=-1, vmax=1, cmap='RdBu')
# axs[1, 1].set_title('$\Omega = 2\pi$ 800 Hz')
# axs[1, 1].set_xlabel('$x\ [\mu m]$')
# axs[1, 1].set_ylabel('shots')

# plt.tight_layout()
# plt.suptitle("Gathered shots")
# plt.show()

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

