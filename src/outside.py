# This script is used to analyze the data outside the bubble
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from util.methods import scriptUsage
from scipy.fft import rfft, rfftfreq
from util.parameters import importParameters

# Data
f, seqs, Omega, knT, Detuning = importParameters()
w = 200 # Thomas-Fermi radius, always the same
sampling_rate = 1.0 # 1/(1 pixel)

chosen_days = scriptUsage()

omega_fft_dict = {}
omega_acf_dict = {}
detuning_fft_dict = {}
detuning_acf_dict = {}

max_length = 0
for day in chosen_days:
    for seq, seqi in enumerate((seqs[day])):
        df_out_left_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/out_left_sorted.csv", header=None)
        df_out_right_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/out_right_sorted.csv", header=None)

        out_left_sorted = df_out_left_sorted.to_numpy().flatten()
        out_right_sorted = df_out_right_sorted.to_numpy().flatten()
        max_size = max(int(max(out_left_sorted) + 1), int(max(2*w - out_right_sorted) + 1))
        if max_size > max_length and max_size < 2*w:
            max_length = max_size

# Common frequency grid for FFT
common_freq_grid = rfftfreq(max_length, d=sampling_rate)

# Common lag grid for ACF
common_lag_grid = np.arange(-max_length+1, max_length)

for day in chosen_days:
    for seq, seqi in enumerate((seqs[day])):
        df_size_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV_sorted.csv", header=None)
        df_center_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/center_sorted.csv", header=None)
        df_out_left_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/out_left_sorted.csv", header=None)
        df_out_right_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/out_right_sorted.csv", header=None)
        df_Z_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/Z_sorted.csv", header=None)
        
        omega = Omega[day][seq]
        detuning = Detuning[day][seq]

        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        b_center_sorted = df_center_sorted.to_numpy().flatten()
        out_left_sorted = df_out_left_sorted.to_numpy().flatten()
        out_right_sorted = df_out_right_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # FFT on outside region
        # if max_length >= w: continue

        outside_left_fft_magnitudes = []
        outside_left_acf_values = []
        outside_right_fft_magnitudes = []
        outside_right_acf_values = []

        # cycle through ordered shots from beginning to end
        for i in range(len(Z)):
            y = Z[i] 
            left = int(out_left_sorted[i])
            right = int(out_right_sorted[i])
            outside_left = y[:left]
            outside_right = y[right:]
            
            # plt.figure()
            # plt.plot(y)
            # plt.plot(np.arange(0, left), outside_left)
            # plt.plot(np.arange(right, 2*w), outside_right)
            # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
            # plt.show()

            N_left = len(outside_left)
            N_right= len(outside_right)

            if N_left > 0 and N_right > 0:
                #---------------------------------------------------------------------------------------------------
                # compute FFT and ACF of the left
                freq_grid_left = rfftfreq(N_left, d=sampling_rate)
                outside_left_fft = rfft(outside_left - np.mean(outside_left)) ## doing FFT on zero-mean signal
                outside_left_spectrum = np.abs(outside_left_fft)

                # outside_left_acf = correlate(outside_left - np.mean(outside_left), outside_left - np.mean(outside_left), mode='full')
                outside_left_acf = correlate(outside_left, outside_left, mode='full')
                outside_left_acf /= np.max(outside_left_acf) # normalize to acf[0] = 1

                # plt.plot(freq_grid, outside_left_spectrum)
                # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
                # plt.show()
                
                # Interpolate onto the common frequency grid
                interpolated_magnitude = np.interp(common_freq_grid, freq_grid_left, outside_left_spectrum)
                outside_left_fft_magnitudes.append(interpolated_magnitude)

                # Compute the lag grid for this signal and iterpolate
                lag_grid = np.arange(-N_left + 1, N_left)
                interpolated_acf = np.interp(common_lag_grid, lag_grid, outside_left_acf)
                outside_left_acf_values.append(interpolated_acf)

                #---------------------------------------------------------------------------------------------------
                # compute FFT and ACF of the right
                freq_grid_right = rfftfreq(N_right, d=sampling_rate)
                outside_right_fft = rfft(outside_right - np.mean(outside_right)) ## doing FFT on zero-mean signal
                outside_right_spectrum = np.abs(outside_right_fft)

                outside_right_acf = correlate(outside_right - np.mean(outside_right), outside_right - np.mean(outside_right), mode='full')
                # outside_right_acf = correlate(outside_right, outside_right, mode='full')
                outside_right_acf /= np.max(outside_right_acf) # normalize to acf[0] = 1

                # plt.plot(freq_grid, outside_right_spectrum)
                # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
                # plt.show()
                
                # Interpolate onto the common frequency grid
                interpolated_magnitude = np.interp(common_freq_grid, freq_grid_right, outside_right_spectrum)
                outside_right_fft_magnitudes.append(interpolated_magnitude)

                # Compute the lag grid for this signal and iterpolate
                lag_grid = np.arange(-N_right + 1, N_right)
                interpolated_acf = np.interp(common_lag_grid, lag_grid, outside_right_acf)
                outside_right_acf_values.append(interpolated_acf)

        outside_left_fft_magnitudes = np.array(outside_left_fft_magnitudes)
        outside_right_fft_magnitudes = np.array(outside_right_fft_magnitudes)
        outside_fft_magnitudes = (outside_left_fft_magnitudes + outside_right_fft_magnitudes)/2
        outside_fft_mean = np.mean(outside_fft_magnitudes, axis=0)
       
        outside_left_acf_values = np.array(outside_left_acf_values)
        outside_right_acf_values = np.array(outside_right_acf_values)
        outside_acf_values = (outside_left_acf_values + outside_right_acf_values)/2
        outside_acf_mean = np.mean(outside_acf_values, axis=0)

        # Store the FFT and ACF magnitudes by omega
        if omega not in omega_fft_dict:
            omega_fft_dict[omega] = []
        omega_fft_dict[omega].append(outside_fft_mean)
        if omega not in omega_acf_dict:
            omega_acf_dict[omega] = []
        omega_acf_dict[omega].append(outside_acf_mean)

        # Store the FFT and ACF magnitudes by detuning
        if detuning not in detuning_fft_dict:
            detuning_fft_dict[detuning] = []
        detuning_fft_dict[detuning].append(outside_fft_mean)
        if detuning not in detuning_acf_dict:
            detuning_acf_dict[detuning] = []
        detuning_acf_dict[detuning].append(outside_acf_mean)

        if int(sys.argv[1]) != -1:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

            # Colormap for FFT
            im1 = axs[0, 0].imshow(np.log(outside_fft_magnitudes[:, 1:]), aspect='auto', extent=[common_freq_grid[1], common_freq_grid[-1], 0, len(Z)-1], origin='lower', cmap='plasma')
            fig.colorbar(im1, ax=axs[0, 0], label='Log Magnitude')
            axs[0, 0].set_title(f"outside FFT of day {day}, sequence {seq}")
            axs[0, 0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
            axs[0, 0].set_ylabel("Shot number")

            # Average FFT
            axs[0, 1].plot(common_freq_grid[1:], outside_fft_mean[1:], '-', label='FFT on background outside')
            axs[0, 1].annotate(f"# of outside shots = {len(Z)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            axs[0, 1].set_title(f"outside FFT average of day {day}, sequence {seq}")
            axs[0, 1].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_xlim(-0.02, 0.52)
            axs[0, 1].legend()

            # Colormap for autocorrelation
            im2 = axs[1, 0].imshow(outside_acf_values, aspect='auto', extent=[common_lag_grid[0], common_lag_grid[-1], 0, len(Z)-1], origin='lower', cmap='plasma')
            fig.colorbar(im2, ax=axs[1, 0], label='ACF')
            axs[1, 0].set_title(f"outside ACF of day {day}, sequence {seq}")
            axs[1, 0].set_xlabel("Lag")
            axs[1, 0].set_ylabel("Shot number")

            # Average Autocorrelation
            axs[1, 1].plot(common_lag_grid, outside_acf_mean, '-', label='ACF on background outside')
            axs[1, 1].annotate(f"# of outside shots = {len(Z)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            axs[1, 1].set_title(f"outside ACF average of day {day}, sequence {seq}")
            axs[1, 1].set_xlabel("Lag")
            axs[1, 1].legend()

            plt.tight_layout()
            # plt.savefig(f"thesis/figures/chap2/outside_fft_acf_day_{day}_seq_{0}.png", dpi=500)
            plt.show()

# FFTs and ACFs as a function of omega
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sorted_omegas = sorted(omega_fft_dict.keys())

# Average FFTs and ACFs with the same omega
for omega in sorted_omegas: 
    fft_list = omega_fft_dict[omega]
    avg_fft = np.mean(fft_list, axis=0)
    axs[0].plot(common_freq_grid[1:], avg_fft[1:], '-', label=fr'$\Omega = {omega}$ Hz')

    acf_list = omega_acf_dict[omega]
    avg_acf = np.mean(acf_list, axis=0)
    axs[1].plot(common_lag_grid, avg_acf, '-', label=fr'$\Omega = {omega}$ Hz')

# Plot FFTs
axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
axs[0].set_yscale('log')
axs[0].set_xlim(-0.02, 0.52)
axs[0].legend()
axs[0].set_title("Average outside FFTs")

# Plot ACFs    
axs[1].set_xlabel("lag")
axs[1].legend()
axs[1].set_title("Average outside ACFs")

plt.tight_layout()
# plt.savefig("thesis/figures/chap2/outside_fft_acf_avg.png", dpi=500)
plt.show()

# FFTs and ACFs as a function of detuning
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sorted_detunings = sorted(detuning_fft_dict.keys())

# Average FFTs and ACFs with the same detuning
# Prepare data for colormap
fft_matrix = []
acf_matrix = []

for detuning in sorted_detunings: 
    fft_list = detuning_fft_dict[detuning]
    avg_fft = np.mean(fft_list, axis=0)
    fft_matrix.append(avg_fft[1:])  # Exclude the zero frequency component

    acf_list = detuning_acf_dict[detuning]
    avg_acf = np.mean(acf_list, axis=0)
    acf_matrix.append(avg_acf)

fft_matrix = np.array(fft_matrix)
acf_matrix = np.array(acf_matrix)

# Plot FFT colormap
im1 = axs[0].imshow(np.log(fft_matrix), aspect='auto', extent=[common_freq_grid[1], common_freq_grid[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im1, ax=axs[0], label='Log Magnitude')
axs[0].set_title("Average outside FFTs")
axs[0].set_xlabel(r"$k/(2\pi) [1/\mu m]$")
axs[0].set_ylabel("$\delta$")

# Plot ACF colormap
im2 = axs[1].imshow(acf_matrix, aspect='auto', extent=[common_lag_grid[0], common_lag_grid[-1], sorted_detunings[0], sorted_detunings[-1]], origin='lower', cmap='plasma')
fig.colorbar(im2, ax=axs[1], label='ACF')
axs[1].set_title("Average outside ACFs")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("$\delta$")

plt.tight_layout()
plt.show()