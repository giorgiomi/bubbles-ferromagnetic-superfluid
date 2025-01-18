# This script is used to analyze the data inside the bubble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate

from util.parameters import import_parameters
import sys

# Data
f, seqs, Omega, knT, detuning = import_parameters()
w = 200
sampling_rate = 1.0

if len(sys.argv) > 1:
    if int(sys.argv[1]) == -1:
        chosen_days = np.arange(len(seqs))
    else:
        chosen_days = [int(sys.argv[1])]
else:
    print(f"Usage: {sys.argv[0]} <chosen_days>\t use chosen_days = -1 for all")
    exit()

omega_fft_dict = {}

max_length = 0
for day in chosen_days:
    for seq, seqi in enumerate((seqs[day])):
        df_size_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/sizeADV_sorted.csv", header=None)
        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        max_size = int(max(b_sizeADV_sorted) + 1)
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
        df_in_left_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/in_left_sorted.csv", header=None)
        df_in_right_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/in_right_sorted.csv", header=None)
        df_Z_sorted = pd.read_csv(f"data/processed/day_{day}/seq_{seq}/Z_sorted.csv", header=None)
        omega = Omega[day][seq]

        b_sizeADV_sorted = df_size_sorted.to_numpy().flatten()
        b_center_sorted = df_center_sorted.to_numpy().flatten()
        in_left_sorted = df_in_left_sorted.to_numpy().flatten()
        in_right_sorted = df_in_right_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # FFT on bubble (inside)
        if max_length >= 2*w: continue

        inside_fft_magnitudes = []
        inside_autocorr_values = []
        # cycle through ordered shots from beginning to end
        for i in range(len(Z)):
            y = Z[i] 
            ## NEW METHOD, with inside estimation
            left = in_left_sorted[i]
            right = in_right_sorted[i]
            start = max(0, int(left))
            end = min(len(y), int(right))
    
            ## OLD METHOD
            # center = b_center_sorted[i]
            # extra_width = 0 # to change where FFT is done
            # width = b_sizeADV_sorted[i] + extra_width
            # start = max(0, int(center - width/2))
            # end = min(len(y), int(center + width/2))

            inside = y[start:end]

            # plt.plot(y)
            # plt.plot(np.arange(start, end), inside)
            # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
            # plt.show()

            # compute FFT of the inside
            N = len(inside)
            # print(day, seq, i, N)
            if N > 0:
                freq_grid = rfftfreq(N, d=sampling_rate)
                inside_fft = rfft(inside - np.mean(inside)) ## doing FFT on zero-mean signal
                inside_spectrum = np.abs(inside_fft)

                # inside_autocorr = correlate(inside - np.mean(inside), inside - np.mean(inside), mode='full')
                inside_autocorr = correlate(inside, inside, mode='full')
                inside_autocorr /= np.max(inside_autocorr) # normalize to acf[0] = 1
                # print(f"len autocorr = {len(inside_autocorr)}")

                # plt.plot(freq_grid, inside_spectrum)
                # plt.title(f"Day {day}, Seq {seq}, Shot {i}")
                # plt.show()
                
                # Interpolate onto the common frequency grid
                interpolated_magnitude = np.interp(common_freq_grid, freq_grid, inside_spectrum)
                inside_fft_magnitudes.append(interpolated_magnitude)

                # Compute the lag grid for this signal and iterpolate
                lag_grid = np.arange(-len(inside) + 1, len(inside))
                interpolated_acf = np.interp(common_lag_grid, lag_grid, inside_autocorr)
                inside_autocorr_values.append(interpolated_acf)

        inside_fft_magnitudes = np.array(inside_fft_magnitudes)
        inside_fft_mean = np.mean(inside_fft_magnitudes, axis=0)
        
        inside_autocorr_values = np.array(inside_autocorr_values)
        inside_autocorr_mean = np.mean(inside_autocorr_values)

        # Store the FFT magnitudes by Omega
        if omega not in omega_fft_dict:
            omega_fft_dict[omega] = []
        omega_fft_dict[omega].append(inside_fft_mean)

        if int(sys.argv[1]) != -1:
            # Colormap for FFT
            plt.figure()
            plt.imshow(np.log(inside_fft_magnitudes[:,1:]), aspect='auto', extent=[common_freq_grid[1], common_freq_grid[-1], 0, len(Z)-1], origin='lower', cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.title(f"inside FFT of day {day}, sequence {seq}")
            plt.xlabel("Frequency")
            plt.ylabel("Shot number")
            plt.show()
            
            # Average FFT
            plt.figure()
            plt.plot(common_freq_grid[1:], inside_fft_mean[1:], '-', label='FFT on background inside')
            plt.annotate(f"# of inside shots = {len(Z)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            plt.title(f"FFT analysis of day {day}, sequence {seq}")
            plt.xlabel("f")
            plt.yscale('log')
            plt.xlim(-0.02, 0.52)
            plt.legend()
            plt.show()

            # Colormap for autocorrelation
            plt.figure()
            plt.imshow(inside_autocorr_values, aspect='auto', extent=[common_lag_grid[0], common_lag_grid[-1], 0, len(Z)-1], origin='lower', cmap='viridis')
            plt.colorbar(label='Autocorrelation')
            plt.title(f"inside Autocorrelation of day {day}, sequence {seq}")
            plt.xlabel("Lag")
            plt.ylabel("Shot number")
            plt.show()
            
            # Average Autocorrelation
            plt.figure()
            plt.plot(common_lag_grid, inside_autocorr_mean, '-', label='Autocorrelation on background inside')
            plt.annotate(f"# of inside shots = {len(Z)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            plt.title(f"Autocorrelation analysis of day {day}, sequence {seq}")
            plt.xlabel("Lag")
            plt.legend()
            plt.show()

# Average all FFTs with the same omega
plt.figure()

# Sort the omega keys
sorted_omegas = sorted(omega_fft_dict.keys())

for omega in sorted_omegas: 
    fft_list = omega_fft_dict[omega]
    avg_fft = np.mean(fft_list, axis=0)
    plt.plot(common_freq_grid[1:], avg_fft[1:], '-', label=fr'$\Omega = {omega}$ Hz')
    
plt.xlabel("f")
plt.yscale('log')
plt.xlim(-0.02, 0.52)
plt.legend()
plt.title("Average inside FFTs")
# plt.savefig("thesis/figures/chap2/insideFFT.png", dpi=500)
plt.show()

## Plot autocorr values
plt.figure()
noise_autocorr_mean = np.mean(inside_autocorr_values, axis=0)
plt.plot(noise_autocorr_mean)
plt.title("Inside autocorrelation mean")
plt.show()
