import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import quadPlot, computeFFT_ACF
from scipy.fft import rfftfreq
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# Data
selected_flag = int(pd.read_csv(f"data/gathered/selected.csv", header=None).to_numpy().flatten()[0])
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200 # Thomas-Fermi radius, always the same
window_len = 20
n_clusters = 20

# Print script purpose
print("Peaks")
# zero_mean_flag = int(input("Zero mean flag: "))

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

omega_vals = [300, 400, 600, 800]
# det_vals = np.unique(raw_detuning)
# omega_vals = [400]
# det_vals = [596.5]

for om in omega_vals:
    indices = np.where((raw_omega == om))[0]
    size = raw_size[indices]
    time = raw_time[indices]
    center = raw_center[indices]
    in_left = raw_in_left[indices]
    in_right = raw_in_right[indices]
    Z = raw_Z[indices]

    # Reshape size for KMeans
    size_reshaped = size.reshape(-1, 1)

    # Apply KMeans clustering by size
    kmeans_size = KMeans(n_clusters=n_clusters, random_state=0).fit(size_reshaped)
    labels_size = kmeans_size.labels_

    # Calculate average and error for each cluster by size
    clustered_s = np.array([np.mean(size[labels_size == i]) for i in range(n_clusters)])
    err_s = np.array([np.std(size[labels_size == i])/np.sqrt(len(size[labels_size == i])) for i in range(n_clusters)])

    # Plot clustered data on ax_cl
    for j in range(n_clusters):
        s = clustered_s[j]
        Z_clustered = Z[labels_size == j]
        inl = in_left[labels_size == j]
        inr = in_right[labels_size == j]

        inside_fft_magnitudes = []
        inside_acf_values = []
        peak_sizes = []
        peak_freqs = []

        for i in range(len(Z_clustered)):
            y = Z_clustered[i] 
            ## NEW METHOD, with inside estimation
            left = inl[i]
            right = inr[i]
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
                inside_fft_magnitudes, inside_acf_values, fft_inside, _ = computeFFT_ACF(1, inside, CFG, CLG, inside_fft_magnitudes, inside_acf_values, window_len)
                # _, inside_acf_values_TRUE, _, _ = computeFFT_ACF(0, inside, CFG, CLG, inside_fft_magnitudes_TRUE, inside_acf_values_TRUE, window_len)
        
        inside_fft_magnitudes = np.array(inside_fft_magnitudes)
        inside_fft_mean = np.mean(inside_fft_magnitudes, axis=0)

        # Find the peak value of the interpolated magnitude between 0.1 and 0.2
        peaks, _ = find_peaks(inside_fft_mean[(CFG >= 0.1) & (CFG <= 0.2)], prominence=0.05)
        if peaks.size > 0:
            # print(peaks)
            # peak_freq = CFG[(CFG >= 0.1) & (CFG <= 0.2)][peak_index]
            plt.figure()
            plt.plot(CFG, inside_fft_mean)
            plt.plot(CFG[(CFG >= 0.1) & (CFG <= 0.2)][peaks], inside_fft_mean[(CFG >= 0.1) & (CFG <= 0.2)][peaks], 'ro')
            plt.title(f"FFT and Peaks for Omega {om}, cluster {j}")
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.show()
        else:
            peak_freq = 0
        # if peak_freq > 0.1 and peak_freq < 0.2:
        #     peak_sizes.append(size[i])
        #     # peak_freqs.append(peak_freq)
        
        # plt.figure()
        # plt.plot(peak_sizes, peak_freqs, '.')
        # plt.title(f"om {om}, cluster {i}")
        # plt.show()