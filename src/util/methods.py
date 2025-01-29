import sys
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate

f, seqs, Omega, knT, detuning, sel_days, sel_seqs = importParameters()

def scriptUsage():
    '''
        Tells the user how to insert command line parameters when running the script
    '''
    if len(sys.argv) > 1:
        if int(sys.argv[1]) == -1:
            chosen_days = sel_days
        else:
            chosen_days = [int(sys.argv[1])]
            if chosen_days[0] not in sel_days:
                print(f"Day {chosen_days[0]} is no good, please use one in {sel_days}")
                exit()
    else:
        print(f"Usage: python3 {sys.argv[0]} <chosen_days>\t\t (use chosen_days = -1 for all selected)")
        exit()
        
    return chosen_days


def quadPlot(day, seq, data, region, CFG, CLG, FFT_mag, FFT_mean, ACF_val, ACF_mean, save_flag):
    '''
        Draws a 2x2 plor showing FFT and ACF of the entire sequence and the average values
        day: day
        seq: sequence
        data: magnetization values
        region: "noise" or "inside" or "outside"
        CFG: Common Frequency Grid
        CLG: Common Lag Grid
        FFT_mag: FFT magnitudes (matrix)
        FFT_mean: FFT average values (array)
        ACF_val: ACF values (matrix)
        ACF_mean: ACF average values (array)
        save_flag: 0 for no, 1 for yes
    '''
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Colormap for FFT
    im1 = axs[0, 0].imshow(FFT_mag, aspect='auto', extent=[CFG[0], CFG[-1], 0, len(data)], origin='lower', cmap='plasma')
    fig.colorbar(im1, ax=axs[0, 0], label='Magnitude')
    axs[0, 0].set_title(region + f" FFT of day {day}, sequence {seq}")
    axs[0, 0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
    axs[0, 0].set_ylabel("Shot number")

    # Average FFT
    axs[0, 1].plot(CFG, FFT_mean, '-', label='FFT on background inside')
    # axs[0, 1].annotate(f"# of inside shots = {len(data)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    axs[0, 1].set_title(region + f" FFT average of day {day}, sequence {seq}")
    axs[0, 1].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
    # axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlim(-0.02, 0.52)
    # axs[0, 1].legend()

    # Colormap for autocorrelation
    im2 = axs[1, 0].imshow(ACF_val, aspect='auto', extent=[CLG[0], CLG[-1], 0, len(data)-1], origin='lower', cmap='plasma')
    fig.colorbar(im2, ax=axs[1, 0], label='ACF')
    axs[1, 0].set_title(region + f" ACF of day {day}, sequence {seq}")
    axs[1, 0].set_xlabel("Lag")
    axs[1, 0].set_ylabel("Shot number")

    # Average Autocorrelation
    axs[1, 1].plot(CLG, ACF_mean, '-', label='ACF on background inside')
    # axs[1, 1].annotate(f"# of inside shots = {len(data)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    axs[1, 1].set_title(region + f" ACF average of day {day}, sequence {seq}")
    axs[1, 1].set_xlabel("Lag")
    # axs[1, 1].legend()

    plt.tight_layout()
    if save_flag:
        plt.savefig(f"thesis/figures/chap2/{region}_fft_acf_day_{day}_seq_{seq}.png", dpi=500)
    return fig

def doublePlot(o_fft_d, o_acf_d, CFG, CLG, region):
    # FFTs and ACFs as a function of omega
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sorted_omegas = sorted(o_fft_d.keys())

    # Average FFTs and ACFs with the same omega
    for omega in sorted_omegas: 
        fft_list = o_fft_d[omega]
        avg_fft = np.mean(fft_list, axis=0)
        axs[0].plot(CFG, avg_fft, '-', label=fr'$\Omega = {omega}$ Hz')

        acf_list = o_acf_d[omega]
        avg_acf = np.mean(acf_list, axis=0)
        axs[1].plot(CLG, avg_acf, '-', label=fr'$\Omega = {omega}$ Hz')

    # Plot FFTs
    axs[0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
    axs[0].set_xlim(-0.02, 0.52)
    axs[0].legend()
    axs[0].set_title(f"Average {region} FFTs")

    # Plot ACFs    
    axs[1].set_xlabel("lag")
    axs[1].legend()
    axs[1].set_title(f"Average {region} ACFs")

    plt.tight_layout()
    return fig

def computeFFT_ACF(zero_mean_flag, data, CFG, CLG, fft_magnitudes, acf_values, w_len):
    '''
        Computes FFT and ACF
        `zero_mean_flag`: 0 for true data, 1 for 0-mean data
        `data`: magnetization values
        `CFG`: Common Frequency Grid
        `CLG`: Common Lag Grid
        `FFT_mag`: FFT magnitudes (matrix)
        `ACF_val`: ACF values (matrix)

    '''
    if zero_mean_flag:
        fft = rfft(data - np.mean(data)) ## doing FFT on zero-mean signal
        acf, lags = myCorrelate(data - np.mean(data), w_len)
    else:
        fft = rfft(data) ## doing FFT on true signal
        acf, lags = myCorrelate(data, w_len)

    # Interpolate onto the common frequency grid
    spectrum = np.abs(fft)
    # spectrum /= np.max(spectrum) # normalization
    freq_grid = rfftfreq(len(data), d=1.0)
    interpolated_magnitude = np.interp(CFG, freq_grid, spectrum)
    max_freq = CFG[np.argmax(interpolated_magnitude)]
    fft_magnitudes.append(interpolated_magnitude)

    # Compute the lag grid for this signal and iterpolate
    acf /= np.max(acf)
    interpolated_acf = np.interp(CLG, lags, acf)
    acf_values.append(interpolated_acf)

    return fft_magnitudes, acf_values, interpolated_magnitude, interpolated_acf

def myCorrelate(data, window_len=40):
    N = len(data)
    if N <= window_len:
        start = 0
        end = N-1
    else:
        start = int((N - window_len)/2)
        end = int((N + window_len)/2)
    # print(start, end)
    ACF = [0] * (window_len * 2 + 1)
    lags = np.arange(-window_len, window_len+1)
    for k in lags:
        for j in range(start, end+1):
            if j+k >= N or j+k < 0:
                continue
            ACF[k + window_len] += data[j]*data[j+k]
    new_ACF = [0] * (window_len + 1)
    new_lags = np.arange(0, window_len+1)
    for k in new_lags:
        new_ACF[k] = (ACF[window_len + k] + ACF[window_len - k])/2
    return new_ACF, new_lags
    