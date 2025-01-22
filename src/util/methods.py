import sys
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate

f, seqs, Omega, knT, detuning, sel_days, sel_seqs = importParameters()

def scriptUsage():
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Colormap for FFT
    im1 = axs[0, 0].imshow(np.log(FFT_mag[:, 1:]), aspect='auto', extent=[CFG[1], CFG[-1], 0, len(data)-1], origin='lower', cmap='plasma')
    fig.colorbar(im1, ax=axs[0, 0], label='Log Magnitude')
    axs[0, 0].set_title(region + f" FFT of day {day}, sequence {seq}")
    axs[0, 0].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
    axs[0, 0].set_ylabel("Shot number")

    # Average FFT
    axs[0, 1].plot(CFG[1:], FFT_mean[1:], '-', label='FFT on background inside')
    # axs[0, 1].annotate(f"# of inside shots = {len(data)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    axs[0, 1].set_title(region + f" FFT average of day {day}, sequence {seq}")
    axs[0, 1].set_xlabel(r"$k/(2\pi)\ [1/\mu m]$")
    axs[0, 1].set_yscale('log')
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

def computeFFT_ACF(zero_mean_flag, data, CFG, fft_magnitudes, acf_values):
    if zero_mean_flag:
        noise_fft = rfft(data - np.mean(data)) ## doing FFT on zero-mean signal
        noise_acf = correlate(data - np.mean(data), data - np.mean(data), mode='full')
    else:
        noise_fft = rfft(data) ## doing FFT on true signal
        noise_acf = correlate(data, data, mode='full')
    
    noise_spectrum = np.abs(noise_fft)
    noise_acf /= np.max(noise_acf)
    noise_freq_grid = rfftfreq(len(data), d=1.0)

    # Interpolate onto the common frequency grid
    interpolated_noise_magnitude = np.interp(CFG, noise_freq_grid, noise_spectrum)
    fft_magnitudes.append(interpolated_noise_magnitude)
    
    # Interpolate onto the common lag grid
    acf_values.append(noise_acf)
    lag_grid = np.arange(-len(data) + 1, len(data))

    return fft_magnitudes, acf_values, lag_grid