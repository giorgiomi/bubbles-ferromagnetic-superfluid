import sys
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
from scipy.optimize import curve_fit
from util.functions import corrGauss

selected_flag = int(input("Enter 1 for selected, 0 for all: "))
f, seqs, Omega, knT, detuning, sel_days, sel_seqs = importParameters(selected_flag)

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
        # acf, lags = myCorrelate(data - np.mean(data), w_len)
        acf = AZcorr(data, int(w_len), 1)
    else:
        fft = rfft(data) ## doing FFT on true signal
        # acf, lags = myCorrelate(data, w_len)
        acf = AZcorr(data, int(w_len), 0)

    lags = np.arange(0, int(w_len)+1)

    # Interpolate onto the common frequency grid
    spectrum = np.abs(fft)
    spectrum /= np.max(spectrum) # normalization
    freq_grid = rfftfreq(len(data), d=1.0)
    interpolated_magnitude = np.interp(CFG, freq_grid, spectrum)
    max_freq = CFG[np.argmax(interpolated_magnitude)]
    fft_magnitudes.append(interpolated_magnitude)

    # Compute the lag grid for this signal and iterpolate
    # acf /= np.max(acf)
    # print(len(CLG), len(lags), len(acf))
    # interpolated_acf = np.interp(CLG, lags, acf)
    acf_values.append(acf)

    return fft_magnitudes, acf_values, interpolated_magnitude, acf

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
    
def AZcorr(x, win, cc):
    rr = []
    cen = int(np.shape(x)[0]/2)
    # print(cen)
    if cc == 1:
        x = np.array(x)-np.mean(np.array(x)[cen-win:cen+win])
    if cc == 0:
        x = x
    for i in np.arange(win+1):
        aap = sum(x[cen-win:cen+win]*x[cen-win:cen+win])
        bbp = sum(x[cen-win+i:cen+win+i]*x[cen-win+i:cen+win+i])
        ccp = sum(x[cen-win:cen+win]*x[cen-win+i:cen+win+i])
        aam = sum(x[cen-win:cen+win]*x[cen-win:cen+win])
        bbm = sum(x[cen-win-i:cen+win-i]*x[cen-win-i:cen+win-i])
        ccm = sum(x[cen-win:cen+win]*x[cen-win-i:cen+win-i])
        rr.append(ccp/np.sqrt(aap*bbp)/2+ccm/np.sqrt(aam*bbm)/2)
    return rr

def groupFitACF_inside(cat_str, cat_data, n_blocks, Z, size, center, window_len, ZMF):
    '''
        `cat_str`: [omega/size/time/detuning/dE] in string format
        `cat_data`: data array
        `n_blocks`: number of blocks to divide the category
        `Z`: magnetization profile shots
        `size`: size array
        `center`: center array
        `ZMF`: Zero-Mean Flag (1 for zero-mean)
    '''
    CLG = np.arange(window_len+1)
    ## GATHER BY CAT
    if cat_str != 'omega':
        # Define cat blocks (e.g., every 10 units of cat)
        cat_block_size = (np.max(cat_data) - np.min(cat_data))/n_blocks
        cat_blocks = np.arange(min(cat_data), max(cat_data) + cat_block_size, cat_block_size)
    else:
        cat_blocks = np.unique(cat_data)

    fig, ax = plt.subplots(figsize=(10, 5), ncols=2)
    ax[0].pcolormesh(Z, vmin = -1, vmax = +1, cmap = 'RdBu')
    ax[0].set_xlabel('$x\ [\mu m]$')
    ax[0].set_ylabel('shots')

    # Sort Z by cat
    sorted_indices = np.argsort(cat_data)
    Z_sorted = Z[sorted_indices]

    # Display the ordered shots
    ax[1].pcolormesh(Z_sorted, vmin=-1, vmax=1, cmap='RdBu')
    ax[1].set_xlabel('$x\ [\mu m]$')
    ax[1].set_ylabel('shots')

    # Add vertical lines to indicate the different blocks
    if cat_str != 'omega':
        for start_cat in cat_blocks:
            end_cat = start_cat + cat_block_size
            block_indices = np.where((cat_data >= start_cat) & (cat_data < end_cat))[0]
            if len(block_indices) > 0:
                ax[1].axhline(y=block_indices[-1], color='k', linestyle='--', linewidth=0.5)
    else:
        for cat_val in cat_blocks:
            block_indices = np.where(cat_data == cat_val)[0]
            if len(block_indices) > 0:
                ax[1].axhline(y=block_indices[-1], color='k', linestyle='--', linewidth=0.5)

    plt.show()

    # Initialize lists to store ACF results for each cat block
    acf_values = []
    cat_new = []

    # Group shots by cat block and compute FFT and ACF
    if cat_str != 'omega':
        for start_cat in cat_blocks:
            end_cat = start_cat + cat_block_size
            shots_in_block = Z[(cat_data >= start_cat) & (cat_data < end_cat)]
            if len(shots_in_block) > 0:
                for i, shot in enumerate(shots_in_block):
                    s = size[(cat_data >= start_cat) & (cat_data < end_cat)][i]
                    c = center[(cat_data >= start_cat) & (cat_data < end_cat)][i]
                    inside = shot[int(c-s/2+10):int(c+s/2-10)]
                    if len(inside) > 4*window_len:
                        if ZMF:
                            acf = AZcorr(inside, int(window_len), 1)
                        else:
                            acf = AZcorr(inside, int(window_len), 0)
                        acf_values.append(acf)
                        cat_new.append(start_cat)
    else:
        for cat_val in cat_blocks:
            shots_in_block = Z[cat_data == cat_val]
            if len(shots_in_block) > 0:
                for i, shot in enumerate(shots_in_block):
                    s = size[cat_data == cat_val][i]
                    c = center[cat_data == cat_val][i]
                    inside = shot[int(c-s/2+10):int(c+s/2-10)]
                    if len(inside) > 4*window_len:
                        if ZMF:
                            acf = AZcorr(inside, int(window_len), 1)
                        else:
                            acf = AZcorr(inside, int(window_len), 0)
                        acf_values.append(acf)
                        cat_new.append(cat_val)

    acf_values = np.array(acf_values)
    cat_new = np.array(cat_new)

    # Compute mean ACF for each cat block
    acf_means = {start_cat: np.mean(acf_values[cat_new == start_cat], axis=0) for start_cat in cat_blocks if start_cat in cat_new}

    # Fit the ACF means to the Gaussian correlation function
    trunc_index = window_len
    fit_params = {}
    fit_errors = {}
    for start_cat, acf_mean in acf_means.items():
        try:
            popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
            fit_params[start_cat] = popt
            fit_errors[start_cat] = np.sqrt(np.diag(pcorr))
        except RuntimeError:
            print(f"Fit failed for {cat_str} block {start_cat}")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ACF means and fits
    colors = plt.cm.viridis(np.linspace(0, 1, len(acf_means)))

    for color, (start_cat, acf_mean) in zip(colors, acf_means.items()):
        ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'{cat_str} {start_cat:.1f}', alpha=0.5)
        if start_cat in fit_params:
            fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[start_cat])
            ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
    ax[0].set_title(f'Mean ACF and fits by {cat_str} block')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()
    # ax[0].legend()

    # Plot the first fit parameter (l1) vs cat
    cats = list(fit_params.keys())
    l1_values = [params[0] for params in fit_params.values()]
    dl1_values = [err[0] for err in fit_errors.values()]

    ax[1].errorbar(cats, l1_values, yerr=dl1_values, marker='o', linestyle='none', capsize=2)
    ax[1].set_title(f'First Fit Parameter ($l_1$) vs {cat_str}')
    ax[1].set_xlabel(f'{cat_str}')
    ax[1].set_ylabel('l1')
    # ax[1].set_yscale('log')

    plt.tight_layout()
    plt.show()

def groupFitACF_outside(cat_str, cat_data, n_blocks, Z, size, center, window_len, ZMF):
    '''
        `cat_str`: [omega/size/time/detuning/dE] in string format
        `cat_data`: data array
        `n_blocks`: number of blocks to divide the category
        `Z`: magnetization profile shots
        `size`: size array
        `center`: center array
        `ZMF`: Zero-Mean Flag (1 for zero-mean)
    '''
    CLG = np.arange(window_len+1)
    ## GATHER BY CAT
    if cat_str != 'omega':
        # Define cat blocks (e.g., every 10 units of cat)
        cat_block_size = (np.max(cat_data) - np.min(cat_data))/n_blocks
        cat_blocks = np.arange(min(cat_data), max(cat_data) + cat_block_size, cat_block_size)
    else:
        cat_blocks = np.unique(cat_data)

    # Initialize lists to store ACF results for each cat block
    acf_values = []
    cat_new = []

    # Group shots by cat block and compute FFT and ACF
    if cat_str != 'omega':
        for start_cat in cat_blocks:
            end_cat = start_cat + cat_block_size
            shots_in_block = Z[(cat_data >= start_cat) & (cat_data < end_cat)]
            if len(shots_in_block) > 0:
                for i, shot in enumerate(shots_in_block):
                    s = size[(cat_data >= start_cat) & (cat_data < end_cat)][i]
                    c = center[(cat_data >= start_cat) & (cat_data < end_cat)][i]
                    left = shot[:int(c-s/2-10)]
                    right = shot[int(c+s/2+10):]
                    if len(left) > 4*window_len and len(right) > 4*window_len:
                        if ZMF:
                            acf_left = AZcorr(left, int(window_len), 1)
                            acf_right = AZcorr(right, int(window_len), 1)
                        else:
                            acf_left = AZcorr(left, int(window_len), 0)
                            acf_right = AZcorr(right, int(window_len), 0)
                        acf_values.append(0.5*(np.array(acf_left) + np.array(acf_right)))
                        cat_new.append(start_cat)
    else:
        for cat_val in cat_blocks:
            shots_in_block = Z[cat_data == cat_val]
            if len(shots_in_block) > 0:
                for i, shot in enumerate(shots_in_block):
                    s = size[cat_data == cat_val][i]
                    c = center[cat_data == cat_val][i]
                    left = shot[:int(c-s/2-10)]
                    right = shot[int(c+s/2+10):]
                    if len(left) > 4*window_len and len(right) > 4*window_len:
                        if ZMF:
                            acf_left = AZcorr(left, int(window_len), 1)
                            acf_right = AZcorr(right, int(window_len), 1)
                        else:
                            acf_left = AZcorr(left, int(window_len), 0)
                            acf_right = AZcorr(right, int(window_len), 0)
                        acf_values.append(0.5*(np.array(acf_left) + np.array(acf_right)))
                        cat_new.append(cat_val)

    acf_values = np.array(acf_values)
    cat_new = np.array(cat_new)

    # Compute mean ACF for each cat block
    acf_means = {start_cat: np.mean(acf_values[cat_new == start_cat], axis=0) for start_cat in cat_blocks if start_cat in cat_new}

    # Fit the ACF means to the Gaussian correlation function
    trunc_index = window_len
    fit_params = {}
    fit_errors = {}
    for start_cat, acf_mean in acf_means.items():
        try:
            popt, pcorr = curve_fit(corrGauss, CLG[:trunc_index], acf_mean[:trunc_index], p0=[1, -0.1])
            fit_params[start_cat] = popt
            fit_errors[start_cat] = np.sqrt(np.diag(pcorr))
        except RuntimeError:
            print(f"Fit failed for {cat_str} block {start_cat}")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ACF means and fits
    colors = plt.cm.viridis(np.linspace(0, 1, len(acf_means)))

    for color, (start_cat, acf_mean) in zip(colors, acf_means.items()):
        ax[0].plot(CLG[:trunc_index], acf_mean[:trunc_index], color=color, label=f'{cat_str} {start_cat:.1f}', alpha=0.5)
        if start_cat in fit_params:
            fitted_curve = corrGauss(CLG[:trunc_index], *fit_params[start_cat])
            ax[0].plot(CLG[:trunc_index], fitted_curve, linestyle='--', color=color)
    ax[0].set_title(f'Mean ACF and fits by {cat_str} block')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()
    # ax[0].legend()

    # Plot the first fit parameter (l1) vs cat
    cats = list(fit_params.keys())
    l1_values = [params[0] for params in fit_params.values()]
    dl1_values = [err[0] for err in fit_errors.values()]

    ax[1].errorbar(cats, l1_values, yerr=dl1_values, marker='o', linestyle='none', capsize=2)
    ax[1].set_title(f'First Fit Parameter ($l_1$) vs {cat_str}')
    ax[1].set_xlabel(f'{cat_str}')
    ax[1].set_ylabel('l1')
    # ax[1].set_yscale('log')

    plt.tight_layout()
    plt.show()