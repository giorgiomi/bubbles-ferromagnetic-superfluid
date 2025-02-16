import sys
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from util.functions import corrGauss, corrExp

# selected_flag = int(input("Enter 1 for selected, 0 for all: "))
# f, seqs, Omega, knT, detuning, sel_days, sel_seqs = importParameters(selected_flag)

def scriptUsage(sel_days):
    '''
        Tells the user how to insert command line parameters when running the script.
        Parameters:
        sel_days (list): A list of valid days that can be selected.
        Returns:
        list: A list containing the chosen days based on the command line input.
        Behavior:
        - If a command line argument is provided:
            - If the argument is -1, all days in sel_days are chosen.
            - Otherwise, the argument is treated as a single day to be chosen.
            - If the chosen day is not in sel_days, an error message is printed and the script exits.
        - If no command line argument is provided, a usage message is printed and the script exits.
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
    fig.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title(region + f" FFT of day {day}, sequence {seq}")
    axs[0, 0].set_xlabel(r"$k\ [1/\mu m]$")
    axs[0, 0].set_ylabel("Shot number")

    # Average FFT
    axs[0, 1].plot(CFG, FFT_mean, '-', label='FFT on background inside')
    # axs[0, 1].annotate(f"# of inside shots = {len(data)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    axs[0, 1].set_title(region + f" FFT average of day {day}, sequence {seq}")
    axs[0, 1].set_xlabel(r"$k\ [1/\mu m]$")
    # axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlim(-0.02, 0.52)
    axs[0, 1].set_ylabel("FFT")
    # axs[0, 1].legend()

    # Colormap for autocorrelation
    im2 = axs[1, 0].imshow(ACF_val, aspect='auto', extent=[CLG[0], CLG[-1], 0, len(data)-1], origin='lower', cmap='plasma')
    fig.colorbar(im2, ax=axs[1, 0])
    axs[1, 0].set_title(region + f" ACF of day {day}, sequence {seq}")
    axs[1, 0].set_xlabel("$\Delta x\ [\mu m]$")
    axs[1, 0].set_ylabel("Shot number")
    axs[1, 0].set_xticks(np.arange(0, 21, 2))

    # Average Autocorrelation
    axs[1, 1].plot(CLG, ACF_mean, '-', label='ACF on background inside')
    # axs[1, 1].annotate(f"# of inside shots = {len(data)}", xy=(0.8, 0.75), xycoords='axes fraction', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    axs[1, 1].set_title(region + f" ACF average of day {day}, sequence {seq}")
    axs[1, 1].set_xlabel("$\Delta x\ [\mu m]$")
    axs[1, 1].set_ylabel("ACF")
    axs[1, 1].set_xticks(np.arange(0, 21, 2))
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
        axs[0].plot(CFG, avg_fft, '-', label=fr'$\Omega_R/2\pi = {omega}$ Hz')

        acf_list = o_acf_d[omega]
        avg_acf = np.mean(acf_list, axis=0)
        axs[1].plot(CLG, avg_acf, '-', label=fr'$\Omega_R/2\pi = {omega}$ Hz')

    # Plot FFTs
    axs[0].set_xlabel(r"$k\ [1/\mu m]$")
    axs[0].set_ylabel("FFT")
    axs[0].set_xlim(-0.02, 0.52)
    axs[0].legend()
    axs[0].set_title(f"Average {region} FFTs")

    # Plot ACFs    
    axs[1].set_xlabel("$\Delta x\ [\mu m]$")
    axs[1].set_ylabel("ACF")
    axs[1].legend()
    axs[1].set_title(f"Average {region} ACFs")
    axs[1].set_xticks(range(0, 21, 2))

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

def groupFitACF(cat_str, cat_data_raw, omega_data, n_blocks, Z_raw, size_raw, center_raw, window_len, ZMF, region):
    '''
        Groups the data and computes the ACF for the inside region of the magnetization profile shots. It then fits the ACF means to a Gaussian correlation function and plots the results.

        Parameters:
        - `cat_str`: [omega/size/time/detuning/dE] in string format
        - `cat_data`: data array
        - `omega_data`: omega data array
        - `n_blocks`: number of blocks to divide the category
        - `Z`: magnetization profile shots
        - `size`: size array
        - `center`: center array
        - `window_len`: window length for ACF computation
        - `ZMF`: Zero-Mean Flag (1 for zero-mean)
        - `region`: [inside/outside] in string format
    '''
    CLG = np.arange(window_len+1)
    # omega_vals = np.unique(omega_data)
    omega_vals = [300, 400, 600, 800]
    fig_fit, ax_fit = plt.subplots(1, 3, figsize=(12, 5))
    fig_pro, ax_pro = plt.subplots(1, len(omega_vals), figsize=(15, 5))
    fig_om, ax_om = plt.subplots(1, 1, figsize=(8, 4))
    colors_om = plt.cm.tab10([0, 1, 2, 3])
    # print(colors_om)
    k = 0

    for om in omega_vals:
        # Filter shots with inside length greater than 4*window_len and omega
        valid_indices = []
        for i, shot in enumerate(Z_raw):
            s = size_raw[i]
            c = center_raw[i]
            omega = omega_data[i]
            if region == 'inside':
                inside = shot[int(c-s/2+10):int(c+s/2-10)]
                if len(inside) > 4 * window_len and omega == om:
                    valid_indices.append(i)
            elif region == 'outside':
                left = shot[:int(c-s/2-10)]
                right = shot[int(c+s/2+10):]
                if len(left) > 4 * window_len and len(right) > 4 * window_len and omega == om:
                    valid_indices.append(i)


        cat_data = cat_data_raw[valid_indices]
        Z = Z_raw[valid_indices]
        size = size_raw[valid_indices]
        center = center_raw[valid_indices]
        print(om, len(cat_data_raw[omega_data == om]), len(cat_data))

        ## GATHER BY CAT
        if cat_str != 'omega':
            # Define cat blocks with a constant number of n_blocks
            sorted_cat_data = np.sort(cat_data)
            cat_blocks = [sorted_cat_data[int(i * len(sorted_cat_data) / n_blocks)] for i in range(n_blocks)]
            # cat_blocks.append(sorted_cat_data[-1] + 1)  # To include the last block
            # print(cat_blocks)
            cat_block_sizes = np.diff(cat_blocks)
            cat_block_sizes = np.append(cat_block_sizes, sorted_cat_data[-1] - cat_blocks[-1])
            # cat_block_sizes = np.append(cat_block_sizes, 0)
            # print(len(cat_block_sizes))
        else:
            cat_blocks = np.unique(cat_data)

        ## PLOT SHOTS, not necessary
        # fig, ax = plt.subplots(figsize=(10, 5), ncols=2)
        # ax[0].pcolormesh(Z, vmin=-1, vmax=+1, cmap='RdBu')
        # ax[0].set_xlabel('$x\ [\mu m]$')
        # ax[0].set_ylabel('shots')

        # # Sort Z by cat
        # sorted_indices = np.argsort(cat_data)
        # Z_sorted = Z[sorted_indices]

        # # Display the ordered shots
        # ax[1].pcolormesh(Z_sorted, vmin=-1, vmax=1, cmap='RdBu')
        # ax[1].set_xlabel('$x\ [\mu m]$')
        # ax[1].set_ylabel('shots')

        # # Add horizontal lines to indicate the different blocks
        # if cat_str != 'omega':
        #     for i, start_cat in enumerate(cat_blocks):
        #         end_cat = start_cat + cat_block_sizes[i]
        #         block_indices = np.where((cat_data[sorted_indices] >= start_cat) & (cat_data[sorted_indices] < end_cat))[0]
        #         # print(start_cat, end_cat, len(block_indices))
        #         if len(block_indices) > 0:
        #             ax[1].axhline(y=block_indices[-1], color='k', linestyle='--', linewidth=0.5)
        # else:
        #     for cat_val in cat_blocks:
        #         block_indices = np.where(cat_data[sorted_indices] == cat_val)[0]
        #         if len(block_indices) > 0:
        #             ax[1].axhline(y=block_indices[-1], color='k', linestyle='--', linewidth=0.5)

        # # plt.show()

        # Initialize lists to store ACF results for each cat block
        acf_values = []
        cat_new = []

        # Group shots by cat block and compute FFT and ACF
        if cat_str != 'omega':
            for i, start_cat in enumerate(cat_blocks):
                end_cat = start_cat + cat_block_sizes[i]
                shots_in_block = Z[(cat_data >= start_cat) & (cat_data < end_cat)]
                # print(i, start_cat, end_cat, len(shots_in_block))
                if len(shots_in_block) > 0:
                    for i, shot in enumerate(shots_in_block):
                        s = size[(cat_data >= start_cat) & (cat_data < end_cat)][i]
                        c = center[(cat_data >= start_cat) & (cat_data < end_cat)][i]
                        if region == 'inside':
                            inside = shot[int(c-s/2+10):int(c+s/2-10)]
                            if ZMF:
                                acf = AZcorr(inside, int(window_len), 1)
                            else:
                                acf = AZcorr(inside, int(window_len), 0)
                            acf_values.append(acf)
                        elif region == 'outside':
                            left = shot[:int(c-s/2-10)]
                            right = shot[int(c+s/2+10):]
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
                        if region == 'inside':
                            inside = shot[int(c-s/2+10):int(c+s/2-10)]
                            if ZMF:
                                acf = AZcorr(inside, int(window_len), 1)
                            else:
                                acf = AZcorr(inside, int(window_len), 0)
                            acf_values.append(acf)
                        elif region == 'outside':
                            left = shot[:int(c-s/2-10)]
                            right = shot[int(c+s/2+10):]
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
        # print(acf_means.keys())

        # Fit the ACF means to the Gaussian correlation function
        fit_params = {}
        fit_errors = {}
        tr_idx = 21 # 21 means all data
        for start_cat, acf_mean in acf_means.items():
            try:
                if region == 'inside':
                    popt, pcorr = curve_fit(corrGauss, CLG[:tr_idx], acf_mean[:tr_idx], p0=[2, -0.1, 2], bounds=((0, -1, 0), (20, 1, 20)))
                elif region == 'outside':
                    popt, pcorr = curve_fit(corrExp, CLG[:tr_idx], acf_mean[:tr_idx], p0=[2, -0.1], bounds=((0, -1), (20, 1)))
                fit_params[start_cat] = popt
                fit_errors[start_cat] = np.sqrt(np.diag(pcorr))
            except RuntimeError:
                print(f"Fit failed for {cat_str} block {start_cat}")

        # Plot fit parameters vs cat
        cats = list(fit_params.keys())
        l1_values = [params[0] for params in fit_params.values()]
        dl1_values = [err[0] for err in fit_errors.values()]
        off_values = [params[1] for params in fit_params.values()]
        doff_values = [err[1] for err in fit_errors.values()]
        if region == 'inside':
            l2_values = [params[2] for params in fit_params.values()]
            dl2_values = [err[2] for err in fit_errors.values()]

        if cat_str == 'omega':
            ax_fit[0].errorbar(1150/np.array(cats), l1_values, yerr=dl1_values, fmt='o', capsize=2, color='tab:blue')
            ax_fit[1].errorbar(1150/np.array(cats), off_values, yerr=doff_values, fmt='o', capsize=2, color='tab:blue')
            ax_fit[0].set_xlabel('$kn/\Omega$')
            ax_fit[1].set_xlabel('$kn/\Omega$')

            if region == 'inside':
                ax_fit[2].errorbar(1150/np.array(cats), l2_values, yerr=dl2_values, fmt='o', capsize=2, color='tab:blue')
                ax_fit[2].set_xlabel('$kn/\Omega$')
        else:
            # Create bins from start_cat to end_cat for each block
            bin_semiwidths = [cat_block_sizes[i]/2 for i in range(len(cat_block_sizes))]
            bin_centers = [cat_blocks[i] + bin_semiwidths[i] for i in range(len(cat_blocks))]
            # print(len(bin_centers), len(bin_semiwidths), len(l1_values))
# 
            ax_fit[0].errorbar(bin_centers, l1_values, xerr=bin_semiwidths, yerr=dl1_values, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
            ax_fit[1].errorbar(bin_centers, off_values, xerr=bin_semiwidths, yerr=doff_values, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
            if region == 'inside':
                ax_fit[2].errorbar(bin_centers, l2_values, xerr=bin_semiwidths, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
                ax_fit[2].set_xlabel(f'{cat_str}')
            ax_fit[0].set_xlabel(f'{cat_str}')
            ax_fit[1].set_xlabel(f'{cat_str}')

        if cat_str == 'time':
            xlim = [0.5, 500]
        elif cat_str == 'size':
            xlim = [25, 280]
        elif cat_str == 'slope':
            xlim = [1e-1, 4e2]
        elif cat_str == 'omega':
            xlim = [1, 4] # kn/omega, not omega
        ax_fit[0].set_title(f'First Fit Parameter ($\ell_1$) vs {cat_str}')
        ax_fit[0].set_ylabel('$\ell_1\ [\mu m]$')
        # ax_fit[0].set_xscale('log')
        ax_fit[0].set_xlim(xlim)
        ax_fit[0].legend()

        ax_fit[1].set_title(f'Second Fit Parameter ($\Delta$) vs {cat_str}')
        ax_fit[1].set_ylabel('$\Delta$')
        # ax_fit[1].set_xscale('log')
        # ax_fit[1].set_yscale('log')
        ax_fit[1].set_xlim(xlim)
        ax_fit[1].legend()

        if region == 'inside':
            ax_fit[2].set_title(f'Third Fit Parameter ($\ell_2$) vs {cat_str}')
            ax_fit[2].set_ylabel('$\ell_2\ [\mu m]$')
            ax_fit[2].set_xscale('log')
            ax_fit[2].set_xlim(xlim)
            ax_fit[2].legend()

        ## Plot ACF means and fits
        if cat_str != 'omega':
            colors = plt.cm.viridis(np.linspace(0, 1, len(acf_means)))
            for color, (start_cat, acf_mean) in zip(colors, acf_means.items()):
                ax_pro[k].plot(CLG, acf_mean, color=color, label=f'{cat_str} {start_cat:.1f}', alpha=0.5)
                if start_cat in fit_params:
                    if region == 'inside':
                        fitted_curve = corrGauss(CLG, *fit_params[start_cat])
                    elif region == 'outside':
                        fitted_curve = corrExp(CLG, *fit_params[start_cat])
                    ax_pro[k].plot(CLG, fitted_curve, linestyle='--', color=color)
            ax_pro[k].set_title(fr'$\Omega_R/2\pi = {om:.0f}$ Hz')
            ax_pro[k].set_xlabel('$\Delta x\ [\mu m]$')
            ax_pro[k].set_ylabel('ACF')
            # ax_pro[k].legend(fontsize='small')
            ax_pro[k].set_xticks(np.arange(0, 21, 4))
        else:
            color = colors_om[k]
            ax_om.plot(CLG, acf_mean, color=color, label=fr'$\Omega_R/2\pi = {om:.0f}$ Hz')
            if region == 'inside':
                fitted_curve = corrGauss(CLG, *fit_params[start_cat])
            elif region == 'outside':
                fitted_curve = corrExp(CLG, *fit_params[start_cat])
            ax_om.plot(CLG, fitted_curve, linestyle='--', color=color)
            ax_om.set_xlabel('$\Delta x\ [\mu m]$')
            ax_om.set_ylabel('ACF')
            ax_om.set_title(f"ACF of {region} shots vs $\Omega_R$")
            ax_om.set_xticks(np.arange(0, 21, 4))
            ax_om.legend()
        k += 1

    fig_fit.tight_layout()
    fig_pro.tight_layout()
    fig_om.tight_layout()
    # fig_pro.savefig(f"thesis/figures/chap2/fit_{cat_str}_{region}.png", dpi=500)
    # fig_fit.savefig(f"thesis/figures/chap2/param_{cat_str}_{region}.png", dpi=500)
    fig_om.savefig(f"thesis/figures/chap2/fit_{cat_str}_{region}.png", dpi=500)
    plt.show()
