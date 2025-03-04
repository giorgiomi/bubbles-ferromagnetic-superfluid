import sys
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from util.functions import corrGauss, corrExp
from sklearn.cluster import KMeans


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
    axs[0, 0].set_ylabel("Size index")

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
    im2 = axs[1, 0].imshow(ACF_val, aspect='auto', extent=[CLG[0], CLG[-1], 0, len(data)-1], origin='lower', cmap='plasma', vmin=0.9, vmax=1)
    fig.colorbar(im2, ax=axs[1, 0])
    axs[1, 0].set_title(region + f" ACF of day {day}, sequence {seq}")
    axs[1, 0].set_xlabel("$\Delta x\ [\mu m]$")
    axs[1, 0].set_ylabel("Size index")
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharey='row')
    sorted_omegas = sorted(o_fft_d.keys())

    # Average FFTs and ACFs with the same omega
    for omega in sorted_omegas: 
        fft_list = o_fft_d[omega]
        avg_fft = np.mean(fft_list, axis=0)
        axs[0, 0].plot(CFG, avg_fft, '-', label=fr'$\Omega_R/2\pi = {omega}$ Hz')
        axs[0, 1].plot(1/np.array(CFG[1:]), avg_fft[1:], '-', label=fr'$\Omega_R/2\pi = {omega}$ Hz')

        acf_list = o_acf_d[omega]
        avg_acf = np.mean(acf_list, axis=0)
        axs[1, 0].plot(1/np.array(CLG[1:]), avg_acf[1:], '-', label=fr'$\Omega_R/2\pi = {omega}$ Hz')
        axs[1, 1].plot(CLG, avg_acf, '-', label=fr'$\Omega_R/2\pi = {omega}$ Hz')

    # Plot FFTs
    axs[0, 0].set_xlabel(r"$k\ [1/\mu m]$")
    axs[0, 0].set_ylabel("FFT")
    axs[0, 0].set_xlim(-0.02, 0.52)
    axs[0, 0].legend()
    axs[0, 0].set_title(f"k-domain average {region} FFTs")

    # Plot FFTs INVERTED-domain
    axs[0, 1].set_xlabel("$1/k\ [\mu m]$")
    # axs[0, 1].set_ylabel("FFT")
    axs[0, 1].legend()
    axs[0, 1].set_title(f"x-domain average {region} FFTs")
    axs[0, 1].set_xticks(range(0, 21, 2))
    axs[0, 1].set_xlim(-0.5, 20.5)
    
    # axs[0, 1].set_xscale('log')

    # Plot ACFs INVERTED-domain  
    axs[1, 0].set_xlabel("$1/\Delta x\ [1/\mu m]$")
    axs[1, 0].set_ylabel("ACF")
    axs[1, 0].legend()
    axs[1, 0].set_title(f"k-domain average {region} ACFs")
    axs[1, 0].set_xlim(-0.02, 0.52)

    # Plot ACFs    
    axs[1, 1].set_xlabel("$\Delta x\ [\mu m]$")
    # axs[1, 1].set_ylabel("ACF")
    axs[1, 1].legend()
    axs[1, 1].set_title(f"x-domain average {region} ACFs")
    axs[1, 1].set_xticks(range(0, 21, 2))
    axs[1, 1].set_xlim(-0.5, 20.5)

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

def groupFitACF(cat_str, cat_data_raw, omega_data, n_blocks, Z_raw, window_len, ZMF, region, in_left_raw, in_right_raw, n_clusters):
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
    colors_om = plt.cm.tab10([0, 1, 2, 3])
    k = 0

    # fig_fit, ax_fit = plt.subplots(1, 3, figsize=(12, 5))
    fig_pro, ax_pro = plt.subplots(1, len(omega_vals), figsize=(15, 5))

    fig_om = plt.figure(figsize=(15, 5))
    ax_om = [plt.subplot(131), plt.subplot(132), plt.subplot(133)]

    if region == 'inside':
        fig_fit = plt.figure(figsize=(12, 5))
        ax_fit = [plt.subplot(131), plt.subplot(132), plt.subplot(133)]
    else:
        fig_fit = plt.figure(figsize=(8, 6))
        ax_fit = [plt.subplot(121), plt.subplot(122)]

    
    if cat_str == 'omega':
        displ_str = r'$\Omega_R/2\pi$ [Hz]'
    elif cat_str == 'size':
        displ_str = r'$\sigma_B\ [\mu$m]'
    elif cat_str == 'time':
        displ_str = r'$t$ [ms]'

    for om in omega_vals:
        # Filter shots with inside length greater than 4*window_len and omega
        valid_indices = []
        for i, shot in enumerate(Z_raw):
            i_l = in_left_raw[i]
            i_r = in_right_raw[i]
            omega = omega_data[i]
            if region == 'inside':
                inside = shot[int(i_l):int(i_r)]
                if len(inside) > 4 * window_len and omega == om:
                    # if (len(inside) > 390):
                    #     print(len(inside))
                    valid_indices.append(i)
            elif region == 'outside':
                left = shot[:int(i_l-20)]
                right = shot[int(i_r+20):]
                if len(left) > 4 * window_len and len(right) > 4 * window_len and omega == om:
                    valid_indices.append(i)

        cat_data = cat_data_raw[valid_indices]
        Z = Z_raw[valid_indices]
        in_left = in_left_raw[valid_indices]
        in_right = in_right_raw[valid_indices]
        print(om, len(cat_data_raw[omega_data == om]), len(cat_data))        

        ## CLUSTER BY CAT
        cat_data_reshaped = cat_data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cat_data_reshaped)
        labels = kmeans.labels_
        # cat_blocks = [cat_data[labels == i] for i in range(n_clusters)]

        # Initialize lists to store ACF results for each cat block
        acf_values = []
        cl_idx = []

        # Group shots by cat block and compute FFT and ACF
        for i in range(n_clusters):
            shots_in_cluster = Z[labels == i]
            if len(shots_in_cluster) > 0:
                for j, shot in enumerate(shots_in_cluster):
                    i_l = in_left[labels == i][j]
                    i_r = in_right[labels == i][j]
                    if region == 'inside':
                        inside = shot[int(i_l):int(i_r)]
                        if ZMF:
                            acf = AZcorr(inside, int(window_len), 1)
                        else:
                            acf = AZcorr(inside, int(window_len), 0)
                        acf_values.append(acf)
                    elif region == 'outside':
                        left = shot[:int(i_l-20)]
                        right = shot[int(i_r+20):]
                        if ZMF:
                            acf_left = AZcorr(left, int(window_len), 1)
                            acf_right = AZcorr(right, int(window_len), 1)
                        else:
                            acf_left = AZcorr(left, int(window_len), 0)
                            acf_right = AZcorr(right, int(window_len), 0)
                        acf_values.append(0.5*(np.array(acf_left) + np.array(acf_right)))
                    cl_idx.append(i)

        acf_values = np.array(acf_values)
        cl_idx = np.array(cl_idx)

        # Compute mean ACF for each cat block
        print(len(acf_values[0 == cl_idx]))
        acf_means = {i: np.mean(acf_values[i == cl_idx], axis=0) for i in range(n_clusters)}
        acf_errs = {i: np.std(acf_values[i == cl_idx], axis=0) / np.sqrt(len(acf_values[i == cl_idx])) for i in range(n_clusters)}
        # print(acf_means.values())

        # Avg cat inside each cluster
        avg_cat = np.array([np.mean(cat_data[labels == i]) for i in range(n_clusters)])
        err_cat = np.array([np.std(cat_data[labels == i])/np.sqrt(len(cat_data[labels == i])) for i in range(n_clusters)])

        # Fit the ACF means to the Gaussian correlation function
        fit_params = {}
        fit_errors = {}
        if cat_str != 'omega':
            if om == 300: tr_idx = 21 # 21 means all data
            else: tr_idx = 12
        else:
            tr_idx = 21
            # if om in [300, 800]: tr_idx = 21
            # else: tr_idx = 14
        tr_idx_out = 12
        for cl_idx, acf_mean in acf_means.items():
            try:
                if region == 'inside':
                    popt, pcorr = curve_fit(corrGauss, CLG[:tr_idx], acf_mean[:tr_idx], p0=[2, -0.1, 12], bounds=((0, -1, 0), (20, 1, 20)))
                    if np.abs(popt[2] - 20.0) < 1e-4 and cat_str != 'omega':
                        # print(f"{om} Bound value")
                        popt[2] = 12
                        popt, pcorr = curve_fit(corrGauss, CLG, acf_mean, p0=popt, bounds=((0, -1, 11.99), (20, 1, 12)))
                elif region == 'outside':
                    popt, pcorr = curve_fit(corrExp, CLG[:tr_idx_out], acf_mean[:tr_idx_out], p0=[2, -0.1], bounds=((0, -1), (20, 1)))
                fit_params[cl_idx] = popt
                fit_errors[cl_idx] = np.sqrt(np.diag(pcorr))
            except RuntimeError:
                print(f"Fit failed for {displ_str} cluster {cl_idx}")

        # Plot fit parameters vs cat
        cats = list(fit_params.keys())
        l1_values = [params[0] for params in fit_params.values()]
        dl1_values = [err[0] for err in fit_errors.values()]
        off_values = [params[1] for params in fit_params.values()]
        doff_values = [err[1] for err in fit_errors.values()]
        if region == 'inside' and len(cats) > 2:
            l2_values = [params[2] for params in fit_params.values()]
            dl2_values = [err[2] for err in fit_errors.values()]

        if cat_str == 'omega':
            if region == 'outside':
                if om == 300:
                    ax_om[1].errorbar(1/np.sqrt(om), l1_values, yerr=dl1_values, fmt='o', capsize=4, color='tab:purple', label='ACF data', markersize=8, elinewidth=2)
                else:
                    ax_om[1].errorbar(1/np.sqrt(om), l1_values, yerr=dl1_values, fmt='o', capsize=4, color='tab:purple', markersize=8, elinewidth=2)
                ax_om[1].set_xlabel('$1/\sqrt{\Omega_R/2\pi}$ [$Hz^{-1/2}$]')
            else:
                ax_om[1].errorbar(om, l1_values, yerr=dl1_values, fmt='o', capsize=4, color='tab:purple', markersize=8, elinewidth=2)
                ax_om[1].set_xlabel(displ_str)
            ax_om[1].set_ylabel(r"$\ell_1\ [\mu $m]")

            ax_om[2].errorbar(om, off_values, yerr=doff_values, fmt='o', capsize=4, color='tab:purple', markersize=8, elinewidth=2)
            ax_om[2].set_xlabel(displ_str)
            ax_om[2].set_ylabel(r"$\Delta$")

            if region == 'inside' and len(cats) > 2:
                ax_om[3].errorbar(om, l2_values, yerr=dl2_values, fmt='o', capsize=2, color='tab:purple')
                ax_om[3].set_xlabel(displ_str)
                ax_om[3].set_ylabel(r"$\ell_2\ [\mu $m]")
        else:
            ax_fit[0].errorbar(avg_cat, l1_values, xerr=err_cat, yerr=dl1_values, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
            if region == 'inside':
                ax_fit[2].errorbar(avg_cat, off_values, xerr=err_cat, yerr=doff_values, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
                l2_values = np.array(l2_values)
                dl2_values = np.array(dl2_values)
                mask = (np.abs(l2_values - 12.0) > 1e-4) & (l2_values < 19.0)
                # print(mask)
                ax_fit[1].errorbar(avg_cat[mask], l2_values[mask], xerr=err_cat[mask], yerr=dl2_values[mask], fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
                ax_fit[2].set_xlabel(f'{displ_str}')
            else:
                ax_fit[1].errorbar(avg_cat, off_values, xerr=err_cat, yerr=doff_values, fmt='o', capsize=2, label=f'$\Omega_R/2\pi = {om}$ Hz')
            
            ax_fit[1].set_xlabel(f'{displ_str}')
            ax_fit[0].set_xlabel(f'{displ_str}')


        if cat_str == 'time':
            xlim = [1, 500]
        elif cat_str == 'size':
            xlim = [25, 300]
        elif cat_str == 'slope':
            xlim = [1e-1, 4e2]
        elif cat_str == 'omega':
            xlim = [1, 4] # kn/omega, not omega

        ax_fit[0].set_ylabel('$\ell_1\ [\mu m]$')
        
        ax_fit[0].legend()

        if cat_str == 'time':
            ax_fit[0].set_xlim(xlim)
            ax_fit[1].set_xlim(xlim)
            ax_fit[0].set_xscale('log')
            ax_fit[1].set_xscale('log')
            if region == 'inside':
                ax_fit[2].set_xscale('log')

        if region == 'inside':
            ax_fit[0].set_xlim(xlim)

            ax_fit[1].set_ylabel('$\ell_2\ [\mu m]$')
            ax_fit[1].set_xlim(xlim)
            ax_fit[1].legend()

            ax_fit[2].set_ylabel('$\Delta$')
            ax_fit[2].set_xlim(xlim)
            ax_fit[2].legend(loc='upper right')
        else:
            if cat_str == 'size':
                ax_fit[0].set_xlim([25, 260])
                ax_fit[1].set_xlim([25, 260])
                ax_fit[0].set_ylim((1,4.2))
                ax_fit[1].set_ylim((0.88,1.01))
            ax_fit[1].legend(loc='lower left')
            ax_fit[1].set_ylabel("$\Delta$")

        ## Plot ACF means and fits
        if cat_str != 'omega':
            sorted_indices = np.argsort(avg_cat)
            sorted_colors = plt.cm.viridis(np.linspace(0, 1, len(acf_means)))
            for idx, color in zip(sorted_indices, sorted_colors):
                if (om == 300 and idx == sorted_indices[0]) or (om == 400 and idx == sorted_indices[-1]):
                    continue
                cl_idx = cats[idx]
                acf_mean = acf_means[cl_idx]
                acf_err = acf_errs[cl_idx]
                ax_pro[k].plot(CLG, acf_mean, color=color, label=f'{cl_idx}', alpha=0.5)
                if cl_idx in fit_params:
                    if region == 'inside':
                        fitted_curve = corrGauss(CLG, *fit_params[cl_idx])
                        ax_pro[k].plot(CLG[:tr_idx], fitted_curve[:tr_idx], linestyle='--', color=color)
                    elif region == 'outside':
                        fitted_curve = corrExp(CLG, *fit_params[cl_idx])
                        ax_pro[k].plot(CLG[:tr_idx_out], fitted_curve[:tr_idx_out], linestyle='--', color=color)
            ax_pro[k].set_title(fr'$\Omega_R/2\pi = {om:.0f}$ Hz')
            ax_pro[k].set_xlabel('$\Delta x\ [\mu m]$')
            ax_pro[k].set_ylabel('ACF')
            # ax_pro[k].legend(fontsize='small')
            ax_pro[k].set_xticks(np.arange(0, 21, 4))
        else:
            color = colors_om[k]
            acf_mean = list(acf_means.values())[0]
            acf_err = list(acf_errs.values())[0]
            # print(acf_mean)
            ax_om[0].errorbar(CLG, acf_mean, yerr=acf_err, color=color, label=fr'$\Omega_R/2\pi = {om:.0f}$ Hz', fmt='o', capsize=2, alpha=0.5)
            if region == 'inside':
                fitted_curve = corrGauss(CLG, *fit_params[cl_idx])
                ax_om[0].plot(CLG[:tr_idx], fitted_curve[:tr_idx], linestyle='--', color=color)
            elif region == 'outside':
                fitted_curve = corrExp(CLG, *fit_params[cl_idx])
                ax_om[0].plot(CLG[:tr_idx_out], fitted_curve[:tr_idx_out], linestyle='--', color=color)
            ax_om[0].set_xlabel('$\Delta x\ [\mu m]$')
            ax_om[0].set_ylabel('ACF')
            # ax_om[0].set_title(f"ACF of {region} shots vs $\Omega_R$")
            ax_om[0].set_xticks(np.arange(0, 21, 4))
            ax_om[0].legend()

        k += 1
    if region == 'outside':
        xi_R = [np.sqrt(0.2741/om)*100 for om in omega_vals]
        ax_om[1].plot([1/np.sqrt(om) for om in omega_vals], xi_R, '-', color='tab:cyan', label=r'$\xi_R$')
    ax_om[1].legend()
    fig_om.suptitle(f"ACF of {region} shots - Fits and parameters $\ell_1$, $\Delta$")
    if region == 'inside':
        fig_fit.suptitle(f"ACF of {region} shots - Fit parameters $\ell_1$, $\ell_2$, $\Delta$")
    else:
        fig_fit.suptitle(f"ACF of {region} shots - Fit parameters $\ell_1$, $\Delta$")
    
    plt.rcParams.update({'font.size': 12})
    fig_fit.tight_layout()
    fig_pro.tight_layout()
    fig_om.tight_layout()
    # fig_pro.savefig(f"thesis/figures/chap2/fit_{cat_str}_{region}.png", dpi=500)
    # fig_fit.savefig(f"thesis/figures/chap2/param_{cat_str}_{region}.png", dpi=500)
    # fig_om.savefig(f"thesis/figures/chap2/fit_{cat_str}_{region}.png", dpi=500)
    plt.show()
