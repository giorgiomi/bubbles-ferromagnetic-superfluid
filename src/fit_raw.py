# Saves data from hdf to csv, with fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from util.parameters import importParameters
from util.functions import bubble, gauss, bubbleshoulder, bubblePieces # fit functions

import os
import warnings
warnings.filterwarnings('ignore')

# Print script purpose
print("\nFitting data, throwing away gaussian fits\n")

# Ask the user for FFT and ACF on true or zero mean signal
save_flag = int(input("Enter 0 for just plotting or 1 for just saving: "))

# Ask the user for selected or processed
selected_flag = int(input("Enter 1 for selected, 0 for all: "))
if selected_flag: 
    str = "selected"
else: 
    str = "processed"

# Data import
f, seqs, Omega, knT, detuning, sel_days, sel_seqs = importParameters(selected_flag)


## Data Analysis

# Parameters
c = 400 # center of the ROI (Region Of Interest)
w = 200 # half width of the ROI
b_check = 20 # half size of the central region used to discriminate the bubble
s_size = 30 # half size of the region to fit the shoulder
threshold = -0.2 # used to discriminate the bubble

# Cycle through days
for fs in sel_days: # all seqs
    if not os.path.exists(f"data/{str}/day_{fs}"):
        os.mkdir(f"data/{str}/day_{fs}") # create a folder for the day
    df12_ = pd.read_hdf(f[fs]) # importing sequence

    #Cycle through sequences
    # for ei, ai in enumerate((seqs[fs])):
    for ei in sel_seqs[fs]:
        ai = seqs[fs][ei]
        # print(fs, ei)
        if not os.path.exists(f"data/{str}/day_{fs}/seq_{ei}"):
            os.mkdir(f"data/{str}/day_{fs}/seq_{ei}") # create a folder for the sequence

        m1 = [] # magnetization UP
        m2 = [] # magnetization DOWN
        time = []
        rep = []

        #Cycle through shots
        for i in ai:
            df12 = df12_[(df12_['sequence_index'].isin([i]))]
            df12 = df12[(df12[('od_remove_thpart', 'N_bec')] > 2e5)] # remove BECs with N < 2e5
            df12 = df12[(df12[('od_remove_thpart', 'N_bec')] < 2e6)] # remove BECs with N > 2e6
            y_axis = 'uW_pulse' # experimental waiting time
            df12 = df12.sort_values(y_axis)
            #print(df12)

            m1 = np.append(m1, df12[('od_remove_thpart', 'm1_1d')]) # add magnetization
            m2 = np.append(m2, df12[('od_remove_thpart', 'm2_1d')]) # add magnetization
            rep = np.append(rep, df12[('Repetition')]) # ??
            time = np.append(time, df12[('uW_pulse')]) # ??
        
        # To array
        m1 = np.array(m1)
        m2 = np.array(m2)
        time = np.array(time)

        # Cleaning for shape and type misbehaviours
        shape = (800, )
        check = type(np.ndarray(1))
        good = np.array([type(p) == check for p in m1]) # type
        good[good] = [p.shape == shape for p in m1[good]] # shape
        M1 = m1[good]
        M2 = m2[good]
        time = time[good]

        # Selecting Region Of Interest
        ROI = np.s_[:, (c - w):(c + w)] # ROI is wide 2 * w
        D = M1 + M2 * 1.3 # density (with adjutsing param)
        M = (M2 * 1.3 - M1) / D # magnetization (with adjusting param)
        D = np.vstack(D)[ROI]
        M = np.vstack(M)[ROI]
        #print(np.shape(M))

        # Cleaning from absurd densities
        mask_D = ((np.sum(D, axis = 1) > 0.1e5) & (np.sum(D, axis = 1) < 8e5) & (np.sum(M, axis = 1) < 8000))
        D = D[mask_D]
        M = M[mask_D]
        time = time[mask_D]

        # Central region
        b_check_ROI = np.s_[:, (w - b_check):(w + b_check)] # Bubble checking for central region
        Mb = np.mean(M[b_check_ROI], axis=1) # average magnetization in the central 2*b_check pixels
        n_shots = len(Mb)

        # Full region
        MK = np.mean(M, axis=1)
        LL = w * (-MK + 1) / 2 # guess bubble size from avg magnetization of full ROI
        
        # Initialising quantities
        xx = np.arange(2 * w)
        b_size = []
        b_sizeADV = []
        b_center = []
        s_width = []
        b_inside_boundary_left = []
        b_inside_boundary_right = []
        b_outside_boundary_left = []
        b_outside_boundary_right = []
        b_exp_slope_left = []
        b_exp_slope_right = []
        b_off = []
        times = np.unique(time) 
        MbList = []
        timeAdvBubble = []

        #Cycle through shots
        for i in np.arange(n_shots):
            #initial values for bubble fitting
            init_amp = (0.7 - Mb[i]) / 2
            init_c1 =  w - LL[i]
            init_c2 =  w + LL[i]
            init_off = (0.7 + Mb[i]) / 2
            init_vals = [init_amp, init_c1, init_c2, init_off, 3, 3] 
            MbList.append(Mb[i])

            # Fitting the curve only if Mb is under the threshold value
            if Mb[i] < threshold:
                try:
                    # Fitting with bubble function
                    best_2arctan, covar_2arctan = curve_fit(bubble, xx, M[i], p0 = init_vals)

                    # Initial values for bubbleshoulder fit
                    init_BS_left = [best_2arctan[0] * 0.7, best_2arctan[1], best_2arctan[3], best_2arctan[4]]
                    init_BS_right = [-best_2arctan[0] * 0.7, best_2arctan[2], best_2arctan[3], best_2arctan[5]]

                    # Left shoulder fit
                    xx_left = xx[int((round(best_2arctan[1])) - s_size) : int(round(best_2arctan[1])) + s_size]
                    Mi_left = M[i][int((round(best_2arctan[1])) - s_size) : int(round(best_2arctan[1])) + s_size]
                    best_BS_left, covar_BS_left = curve_fit(bubbleshoulder, xx_left, Mi_left, p0 = init_BS_left)

                    # Right shoulder fit
                    xx_right = xx[int((round(best_2arctan[2])) - s_size) : int(round(best_2arctan[2])) + s_size]
                    Mi_right = M[i][int((round(best_2arctan[2])) - s_size) : int(round(best_2arctan[2])) + s_size]
                    best_BS_right, covar_BS_right = curve_fit(bubbleshoulder, xx_right, Mi_right, p0 = init_BS_right)

                    # Fitting with new function
                    best_pieces, covar_pieces = curve_fit(bubblePieces, xx, M[i], p0=[-1, 0.05, 0.05, init_c1, init_c2, 0.7, 0, 0])
                    chi_squared = np.sum((M[i] - bubblePieces(xx, *best_pieces)) ** 2)

                    # if not save_flag and best_BS_left[3] < 4 and best_BS_left[3] > 0: 
                    if not save_flag:
                        # print(f"{best_BS_left[1]:.2f} +/- {best_BS_left[3]:.2f}")
                        print(chi_squared)
                        # Plot bubble and bubbleshoulder fit
                        plt.figure(figsize=(8,4))
                        plt.plot(xx, M[i], label="Data")
                        plt.plot(xx, bubble(xx, *best_2arctan), label="Double arctan fit")
                        plt.plot(xx, bubblePieces(xx, *best_pieces), label="Piecewise fit")
                        plt.plot(xx_left, bubbleshoulder(xx_left, *best_BS_left), label="Left shoulder fit")
                        plt.plot(xx_right, bubbleshoulder(xx_right, *best_BS_right), label="Right shoulder fit")
                        plt.title(f'Day: {fs}, Sequence: {ei}, Shot: {i}')
                        # plt.text(0.7, 0.1, f'Chi-squared: {chi_squared:.2f}', transform=plt.gca().transAxes,
                        #          fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
                        plt.xlabel('$x\ [\mu m]$')
                        plt.ylabel('$Z(x)$')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig('thesis/figures/chap2/arctan_fit.png', dpi=500)
                        plt.show()

                    # Selection on shots
                    # if best_BS_left[3] < 4 and best_BS_left[3] > 0:
                    if True:
                        b_center.append(int(best_BS_right[1] / 2 + best_BS_left[1] / 2))
                        b_size.append(best_2arctan[2] - best_2arctan[1])
                        b_sizeADV.append(best_BS_right[1] - best_BS_left[1])
                        s_width.append(0.5*(best_BS_left[3] + best_BS_right[3]))

                        b_off.append(best_pieces[5])

                        b_exp_slope_left.append(best_pieces[1])
                        b_exp_slope_right.append(best_pieces[2])
                        b_inside_boundary_left.append(best_pieces[3]) # defines the inside region with bubblePieces
                        b_inside_boundary_right.append(best_pieces[4]) # defines the inside region with bubblePieces
                        
                        b_outside_boundary_left.append(best_BS_left[1] - 2*best_BS_left[3]) # defines the outside region with BS fit
                        b_outside_boundary_right.append(best_BS_right[1] + 2*best_BS_right[3]) # defines the outside region with BS fit
                    else:
                        b_size.append(0)
                        b_sizeADV.append(0)
                        b_center.append(w)
                        s_width.append(0)
                        b_inside_boundary_left.append(w)
                        b_inside_boundary_right.append(w)
                        b_outside_boundary_left.append(0)
                        b_outside_boundary_right.append(2*w)
                        b_exp_slope_left.append(0)
                        b_exp_slope_right.append(0)
                        b_off.append(0)

                    #print('Arctan fit working')
                
                except:
                    # print('Arctan fit does not work, going with gaussian')

                    # THROW AWAY GAUSSIAN FITS
                    # Gaussian fit
                    # best_GS, covar_GS = curve_fit(gauss, xx, M[i], p0 = [2, w, 10, .7])
                    # chi_squared = np.sum((M[i] - gauss(xx, *best_GS)) ** 2)

                    # if not save_flag:
                    #     # Plot gaussian fit
                    #     plt.plot(xx, M[i], label="Data")
                    #     plt.plot(xx, gauss(xx, *best_GS), label="Gaussian fit")
                    #     plt.title(f'Day: {fs}, Sequence: {ei}, Shot: {i}')
                    #     # plt.text(0.7, 0.1, f'Chi-squared: {chi_squared:.2f}', transform=plt.gca().transAxes,
                    #     #          fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
                    #     plt.xlabel('$x\ [\mu m]$')
                    #     plt.ylabel('$Z(x)$')
                    #     plt.legend()
                    #     plt.savefig('thesis/figures/chap2/gaussian_fit.png', dpi=500)
                    #     plt.show()

                    
                    # b_size.append(best_GS[2] * 2.355)
                    # b_sizeADV.append(best_GS[2] * 2.355)
                    # # print(best_GS[2] * 2.355)
                    # b_center.append(best_GS[1])

                    # b_inside_boundary_left.append(best_GS[1] - best_GS[2] * 0.5) # defines the inside region with GS fit
                    # b_inside_boundary_right.append(best_GS[1] + best_GS[2] * 0.5) # defines the inside region with GS fit
                    
                    # b_outside_boundary_left.append(best_GS[1] - best_GS[2] * 2) # defines the outside region with GS fit
                    # b_outside_boundary_right.append(best_GS[1] + best_GS[2] * 2) # defines the outside region with GS fit

                    b_size.append(0)
                    b_sizeADV.append(0)
                    b_center.append(w)
                    s_width.append(0)
                    b_inside_boundary_left.append(w)
                    b_inside_boundary_right.append(w)
                    b_outside_boundary_left.append(0)
                    b_outside_boundary_right.append(2*w)
                    b_exp_slope_left.append(0)
                    b_exp_slope_right.append(0)
                    b_off.append(0)

            # Over the threshold value, the bubble is not formed, hence everything set to 0
            else: 
                #print("Over threshold")
                b_size.append(0)
                b_sizeADV.append(0)
                b_center.append(w)
                s_width.append(0)
                b_inside_boundary_left.append(w)
                b_inside_boundary_right.append(w)
                b_outside_boundary_left.append(0)
                b_outside_boundary_right.append(2*w)
                b_exp_slope_left.append(0)
                b_exp_slope_right.append(0)
                b_off.append(0)
        
        b_size = np.array(b_size)
        b_sizeADV = np.array(b_sizeADV) 
        b_center = np.array(b_center)
        s_width = np.array(s_width)
        b_inside_boundary_left = np.array(b_inside_boundary_left)
        b_inside_boundary_right = np.array(b_inside_boundary_right)
        b_outside_boundary_left = np.array(b_outside_boundary_left)
        b_outside_boundary_right = np.array(b_outside_boundary_right)
        b_exp_slope_left = np.array(b_exp_slope_left)
        b_exp_slope_right = np.array(b_exp_slope_right)
        b_off = np.array(b_off)

        if save_flag:
            print(f"\rSaving fitted data on data/{str}/day_{fs}/seq_{ei}/", end="")
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/center.csv", b_center, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/slope.csv", s_width, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/time.csv", time, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/sizeADV.csv", b_sizeADV, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/magnetization.csv", M, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/density.csv", D, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/off.csv", b_off, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/in_left.csv", b_inside_boundary_left, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/in_right.csv", b_inside_boundary_right, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/out_left.csv", b_outside_boundary_left, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/out_right.csv", b_outside_boundary_right, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/exp_left.csv", b_exp_slope_left, delimiter=',')
            np.savetxt(f"data/{str}/day_{fs}/seq_{ei}/exp_right.csv", b_exp_slope_right, delimiter=',')

