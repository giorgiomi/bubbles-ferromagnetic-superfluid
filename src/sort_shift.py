# This script is used to analyze the data saved by save_data.py (from hdf to csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import scriptUsage

# Data
selected_flag = int(input("Enter 1 for selected, 0 for all: "))
if selected_flag:
    str = 'selected'
else:
    str = 'processed'
f, seqs, Omega, knT, Detuning, sel_days, sel_seq = importParameters(selected_flag)
w = 200
chosen_days = scriptUsage(sel_days)

# Print script purpose
print("\nSorting data with bubble width (and shifting)\n")

# Ask the user for FFT and ACF on true or zero mean signal
save_flag = int(input("Enter 0 for just plotting or 1 for just saving: "))

for day in chosen_days:
    # for seq, seqi in enumerate((seqs[day])):
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_center = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/center.csv")
        df_time = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/time.csv")
        df_size = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/sizeADV.csv")
        df_M = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/magnetization.csv")
        df_in_left = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/in_left.csv")
        df_in_right = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/in_right.csv")
        df_out_left = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/out_left.csv")
        df_out_right = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/out_right.csv")
        df_exp_left = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/exp_left.csv")
        df_exp_right = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/exp_right.csv")
        df_off = pd.read_csv(f"data/{str}/day_{day}/seq_{seq}/off.csv")

        b_center = df_center.to_numpy().flatten()
        time = df_time.to_numpy().flatten()
        b_sizeADV = df_size.to_numpy().flatten()
        in_left = df_in_left.to_numpy().flatten()
        in_right = df_in_right.to_numpy().flatten()
        out_left = df_out_left.to_numpy().flatten()
        out_right = df_out_right.to_numpy().flatten()
        exp_left = df_exp_left.to_numpy().flatten()
        exp_right = df_exp_right.to_numpy().flatten()
        off = df_off.to_numpy().flatten()
        M = df_M.to_numpy()

        # Sorting the bubble
        Zlist = np.argsort(b_sizeADV)
        Z = (M[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        b_sizeADV_sorted = (b_sizeADV[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        b_center_sorted = (b_center[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        time_sorted = (time[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        in_left_sorted = (in_left[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        in_right_sorted = (in_right[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        out_left_sorted = (out_left[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        out_right_sorted = (out_right[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        exp_left_sorted = (exp_left[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        exp_right_sorted = (exp_right[Zlist])[np.where(b_sizeADV[Zlist] > 0)]
        off_sorted = (off[Zlist])[np.where(b_sizeADV[Zlist] > 0)]

        # Shifting
        # length = 2 * w
        # shift = - b_center_sorted + length/2
        # max_shift = np.max(np.abs(shift))
        # Z_shifted = np.zeros((len(Z), length + 2 * int(max_shift)))
        # for i in np.arange(len(Z_shifted)):
        #     Z_shifted[i, (int(max_shift + int(shift[i]))) : (int(max_shift) + int(shift[i]) + length)] = Z[i]
        
        # save Z, Z_shifted on file
        if save_flag:
            print(f"\rSaving sorted data on data/{str}/day_{day}/seq_{seq}/", end="")
            # np.savetxt(f"data/{str}/day_{day}/seq_{seq}/Z_shifted.csv", Z_shifted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/Z_sorted.csv", Z, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/center_sorted.csv", b_center_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/time_sorted.csv", time_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/sizeADV_sorted.csv", b_sizeADV_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/in_left_sorted.csv", in_left_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/in_right_sorted.csv", in_right_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/exp_left_sorted.csv", exp_left_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/exp_right_sorted.csv", exp_right_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/out_left_sorted.csv", out_left_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/out_right_sorted.csv", out_right_sorted, delimiter=',')
            np.savetxt(f"data/{str}/day_{day}/seq_{seq}/off_sorted.csv", off_sorted, delimiter=',')

        else:
            # # Plotting the bubble (unsorted)
            fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})
            ax[0].pcolormesh(M, vmin = -1, vmax = +1, cmap = 'RdBu')
            ax[0].set_title('Unsorted shots')
            ax[0].set_xlabel('$x\ [\mu m]$')
            ax[0].set_ylabel('shots')

            # # Plotting the bubble (sorted)
            # im = ax[1].pcolormesh(Z_shifted, vmin=-1, vmax=1, cmap='RdBu')
            # cbar = fig.colorbar(im, cax=ax[2])
            # cbar.set_label('M', rotation=270)
            # ax[1].set_title('Sorted shots')
            # ax[1].set_xlabel('x')
            # fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
            # # plt.savefig("thesis/figures/chap2/shot_sorting.png", dpi=500)
            # plt.show()

