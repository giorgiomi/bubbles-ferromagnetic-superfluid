# This script is used to shift the bubbles on their border
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.parameters import importParameters
from util.methods import scriptUsage
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle

# Data
f, seqs, Omega, knT, detuning, sel_days, sel_seq = importParameters()
w = 200
chosen_days = scriptUsage()
wind = 50

for day in chosen_days:
    for seq in sel_seq[day]:
        seqi = seqs[day][seq]
        df_center_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/center_sorted.csv")
        df_size_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/sizeADV_sorted.csv")
        df_Z_sorted = pd.read_csv(f"data/selected/day_{day}/seq_{seq}/Z_sorted.csv")

        b_center = df_center_sorted.to_numpy().flatten()
        b_size = df_size_sorted.to_numpy().flatten()
        Z = df_Z_sorted.to_numpy()

        # Shifting to RIGHT BORDER
        shift = - b_center + w - b_size / 2
        Z_shifted_right = np.array([np.roll(Z[i], int(shift[i])) for i in range(len(Z))])
        # Z_shifted_right = Z_shifted_right[:,175:225]

        # Plotting the bubble (shifted RIGHT BORDER)
        fig, ax = plt.subplots(figsize=(10, 5), ncols=3, gridspec_kw={'width_ratios': [1, 1, 0.05]})
        im = ax[0].pcolormesh(np.arange(Z_shifted_right.shape[1]), np.arange(Z_shifted_right.shape[0]), Z_shifted_right, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z')
        ax[0].set_title('Shifted shots to right border')
        ax[0].set_xlabel(r'$\tilde{x}\ [\mu m]$')

        # Shifting to LEFT BORDER
        shift = - b_center + w + b_size / 2
        Z_shifted_left = np.array([np.roll(Z[i], int(shift[i])) for i in range(len(Z))])
        # Z_shifted_left = Z_shifted_left[:,175:225]

        # Plotting the bubble (shifted LEFT BORDER)
        im = ax[1].pcolormesh(np.arange(Z_shifted_left.shape[1]), np.arange(Z_shifted_left.shape[0]), Z_shifted_left, vmin=-1, vmax=1, cmap='RdBu')
        cbar = fig.colorbar(im, cax=ax[2])
        cbar.set_label('Z')
        ax[1].set_title('Shifted shots to left border')
        ax[1].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        fig.suptitle(f"Experiment realization of day {day}, sequence {seq}")
        # plt.savefig("thesis/figures/chap2/shot_shifting_border.png", dpi=500)
        plt.show()

        
        ## MANUAL MUMBO JUMBO
        # # Create a global variable to store the adjusted image
        # aligned_image = Z_shifted_left.copy()

        # # Function to update the plot
        # def update_plot():
        #     im.set_data(aligned_image)  # Update the image data
        #     highlight_box.set_y(selected_row - 0.5)  # Update the y-position of the box            
        #     fig.canvas.draw_idle()  # Redraw the canvas

        # # Function to shift rows
        # def shift_row(row, amount):
        #     global aligned_image
        #     aligned_image[row] = np.roll(aligned_image[row], amount)  # Shift the row
        #     update_plot()
        
        #  # Highlight the selected row
        # def highlight_selected_row():
        #     global aligned_image, selected_row
        #     highlighted_image = aligned_image.copy()
        #     highlighted_image[selected_row] = np.max(aligned_image)  # Highlight the selected row
        #     highlighted_image[selected_row, :] = np.max(aligned_image)  # Highlight the entire row
        #     im.set_data(highlighted_image)  # Update the image data
        #     fig.canvas.draw_idle()  # Redraw the canvas

        # # Event handler for key presses
        # def on_key(event):
        #     global selected_row
        #     if event.key == "up":
        #         selected_row = (selected_row - 1) % aligned_image.shape[0]  # Select previous row
        #         update_plot()  # Update the plot
        #     elif event.key == "down":
        #         selected_row = (selected_row + 1) % aligned_image.shape[0]  # Select next row
        #         update_plot()  # Update the plot
        #     elif event.key == "left":
        #         shift_row(selected_row, -1)  # Shift selected row left
        #     elif event.key == "right":
        #         shift_row(selected_row, 1)  # Shift selected row right
        #     elif event.key == "s":  # Save the figure when 's' is pressed
        #         save_figure()
        
        # # Function to save the figure
        # def save_figure():
        #     filename = "MANUAL_MAGIC.png"  # Change the filename as needed
        #     plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save with high resolution
        #     print(f"Figure saved as {filename}")

        # # Initial setup
        # fig, ax = plt.subplots()
        # selected_row = 0  # Start with the first row selected
        # im = ax.imshow(aligned_image, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)  # Initialize the image
        # cbar = plt.colorbar(im, ax=ax, orientation='vertical')  # Create the colorbar
        # cbar.set_label('Z')

        # # Add a rectangle to highlight the selected row
        # highlight_box = Rectangle(
        #     (-0.5, selected_row - 0.5),  # Bottom-left corner (start of the row)
        #     aligned_image.shape[1],  # Width (number of columns)
        #     1,  # Height (one row)
        #     edgecolor='black',  # Border color
        #     facecolor='none',  # Transparent fill
        #     linewidth=0.5,  # Thickness of the border
        # )
        # ax.add_patch(highlight_box)

        # plt.title("Use Arrow Keys to Align Rows")
        # plt.xlabel("Pixels")
        # plt.ylabel("Rows")

        # # Connect keyboard events
        # fig.canvas.mpl_connect("key_press_event", on_key)

        # plt.show()

        # Plotting all magnetization profiles shifted
        fig, ax = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)
        
        # Plotting all magnetization profiles shifted to the left
        for i in range(len(Z_shifted_left)):
            ax[0].plot(np.arange(Z_shifted_left.shape[1]), Z_shifted_left[i], alpha=0.05)
        ax[0].plot(np.arange(Z_shifted_left.shape[1]), np.mean(Z_shifted_left, axis=0), label='mean', color='black')
        ax[0].set_title('All magnetization profiles shifted left')
        ax[0].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        ax[0].set_ylabel('Magnetization')
        ax[0].set_xlim(0, 250)
        ax[0].legend()

        # Plotting all magnetization profiles shifted to the right
        for i in range(len(Z_shifted_right)):
            ax[1].plot(np.arange(Z_shifted_right.shape[1]), Z_shifted_right[i], alpha=0.05)
        ax[1].plot(np.arange(Z_shifted_right.shape[1]), np.mean(Z_shifted_right, axis=0), label='mean', color='black')
        ax[1].set_title('All magnetization profiles shifted right')
        ax[1].set_xlabel(r'$\tilde{x}\ [\mu m]$')
        ax[1].set_xlim(150, 400)
        ax[1].legend()

        fig.suptitle(f"All magnetization profiles shifted for day {day}, sequence {seq}")
        plt.show()
