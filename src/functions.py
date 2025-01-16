import numpy as np

# Fitting to 2 arctans
def bubble(x, amp, center1, center2, offset, width1, width2):
    return - amp * ((2/np.pi) * np.arctan((x - center1) / width1) - (2/np.pi) * np.arctan((x - center2) / width2) - 1) + offset

# Fitting to Gaussian
def gauss(x, amp, center, width, offset):
    return - amp * (np.exp(-(x - center)**2 / 2 / width**2)) + offset

# Fitting to 1 arctan (only one bubble shoulder)
def bubbleshoulder(x, amp, cen1, offset, wid1):
    return - amp * (np.arctan((x - cen1) / wid1)) / (np.pi / 2) + offset