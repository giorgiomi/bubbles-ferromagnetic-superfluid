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

def bubblePieces(x, A, w0, w1, x0, x1, d0, d1):
    x = np.asarray(x)
    result = np.empty_like(x, dtype=float)
    
    mask1 = x < x0
    mask2 = (x >= x0) & (x < x1)
    mask3 = x >= x1
    
    result[mask1] = (A - d0) * np.exp((x[mask1] - x0)/w0) + d0
    result[mask2] = A
    result[mask3] = (A - d1) * np.exp(-(x[mask3] - x1)/w1) + d1
    
    return result

def corrGauss(x, l1, off, l2):
    ex = 1.7
    # gauss corrected by exp factor 1.7 instead of 2.0
    return np.cos(x/l2) * (1 - off) * np.exp(-(x/l1)**ex) + off
    # return (1 - off) * np.exp(-(x/l1)**ex) + off
    # return (1 - off - A) * np.exp(-(x/l1)**ex) + A * np.cos(x/l2) + off

def corrExp(x, l1, off):
    return (1 - off) * np.exp(-(x/l1)) + off