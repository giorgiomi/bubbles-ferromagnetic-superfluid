import sys
import numpy as np
from util.parameters import importParameters

f, seqs, Omega, knT, detuning = importParameters()

def scriptUsage():
    if len(sys.argv) > 1:
        if int(sys.argv[1]) == -1:
            chosen_days = np.arange(len(seqs))
        else:
            chosen_days = [int(sys.argv[1])]
    else:
        print(f"Usage: python3 {sys.argv[0]} <chosen_days>\t\t (use chosen_days = -1 for all)")
        exit()
        
    return chosen_days