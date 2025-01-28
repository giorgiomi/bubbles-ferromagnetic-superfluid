import matplotlib.pyplot as plt
import numpy as np

def V_MF(Z, param):
    omega = param[0]
    delta = param[1]
    k = param[2]
    n = param[3]
    return -(np.abs(k)*n*Z**2 + 2*omega*np.sqrt(1-Z**2) + 2*delta*Z)

def V_MF_prime(Z, param):
    omega = param[0]
    delta = param[1]
    k = param[2]
    n = param[3]
    return np.abs(k)*n*Z - omega*Z*np.sqrt(1-Z**2) + delta

Z = np.linspace(-1, 1, 1000)
plt.figure()
plt.plot(Z, V_MF(Z, [2, -1, -5 , 1]))
plt.xlabel("Z")
plt.ylabel("$V_{MF}$")
plt.show()