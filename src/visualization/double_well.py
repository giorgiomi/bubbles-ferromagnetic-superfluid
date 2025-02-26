import matplotlib.pyplot as plt
import numpy as np

def V_MF(Z, param):
    omega = param[0]
    delta = param[1]
    k = param[2]
    n = param[3]
    return -(np.abs(k)*n*Z**2 + 2*omega*np.sqrt(1-Z**2) + 2*delta*Z)

omega = 400
k = -1
n = 1150
delta_B = -1150-475
delta_eff = delta_B + np.abs(k)*n
print(delta_eff)

par = [omega, delta_eff, k, n]
# print(np.sqrt(300*(1150-300)))

Z_max = delta_eff/(omega+k*n)
Z_min1 = 1-0.5*(omega/(delta_eff-k*n))**2
Z_min2 = -1+0.5*(omega/(-delta_eff-k*n))**2
print(Z_min1, Z_min1**2, np.sqrt(1-Z_min1**2))

plt.figure()
Z = np.linspace(-1, 1, 1000)
plt.plot(Z, V_MF(Z, par)/1000, color='tab:gray')
plt.plot(Z_max, V_MF(Z_max, par)/1000, 'o', color='tab:green', markersize=8)
plt.plot(Z_min1, V_MF(Z_min1, par)/1000, 'o', color='tab:blue', markersize=8)
plt.plot(Z_min2, V_MF(Z_min2, par)/1000, 'o', color='tab:red', markersize=8)
plt.axhline(y=V_MF(Z_min1, par)/1000, color='tab:blue', linestyle='--')
plt.axhline(y=V_MF(Z_max, par)/1000, color='tab:red', linestyle='--')
plt.xlabel("$Z$")
# plt.ylabel("$E_{MF}/\hbar$ [Hz]")
plt.ylabel("$E_{MF}/\hbar$ [kHz]")
# plt.ylim(np.min(V_MF(Z, par))/1000, np.max(V_MF(Z, par))/1000)
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.1f}'))
plt.show()
