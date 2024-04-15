import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit

# Sine wave
x = np.linspace(-2,2,75)
y = 0.5*np.sin(2*np.pi*x) + 0.2*np.random.randn(len(x))

yf = rfft(y)
yf_freq = rfftfreq(len(y))

y_fit, ycov = curve_fit(lambda t, A, f, phi: A*np.sin(2*np.pi*f*t + phi), x, y, p0=[0.5, 1, 0])
fit_freq = y_fit[1]

fig, ax = plt.subplots(1, 2)
ax[0].plot(x,y)
ax[0].plot(x, y_fit[0]*np.sin(2*np.pi*y_fit[1]*x + y_fit[2]))
ax[0].grid()
ax[0].set_xlabel('x')
ax[0].set_title('Signal')

ax[1].plot(yf_freq, np.abs(yf))
ax[1].grid()
ax[1].set_xlabel('f')
#ax[1].axvline(fit_freq/4/np.pi, color='r', linestyle='--')
ax[1].set_title('FFT')
#ax[1].set_xscale('log')
#ax[1].set_yscale('log')

plt.show()