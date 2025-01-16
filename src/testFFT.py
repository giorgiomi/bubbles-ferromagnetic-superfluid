# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import rfft, rfftfreq
# from scipy.optimize import curve_fit

# # Sine wave
# x = np.linspace(-2,2,75)
# y = 0.5*np.sin(2*np.pi*x) + 0.2*np.random.randn(len(x))

# yf = rfft(y)
# yf_freq = rfftfreq(len(y))

# y_fit, ycov = curve_fit(lambda t, A, f, phi: A*np.sin(2*np.pi*f*t + phi), x, y, p0=[0.5, 1, 0])
# fit_freq = y_fit[1]

# fig, ax = plt.subplots(1, 2)
# ax[0].plot(x,y)
# ax[0].plot(x, y_fit[0]*np.sin(2*np.pi*y_fit[1]*x + y_fit[2]))
# ax[0].grid()
# ax[0].set_xlabel('x')
# ax[0].set_title('Signal')

# ax[1].plot(yf_freq, np.abs(yf))
# ax[1].grid()
# ax[1].set_xlabel('f')
# #ax[1].axvline(fit_freq/4/np.pi, color='r', linestyle='--')
# ax[1].set_title('FFT')
# #ax[1].set_xscale('log')
# #ax[1].set_yscale('log')

# plt.show()



from scipy.fft import rfft, rfftfreq, fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

# Number of sample points
N = 50

# sample spacing
T = 1.0
x = np.linspace(0.0, N*T, N, endpoint=False)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
y = np.sin(x) - 10.0
print(len(y), len(x))

plt.plot(x, y)
plt.show()

yf = rfft(y)
xf = rfftfreq(len(y), d=1)

plt.plot(xf, np.abs(yf))
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.yscale('log')
plt.show()

print(len(yf), len(xf))