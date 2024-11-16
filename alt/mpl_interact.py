import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mplcursors

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of time
f0 = 50
f1 = 200
signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t) * t)

frequencies, times, Sxx = spectrogram(signal, fs)

# create figure
fig, ax = plt.subplots()
cax = ax.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]')
plt.colorbar(cax, label='Intensity [dB]')

# add interactive cursor
mplcursors.cursor(cax, hover=True)

# show plot
plt.title('Interactive Spectrogram')
plt.show()

