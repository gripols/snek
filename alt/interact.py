import numpy as np
import plotly.graph_objects as go
from scipy.signal import spectrogram

# generate sample signal (same as before)
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of time
f0 = 50
f1 = 200
signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t) * t)

# calculate spectrogram
frequencies, times, Sxx = spectrogram(signal, fs)

# create plotly figure
fig = go.Figure(data=go.Heatmap(
    z=10 * np.log10(Sxx),
    x=times,
    y=frequencies,
    colorscale='Viridis'
))

# update layout
fig.update_layout(
    title='Interactive Spectrogram',
    xaxis_title='Time [s]',
    yaxis_title='Frequency [Hz]',
    coloraxis_colorbar=dict(title='Intensity [dB]')
)


fig.show()
