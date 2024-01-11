import matplotlib.pyplot as plt
from scipy import signal
import pyyawt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def butterworth_bandpass_filter(signal_data, low_cutoff_freq=10, high_cutoff_freq=250, order=4, fs=1000): #No entiendo porque el filtro paso bando no funciona correctamente
    # Normalizar las frecuencias de corte
    normalized_low_cutoff_freq = low_cutoff_freq / (0.5 * fs)
    normalized_high_cutoff_freq = high_cutoff_freq / (0.5 * fs)

    # Diseñar el filtro Butterworth
    b, a = signal.butter(order, [normalized_low_cutoff_freq, normalized_high_cutoff_freq], btype='band', analog=False, output='ba')
    zi = signal.lfilter_zi(b, a)

    # Filtrar la señal
    filtered_signal, _ = signal.lfilter(b, a, signal_data, zi=zi)

    return filtered_signal


def butterworth_filter(signal_data, order=4, cutoff_freq=50, fs=1000):
    # Normalizar la frecuencia de corte
    normalized_cutoff_freq = cutoff_freq / (0.5 * fs)

    # Diseñar el filtro Butterworth
    b, a = signal.butter(order, normalized_cutoff_freq, btype='low', analog=False, output='ba')
    zi = signal.lfilter_zi(b, a)

    # Filtrar la señal
    filtered_signal, _ = signal.lfilter(b, a, signal_data, zi=zi)

    return filtered_signal

def denoisewavelet(x1, level=5):
    xd, cxd, lxd = pyyawt.wden(x1, 'minimaxi', 's', 'mln', level, 'db5')
    return xd

def plot_original_and_filtered_signals(timestamp, original_signal, filtered_signal, start_index=100):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamp[start_index:], original_signal[start_index:], label='Señal Original')
    plt.plot(timestamp[start_index:], filtered_signal[start_index:], label='Señal Filtrada')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (µV)')
    plt.legend()
    plt.title('Comparación entre Señal Original y Señal Filtrada')
    plt.grid(True)
    plt.show()

def rectification_normalization(timestamps, emg_signals, plot):
    # Calcular el valor absoluto de la señal
    rectified_signals = [np.abs(emg_signal[100:]) for emg_signal in emg_signals]

    # Normalizar los valores de cada canal desde -1 hasta 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_signals = [scaler.fit_transform(rectified_signal.reshape(-1, 1)).ravel() for rectified_signal in rectified_signals]

def graficar_señal_emg(timestamps, emg_signals, labels=None):
    plt.figure(figsize=(12, 6))

    if labels is None:
        labels = [f'Señal {i + 1}' for i in range(len(emg_signals))]

    for i, signal in enumerate(emg_signals):
        plt.plot(timestamps, signal, label=labels[i])

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.title('Señales EMG')
    plt.grid(True)
    plt.show()
    

