def butterworth_bandpass_filter(signal_data, low_cutoff_freq=20, high_cutoff_freq=450, order=2, fs=1100):
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

def apply_filter(emg_data, filter_type):
    if filter_type == 'lowpass':
        return butterworth_filter(emg_data)
    elif filter_type == 'bandpass':
        return butterworth_bandpass_filter(emg_data)
    else:
        raise ValueError("Invalid filter_type. Choose 'lowpass' or 'bandpass'.")

def valor_abs(array):

    return [abs(num) for num in array]
