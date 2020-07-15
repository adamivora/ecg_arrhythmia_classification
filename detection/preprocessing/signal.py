from scipy.signal import butter, sosfiltfilt


def butter_bandpass(lowcut, highcut, fs, order):
    """
    Butterworth bandpass filter design.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', analog=False, output='sos')


def bandpass(signal, lowcut, highcut, fs, order=5):
    """
    Butterworth bandpass zero-phase filter using the `butter_bandpass` filter design.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, signal)
