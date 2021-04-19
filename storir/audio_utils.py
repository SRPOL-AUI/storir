import numpy as np


def decibels_to_gain(decibels: float):
    """
    Change unit from decibels to gains.
    Args:
        decibels: value in decibels.

    Returns:
        value in gains.
    """
    return 10 ** (decibels / 20)


def peak_norm(audio: np.ndarray):
    """
    Audio normalisation with respect to highest peak to obtain (-1, 1) amplitudes.
    Args:
        audio: signal to be normalised

    Returns:
        normalised signal
    """
    return audio / np.max(np.abs(audio))

