import numpy as np


def calculate_drr_energy_ratio(y, direct_sound_idx):
    """
    Calculates the direct to reverberant sound energy ratio.
    Args:
        y: energetic impulse response
        direct_sound_idx: index of the initial sound ray

    Returns:
        drr energy ratio
    """
    # everything up to the given idx is summed up and treated as direct sound energy
    direct = sum(y[:direct_sound_idx + 1])
    reverberant = sum(y[direct_sound_idx + 1:])
    drr = 10 * np.log10(direct / reverberant)
    return drr


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

