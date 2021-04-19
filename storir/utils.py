import numpy as np


def calculate_direct_to_reverberant_ratio(data, direct_sound_idx):
    """
    Calculates the direct to reverberant sound energy ratio.
    :param data: energetic impulse response
    :param direct_sound_idx: index of the initial sound ray
    """
    # everything up to the given idx is summed up and treated as direct sound energy
    direct = sum(data[:direct_sound_idx + 1])
    reverberant = sum(data[direct_sound_idx + 1:])
    drr = 10 * np.log10(direct / reverberant)
    return drr


def decibels_to_gain(decibels):
    return 10 ** (decibels / 20)


def peak_norm(audio):
    return audio / np.max(np.abs(audio))


def calculate_drr(rt60):
    return int(rt60 * (- 1 / 100)) + np.random.randint(0, np.ceil(rt60 * (1 / 100)))
