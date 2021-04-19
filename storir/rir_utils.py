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


def thin_out_reflections(y, start_idx, end_idx, rate):
    """
    Randomly deletes a fraction of sound rays in a specified time window.
    Args:
        y: energetic impulse response
        start_idx: time window starting sample index
        end_idx: time window ending sample index
        rate: the fraction of sound rays to delete

    Returns:
        energetic IR without fraction of sound rays in specified interval
    """
    ray_indices = [idx for idx in range(start_idx, end_idx + 1) if y[idx] != 0]
    num_rays = int(len(ray_indices) * rate)
    assert num_rays >= 1
    random_subset = np.random.choice(ray_indices, num_rays, replace=False)
    y[random_subset] = 0
    return y
