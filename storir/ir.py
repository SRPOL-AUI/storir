import numpy as np
from storir.audio_utils import decibels_to_gain
from storir.rir_utils import calculate_drr_energy_ratio, thin_out_reflections


class ImpulseResponse:

    def __init__(
            self,
            rt60: float,
            edt: float,
            itdg: float,
            er_duration: float,
            drr: float,
    ):
        """
        Energetic stochastic impulse response.
        Args:
            rt60: reverberation time [ms]
            edt: early decay time [ms]
            itdg: initial time delay gap [ms]
            er_duration: early reflections duration [ms]
            drr: direct to reverberant energy ratio [dB]
        """
        self.rt60 = rt60
        self.edt = edt
        self.itdg = itdg
        self.er_duration = er_duration
        self.drr = drr
        if self.rt60 <= self.edt:
            raise ValueError('RT60 needs to be longer than EDT.')

    def generate(self, sampling_rate):
        energetic = self._get_noise(sampling_rate).astype('float32')
        energetic, dsi, ersi, erei = self._get_edt_and_rt60_slope(energetic, sampling_rate)
        energetic = self._randomize_reflections(energetic, dsi, ersi, erei, sampling_rate)
        return energetic[dsi:]

    def _get_noise(self, sampling_rate):
        # initialize random noise (10 dB range)
        num_samples = self._get_num_samples(self.rt60, sampling_rate)
        noise = np.random.random_sample(size=num_samples) * 10 - 5
        return noise

    def _get_edt_and_rt60_slope(self, y, sampling_rate):
        """
        Shapes a random vector so it has slope specified by EDT and RT60.
        """

        edt_num_samples = self._get_num_samples(self.edt, sampling_rate)
        rt60_num_samples = self._get_num_samples(self.rt60, sampling_rate)
        er_duration_num_samples = self._get_num_samples(self.er_duration, sampling_rate)

        # shape the EDT slope of the IR
        y[:edt_num_samples - 1] -= np.arange(0, edt_num_samples - 1)
        y[edt_num_samples - 1:] -= (edt_num_samples - 1)  # last sample of EDT
        y = y * 10 / edt_num_samples

        # shape the RT60 slope of the IR (after EDT)
        k = np.arange(edt_num_samples, rt60_num_samples)
        y[edt_num_samples:rt60_num_samples] -= (k - (edt_num_samples + 1)) * 50 / rt60_num_samples

        y -= max(y)  # change scale to dBFS (0 dB becomes the maximal level)
        y = decibels_to_gain(y) ** 2

        # assign values to specific time points in the IR
        direct_sound_idx = np.argmax(y)

        # if any of the parameters like er_duration set in config exceed the length
        # of the whole IR than we just treat the last idx of the IR as the start/end point
        # (if the parameters are set logically it will never happen)
        er_start_idx = min(direct_sound_idx + 1, len(y) - 1)
        er_end_idx = min(er_start_idx + er_duration_num_samples, len(y) - 1)
        return y, direct_sound_idx, er_start_idx, er_end_idx

    def _randomize_reflections(self, y, direct_sound_idx, early_ref_start, early_ref_end, sampling_rate):
        """
        Creates time gaps between incoming sound rays of the energetic impulse response
        in a way that the DRR condition is met as closely as possible.
        """
        y = self._create_initial_time_delay_gap(y, direct_sound_idx, sampling_rate)

        # create a 1 dB margin for error (we will never hit the exact drr value)
        drr_low = self.drr - .5
        drr_high = self.drr + .5

        current_drr = calculate_drr_energy_ratio(y=y, direct_sound_idx=direct_sound_idx)

        if current_drr > drr_high:
            return y

        while drr_low > current_drr:

            # thin out early reflections
            y = thin_out_reflections(y=y,
                                     start_idx=early_ref_start,
                                     end_idx=early_ref_end,
                                     rate=1/8)

            # thin out reverberation tail
            y = thin_out_reflections(y=y,
                                     start_idx=early_ref_end,
                                     end_idx=len(y) - 1,
                                     rate=1 / 10)

            previous_drr = current_drr
            current_drr = calculate_drr_energy_ratio(y=y, direct_sound_idx=direct_sound_idx)

            # if thinning out reflections did not decrease the DRR it means
            # that the maximal DRR possible has been reached
            if np.isclose(previous_drr, current_drr):
                break

        return y

    def _create_initial_time_delay_gap(self, y, direct_sound_idx, sampling_rate):
        """
        Creates a time gap between the initial sound ray (direct sound), and the rest of the reverberant rays.
        """
        # if itdg exceeds the length of the whole IR than we just
        # treat the last idx of the IR as the end point
        # (if the parameters are set logically it will never happen)
        itdg_num_samples = self._get_num_samples(self.itdg, sampling_rate)
        itdg_end_idx = min(direct_sound_idx + 1 + itdg_num_samples, len(y) - 1)
        y[direct_sound_idx + 1:itdg_end_idx] = 0
        return y

    @staticmethod
    def _get_num_samples(param, sampling_rate):
        return int((param / 1000) * sampling_rate)
