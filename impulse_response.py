import numpy as np

import utils


class ImpulseResponse:
    def __init__(self,
                 rt60,
                 edt,
                 itdg,
                 drr,
                 er_duration,
                 sr):
        '''
        Generates a stochastic impulse response (energetic)
        based on the given parameters.
        :param rt60: reverberation time [ms]
        :param edt: early decay time [ms]
        :param itdg: initial time delay gap [ms]
        :param drr: direct to reverberant energy ratio [dB]
        :param er_duration: early reflections duration [ms]
        :param sr: sampling rate [Hz]
        '''
        self.rt60 = rt60
        self.edt = edt
        self.itdg = itdg
        self.drr = drr
        self.er_duration = er_duration
        self.sr = sr

        self.edt_samples = int((self.edt / 1000) * self.sr)
        self.rt60_samples = int((self.rt60 / 1000) * self.sr)
        self.itdg_samples = int((self.itdg / 1000) * self.sr)
        self.er_duration_samples = int((self.er_duration / 1000) * self.sr)

        self.direct_sound_idx = None
        self.early_reflections_start_idx = None
        self.early_reflections_end_idx = None

        self.energetic = None

        self.generate()

    def generate(self):
            '''
            Generates an energetic impulse response.
            '''
            self.energetic = self.__generate_initial_random_noise()
            self.energetic = self.energetic.astype('float32')
            self.energetic = self.__create_edt_and_rt60_slope()
            self.__randomize_reflections()
            self.energetic = self.energetic[self.direct_sound_idx:]

    def __generate_initial_random_noise(self):
        # initialize random noise (10 dB range)
        data = (np.random.random_sample(size=self.rt60_samples) * 10) - 5

        return data

    def __create_edt_and_rt60_slope(self):
            '''
            Shapes a random vector, so that it has a slope specified
            by the EDT and RT60 parameters.
            '''
            assert self.rt60 > self.edt, 'RT60 has to be longer than EDT!'

            for k in range(self.edt_samples):
                if k == self.edt_samples - 1:  # last sample of EDT
                    # decrease the level of the rest of the random sequence
                    # later to be shaped based on the rt60 value
                    self.energetic[k:] -= k * 10 / self.edt_samples
                    break
                # shape the EDT slope of the IR
                self.energetic[k] -= k * 10 / self.edt_samples

            for k in range(self.edt_samples, self.rt60_samples):
                # shape the RT60 slope of the ir (after EDT)
                self.energetic[k] -= (k - (self.edt_samples + 1)) * 50 / self.rt60_samples

            self.energetic -= max(self.energetic)  # change scale to dBFS (0 dB becomes the maximal level)
            self.energetic = utils.decibels_to_gain(self.energetic)
            self.energetic **= 2

            # assign values to specific time points in the IR
            self.direct_sound_idx = np.argmax(self.energetic)

            # if any of the parameters like er_duration set in config exceed the length
            # of the whole IR than we just treat the last idx of the IR as the start/end point
            # (if the parameters are set logically it will never happen)
            self.early_reflections_start_idx = min(self.direct_sound_idx + 1, len(self.energetic) - 1)
            self.early_reflections_end_idx = min(self.early_reflections_start_idx + self.er_duration_samples,
                                                 len(self.energetic) - 1)

            return self.energetic

    def __randomize_reflections(self):
        '''
        Creates time gaps between incoming sound rays of the energetic impulse response
        in a way that the DRR condition is met as closely as possible.
        '''
        self.__create_initial_time_delay_gap()

        # create a 1 dB margin for error (we will never hit the exact drr value)
        drr_low = self.drr - .5
        drr_high = self.drr + .5
        randomization_epochs = 0
        current_drr = utils.direct_to_reverberant_ratio(data=self.energetic,
                                                        direct_sound_idx=self.direct_sound_idx)

        while not drr_low <= current_drr:
            # print(f'Epoch {randomization_epochs + 1}, DRR = {current_drr:.2f}')

            if randomization_epochs == 0 and current_drr > drr_high:
                break

            self.__thin_out_early_reflections()
            self.__thin_out_reverberation_tail()

            previous_drr = current_drr
            current_drr = utils.direct_to_reverberant_ratio(data=self.energetic,
                                                            direct_sound_idx=self.direct_sound_idx)

            # if thinning out reflections did not decrease the DRR it means
            # that the maximal DRR possible has been reached
            if previous_drr == current_drr:
                break

            randomization_epochs += 1

    def __create_initial_time_delay_gap(self):
        '''
        Creates a time gap between the initial sound ray (direct sound), and the rest of the reverberant rays.
        '''
        # if itdg exceeds the length of the whole IR than we just
        # treat the last idx of the IR as the end point
        # (if the parameters are set logically it will never happen)
        itdg_end_idx = min(self.direct_sound_idx + 1 + self.itdg_samples, len(self.energetic) - 1)

        self.energetic[self.direct_sound_idx + 1:itdg_end_idx] = 0

    def __thin_out_early_reflections(self):
        self.__thin_out_reflections(start_idx=self.early_reflections_start_idx,
                                    end_idx=self.early_reflections_end_idx,
                                    rate=1 / 8)

    def __thin_out_reverberation_tail(self):
        self.__thin_out_reflections(start_idx=self.early_reflections_end_idx,
                                    end_idx=len(self.energetic) - 1,
                                    rate=1 / 10)

    def __thin_out_reflections(self, start_idx, end_idx, rate):
        """
        Randomly deletes a fraction of sound rays in a specified time widnow.
        :param start_idx: time window starting sample index
        :param end_idx: time window ending sample index
        :param rate: the fraction of sound rays to delete

        """
        detected_ray_indices = np.nonzero(self.energetic)[0]
        # include only the desired part of the IR
        detected_ray_indices = [idx for idx in detected_ray_indices if
                                start_idx <= idx <= end_idx]
        rays_to_delete_indices = np.random.choice(detected_ray_indices,
                                                  int(len(detected_ray_indices) * rate),
                                                  replace=False)

        for idx in rays_to_delete_indices:
            self.energetic[idx] = 0


if __name__ == '__main__':
    # example configuration
    rt60 = 500
    edt = 50
    itdg = 3
    drr = int(rt60 * (- 1 / 100)) + np.random.randint(0, np.ceil(rt60 * (1 / 100)))
    er_duration = 80
    sr = 16000

    ir = ImpulseResponse(rt60=rt60,
                         edt=edt,
                         itdg=itdg,
                         drr=drr,
                         er_duration=er_duration,
                         sr=sr)

    # extract the energetic impulse response
    output = ir.energetic
