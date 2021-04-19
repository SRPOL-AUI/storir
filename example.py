from storir import ImpulseResponse
import numpy as np

if __name__ == '__main__':
    # example configuration
    rt60 = 500
    edt = 50
    itdg = 3
    er_duration = 80
    drr = int(rt60 * (- 1 / 100)) + np.random.randint(0, np.ceil(rt60 * (1 / 100)))
    sr = 16000

    rir = ImpulseResponse(rt60=rt60,
                          edt=edt,
                          itdg=itdg,
                          drr=drr,
                          er_duration=er_duration)

    # get 5 impulse responses
    for _ in range(5):
        output = rir.generate(sampling_rate=sr)
        print(output.shape)
