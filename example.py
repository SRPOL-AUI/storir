from storir import ImpulseResponse

if __name__ == '__main__':
    # example configuration
    rt60 = 500
    edt = 50
    itdg = 3
    er_duration = 80
    sr = 16000

    rir = ImpulseResponse(rt60=rt60,
                          edt=edt,
                          itdg=itdg,
                          er_duration=er_duration)

    # get 5 impulse responses
    for _ in range(5):
        output = rir.generate(sampling_rate=sr)
        print(output.shape)
