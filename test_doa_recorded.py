from __future__ import division
from scipy.io import wavfile
import os, sys, getopt
import json

import pyroomacoustics as pra

import doa
from tools import *
from experiment import arrays, calculate_speed_of_sound

if __name__ == '__main__':

    # parse arguments
    argv = sys.argv[1:]
    algo = None
    rec_file = None
    n_bands = None
    try:
        opts, args = getopt.getopt(argv, "ha:f:b:", ["algo=", "file=", "n_bands"])
    except getopt.GetoptError:
        print('test_doa_recorded.py -a <algo> -f <file> -b <n_bands>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test_doa_recorded.py -a <algo> -f <file> -b <n_bands>')
            sys.exit()
        elif opt in ("-a", "--algo"):
            algo = int(arg)
        elif opt in ("-f", "--file"):
            rec_file = arg
        elif opt in ("-b", "--n_bands"):
            n_bands = int(arg)

    # We should make this the default structure
    # it can be applied by copying/downloading the data or creating a symbolic link
    exp_folder = './recordings/20160831/'

    # Get the speakers and microphones grounndtruth locations
    sys.path.append(exp_folder)
    from edm_to_positions import twitters

    array_str = 'pyramic'
    #array_str = 'compactsix'

    if array_str == 'pyramic':

        twitters.center('pyramic')

        R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)
        mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()
        mic_array += twitters[['pyramic']]

        rec_folder = exp_folder + 'data_pyramic/segmented/'

    elif array_str == 'compactsix':

        twitters.center('compactsix')

        R_flat_I = range(6)
        mic_array = arrays['compactsix_circular_1'][:, R_flat_I].copy()
        mic_array += twitters[['compactsix']]
        rec_folder = exp_folder + 'data_compactsix/segmented/'

    fs = 16000

    num_mic = mic_array.shape[1]  # number of microphones
    K = rec_file.count('-') + 1  # Real number of sources
    K_est = K  # Number of sources to estimate

    # Open the protocol json file
    with open(exp_folder + 'protocol.json') as fd:
        exp_data = json.load(fd)

    # These parameters could be extracted from a JSON file
    # Experiment related parameters
    temp = exp_data['conditions']['temperature']
    hum = exp_data['conditions']['humidity']
    c = calculate_speed_of_sound(temp, hum)
    # save parameters
    save_fig = False
    save_param = True
    fig_dir = './result/'

    # Check if the directory exists
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # algorithm parameters
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'
    fft_size = 256  # number of FFT bins
    frame_shift_step = np.int(fft_size / 1.)
    M = 24  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

    # ----------------------------
    # Perform direction of arrival
    phi_plt = np.linspace(0, 2*np.pi, num=721, dtype=float, endpoint=False)
    freq_range = [2300., 4300.]

    # Hand-picked frequencies for the two speech signals used
    freq_hz_s1 = [130., 266., 406., 494., 548., 682., 823., 960., 1100., 1236., 1500., 2229., 2577., 3182.]
    freq_hz_s2 = [200., 394., 518., 611., 724., 866., 924., 1042., 1884., 2094., 2441., 2794., 3351., 4122.]
    freq_hz_s3 = [200., 400., 600., 700., 875., 1450., 1640., 2100., 2450.]

    #freq_hz = np.array([ 1100., 1450., 1884., 2100., 2441., 2577., 3182., 3351, 4122., 4365, 4520])
    freq_hz = np.array([ 1884., 2100., 2441., 2577., 3182., 3351, 4122., 4365, ])

    # works well for 1 and 2 sources
    #freq_hz = np.array([ 2441., 2577., 3182., 3351, 4122.])

    # works well for 3 sources
    #freq_hz = np.array([2812.5, 3187.5, 3312., 3375., 4125., 4140.])

    #freq_hz = np.array([705.6, 1237., 1633., 2441., 2577., 3182., 3351., 4122., 5500., 6000.])

    freq_hz = np.array([2300., 2441., 2577.,  3182., 3351, 4122.])

    #freq_hz = np.linspace(freq_range[0], freq_range[1], n_bands)

    freq_bins = np.array([int(np.round(f / fs * fft_size)) for f in freq_hz])
    freq_bins = np.unique(freq_bins)[-n_bands:]

    print('Selected frequencies: {0} Hertz'.format(freq_bins / fft_size * fs))

    # Import speech signal
    # -------------------------
    if K == 1:
        filename = rec_folder + 'one_speaker/' + rec_file + '.wav'
    elif K == 2:
        filename = rec_folder + 'two_speakers/' + rec_file + '.wav'
    elif K == 3:
        filename = rec_folder + 'three_speakers/' + rec_file + '.wav'
    fs_file, rec_signals = wavfile.read(filename)
    fs_silence, rec_silence = wavfile.read(rec_folder + 'silence.wav')

    if fs_file != fs_silence:
        raise ValueError('Weird: fs of signals and silence are different...')

    # Resample the files if required
    if fs_file != fs:
        print 'Resampling signals'
        from scikits.samplerate import resample

        resampled_signals = []
        resampled_silence = []
        for i in R_flat_I:
            resampled_signals.append(
                pra.highpass(
                    resample(rec_signals[:, i], fs / fs_file, 'sinc_best'),
                    fs,
                    fc=150.
                )
            )
            resampled_silence.append(
                pra.highpass(
                    resample(rec_silence[:, i], fs / fs_file, 'sinc_best'),
                    fs,
                    fc=150.
                )
            )
        speech_signals = np.array(resampled_signals, dtype=np.float).T
        silence = np.array(resampled_silence, dtype=np.float).T

    else:
        print('No need to resample signals')
        speech_signals = np.array(rec_signals[:, R_flat_I], dtype=np.float32)
        silence = np.array(rec_silence[:, R_flat_I], dtype=np.float32)

        # highpass filter at 150
        for s in speech_signals.T:
            s[:] = pra.highpass(s, fs, fc=150.)
        for s in silence.T:
            s[:] = pra.highpass(s, fs, fc=150.)

    # Normalize the amplitude
    n_factor = 0.95 / np.max(np.abs(speech_signals))
    speech_signals *= n_factor
    silence *= n_factor

    # stft window
    win_stft = np.hanning(fft_size)

    # estimate noise floor
    y_noise_stft = []
    for k in range(num_mic):
        y_stft = pra.stft(silence[:, k], fft_size, frame_shift_step,
                          transform=rfft, win=win_stft).T / np.sqrt(fft_size)
        y_noise_stft.append(y_stft)
    y_noise_stft = np.array(y_noise_stft)
    noise_floor = np.mean(np.abs(y_noise_stft) ** 2)

    # estimate SNR in dB (on 1st microphone)
    noise_var = np.mean(np.abs(silence) ** 2)
    sig_var = np.mean(np.abs(speech_signals) ** 2)
    # rought estimate of SNR
    SNR = 10 * np.log10((sig_var - noise_var) / noise_var)
    print('Estimated SNR: ' + str(SNR))

    # Compute DFT of snapshots
    # -------------------------
    y_mic_stft = []
    for k in range(num_mic):
        y_stft = pra.stft(speech_signals[:, k], fft_size, frame_shift_step,
                          transform=rfft, win=win_stft).T / np.sqrt(fft_size)
        y_mic_stft.append(y_stft)
    y_mic_stft = np.array(y_mic_stft)

    energy_level = np.abs(y_mic_stft) ** 2

    # True direction of arrival
    # -------------------------
    sources = rec_file.split('-')
    phi_ks = np.array([twitters.doa(array_str, sources[k])[0] for k in range(K)])
    phi_ks[phi_ks < 0] = phi_ks[phi_ks < 0] + 2 * np.pi

    # create DOA object
    if algo == 1:
        algo_name = 'SRP-PHAT'
        d = doa.SRP(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, c=c,
                    theta=phi_plt)
    if algo == 2:
        algo_name = 'MUSIC'
        d = doa.MUSIC(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, c=c,
                      theta=phi_plt)
    elif algo == 3:
        algo_name = 'CSSM'
        d = doa.CSSM(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, c=c,
                     theta=phi_plt, num_iter=10)
    elif algo == 4:
        algo_name = 'WAVES'
        d = doa.WAVES(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, c=c,
                      theta=phi_plt, num_iter=10)
    elif algo == 5:
        algo_name = 'TOPS'
        d = doa.TOPS(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, c=c,
                     theta=phi_plt)
    elif algo == 6:
        algo_name = 'FRI'
        d = doa.FRI(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, c=c, 
            theta=phi_plt, max_four=M, noise_floor=noise_floor, noise_margin=0.0)

    # perform localization
    print 'Applying ' + algo_name + '...'
    # d.locate_sources(y_mic_stft, freq_bins=freq_bins)
    '''
    if isinstance(d, doa.TOPS) or isinstance(d, doa.WAVES) or isinstance(d, doa.MUSIC) or isinstance(d, doa.CSSM):
        d.locate_sources(y_mic_stft, freq_range=freq_range)
    else:
        print 'using bins'
        d.locate_sources(y_mic_stft, freq_bins=freq_bins)
    '''
    d.locate_sources(y_mic_stft, freq_bins=freq_bins)

    # print reconstruction results
    recon_err, sort_idx = polar_distance(phi_ks, d.phi_recon)
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})

    print('Reconstructed spherical coordinates (in degrees) and amplitudes:')
    if d.num_src > 1:
        #d.phi_recon = d.phi_recon[sort_idx[:,1]]
        print('Original azimuths   : {0}'.format(np.degrees(
            phi_ks[sort_idx[:, 0]])))
            #phi_ks)))
        print('Detected azimuths   : {0}'.format(np.degrees(
            d.phi_recon[sort_idx[:, 1]])))
            #d.phi_recon)))
    else:
        print('Original azimuths   : {0}'.format(np.degrees(phi_ks)))
        print('Detected azimuths   : {0}'.format(np.degrees(d.phi_recon)))

    if isinstance(d, doa.FRI):
        #d.alpha_recon = d.alpha_recon[:,sort_idx[:,1]]
        print d.alpha_recon.shape
        if K > 1:
            print('Reconstructed amplitudes : \n{0}\n'.format(d.alpha_recon.squeeze()))
        else:
            print('Reconstructed amplitudes : \n{0}\n'.format(d.alpha_recon.squeeze()))

    print('Reconstruction error     : {0:.3e}'.format(np.degrees(recon_err)))

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)

    # plot results
    file_name = (fig_dir + 'polar_sources_{0}_numMic_{1}_' +
                 '_locations' + '.pdf').format(repr(rec_file), repr(num_mic))

    # plot response (for FRI one subband)
    d.polar_plt_dirac(phi_ks, file_name=file_name)

    plt.show()
