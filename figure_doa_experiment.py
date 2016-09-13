from __future__ import division
from experiment import select_bands

def parallel_loop(filename, algo_names, pmt):
    '''
    This is one loop of the computation
    extracted for parallelization
    '''

    # We need to do a bunch of imports
    import pyroomacoustics as pra
    import os
    import numpy as np
    from scipy.io import wavfile
    import mkl as mkl_service
    import copy

    import doa
    from tools import rfft

    # for such parallel processing, it is better 
    # to deactivate multithreading in mkl
    mkl_service.set_num_threads(1)

    # exctract the speaker names from filename
    name = os.path.splitext(os.path.basename(filename))[0]
    sources = name.split('-')

    # number of sources
    K = len(sources)

    # Import speech signal
    fs_file, rec_signals = wavfile.read(filename)

    # sanity check
    if pmt['fs'] != fs_file:
        raise ValueError('The sampling frequency of the files doesn''t match that of the script')
    
    speech_signals = np.array(rec_signals[:,pmt['mic_select']], dtype=np.float32)

    # Remove the DC bias
    for s in speech_signals.T:
        s[:] = pra.highpass(s, pmt['fs'], 100.)

    if pmt['stft_win']:
        stft_win = np.hanning(pmt['nfft'])
    else:
        stft_win = None

    # Normalize the amplitude
    speech_signals *= pmt['scaling']

    # Compute STFT of signal
    # -------------------------
    y_mic_stft = []
    for k in range(speech_signals.shape[1]):
        y_stft = pra.stft(speech_signals[:, k], pmt['nfft'], pmt['stft_hop'],
                          transform=rfft, win=stft_win).T / np.sqrt(pmt['nfft'])
        y_mic_stft.append(y_stft)
    y_mic_stft = np.array(y_mic_stft)

    # estimate SNR in dB (on 1st microphone)
    sig_var = np.var(speech_signals)
    SNR = 10*np.log10( (sig_var - pmt['noise_var']) / pmt['noise_var'] )

    freq_bins = copy.copy(pmt['freq_bins'][K-1])

    # dict for output
    phi_recon = {}

    for alg in algo_names:

        # Use the convenient dictionary of algorithms defined
        d = doa.algos[alg](
                L=pmt['mic_array'], 
                fs=pmt['fs'], 
                nfft=pmt['nfft'], 
                num_src=K, 
                c=pmt['c'], 
                theta=pmt['phi_grid'], 
                max_four=pmt['M'], 
                num_iter=pmt['num_iter'],
                G_iter = pmt['G_iter']
                )

        # perform localization
        d.locate_sources(y_mic_stft, freq_bins=freq_bins[alg])

        # store result
        phi_recon[alg] = d.phi_recon

    return SNR, sources, phi_recon


if __name__ == '__main__':

    import numpy as np
    from scipy.io import wavfile
    import os, sys, getopt
    import time
    import json

    import ipyparallel as ip

    import pyroomacoustics as pra

    import doa
    from tools import rfft
    from experiment import arrays, calculate_speed_of_sound

    # We should make this the default structure
    # it can be applied by copying/downloading the data or creating a symbolic link
    exp_folder = './recordings/20160908/'

    # Open the protocol json file
    with open(exp_folder + 'protocol.json') as fd:
        exp_data = json.load(fd)

    # Get the speakers and microphones grounndtruth locations
    sys.path.append(exp_folder)
    from edm_to_positions import twitters

    array_str = 'pyramic'
    #array_str = 'compactsix'

    if array_str == 'pyramic':

        twitters.center('pyramic')

        # subselect the flat part of the array
        R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)

        # get array geometry
        mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()
        mic_array += twitters[['pyramic']]

        # data subfolder
        rec_folder = exp_folder + 'data_pyramic/segmented/'

    elif array_str == 'compactsix':

        twitters.center('compactsix')

        R_flat_I = range(6)
        mic_array = arrays['compactsix_circular_1'][:,R_flat_I].copy()
        mic_array += twitters[['compactsix']]
        rec_folder = exp_folder + 'data_compactsix/segmented/'
        missing_rec = ()

    # General parameters
    fs = 16000

    # Define the algorithms to run
    algo_names = ['SRP', 'MUSIC', 'CSSM', 'WAVES', 'TOPS', 'FRI']

    # Experiment related parameters
    temp = exp_data['conditions']['temperature']
    hum = exp_data['conditions']['humidity']
    c = calculate_speed_of_sound(temp, hum)

    # algorithm parameters
    parameters = {
            'mic_array' : mic_array,  # The array geometry
            'mic_select': R_flat_I,   # A subselection of microphones
            'fs' : 16000,  # the sampling frequency
            'nfft': 256,   # The FFT size
            'stft_hop': 256, # the number of samples between two stft frames
            'stft_win': True, # Use a hanning window for the STFT
            'c': c,        # The speed of sound
            'M' : 24,      # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2
            'num_iter' : 10,  # Maximum number of iterations for algorithms that require them
            'stop_cri' : 'max_iter',  # stropping criterion for FRI ('mse' or 'max_iter')
            'G_iter' : 1,   # Maximum of liner mapping update iterations
            }

    # The frequency grid for the algorithms requiring a grid search
    parameters['phi_grid'] = np.linspace(0, 2*np.pi, num=721, dtype=float, endpoint=False)

    # ----------------------------
    # The mighty frequency band selection

    # Old 'magic' bands
    #freq_hz = np.array([2300., 2441., 2577., 3182., 3351, 4122.])

    # Choose the frequency range to use
    # These were chosen empirically to give good performance
    parameters['freq_range'] = {
            'MUSIC': [2500., 4500.],
            'SRP':   [2500., 4500.],
            'CSSM':  [2500., 4500.],
            'WAVES': [3000., 4000.],
            'TOPS':  [100., 5000.],
            'FRI':   [2500., 4500.],
            }

    parameters['n_bands'] = {
            'MUSIC' : 20,
            'SRP' :   20,
            'CSSM' :  10,
            'WAVES' : 10,
            'TOPS' :  60,
            'FRI' :   20,
            }

    # Band selection
    parameters['freq_bins'] = []
    for K in [1,2,3]:

        # the samples are used to select the frequencies
        samples = ['experiment/samples/fq_sample{}.wav'.format(i) for i in range(K)]

        parameters['freq_bins'].append({})

        for algo in algo_names:
            # Call the band selection routine
            # The routine averages the speech signals
            # then splits the range into n_bands equal size bands
            # and picks the bin with largest power in each band
            freq_hz, freq_bins = select_bands(
                    samples, 
                    parameters['freq_range'][algo], 
                    parameters['fs'], 
                    parameters['nfft'],
                    parameters['stft_win'], 
                    parameters['n_bands'][algo],
                    div = 1)
            parameters['freq_bins'][-1][algo] = freq_bins
            print K, algo, 'Number of bins', freq_bins.shape[0]

    #-----------------------------------------------
    # Get the silence file to use for SNR estimation
    fs_silence, rec_silence = wavfile.read(rec_folder + 'silence.wav')
    silence = np.array(rec_silence[:,R_flat_I], dtype=np.float32)
    for s in silence.T:
        s[:] = s - s.mean()

    # This is a scaling factor to apply to raw signals
    parameters['scaling'] = np.sqrt(0.1 / np.var(silence))
    silence *= parameters['scaling']

    # Compute noise variance for later SNR estimation
    parameters['noise_var'] = np.var(silence)

    # The folders for the different numbers of speakers
    spkr_2_folder = { 1: 'one_speaker/', 2: 'two_speakers/', 3: 'three_speakers/' }

    # collect all filenames
    filenames = []
    for K in range(1,4):
        fldr = rec_folder + spkr_2_folder[K]
        filenames += [fldr + name for name in os.listdir(rec_folder + spkr_2_folder[K])]

    # Start the parallel processing
    c = ip.Client()
    NC = len(c.ids)
    print NC,'workers on the job'

    # replicate some parameters
    algo_names_ls = [algo_names]*len(filenames)
    params_ls = [parameters]*len(filenames)

    # dispatch to workers
    out = c[:].map_sync(parallel_loop, filenames, algo_names_ls, params_ls)

    # Save the result to a file
    date = time.strftime("%Y%m%d-%H%M%S")
    np.savez('data/{}_doa_experiment.npz'.format(date), filenames=filenames, parameters=parameters, algo_names=algo_names, out=out) 

