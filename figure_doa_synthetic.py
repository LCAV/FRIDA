from __future__ import division

def parallel_loop(algo_names, pmt, args):
    '''
    This is one loop of the computation
    extracted for parallelization
    '''

    number_sources = args[0]
    SNR = args[1]
    n_bands = args[2]

    # We need to do a bunch of imports
    import pyroomacoustics as pra
    import os
    import numpy as np
    from scipy.io import wavfile
    import mkl as mkl_service

    import doa
    from tools import rfft, polar_error, polar_distance, gen_sig_at_mic_stft, gen_diracs_param

    # for such parallel processing, it is better 
    # to deactivate multithreading in mkl
    mkl_service.set_num_threads(1)

    # number of sources
    K = number_sources

    # Generate "groundtruth" Diracs at random
    alpha_gt, phi_gt, time_stamp = gen_diracs_param(
            K, positive_amp=True, log_normal_amp=False,
            semicircle=False, save_param=False
            )

    # generate complex base-band signal received at microphones
    y_mic_stft, y_mic_stft_noiseless = \
            gen_sig_at_mic_stft(phi_gt, alpha_gt, pmt['mic_array'][:2,:], SNR,
                            pmt['fs'], fft_size=pmt['nfft'], Ns=pmt['num_snapshots'])

    # select frequency bins
    freq_hz = np.linspace(pmt['freq_range'][0], pmt['freq_range'][1], n_bands)
    freq_bins = np.unique(
            np.array([int(np.round(f / pmt['fs'] * pmt['nfft'])) 
                for f in freq_hz])
            )

    # dict for output
    phi = { 'groundtruth': phi_gt, }
    alpha = { 'groundtruth': alpha_gt, }
    phi_errors = {}
    alpha_errors = {}

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
                num_iter=pmt['num_iter']
                )

        # perform localization
        d.locate_sources(y_mic_stft, freq_bins=freq_bins)

        # sort out confusion
        recon_err, sort_idx = polar_distance(phi_gt, d.phi_recon)

        # errors
        phi_errors[alg] = polar_error(phi_gt[sort_idx[:,0]], d.phi_recon[sort_idx[:,1]])

        # store result
        phi[alg] = d.phi_recon

        if alg == 'FRI':
            alpha_errors[alg] = np.abs(alpha_gt[sort_idx[:,0]] - d.alpha_recon[sort_idx[:,1]])
            alpha[alg] = d.alpha_recon[sort_idx[:,1]]

    return phi, phi_errors, alpha, alpha_errors, len(freq_bins)


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

    # parse arguments
    algo_names = ['SRP', 'MUSIC', 'CSSM', 'WAVES', 'TOPS', 'FRI']
    num_sources = range(1,3+1)
    SNRs = [-5, 0, 5, 10, 15, 20]
    n_bands = [2,4]
    loops = 10
    
    # We use the same array geometry as in the experiment
    array_str = 'pyramic'
    #array_str = 'compactsix'

    if array_str == 'pyramic':

        # subselect the flat part of the array
        R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)

        # get array geometry
        mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()

    elif array_str == 'compactsix':

        R_flat_I = range(6)
        mic_array = arrays['compactsix_circular_1'][:,R_flat_I].copy()

    # algorithm parameters
    parameters = {
            'mic_array' : mic_array,  # The array geometry
            'mic_select': R_flat_I,   # A subselection of microphones
            'fs' : 16000,  # the sampling frequency
            'nfft': 256,   # The FFT size
            'stft_hop': 256, # the number of samples between two stft frames
            'stft_win': True, # Use a hanning window for the STFT
            'num_snapshots': 256, # The number of snapshots to compute covariance  matrix
            'c': 343.,        # The speed of sound
            'M' : 24,      # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2
            'num_iter' : 10,  # Maximum number of iterations for algorithms that require them
            'stop_cri' : 'max_iter',  # stropping criterion for FRI ('mse' or 'max_iter')
            'freq_range': [2000., 4000.],
            }

    # The frequency grid for the algorithms requiring a grid search
    parameters['phi_grid'] = np.linspace(0, 2*np.pi, num=721, dtype=float, endpoint=False)

    # build the combinatorial argument list
    args = []
    for K in num_sources:
        for SNR in SNRs:
            for B in n_bands:
                for epoch in range(loops):
                    args.append((K, SNR, B))

    # Start the parallel processing
    c = ip.Client()
    NC = len(c.ids)
    print NC,'workers on the job'

    # replicate some parameters
    algo_names_ls = [algo_names]*len(args)
    params_ls = [parameters]*len(args)

    # dispatch to workers
    out = c[:].map_sync(parallel_loop, algo_names_ls, params_ls, args)

    # Save the result to a file
    date = time.strftime("%Y%m%d-%H%M%S")
    np.savez('data/{}_doa_synthetic.npz'.format(date), args=args, parameters=parameters, algo_names=algo_names, out=out) 

