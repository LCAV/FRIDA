from __future__ import division

def parallel_loop(algo_names, pmt, args):
    '''
    This is one loop of the computation
    extracted for parallelization
    '''

    SNR = args[0]
    separation_angle = args[1]
    look = args[2]
    seed = args[3]

    # We need to do a bunch of imports
    import pyroomacoustics as pra
    import os
    import numpy as np
    from scipy.io import wavfile
    import mkl as mkl_service

    import doa
    from tools import rfft, polar_error, polar_distance, gen_sig_at_mic_stft, gen_diracs_param

    # initialize local RNG seed
    np.random.seed(seed)

    # for such parallel processing, it is better 
    # to deactivate multithreading in mkl
    mkl_service.set_num_threads(1)

    # number of sources is always two
    K = 2

    # The ground truth angles, same power sources
    alpha_gt = np.ones(2, dtype=np.float32)
    phi_gt = np.array([look, look + separation_angle], dtype=np.float32)

    # generate complex base-band signal received at microphones
    y_mic_stft, y_mic_stft_noiseless = \
            gen_sig_at_mic_stft(phi_gt, alpha_gt, pmt['mic_array'][:2,:], SNR,
                            pmt['fs'], fft_size=pmt['nfft'], Ns=pmt['num_snapshots'])

    # dict for output
    phi = { 'groundtruth': phi_gt, }
    phi_errors = {}

    for alg in algo_names:

        # select frequency bins uniformly in the range
        freq_hz = np.linspace(pmt['freq_range'][alg][0], pmt['freq_range'][alg][1], pmt['n_bands'][alg])
        freq_bins = np.unique(
                np.array([int(np.round(f / pmt['fs'] * pmt['nfft'])) 
                    for f in freq_hz])
                )

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
	if K > 1:
	    phi_errors[alg] = polar_error(phi_gt[sort_idx[:,0]], d.phi_recon[sort_idx[:,1]])
        else:
	    phi_errors[alg] = polar_error(phi_gt, d.phi_recon)

        # store result
        phi[alg] = d.phi_recon

    return phi, phi_errors 


if __name__ == '__main__':

    '''
    This experiment will examine the discrimination power
    of different algorithms for sources closely spaced
    '''

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
    SNRs = [0]
    separation_angle = np.pi / 2.**np.arange(6.,0.5,-0.5)
    look_direction = np.linspace(0, 2.*np.pi/3., 120., endpoint=False)  # 120 because of array sym
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
            'seed': 12345,
            }

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


    # The frequency grid for the algorithms requiring a grid search
    parameters['phi_grid'] = np.linspace(0, 2*np.pi, num=721, dtype=float, endpoint=False)

    # seed the original RNG
    np.random.seed(parameters['seed'])

    # build the combinatorial argument list
    args = []
    for SNR in SNRs:
        for phi in separation_angle:
            for look in look_direction:
                for epoch in range(loops):
                    seed = np.random.randint(4294967295, dtype=np.uint32)
                    args.append( (SNR, phi, look, seed) )

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
    np.savez('data/{}_doa_separation.npz'.format(date), args=args, parameters=parameters, algo_names=algo_names, out=out) 

