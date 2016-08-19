''' 
Wrapper for the MKL FFT routines.

Inspiration from:
http://stackoverflow.com/questions/11752898/threaded-fft-in-enthought-python
'''

import numpy as np
import ctypes as _ctypes
import os

from dftidefs import *

def load_libmkl():
    if os.name == 'posix':
        try:
            lib_mkl = os.getenv('LIBMKL')
            return _ctypes.cdll.LoadLibrary(lib_mkl)
        except:
            pass
        try:
            return _ctypes.cdll.LoadLibrary("libmkl_rt.dylib")
        except:
            raise ValueError('MKL Library not found')

    else:
        try:
            return _ctypes.cdll.LoadLibrary("mk2_rt.dll")
        except:
            raise ValueError('MKL Library not found')

mkl = load_libmkl()


def mkl_rfft(a, n=None, axis=-1, norm=None, direction='forward', out=None):
    ''' 
    Forward one-dimensional double-precision real-complex FFT.
    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    if axis == -1:
        axis = a.ndim-1

    # This code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert (axis < a.ndim and axis >= -1)
    assert (direction == 'forward' or direction == 'backward')
    if direction == 'forward':
        assert a.dtype == np.float32 or a.dtype == np.float64
    else:
        assert a.dtype == np.complex64 or a.dtype == np.complex128

    order = 'C'
    if a.flags['F_CONTIGUOUS'] and not a.flags['C_CONTIGUOUS']:
        order = 'F'

    # Add zero padding or truncate if needed (incurs memory copy)
    if n is not None:
        m = n if direction == 'forward' else (n//2 + 1)
        if a.shape[axis] < m:
            # pad axis with zeros
            pad_width = np.zeros((a.ndim, 2), dtype=np.int)
            pad_width[axis,1] = m - a.shape[axis]
            a = np.pad(a, pad_width, mode='constant')
        elif a.shape[axis] > m:
            # truncate along axis
            b = np.swapaxes(a, axis, 0)[:m,]
            a = np.swapaxes(b, 0, axis).copy()
    elif direction == 'forward':
        n = a.shape[axis]

    elif direction == 'backward':
        n = 2*(a.shape[axis]-1)

    # determine output type
    if direction == 'backward':
        out_type = np.float64
        if a.dtype == np.complex64:
            out_type = np.float32
    elif direction == 'forward':
        out_type = np.complex128
        if a.dtype == np.float32:
            out_type = np.complex64

    # Configure output array
    assert a is not out
    if out is not None:
        assert out.dtype == out_type
        for i in xrange(a.ndim):
            if i != axis:
                assert a.shape[i] == out.shape[i]
        if direction == 'forward':
            assert (n//2 + 1) == out.shape[axis]
        else:
            assert out.shape[axis] == n
        assert not np.may_share_memory(a, out)
    else:
        size = list(a.shape)
        size[axis] = n//2 + 1 if direction == 'forward' else n
        out = np.empty(size, dtype=out_type, order=order)

    # Define length, number of transforms strides
    length = _ctypes.c_int(n)
    n_transforms = _ctypes.c_int(np.prod(a.shape)/a.shape[axis])

    # For strides, the C type used *must* be int64
    strides = (_ctypes.c_int64*2)(0, a.strides[axis]/a.itemsize)
    if a.ndim == 2:
        if axis == 0:
            distance = _ctypes.c_int(a.strides[1]/a.itemsize)
            out_distance = _ctypes.c_int(out.strides[1]/out.itemsize)
        else:
            distance = _ctypes.c_int(a.strides[0]/a.itemsize)
            out_distance = _ctypes.c_int(out.strides[0]/out.itemsize)

    double_precision = True
    if (direction == 'forward' and a.dtype == np.float32) or (direction == 'backward' and a.dtype == np.complex64):
        double_precision = False

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    if not double_precision:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_REAL, _ctypes.c_int(1), length)
    else:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_REAL, _ctypes.c_int(1), length)

    # set the storage type
    mkl.DftiSetValue(Desc_Handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)

    # set normalization factor
    if norm == 'ortho':
        if not double_precision:
            scale = _ctypes.c_double(1/np.sqrt(n))
        else:
            scale = _ctypes.c_double(1/np.sqrt(n))
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if not double_precision:
            scale = _ctypes.c_double(1./n)
        else:
            scale = _ctypes.c_double(1./n)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_DISTANCE, out_distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    # Not-in-place FFT
    mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

    mkl.DftiCommitDescriptor(Desc_Handle)
    
    fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


def mkl_fft(a, n=None, axis=-1, norm=None, direction='forward', out=None):
    ''' 
    Forward/Backward one-dimensional single/double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    # This code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert axis < a.ndim and axis >= -1

    # Add zero padding if needed (incurs memory copy)
    if n is not None and n != a.shape[axis]:
        pad_width = np.zeros((a.ndim, 2), dtype=np.int)
        pad_width[axis,1] = n - a.shape[axis]
        a = np.pad(a, pad_width, mode='constant')

    if n is not None:
        if a.shape[axis] < n:
            # pad axis with zeros
            pad_width = np.zeros((a.ndim, 2))
            pad_width[axis,1] = m - a.shape[axis]
            a = np.pad(x, pad_width, mode='constant')
        elif a.shape[axis] > n:
            # truncate along axis
            b = np.swapaxes(a, axis, 0)[:m,]
            a = np.swapaxes(b, 0, axis).copy()

    # Convert input to complex data type if real (also memory copy)
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)

    # Configure in-place vs out-of-place
    inplace = False
    if out is a:
        inplace = True
    elif out is not None:
        assert out.dtype == a.dtype
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Define length, number of transforms strides
    length = _ctypes.c_int(a.shape[axis])
    n_transforms = _ctypes.c_int(np.prod(a.shape)/a.shape[axis])
    
    # For strides, the C type used *must* be int64
    strides = (_ctypes.c_int64*2)(0, a.strides[axis]/a.itemsize)
    if a.ndim == 2:
        if axis == 0:
            distance = _ctypes.c_int(a.strides[1]/a.itemsize)
        else:
            distance = _ctypes.c_int(a.strides[0]/a.itemsize)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(1), length)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(1), length)

    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1/np.sqrt(a.shape[axis]))
        else:
            scale = _ctypes.c_double(1/np.sqrt(a.shape[axis]))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1./a.shape[axis])
        else:
            scale = _ctypes.c_double(1./a.shape[axis])
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        # In-place FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        # Not-in-place FFT
        mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )


    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


def mkl_fft2(a, norm=None, direction='forward', out=None):
    ''' 
    Forward two-dimensional double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    # convert input to complex data type if real (also memory copy)
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)

    # Configure in-place vs out-of-place
    inplace = False
    if out is a:
        inplace = True
    elif out is not None:
        assert out.dtype == a.dtype
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_int64*2)(*a.shape)
   
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)


    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_double(1.0/np.sqrt(np.prod(a.shape)))
        else:
            scale = _ctypes.c_double(1.0/np.sqrt(np.prod(a.shape)))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_double(1.0/np.prod(a.shape))
        else:
            scale = _ctypes.c_double(1.0/np.prod(a.shape))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)


    # Set input strides if necessary
    if not a.flags['C_CONTIGUOUS']:
        in_strides = (_ctypes.c_int*3)(0, a.strides[0]/a.itemsize, a.strides[1]/a.itemsize)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        # In-place FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )


    else:
        # Not-in-place FFT
        mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

        # Set output strides if necessary
        if not out.flags['C_CONTIGUOUS']:
            out_strides = (_ctypes.c_int*3)(0, out.strides[0]/out.itemsize, out.strides[1]/out.itemsize)
            mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out

def rfft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_rfft(a, n=n, axis=axis, norm=norm, direction='forward', out=out)

def irfft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_rfft(a, n=n, axis=axis, norm=norm, direction='backward', out=out)

def fft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='forward', out=out)

def ifft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='backward', out=out)

def fft2(a, norm=None, out=None):
    return mkl_fft2(a, norm=norm, direction='forward', out=out)

def ifft2(a, norm=None, out=None):
    return mkl_fft2(a, norm=norm, direction='backward', out=out)


def cce2full(A):

    # Assume all square for now

    N = A.shape
    N_half = N[0]//2 + 1
    out = np.empty((A.shape[0], A.shape[0]), dtype=A.dtype)
    out[:, :N_half] = A

    out[1:, N_half:] = np.rot90(A[1:, 1:-1], 2).conj()

    # Complete the first row
    out[0, N_half:] = A[0, -2:0:-1].conj()

    return out


def mkl_rfft2(a, norm=None, direction='forward', out=None):
    ''' 
    Forward two-dimensional double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    assert (a.dtype == np.float32) or (a.dtype == np.float64)

    out_type = np.complex128
    if a.dtype == np.float32:
        out_type = np.complex64

    n = a.shape[1]

    # Allocate memory if needed
    if out is not None:
        assert out.dtype == out_type
        assert out.shape[1] == n//2 + 1
        assert not np.may_share_memory(a, out)
    else:
        size = list(a.shape)
        size[1] = n//2 + 1
        out = np.empty(size, dtype=out_type)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_int64*2)(*a.shape)
   
    if a.dtype == np.float32:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_REAL, _ctypes.c_int(2), dims)
    elif a.dtype == np.float64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_REAL, _ctypes.c_int(2), dims)

    # Set the storage type
    mkl.DftiSetValue(Desc_Handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)

    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.float32:
            scale = _ctypes.c_float(1.0/np.sqrt(np.prod(a.shape)))
        else:
            scale = _ctypes.c_double(1.0/np.sqrt(np.prod(a.shape)))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.float64:
            scale = _ctypes.c_float(1.0/np.prod(a.shape))
        else:
            scale = _ctypes.c_double(1.0/np.prod(a.shape))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    
    # For strides, the C type used *must* be int64
    in_strides = (_ctypes.c_int64*3)(0, a.strides[0]/a.itemsize, a.strides[1]/a.itemsize)
    out_strides = (_ctypes.c_int64*3)(0, out.strides[0]/out.itemsize, out.strides[1]/out.itemsize)
    
    # mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))
    mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    # Not-in-place FFT
    mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

    # Set output strides if necessary
    if not out.flags['C_CONTIGUOUS']:
        out_strides = (_ctypes.c_int*3)(0, out.strides[0]/out.itemsize, out.strides[1]/out.itemsize)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

    mkl.DftiCommitDescriptor(Desc_Handle)
    fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


if __name__ == "__main__":

    import time

    n_iter = 200
    N = 256

    np.seterr(all='raise')

    A = np.complex64(np.random.randn(N, N))
    C = np.zeros((N, N), dtype='complex64')
    start_time = time.time()
    for i in range(n_iter):
        C += np.fft.fft(A)
    print("--- %s seconds ---" % (time.time() - start_time))

    A = np.complex64(np.random.randn(N, N))
    C = np.zeros((N, N), dtype='complex64')
    start_time = time.time()
    for i in range(n_iter):
        C += fft(A)
    print("--- %s seconds ---" % (time.time() - start_time))

    A = np.float32(np.random.randn(N, N))
    C = np.zeros((N, N//2+1), dtype='complex64')
    start_time = time.time()
    for i in range(n_iter):
        C += mkl_rfft(A)
    print("--- %s seconds ---" % (time.time() - start_time))

