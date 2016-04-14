# Copyright 2014 Knowledge Economy Developments Ltd
# Copyright 2014 David Wells
#
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

from pyfftw.interfaces import scipy_fftpack
import pyfftw
import numpy

try:
    import scipy
    import scipy.fftpack
    import scipy.signal

except ImportError:
    scipy_missing = True

else:
    scipy_missing = False

import unittest
from .test_pyfftw_base import run_test_suites
from . import test_pyfftw_numpy_interface

'''pyfftw.interfaces.scipy_fftpack wraps pyfftw.interfaces.numpy_fft and
implements the dct and dst functions.

All the tests here just check that the call is made correctly.
'''

funcs = ('fft','ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 
           'rfft', 'irfft')

acquired_names = ('diff', 'tilbert', 'itilbert', 'hilbert',
        'ihilbert', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff', 'shift',
        'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'convolve',
        '_fftpack')

def make_complex_data(shape, dtype):
    ar, ai = dtype(numpy.random.randn(2, *shape))
    return ar + 1j*ai

def make_r2c_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))

def make_c2r_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))

make_complex_data = test_pyfftw_numpy_interface.make_complex_data

complex_dtypes = test_pyfftw_numpy_interface.complex_dtypes
real_dtypes = test_pyfftw_numpy_interface.real_dtypes

def numpy_fft_replacement(a, s, axes, overwrite_input, planner_effort, 
        threads, auto_align_input, auto_contiguous):

    return (a, s, axes, overwrite_input, planner_effort, 
        threads, auto_align_input, auto_contiguous)

io_dtypes = {
        'complex': (complex_dtypes, make_complex_data),
        'r2c': (real_dtypes, make_r2c_real_data),
        'c2r': (real_dtypes, make_c2r_real_data)}

@unittest.skipIf(scipy_missing, 'scipy is not installed, so this feature is'
                 'unavailable')
class InterfacesScipyFFTPackTestSimple(unittest.TestCase):
    ''' A really simple test suite to check simple implementation.
    '''

    def test_scipy_overwrite(self):

        new_style_scipy_fftn = False
        try:
            scipy_fftn = scipy.signal.signaltools.fftn
            scipy_ifftn = scipy.signal.signaltools.ifftn
        except AttributeError:
            scipy_fftn = scipy.fftpack.fftn
            scipy_ifftn = scipy.fftpack.ifftn
            new_style_scipy_fftn = True

        a = pyfftw.empty_aligned((128, 64), dtype='complex128', n=16)
        b = pyfftw.empty_aligned((128, 64), dtype='complex128', n=16)

        a[:] = (numpy.random.randn(*a.shape) + 
                1j*numpy.random.randn(*a.shape))
        b[:] = (numpy.random.randn(*b.shape) + 
                1j*numpy.random.randn(*b.shape))


        scipy_c = scipy.signal.fftconvolve(a, b)

        if new_style_scipy_fftn:
            scipy.fftpack.fftn = scipy_fftpack.fftn
            scipy.fftpack.ifftn = scipy_fftpack.ifftn

        else:
            scipy.signal.signaltools.fftn = scipy_fftpack.fftn
            scipy.signal.signaltools.ifftn = scipy_fftpack.ifftn

        scipy_replaced_c = scipy.signal.fftconvolve(a, b)

        self.assertTrue(numpy.allclose(scipy_c, scipy_replaced_c))

        if new_style_scipy_fftn:
            scipy.fftpack.fftn = scipy_fftn
            scipy.fftpack.ifftn = scipy_ifftn

        else:
            scipy.signal.signaltools.fftn = scipy_fftn
            scipy.signal.signaltools.ifftn = scipy_ifftn

    def test_funcs(self):

        for each_func in funcs:
            func_being_replaced = getattr(scipy_fftpack, each_func)

            #create args (8 of them)
            args = []
            for n in range(8):
                args.append(object())

            args = tuple(args)

            try:
                setattr(scipy_fftpack, each_func, 
                        numpy_fft_replacement)

                return_args = getattr(scipy_fftpack, each_func)(*args)
                for n, each_arg in enumerate(args):
                    # Check that what comes back is what is sent
                    # (which it should be)
                    self.assertIs(each_arg, return_args[n])
            except:
                raise

            finally:
                setattr(scipy_fftpack, each_func, 
                        func_being_replaced)

    def test_acquired_names(self):
        for each_name in acquired_names: 

            fftpack_attr = getattr(scipy.fftpack, each_name)
            acquired_attr = getattr(scipy_fftpack, each_name)

            self.assertIs(fftpack_attr, acquired_attr)

@unittest.skipIf(scipy_missing, 'scipy is not installed, so this feature is'
                 'unavailable')
class InterfacesScipyFFTTest(unittest.TestCase):
    ''' Class template for building the scipy real to real tests.
    '''

    # unittest is not very smart and will always turn this class into a test,
    # even though it is not on the list. Hence mark test-dependent values as
    # constants (so this particular test ends up being run twice).
    func_name = 'dct'
    floating_type = numpy.float64

    def setUp(self):
        self.scipy_func = getattr(scipy.fftpack, self.func_name)
        self.pyfftw_func = getattr(scipy_fftpack, self.func_name)
        self.ndims = numpy.random.randint(1, high=3)
        self.shape = numpy.random.randint(2, high=10, size=self.ndims)
        self.data = numpy.random.rand(*self.shape).astype(floating_type)
        self.data_copy = self.data.copy()

    def test_unnormalized(self):
        '''Test unnormalized pyfftw transformations against their scipy
        equivalents.
        '''
        for transform_type in range(1, 4):
            data_hat_p = self.pyfftw_func(self.data, type=transform_type,
                                          overwrite_x=False)
            self.assertEqual(numpy.linalg.norm(self.data - self.data_copy), 0.0)
            data_hat_s = self.scipy_func(self.data, type=transform_type,
                                         overwrite_x=False)
            self.assertTrue(numpy.allclose(data_hat_p, data_hat_s))

    def test_normalized(self):
        '''Test normalized against scipy results. Note that scipy does
        not support normalization for all transformations.
        '''
        for transform_type in range(1, 4):
            data_hat_p = self.pyfftw_func(self.data, type=transform_type,
                                          norm='ortho',
                                          overwrite_x=False)
            self.assertEqual(numpy.linalg.norm(self.data - self.data_copy), 0.0)
            try:
                data_hat_s = self.scipy_func(self.data, type=transform_type,
                                             norm='ortho',
                                             overwrite_x=False)
                self.assertTrue(numpy.allclose(data_hat_p, data_hat_s))
            except NotImplementedError:
                return None

    def test_normalization_inverses(self):
        '''Test normalization in all of the pyfftw scipy wrappers.
        '''
        for transform_type in range(1, 4):
            inverse_type = {1: 1, 2: 3, 3:2}[transform_type]
            forward = self.pyfftw_func(self.data, type=transform_type,
                                       norm='ortho',
                                       overwrite_x=False)
            result = self.pyfftw_func(forward, type=inverse_type,
                                      norm='ortho',
                                      overwrite_x=False)
            self.assertTrue(numpy.allclose(self.data, result))


built_classes = []
# Construct the r2r test classes.
for floating_type, floating_name in [[numpy.float32, 'Float32'],
                                     [numpy.float64, 'Float64']]:
    for transform_name in ('dct', 'idct', 'dst', 'idst'):
        class_name = ('InterfacesScipyFFTTest' + transform_name.upper()
                      + floating_name)

        globals()[class_name] = type(class_name, (InterfacesScipyFFTTest,),
                                    {'func_name': transform_name,
                                     'float_type': floating_type})

        built_classes.append(globals()[class_name])


# Construct the test classes derived from the numpy tests.
for each_func in funcs:

    class_name = 'InterfacesScipyFFTPackTest' + each_func.upper()

    parent_class_name = 'InterfacesNumpyFFTTest' + each_func.upper()
    parent_class = getattr(test_pyfftw_numpy_interface, parent_class_name)

    class_dict = {'validator_module': scipy.fftpack, 
                'test_interface': scipy_fftpack,
                'io_dtypes': io_dtypes,
                'overwrite_input_flag': 'overwrite_x',
                'default_s_from_shape_slicer': slice(None)}

    globals()[class_name] = type(class_name,
            (parent_class,), class_dict)

    built_classes.append(globals()[class_name])

built_classes = tuple(built_classes)

test_cases = (
        InterfacesScipyFFTPackTestSimple,) + built_classes

test_set = None
#test_set = {'InterfacesScipyFFTPackTestIFFTN': ['test_auto_align_input']}


if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
