# Copyright 2014 Knowledge Economy Developments Ltd
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

cimport numpy as np
cimport cpu
from libc.stdint cimport intptr_t


cdef int _simd_alignment = cpu.simd_alignment()

#: The optimum SIMD alignment in bytes, found by inspecting the CPU.
simd_alignment = _simd_alignment

#: A tuple of simd alignments that make sense for this cpu
if _simd_alignment == 16:
    _valid_simd_alignments = (16,)

elif _simd_alignment == 32:
    _valid_simd_alignments = (16, 32)

else:
    _valid_simd_alignments = ()

cpdef n_byte_align_empty(shape, n, dtype='float64', order='C'):
    '''n_byte_align_empty(shape, n, dtype='float64', order='C')

    Function that returns an empty numpy array
    that is n-byte aligned.

    The alignment is given by the second argument, ``n``.
    The rest of the arguments are as per :func:`numpy.empty`.
    '''
    
    itemsize = np.dtype(dtype).itemsize

    # Apparently there is an issue with numpy.prod wrapping around on 32-bits
    # on Windows 64-bit. This shouldn't happen, but the following code 
    # alleviates the problem.
    if not isinstance(shape, (int, np.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension
    
    else:
        array_length = shape

    # Allocate a new array that will contain the aligned data
    _array_aligned = np.empty(array_length*itemsize+n, dtype='int8')
    
    # We now need to know how to offset _array_aligned 
    # so it is correctly aligned
    _array_aligned_offset = (n-<intptr_t>np.PyArray_DATA(_array_aligned))%n

    array = np.frombuffer(
            _array_aligned[_array_aligned_offset:_array_aligned_offset-n].data,
            dtype=dtype).reshape(shape, order=order)
    
    return array

cpdef n_byte_align(array, n, dtype=None):
    ''' n_byte_align(array, n, dtype=None)

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is a passed parameter. If it is, the array is
    returned without further ado.  If it is not, a new array is created and
    the data copied in, but aligned on the n-byte boundary.

    ``dtype`` is an optional argument that forces the resultant array to be
    of that dtype.
    '''
    
    if not isinstance(array, np.ndarray):
        raise TypeError('Invalid array: n_byte_align requires a subclass '
                'of ndarray')

    if dtype is not None:
        if not array.dtype == dtype:
            update_dtype = True
    
    else:
        dtype = array.dtype
        update_dtype = False
    
    # See if we're already n byte aligned. If so, do nothing.
    offset = <intptr_t>np.PyArray_DATA(array) %n

    if offset is not 0 or update_dtype:

        _array_aligned = n_byte_align_empty(array.shape, n, dtype)

        _array_aligned[:] = array

        array = _array_aligned.view(type=array.__class__)
    
    return array

cpdef is_n_byte_aligned(array, n):
    ''' is_n_byte_aligned(array, n)

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is a passed parameter, returning ``True`` if it is,
    and ``False`` if it is not.
    '''
    if not isinstance(array, np.ndarray):
        raise TypeError('Invalid array: is_n_byte_aligned requires a subclass '
                'of ndarray')

    # See if we're n byte aligned.
    offset = <intptr_t>np.PyArray_DATA(array) %n

    return not bool(offset)
