#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2014 David Wells
#
# David Wells
# drwells <at> vt.edu
from __future__ import division
import itertools as it
import random as rand
import unittest
import numpy as np
import pyfftw

discrete_sine_directions = ['FFTW_RODFT00', 'FFTW_RODFT01', 'FFTW_RODFT10',
                            'FFTW_RODFT11']

discrete_cosine_directions = ['FFTW_REDFT00', 'FFTW_REDFT01', 'FFTW_REDFT10',
                              'FFTW_REDFT11']

real_transforms = discrete_sine_directions + discrete_cosine_directions

normalisation_lookup = {
    'FFTW_RODFT00': lambda n: 2*(n + 1),
    'FFTW_RODFT01': lambda n: 2*n,
    'FFTW_RODFT10': lambda n: 2*n,
    'FFTW_RODFT11': lambda n: 2*n,
    'FFTW_REDFT00': lambda n: 2*(n - 1),
    'FFTW_REDFT01': lambda n: 2*n,
    'FFTW_REDFT10': lambda n: 2*n,
    'FFTW_REDFT11': lambda n: 2*n,
}

inverse_lookup = {
    'FFTW_RODFT00': 'FFTW_RODFT00',
    'FFTW_RODFT01': 'FFTW_RODFT10',
    'FFTW_RODFT10': 'FFTW_RODFT01',
    'FFTW_RODFT11': 'FFTW_RODFT11',
    'FFTW_REDFT00': 'FFTW_REDFT00',
    'FFTW_REDFT01': 'FFTW_REDFT10',
    'FFTW_REDFT10': 'FFTW_REDFT01',
    'FFTW_REDFT11': 'FFTW_REDFT11',
}

interpolated_function_lookup = {
    'FFTW_RODFT00': lambda k, x: np.sin(np.pi*(k + 1)*x),
    'FFTW_RODFT01': lambda k, x: np.sin(np.pi*(k + 0.5)*x),
    'FFTW_RODFT10': lambda k, x: np.sin(np.pi*(k + 1)*x),
    'FFTW_RODFT11': lambda k, x: np.sin(np.pi*(k + 0.5)*x),

    'FFTW_REDFT00': lambda k, x: np.cos(np.pi*k*x),
    'FFTW_REDFT10': lambda k, x: np.cos(np.pi*k*x),
    'FFTW_REDFT01': lambda k, x: np.cos(np.pi*(k + 0.5)*x),
    'FFTW_REDFT11': lambda k, x: np.cos(np.pi*(k + 0.5)*x),
}

nodes_lookup = {
    'FFTW_RODFT00': lambda n: np.arange(n + 2)[1:-1]/(n + 1),
    'FFTW_RODFT01': lambda n: np.arange(1, n + 1)/n,
    'FFTW_RODFT10': lambda n: (np.arange(n) + 0.5)/n,
    'FFTW_RODFT11': lambda n: (np.arange(n) + 0.5)/n,

    'FFTW_REDFT00': lambda n: np.arange(n)/(n - 1),
    'FFTW_REDFT10': lambda n: (np.arange(n) + 0.5)/n,
    'FFTW_REDFT01': lambda n: np.arange(n)/n,
    'FFTW_REDFT11': lambda n: (np.arange(n) + 0.5)/n,
}

class TestRandomRealTransforms(unittest.TestCase):
    def test_normalisation(self):
        for _ in range(50):
            testcase = random_testcase()
            self.assertTrue(testcase.test_normalisation())

    def test_exact_data(self):
        for _ in range(50):
            testcase = random_testcase()
            self.assertTrue(testcase.test_against_exact_data())

    def test_random_data(self):
        for _ in range(50):
            testcase = random_testcase()
            self.assertTrue(testcase.test_against_random_data())

def test_lookups():
    """Test that the lookup tables correctly pair node choices and
    function choices for using the DCT/DST as interpolators.
    """
    n = rand.randint(10, 20)
    j = rand.randint(5, n) - 3
    for transform in real_transforms:
        nodes   = nodes_lookup[transform](n)
        data    = interpolated_function_lookup[transform](j, nodes)
        output  = np.empty_like(data)
        plan    = pyfftw.FFTW(data, output, direction=[transform])
        data[:] = interpolated_function_lookup[transform](j, nodes)
        plan.execute()
        tol = 4*j*n*1e-16
        if transform == 'FFTW_RODFT00':
            assert abs(output[j] - n - 1) < tol
        elif transform == 'FFTW_REDFT00':
            assert abs(output[j] - n + 1) < tol
        else:
            assert abs(output[j] - n) < tol

class TestRealTransform(object):
    def __init__(self, directions, dims, axes=None, noncontiguous=True):
        """
        Arguments:

        - `directions`: List of abbreviated directions, like 'O11' or 'E01'.

        - `dims`: Shape of the data.

        - `axes`: Axes on which to take the transformation. Defaults to the
                  number of directions.
        """
        if axes is None:
            self.axes = tuple(range(len(directions)))
        else:
            self.axes = axes
        for dim in dims:
            if dim < 3:
                raise NotImplementedError("Due to complications with the DCT1, "
                                          "arrays must be of length at least "
                                          "three.")

        if len(self.axes) != len(directions):
            raise ValueError("There must be exactly one axis per direction.")

        self.directions = directions
        self.inverse_directions = [inverse_lookup[direction]
                                    for direction in directions]
        self.dims = dims
        self._normalisation_factor = 1.0
        for index, axis in enumerate(self.axes):
            dim = self.dims[axis]
            direction = self.directions[index]
            self._normalisation_factor *= normalisation_lookup[direction](dim)

        if noncontiguous:
            self._input_array = empty_noncontiguous(dims)
            self._output_array = empty_noncontiguous(dims)
        else:
            self._input_array = np.zeros(dims)
            self._output_array = np.zeros(dims)
        self.plan = pyfftw.FFTW(self._input_array, self._output_array,
            axes=self.axes, direction=self.directions)
        self.inverse_plan = pyfftw.FFTW(self._input_array, self._output_array,
            axes=self.axes, direction=self.inverse_directions)

    def test_normalisation(self):
        return self._normalisation_factor == float(self.plan._get_N())

    def test_against_random_data(self):
        data = np.random.rand(*self.dims)
        self._input_array[:] = data
        self.plan.execute()
        self._input_array[:] = self._output_array[:]
        self.inverse_plan.execute()

        data *= self._normalisation_factor
        err = np.mean(np.abs(data - self._output_array))/self._normalisation_factor
        return err < 10e-8

    def test_against_exact_data(self):
        points = grid(self.dims, self.axes, self.directions)
        data   = np.ones_like(points[0])
        wavenumbers = list()
        factors = list()

        for index, axis in enumerate(self.axes):
            # Simplification: don't test constant terms. They are weird.
            if self.directions[index] in discrete_cosine_directions:
                wavenumber_min = 1
                wavenumber_max = self.dims[axis] - 2
            else:
                wavenumber_min = 0
                wavenumber_max = self.dims[axis] - 2
            _wavenumbers = sorted({rand.randint(wavenumber_min, wavenumber_max)
                                 for _ in range(self.dims[axis])})
            _factors = [rand.randint(1, 8) for _ in _wavenumbers]
            interpolated_function = interpolated_function_lookup[
                self.directions[index]]
            data *= sum((factor*interpolated_function(wavenumber, points[axis])
                         for factor, wavenumber in zip(_factors, _wavenumbers)))
            wavenumbers.append(np.array(_wavenumbers))
            factors.append(np.array(_factors))

        self._input_array[:] = data
        self.plan.execute()

        # zero all of the entries that do not correspond to a wavenumber.
        exact_coefficients = np.ones(data.shape)
        for index, axis in enumerate(self.axes):
            dim = self.dims[axis]
            sp = list(it.repeat(Ellipsis, len(data.shape)))
            zero_indicies = (np.array(list(set(np.arange(0, dim))
                                      - set(wavenumbers[index]))))
            if len(zero_indicies) == 0:
                pass
            else:
                sp[axis] = zero_indicies
                mask = np.ones(data.shape)
                mask[sp] = 0.0
                exact_coefficients *= mask

        # create the 'known' array of interpolation coefficients.
        normalisation = self.plan.N/(2**len(self.axes))
        for index, axis in enumerate(self.axes):
            for factor, wavenumber in zip(factors[index], wavenumbers[index]):
                sp = list(it.repeat(Ellipsis, len(data.shape)))
                sp[axis] = wavenumber
                exact_coefficients[sp] *= factor

        error = np.mean(np.abs(self._output_array/normalisation
                              - exact_coefficients))
        return error < 1e-8

def random_testcase():
    ndims = rand.randint(1, 5)

    axes = list()
    directions = list()
    dims = list()
    for dim in range(ndims):
        if ndims > 3:
            dims.append(rand.randint(3, 10))
        else:
            dims.append(rand.randint(3, 100))
        # throw out some dimensions randomly
        if rand.choice([True, True, False]):
            directions.append(rand.choice(real_transforms))
            axes.append(dim)

    if len(axes) == 0:
        # reroll.
        return random_testcase()
    else:
        return TestRealTransform(directions, dims, axes=axes)

def meshgrid(*x):
    if len(x) == 1:
        # necessary for one-dimensional case to work correctly. x is a
        # tuple due to the * operator.
        return x
    else:
        args = np.atleast_1d(*x)
        s0 = (1,)*len(args)
        return map(np.squeeze,
            np.broadcast_arrays(*[x.reshape(s0[:i] + (-1,) + s0[i + 1::])
                                  for i, x in enumerate(args)]))

def grid(shape, axes, directions, aspect_ratio=None):
    grids = [np.linspace(1, 2, dim) for dim in shape]
    for index, (axis, direction) in enumerate(zip(axes, directions)):
        grids[axis] = nodes_lookup[direction](shape[axes[index]])

    return np.array(meshgrid(*grids))

def empty_noncontiguous(shape):
    """Create a non-contiguous empty array with shape `shape`.
    """
    offsets = lambda s: [rand.randint(0, 3) for _ in s]
    strides = lambda s: [rand.randint(1, 3) for _ in s]
    parent_left_offsets = offsets(shape)
    parent_right_offsets = offsets(shape)
    parent_strides = strides(shape)

    parent_shape = list()
    child_slice = list()
    for index, length in enumerate(shape):
        left_offset = parent_left_offsets[index]
        right_offset = parent_right_offsets[index]
        stride = parent_strides[index]
        parent_shape.append(left_offset + stride*length + right_offset)
        if right_offset == 0:
            child_slice.append(slice(left_offset, None, stride))
        else:
            child_slice.append(slice(left_offset, -1*right_offset, stride))

    child = np.empty(parent_shape)[child_slice]
    if list(child.shape) != list(shape):
        raise ValueError("The shape of the noncontiguous array is incorrect."
                         " This is a bug.")
    return child

if __name__ == '__main__':
    unittest.main()
