import pytest

import cupy

import numpy
import scipy.interpolate  # NOQA

import cupyx.scipy.interpolate
from cupy import testing
from cupy.testing import (
    assert_array_equal,
    numpy_cupy_allclose,
)


@testing.gpu()
@testing.with_requires("scipy")
class TestBarycentric:

    def setup_method(self):
        self.true_poly = cupy.poly1d([-2, 3, 1, 5, -4])
        self.test_xs = cupy.linspace(-1, 1, 100)
        self.xs = cupy.linspace(-1, 1, 5)
        self.ys = self.true_poly(self.xs)

    def test_lagrange(self):
        P = cupyx.scipy.interpolate.BarycentricInterpolator(self.xs,
                                                            self.ys)
        testing.assert_allclose(self.true_poly(self.test_xs),
                                P(self.test_xs), rtol=1e-6)

    def test_scalar(self):
        P = cupyx.scipy.interpolate.BarycentricInterpolator(self.xs,
                                                            self.ys)
        testing.assert_allclose(self.true_poly(cupy.array(7)),
                                               P(cupy.array(7)))
        testing.assert_allclose(self.true_poly(7), P(7))

    def test_delayed(self):
        P = cupyx.scipy.interpolate.BarycentricInterpolator(self.xs[:3],
                                                            self.ys[:3])
        P.add_xi(self.xs[3:], self.ys[3:])
        testing.assert_allclose(self.true_poly(self.test_xs),
                                               P(self.test_xs))
