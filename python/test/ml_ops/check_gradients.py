# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import numpy as np
from collections import OrderedDict


def compute_jacobian_finite_differences(x0, fn, epsilon):
    """Computes the Jacobian using finite differences

    x0:      The positions at which to compute J.

    fn:      A function of the form fn(x) which returns a single numpy array.

    epsilon: A scalar or an array that can be broadcasted to the same
             shape as x0.
    """
    dtype = x0.dtype
    y0 = fn(x0)
    h = np.zeros_like(x0)
    J = np.zeros((x0.size, y0.size), dtype=dtype)

    epsilon_arr = np.broadcast_to(epsilon, x0.shape)

    for i in range(x0.size):
        eps = epsilon_arr.flat[i]
        h.flat[i] = eps
        J[i, :] = ((fn(x0 + h) - y0) / eps).flat
        h.flat[i] = 0

    return J


def compute_jacobian_analytical(x0, y_shape, fn_grad, y_bp=None):
    """Computes the analytical Jacobian

    x0:      The position at which to compute J.

    y_shape: The shape of the backpropagated value, i.e. the shape of
             the output of the corresponding function 'fn'.

    fn_grad: The gradient of the original function with the form
             x_grad = fn_grad(y_bp, x0) where 'y_bp' is the backpropagated
             value and 'x0' is the original input to 'fn'. The output of
             the function is the gradient of x wrt to y.

    y_bp:    Optional array with custom values for individually scaling
             the gradients.

    """
    dtype = x0.dtype
    y_size = 1
    for k in y_shape:
        y_size *= k

    J = np.zeros((x0.size, y_size), dtype=dtype)

    y = np.zeros(y_shape, dtype=dtype)

    y_bp_arr = np.broadcast_to(y_bp, y_shape) if not y_bp is None else np.ones(
        y_shape, dtype=dtype)

    for j in range(y_size):
        y.flat[j] = y_bp_arr.flat[j]
        J[:, j] = fn_grad(y, x0).flat
        y.flat[j] = 0

    return J


def check_gradients(x0,
                    fn,
                    fn_grad,
                    epsilon=1e-6,
                    rtol=1e-3,
                    atol=1e-5,
                    debug_outputs=OrderedDict()):
    """Checks if the numerical and analytical gradients are compatible for a function 'fn'

    x0:      The position at which to compute the gradients.

    fn:      A function of the form fn(x) which returns a single numpy array.

    fn_grad: The gradient of the original function with the form
             x_grad = fn_grad(y_bp, x0) where 'y_bp' is the backpropagated
             value and 'x0' is the original input to 'fn'. The output of
             the function is the gradient of x wrt to y.

    epsilon: A scalar or an array that can be broadcasted to the same
             shape as x0. This is used for computing the numerical Jacobian

    rtol:    The relative tolerance parameter used in numpy.allclose()

    atol:    The absolute tolerance parameter used in numpy.allclose()

    debug_outputs: Output variable which stores additional outputs useful for
                   debugging in a dictionary.
    """
    dtype = x0.dtype

    y = fn(x0)  # compute y to get the shape
    grad = fn_grad(np.zeros(y.shape, dtype=dtype), x0)

    grad_shape_correct = x0.shape == grad.shape

    if not grad_shape_correct:
        print(
            'The shape of the gradient [{0}] does not match the shape of "x0" [{1}].'
            .format(grad.shape, x0.shape))

    zero_grad = np.count_nonzero(grad) == 0

    if not zero_grad:
        print('The gradient is not zero for a zero backprop vector.')

    ana_J = compute_jacobian_analytical(x0, y.shape, fn_grad)
    ana_J2 = compute_jacobian_analytical(x0, y.shape, fn_grad,
                                         2 * np.ones(y.shape, dtype=x0.dtype))

    num_J = compute_jacobian_finite_differences(x0, fn, epsilon)

    does_scale = np.allclose(0.5 * ana_J2, ana_J, rtol, atol)
    isclose = np.allclose(ana_J, num_J, rtol, atol)
    ana_J_iszero = np.all(ana_J == 0)

    if ana_J_iszero and not np.allclose(num_J, np.zeros_like(num_J), rtol,
                                        atol):
        print(
            'The values of the analytical Jacobian are all zero but the values of the numerical Jacobian are not.'
        )
    elif not does_scale:
        print(
            'The gradients do not scale with respect to the backpropagated values.'
        )

    if not isclose:
        print('The gradients are not close to the numerical Jacobian.')

    debug_outputs.update(
        OrderedDict([
            ('isclose', isclose),
            ('does_scale', does_scale),
            ('ana_J_iszero', ana_J_iszero),
            ('grad_shape_correct', grad_shape_correct),
            ('zero_grad', zero_grad),
            ('ana_J', ana_J),
            ('num_J', num_J),
            ('absdiff', np.abs(ana_J - num_J)),
        ]))

    result = isclose and does_scale
    return result
