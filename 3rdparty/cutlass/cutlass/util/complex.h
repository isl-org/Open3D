/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuComplex.h>
#include "cutlass/cutlass.h"
#include <iosfwd>

namespace cutlass {
namespace platform {

//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for CUDA complex types
//

/// Returns the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
float const &real(cuFloatComplex const &z) { return z.x; }

/// Returns the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
float &real(cuFloatComplex &z) { return z.x; }

/// Returns the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
double const &real(cuDoubleComplex const &z) { return z.x; }

/// Returns the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
double &real(cuDoubleComplex &z) { return z.x; }

/// Returns the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
float const &imag(cuFloatComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
float &imag(cuFloatComplex &z) { return z.y; }

/// Returns the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
double const &imag(cuDoubleComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
CUTLASS_HOST_DEVICE
double &imag(cuDoubleComplex &z) { return z.y; }

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for representing and manipulating complex numbers with conversions from built-in CUDA
/// complex types.
template <typename T>
class complex {
 public:
  /// Type alias for scalar type
  typedef T value_type;

 private:
  //
  // Data members
  //

  /// Real part
  T _real;

  /// Imaginary part
  T _imag;

 public:
//
// Methods
//

/// Constructor
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  complex(T r = T(0), T i = T(0)) : _real(r), _imag(i) {}

/// Conversion from cuFloatComplex
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  complex(cuFloatComplex const &z) : _real(platform::real(z)), _imag(platform::imag(z)) {}

/// Conversion from cuDoubleComplex
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  complex(cuDoubleComplex const &z) : _real(platform::real(z)), _imag(platform::imag(z)) {}

/// Accesses the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  T const &real() const { return _real; }

/// Accesses the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  T &real() { return _real; }

/// Accesses the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  T const &imag() const { return _imag; }

/// Accesses the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  T &imag() { return _imag; }

/// Converts to cuFloatComplex
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  operator cuFloatComplex() const { return make_cuFloatComplex(real(), imag()); }

/// Converts to cuDoubleComplex
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  operator cuDoubleComplex() const { return make_cuDoubleComplex(real(), imag()); }
};

//
// Accessors for complex template
//

/// Returns the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T const &real(complex<T> const &z) {
  return z.real();
}

/// Returns the real part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T &real(complex<T> &z) {
  return z.real();
}

/// Returns the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T const &imag(complex<T> const &z) {
  return z.imag();
}

/// Returns the imaginary part of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T &imag(complex<T> &z) {
  return z.imag();
}

//
// Output operators
//

template <typename T>
std::ostream &operator<<(std::ostream &out, complex<T> const &z) {
  T _r = real(z);
  T _i = imag(z);
  return out << _r << "+i" << _i;
}

//
// Non-member operators defined for complex types
//

/// Equality operator
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE bool operator==(complex<T> const &lhs, complex<T> const &rhs) {
  return real(lhs) == (rhs) && imag(lhs) == imag(rhs);
}

/// Inequality operator
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE bool operator!=(complex<T> const &lhs, complex<T> const &rhs) {
  return !(lhs == rhs);
}

/// Addition
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator+(complex<T> const &lhs, complex<T> const &rhs) {
  return complex<T>(real(lhs) + real(rhs), imag(lhs) + imag(rhs));
}

/// Subtraction
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator-(complex<T> const &lhs, complex<T> const &rhs) {
  return complex<T>(real(lhs) - real(rhs), imag(lhs) - imag(rhs));
}

/// Multiplication
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator*(complex<T> const &lhs, complex<T> const &rhs) {
  return complex<T>(real(lhs) * real(rhs) - imag(lhs) * imag(rhs),
                    real(lhs) * imag(rhs) + imag(lhs) * real(rhs));
}

/// Scalar Multiplication
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator*(complex<T> const &lhs, T const &s) {
  return complex<T>(real(lhs) * s, imag(lhs) * s);
}

/// Scalar Multiplication
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator*(T const &s, complex<T> const &rhs) {
  return complex<T>(s * real(rhs), s * imag(rhs));
}

/// Division
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator/(complex<T> const &lhs, complex<T> const &rhs) {
  T d = (real(rhs) * (rhs) + imag(rhs) * imag(rhs));

  return complex<T>((real(lhs) * (rhs) + imag(lhs) * imag(rhs)) / d,
                    (imag(lhs) * (rhs)-real(lhs) * imag(rhs)) / d);
}

/// Scalar Division
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator/(complex<T> const &lhs, T const &s) {
  return complex<T>(real(lhs) / s, imag(lhs) / s);
}

/// Scalar divided by complex
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> operator/(T const &s, complex<T> const &rhs) {
  T d = (real(rhs) * (rhs) + imag(rhs) * imag(rhs));

  return complex<T>((s * (rhs)) / d, -(s * imag(rhs)) / d);
}

/// Addition
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> &operator+=(complex<T> &lhs, complex<T> const &rhs) {
  lhs = (lhs + rhs);
  return lhs;
}

/// Subtraction
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> &operator-=(complex<T> &lhs, complex<T> const &rhs) {
  lhs = (lhs - rhs);
  return lhs;
}

/// Multiplication
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> &operator*=(complex<T> &lhs, complex<T> const &rhs) {
  lhs = (lhs * rhs);
  return lhs;
}

/// Scalar multiplication
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> &operator*=(complex<T> &lhs, T s) {
  lhs = (lhs * s);
  return lhs;
}

/// Division
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> &operator/=(complex<T> &lhs, complex<T> const &rhs) {
  lhs = (lhs / rhs);
  return lhs;
}

//
// Non-member functions defined for complex numbers
//

/// Returns the magnitude of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T abs(complex<T> const &z) {
  return sqrt(norm(z));
}

/// Returns the magnitude of the complex number
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T arg(complex<T> const &z) {
  return atan2(imag(z), real(z));
}

/// Returns the squared magnitude
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE T norm(complex<T> const &z) {
  return real(z) * real(z) + imag(z) * imag(z);
}

/// Returns the complex conjugate
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> conj(complex<T> const &z) {
  return complex<T>(real(z), -imag(z));
}

/// Projects the complex number z onto the Riemann sphere
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> proj(complex<T> const &z) {
  T d = real(z) * real(z) + imag(z) * imag(z) + T(1);
  return complex<T>((T(2) * real(z)) / d, (T(2) * imag(z)) / d);
}

/// Returns a complex number with magnitude r and phase theta
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> polar(T const &r, T const &theta = T()) {
  return complex<T>(r * cos(theta), r * sin(theta));
}

/// Computes the complex exponential of z.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> exp(complex<T> const &z) {
  return complex<T>(real(z) * cos(imag(z)), real(z) * sin(imag(z)));
}

/// Computes the complex exponential of z.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> log(complex<T> const &z) {
  return complex<T>(log(abs(z)), arg(z));
}

/// Computes the complex exponential of z.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> log10(complex<T> const &z) {
  return log(z) / T(log(T(10)));
}

/// Computes the square root of complex number z
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> sqrt(complex<T> const &z) {
  return sqrt(T(2)) / T(2) *
         complex<T>(sqrt(sqrt(norm(z)) + real(z)),
                    (imag(z) < 0 ? T(-1) : T(1)) * sqrt(sqrt(norm(z)) - real(z)));
}

/// Computes the cosine of complex z.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> cos(complex<T> const &z) {
  return (exp(z) + exp(-z)) / T(2);
}

/// Computes the sin of complex z.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
template <typename T>
CUTLASS_HOST_DEVICE complex<T> sin(complex<T> const &z) {
  return (exp(-z) - exp(z)) * complex<T>(T(0), T(1) / T(2));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace platform
}  // namespace cutlass
