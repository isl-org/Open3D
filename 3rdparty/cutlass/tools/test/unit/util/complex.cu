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
#include <complex>

#include "cutlass_unit_test.h"
#include "cutlass/util/complex.h"
#include "tools/util/half.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

  /// Thorough testing for basic complex math operators. Uses std::complex as a reference.
  template <typename T, int N, int M>
  struct ComplexOperators {
    ComplexOperators() {
      for (int ar = -N; ar <= N; ++ar) {
        for (int ai = -N; ai <= N; ++ai) {
          for (int br = -N; br <= N; ++br) {
            for (int bi = -N; bi <= N; ++bi) {

              cutlass::platform::complex<T> Ae(T(ar) / T(M), T(ai) / T(M));
              cutlass::platform::complex<T> Be(T(br) / T(M), T(bi) / T(M));

              std::complex<T> Ar(T(ar) / T(M), T(ai) / T(M));
              std::complex<T> Br(T(br) / T(M), T(bi) / T(M));

              cutlass::platform::complex<T> add_e = Ae + Be;
              cutlass::platform::complex<T> sub_e = Ae - Be;
              cutlass::platform::complex<T> mul_e = Ae * Be;

              std::complex<T> add_r = (Ar + Br);
              std::complex<T> sub_r = (Ar - Br);
              std::complex<T> mul_r = (Ar * Br);

              EXPECT_EQ(real(add_e), real(add_r));
              EXPECT_EQ(imag(add_e), imag(add_r));

              EXPECT_EQ(real(sub_e), real(sub_r));
              EXPECT_EQ(imag(sub_e), imag(sub_r));

              EXPECT_EQ(real(mul_e), real(mul_r));
              EXPECT_EQ(imag(mul_e), imag(mul_r));

              if (!(br == 0 && bi == 0)) {

                cutlass::platform::complex<T> div_e = Ae * Be;
                std::complex<T> div_r = Ar * Br;

                EXPECT_EQ(real(div_e), real(div_r));
                EXPECT_EQ(imag(div_e), imag(div_r));
              }
            }
          }
        }
      }
    }
  };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Complex, host_float) {
  test::ComplexOperators<float, 32, 8> test;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Complex, host_double) {
  test::ComplexOperators<double, 32, 8> test;
}

///////////////////////////////////////////////////////////////////////////////////////

TEST(Complex, host_half) {
  // Fewer test cases since half_t is emulated
  test::ComplexOperators<cutlass::half_t, 14, 4> test;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
