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
/*! \file
\brief Test environment for batched reduction
*/

#pragma once
#include "cutlass/matrix_traits.h"
#include "cutlass/util/platform.h"

#include "tools/util/host_matrix.h"
#include "tools/util/host_matrix_view.h"
#include "tools/util/host_tensor.h"
#include "tools/util/tensor_view_io.h"
#include "tools/util/type_traits.h"

#include <assert.h>

namespace test {

  inline cublasOperation_t convert(cutlass::MatrixLayout::Kind layout) {
    switch (layout) {
    case cutlass::MatrixLayout::kRowMajor:
      return CUBLAS_OP_T;
    case cutlass::MatrixLayout::kColumnMajor:
      return CUBLAS_OP_N;
    default:
      break;
    }
    return CUBLAS_OP_N;
  }

  inline cutlass::MatrixLayout::Kind convert(cublasOperation_t transform) {
    switch (transform) {
    case CUBLAS_OP_T:
      return cutlass::MatrixLayout::kRowMajor;
    case CUBLAS_OP_N:
      return cutlass::MatrixLayout::kColumnMajor;
    default:
      break;
    }
    return cutlass::MatrixLayout::kColumnMajor;
  }

  /// Testbed for evaluating batched reduction
  template <
    typename AType,
    typename CType,
    typename DType,
    typename ScalarAlpha,
    typename ScalarBeta,
    typename ScalarAccum,
    // input matrix depth size to be sumed
    int ReductionSize
  >
    struct BatchedReductionTestbed {
    //
    // Type definitions
    //
    /// Host tensor for operand C
    typedef cutlass::HostTensor<AType, 3> HostTensorA;

    /// Host tensor for operand C
    typedef cutlass::HostMatrix<CType> HostMatrixC;

    /// Host tensor for operand D
    typedef cutlass::HostMatrix<DType> HostMatrixD;

    /// Generates random elements
    template <typename T>
    struct RandomGenerator {
      RandomGenerator(int seed = -1, bool only_ones_ = false) : only_ones(only_ones_) { srand(seed); }

      T operator()() {
        if (only_ones) {
          return T(1);
        }
        else {
          int val = (rand() % 16) - 8;
          return T(val);
        }
      }

      bool only_ones;
    };

    template <typename T>
    struct RandomBitGenerator {
      RandomBitGenerator(int seed = -1) { srand(seed); }

      T operator()() {
        uint32_t val = 0;
        for (int i = 0; i < 32; i++) {
          val |= rand() % 2;
          val <<= 1;
        }
        return T(val);
      }
    };

    //
    // Data members
    //

    /// input/output number of rows
    int m;

    /// input/output number of columns
    int n;

    /// A matrix operand, always column major, no trans
    HostTensorA A;

    /// C matrix operand, always column major, no trans
    HostMatrixC C;

    /// D matrix operand, always column major, no trans
    HostMatrixD D;

    /// Reference 
    cutlass::HostTensor<AType, 3> ref_A;

    ///
    cutlass::HostMatrix<CType> ref_C;

    /// Reference result computed on the host
    cutlass::HostMatrix<DType> ref_D;

    /// lda
    int lda;

    /// ldc
    int ldc;

    /// ldd
    int ldd;

    /// Linear scalaring factor
    ScalarAlpha alpha;

    /// Linear scaling factor
    ScalarBeta beta;

    /// stride between two element that will be sumed
    long long int reduction_stride;

    //
    // Static helpers
    //

    /// Helper to resize a matrix with a given size and layout
    template <typename T>
    static void resize(cutlass::HostMatrix<T>& tensor,
      int rows,
      int columns,
      cublasOperation_t layout,
      int ldm = 0,
      bool device_backed = true) {

      tensor.resize(cutlass::make_Coord(rows, columns), convert(layout), ldm, device_backed);
    }

    template <typename T>
    static void resize(cutlass::HostTensor<T, 3>& tensor,
      int rows,
      int columns,
      int batches,
      cublasOperation_t layout,
      int ldm,
      long long int batch_stride,
      bool device_backed = true) {
      assert(CUBLAS_OP_N == layout);
      //tensor.resize(cutlass::make_Coord(rows, columns), convert(layout), ldm, device_backed);
      tensor.reset(cutlass::make_Coord(static_cast<int>(batch_stride), ldm, 1), /*stride, slowest moving dim on the left*/
        cutlass::make_Coord(batches, columns, rows), /*size, slowest moving dim on the left*/
        device_backed);
    }


    //
    // Methods
    //

    /// Ctor.
    BatchedReductionTestbed(int m_,
      int n_,
      int lda_,
      int ldc_,
      int ldd_,
      typename cutlass::TypeTraits<ScalarAlpha>::host_type alpha_ =
      typename cutlass::TypeTraits<ScalarAlpha>::host_type(2),
      typename cutlass::TypeTraits<ScalarAlpha>::host_type beta_ =
      typename cutlass::TypeTraits<ScalarAlpha>::host_type(3))
      : m(m_),
      n(n_),
      lda(lda_),
      ldc(ldc_),
      ldd(ldd_),
      alpha(alpha_),
      beta(beta_),
      reduction_stride(ldc_ * n_) {
      /// column major, batch along rows
      resize(A, m_, n_, ReductionSize, CUBLAS_OP_N, lda_, reduction_stride, true);
      resize(C, m_, n_, CUBLAS_OP_N, ldc_, true);
      resize(D, m_, n_, CUBLAS_OP_N, ldd_, true);
      resize(ref_A, m_, n_, ReductionSize, CUBLAS_OP_N, lda_, reduction_stride, false);
      resize(ref_C, m_, n_, CUBLAS_OP_N, ldc_, false);
      resize(ref_D, m_, n_, CUBLAS_OP_N, ldd_, false);
    }

    /// Dtor
    ~BatchedReductionTestbed() { }

    /// Getters
    /// Returns a pointer to the C operand
    typename HostTensorA::DeviceType* ptr_A() const { return A.device_data(); }
    /// Returns a pointer to the C operand
    typename HostMatrixC::DeviceType* ptr_C() const { return C.device_data(); }
    /// Returns a pointer to the D operand
    typename HostMatrixD::DeviceType* ptr_D() const { return D.device_data(); }

    ///
    int M() const { return m; }
    ///
    int N() const { return n; }
    ///
    int get_lda() const { return lda; }
    ///
    int get_ldc() const { return ldc; }
    ///
    int get_ldd() const { return ldd; }
    ///
    ScalarAlpha get_alpha() const { return alpha; }
    ///
    ScalarBeta get_beta() const { return beta; }
    ///
    long long int get_reduction_stride() const { return reduction_stride; }

    /// Initializes data, randomly
    void initialize(int seed = -1) {
      A.fill_random(RandomGenerator<AType>(seed + 7));
      //A.fill(3);
      C.fill_random(RandomGenerator<CType>(seed));
      //C.fill(1);
      D.fill_random(RandomGenerator<DType>(seed + 11));
      //D.fill(2);
    }

    /// compute_host
    void compute_host() {
      ref_A.fill(A);
      ref_C.fill(C);
      ref_D.fill(D);
      /// D = alpha * reduction(A) + beta * C

      for (int m_idx = 0; m_idx < m; m_idx++) {
        for (int n_idx = 0; n_idx < n; n_idx++) {
          ScalarAccum accum = static_cast<ScalarAccum>(0.0);
          for (int r_idx = 0; r_idx < static_cast<int>(ReductionSize); r_idx++) {
            accum += static_cast<ScalarAccum>(ref_A.at(cutlass::make_Coord(r_idx, n_idx, m_idx)));
          }
          ref_D.at(cutlass::make_Coord(m_idx, n_idx)) = static_cast<DType>(
                                                        alpha * static_cast<ScalarAlpha>(accum) +
                                                        beta * static_cast<ScalarBeta>(ref_C.at(cutlass::make_Coord(m_idx, n_idx)))
                                                        );
        }
      }
    }

    /// Verifies the contents of C equal the host-side reference
    bool verify_with_host() {
      compute_host();
      D.sync_host();
      bool passed = D.bit_equals(ref_D);
      return passed;
    }
  };

} //namespace test
