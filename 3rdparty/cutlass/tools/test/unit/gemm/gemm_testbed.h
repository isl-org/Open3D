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
    \brief Test environment for GEMM
*/

#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>

#include <cublas_v2.h>

#include "cutlass/matrix_traits.h"
#include "cutlass/util/platform.h"
#include "cutlass/gemm/gemm_coord.h"

#include "tools/util/host_matrix.h"
#include "tools/util/host_matrix_view.h"
#include "tools/util/tensor_view_io.h"
#include "tools/util/type_traits.h"

#include "tools/util/reference/host/gemm.h"
#include "tools/util/reference/device/gemm.h"
#include "tools/util/reference/host/tensor_elementwise.h"

//////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

template <cutlass::GemmOperand::Kind kOperand_,
          cutlass::MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename WmmaShape_>
struct WmmaMatrix;

}  // namespace cutlass

//////////////////////////////////////////////////////////////////////////////////////////

namespace test {

//////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct GemmTestbedTraits : public cutlass::TypeTraits<T> {};

template <cutlass::GemmOperand::Kind kOperand_,
          cutlass::MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename WmmaShape_>
struct GemmTestbedTraits<cutlass::WmmaMatrix<kOperand_, kLayout_, Scalar_, WmmaShape_> > {
  static cudaDataType_t const cublas_type = cutlass::TypeTraits<Scalar_>::cublas_type;
  typedef typename cutlass::TypeTraits<Scalar_>::host_type host_type;
  typedef typename cutlass::TypeTraits<Scalar_>::device_type device_type;
  static inline double remove_negative_zero(double x) { return x == -0.0 ? 0.0 : x; }
  static inline double to_print(double x) { return x; }
};

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

//////////////////////////////////////////////////////////////////////////////////////////

/// Testbed for evaluating real-valued GEMMs
template <typename AType, typename BType, typename CType, typename Accumulator, typename Scalar>
struct GemmTestbed {
  //
  // Type definitions
  //

  /// Host tensor for operand A
  typedef cutlass::HostMatrix<AType> HostMatrixA;

  /// Host tensor for operand B
  typedef cutlass::HostMatrix<BType> HostMatrixB;

  /// Host tensor for operand C
  typedef cutlass::HostMatrix<CType> HostMatrixC;

  /// Functor to print errors
  struct PrintErrors {
    /// Equivalently sized integer type
    typedef typename GemmTestbedTraits<CType>::integer_type integer_t;

    /// Output stream to write to
    std::ostream& out;

    /// Reference tensor view
    HostMatrixC const& reference;

    /// Computed tensor view
    HostMatrixC const& experimental;

    /// Errors greater than or this amount result in printing
    integer_t ulps_threshold;

    ///
    PrintErrors(std::ostream& _out,
                HostMatrixC const& _reference,
                HostMatrixC const& _experimental,
                integer_t _ulps_threshold = 1)
        : out(_out),
          reference(_reference),
          experimental(_experimental),
          ulps_threshold(_ulps_threshold) {}

    /// Compares one element
    void operator()(CType const& element, typename HostMatrixC::TensorCoord coord) {
      CType exp = experimental.at(coord);
      CType ref = reference.at(coord);

      int64_t int_exp = 0;
      int64_t int_ref = 0;

      *reinterpret_cast<CType*>(&int_exp) = exp;
      *reinterpret_cast<CType*>(&int_ref) = ref;

      integer_t ulps = integer_t(int_exp - int_ref);

      if (std::abs(ulps) >= ulps_threshold) {
        // width in hexadecimal digits of value
        int const width = sizeof(integer_t) * 2;

        double relative = double(exp) - double(ref);
        if (ref != CType(0)) {
          relative /= double(ref);
        }

        out << "[" << coord << "] expected: " << GemmTestbedTraits<CType>::to_print(ref) << " (0x"
            << std::hex << std::setw(width) << std::setfill('0') << integer_t(int_ref) << std::dec
            << ")"
            << ",  got: " << GemmTestbedTraits<CType>::to_print(exp) << " (0x" << std::hex
            << std::setw(width) << std::setfill('0') << integer_t(int_exp) << std::dec << ")"
            << "  relative error: " << relative << ", ulps: " << ulps << "\n";
      }
    }
  };

  /// Generates random elements
  template <typename T>
  struct RandomGenerator {
    RandomGenerator(int seed = -1, bool only_ones_ = false) : only_ones(only_ones_) { srand(seed); }

    T operator()() {
      if (only_ones) {
        return T(1);
      } else {
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

  /// Status
  cublasStatus_t status;

  /// cuBLAS handle
  cublasHandle_t handle;

  /// cuBLAS GEMM algorithm selector
  cublasGemmAlgo_t algorithm;

  /// Problem size as a GemmCoord
  cutlass::gemm::GemmCoord problem_size;

  /// A matrix operand
  HostMatrixA A;

  /// Layout of A matrix
  cublasOperation_t layout_A;

  /// B matrix operand
  HostMatrixB B;

  /// Layout of B matrix
  cublasOperation_t layout_B;

  /// C matrix operand
  HostMatrixC C_initial;

  /// Reference result computed on the host
  HostMatrixC ref_host;

  /// Reference result computed on the device
  HostMatrixC ref_device;

  /// Reference result computed with cublas
  HostMatrixC ref_cublas;

  /// Computed result
  HostMatrixC computed;

  /// Linear scalaring factor
  Scalar alpha;

  /// Linear scaling factor
  Scalar beta;

  /// batch count
  int batch_count;

  /// partitionK count
  int partitionK_count;

  /// each partition should be mulitples of partitionK_multiple
  int partitionK_multiple;

  /// distance between A[i] and A[i+1] for strided batched gemm
  long long int batch_stride_A;

  /// distance between B[i] and B[i+1] for strided batched gemm
  long long int batch_stride_B;

  /// distance between C[i] and C[i+1] for strided batched gemm
  long long int batch_stride_C;

  //
  // Static helpers
  //

  /// Helper to resize a matrix with a given size and layout
  template <typename T>
  static void resize(cutlass::HostMatrix<T>& tensor,
                     int rows,
                     int columns,
                     cublasOperation_t layout,
                     int ldm = 0) {

    tensor.resize(cutlass::make_Coord(rows, columns), convert(layout), ldm);
  }

  //
  // Methods
  //

  /// Constructs a workspace for verifying GEMM, assumes
  /// dense packing.
  GemmTestbed(int M_,
              int N_,
              int K_,
              cublasOperation_t layout_a,
              cublasOperation_t layout_b,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0),
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              cublasOperation_t layout_c = CUBLAS_OP_N)
      : problem_size(K_, N_, M_, 1),
        layout_A(layout_a),
        layout_B(layout_b),
        alpha(alpha_),
        beta(beta_),
        algorithm(algorithm_),
        batch_count(1),
        partitionK_count(1),
        partitionK_multiple(1),
        batch_stride_A(static_cast<long long int>(0)),
        batch_stride_B(static_cast<long long int>(0)),
        batch_stride_C(static_cast<long long int>(0)) {

    #if CUTLASS_ENABLE_CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw cutlass::cuda_exception("Failed to create CUBLAS handle");
    }
    #else
    status = CUBLAS_STATUS_NOT_INITIALIZED;
    #endif

    resize(A, M_, K_, layout_a);
    resize(B, K_, N_, layout_b);
    resize(C_initial, M_, N_, layout_c);
    resize(ref_host, M_, N_, layout_c);
    resize(ref_device, M_, N_, layout_c);
    resize(ref_cublas, M_, N_, layout_c);
    resize(computed, M_, N_, layout_c);
  }

  /// Constructs a workspace for verifying GEMM, assumes
  /// dense packing.
  GemmTestbed(cublasHandle_t handle_,
              int M_,
              int N_,
              int K_,
              cublasOperation_t layout_a,
              cublasOperation_t layout_b,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0),
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              cublasOperation_t layout_c = CUBLAS_OP_N)
      : status(CUBLAS_STATUS_SUCCESS),
        handle(handle_),
        problem_size(K_, N_, M_, 1),
        layout_A(layout_a),
        layout_B(layout_b),
        alpha(alpha_),
        beta(beta_),
        algorithm(algorithm_),
        batch_count(1),
        partitionK_count(1),
        partitionK_multiple(1),
        batch_stride_A(static_cast<long long int>(0)),
        batch_stride_B(static_cast<long long int>(0)),
        batch_stride_C(static_cast<long long int>(0)) {

    resize(A, M_, K_ * batch_count, layout_a);
    resize(B, K_ * batch_count, N_, layout_b);
    resize(C_initial, M_, N_ * batch_count, layout_c);
    resize(ref_host, M_, N_ * batch_count, layout_c);
    resize(ref_device, M_, N_ * batch_count, layout_c);
    resize(ref_cublas, M_, N_ * batch_count, layout_c);
    resize(computed, M_, N_ * batch_count, layout_c);
  }

  /// Constructs a workspace for verifying GEMM with arbitrary strides
  GemmTestbed(int M_,
              int N_,
              int K_,
              int lda,
              int ldb,
              int ldc,
              cublasOperation_t layout_a,
              cublasOperation_t layout_b,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0),
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              cublasOperation_t layout_c = CUBLAS_OP_N)
      : problem_size(K_, N_, M_, 1),
        layout_A(layout_a),
        layout_B(layout_b),
        alpha(alpha_),
        beta(beta_),
        algorithm(algorithm_),
        batch_count(1),
        partitionK_count(1),
        partitionK_multiple(1),
        batch_stride_A(static_cast<long long int>(0)),
        batch_stride_B(static_cast<long long int>(0)),
        batch_stride_C(static_cast<long long int>(0)) {
    
    #if CUTLASS_ENABLE_CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw cutlass::cuda_exception("Failed to create CUBLAS handle");
    }
    #else
    status = CUBLAS_STATUS_NOT_INITIALIZED;
    #endif

    resize(A, M_, K_, layout_a, lda);
    resize(B, K_, N_, layout_b, ldb);
    resize(C_initial, M_, N_, layout_c, ldc);
    resize(ref_host, M_, N_, layout_c, ldc);
    resize(ref_device, M_, N_, layout_c, ldc);
    resize(ref_cublas, M_, N_, layout_c, ldc);
    resize(computed, M_, N_, layout_c, ldc);
  }

  /// Constructs a workspace for verifying GEMM with arbitrary strides
  GemmTestbed(cublasHandle_t handle_,
              int M_,
              int N_,
              int K_,
              int ldc,
              cublasOperation_t layout_a,
              int lda,
              cublasOperation_t layout_b,
              int ldb,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0),
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              cublasOperation_t layout_c = CUBLAS_OP_N)
      : status(CUBLAS_STATUS_SUCCESS),
        handle(handle_),
        problem_size(K_, N_, M_, 1),
        alpha(alpha_),
        beta(beta_),
        algorithm(algorithm_),
        batch_count(1),
        partitionK_count(1),
        partitionK_multiple(1),
        batch_stride_A(static_cast<long long int>(0)),
        batch_stride_B(static_cast<long long int>(0)),
        batch_stride_C(static_cast<long long int>(0)) {

    resize(A, M_, K_ * batch_count, layout_a);
    resize(B, K_ * batch_count, N_, layout_b);
    resize(C_initial, M_, N_ * batch_count, layout_c);
    resize(ref_host, M_, N_ * batch_count, layout_c);
    resize(ref_device, M_, N_ * batch_count, layout_c);
    resize(ref_cublas, M_, N_ * batch_count, layout_c);
    resize(computed, M_, N_ * batch_count, layout_c);
  }

  /// Constructs a workspace for verifying strided batched GEMM, assumes
  /// dense packing.
  /// batches are "concated" along K for matrix A and matrix B, and along N for matrix C
  /// a full implementation of strided batched GEMM should handle other corner cases
  GemmTestbed(int M_,
              int N_,
              int K_,
              int batch_count_,
              cublasOperation_t layout_a,
              cublasOperation_t layout_b,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0),
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              cublasOperation_t layout_c = CUBLAS_OP_N)
      : problem_size(K_, N_, M_, batch_count_),
        layout_A(layout_a),
        layout_B(layout_b),
        alpha(alpha_),
        beta(beta_),
        algorithm(algorithm_),
        batch_count(batch_count_),
        partitionK_count(1),
        partitionK_multiple(1) {

    #if CUTLASS_ENABLE_CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw cutlass::cuda_exception("Failed to create CUBLAS handle");
    }
    #else
    status = CUBLAS_STATUS_NOT_INITIALIZED;
    #endif

    resize(A, M_, K_ * batch_count, layout_a);
    resize(B, K_ * batch_count, N_, layout_b);
    resize(C_initial, M_, N_ * batch_count, layout_c);
    resize(ref_host, M_, N_ * batch_count, layout_c);
    resize(ref_device, M_, N_ * batch_count, layout_c);
    resize(ref_cublas, M_, N_ * batch_count, layout_c);
    resize(computed, M_, N_ * batch_count, layout_c);

    batch_stride_A = (layout_a == CUBLAS_OP_N) ? M_ * K_ : K_;
    batch_stride_B = (layout_b == CUBLAS_OP_N) ? K_ : K_ * N_;
    batch_stride_C = M_ * N_;
  }

  /// Constructs a workspace for verifying partitionedK GEMM, assumes
  /// dense packing.
  /// in partitionedK GEMM, the K is partitioned by partitionK_size
  /// each partition is of the same size, except for the last partition
  /// each partition, except for the last one, is of size K / partitionK_count
  /// if K is not divisible by partitionK_size, the last partitionK = K % partitionK_count + K / partitionK_count
  GemmTestbed(int M_,
              int N_,
              std::pair<int, int> K_pair_, /*(k, partitionK_count)*/
              int partitionK_multiple_, /*each partition should be mulitiple of partitionK_multiple*/
              cublasOperation_t layout_a,
              cublasOperation_t layout_b,
              Scalar alpha_ = Scalar(1),
              Scalar beta_ = Scalar(0),
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              cublasOperation_t layout_c = CUBLAS_OP_N)
    : problem_size(K_pair_.first, N_, M_, 1),
    layout_A(layout_a),
    layout_B(layout_b),
    alpha(alpha_),
    beta(beta_),
    algorithm(algorithm_),
    batch_count(1),
    partitionK_count(K_pair_.second),
    partitionK_multiple(partitionK_multiple_) {

    #if CUTLASS_ENABLE_CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw cutlass::cuda_exception("Failed to create CUBLAS handle");
    }
    #else
    status = CUBLAS_STATUS_NOT_INITIALIZED;
    #endif

    resize(A, M_, K_pair_.first, layout_a);
    resize(B, K_pair_.first, N_, layout_b);
    resize(C_initial, M_, N_ * partitionK_count, layout_c);
    resize(ref_host, M_, N_ * partitionK_count, layout_c);
    resize(ref_device, M_, N_ * partitionK_count, layout_c);
    resize(ref_cublas, M_, N_ * partitionK_count, layout_c);
    resize(computed, M_, N_ * partitionK_count, layout_c);

    // we can use a combination of batched stried gemm and regular gemm
    // to simulation partitionedK, which is what we will do for reference code
    int partitionK_size = K() / partitionK_count;
    partitionK_size = partitionK_size - (partitionK_size % partitionK_multiple);
    batch_stride_A = (layout_a == CUBLAS_OP_N) ? M_ * partitionK_size : partitionK_size;
    batch_stride_B = (layout_b == CUBLAS_OP_N) ? partitionK_size : partitionK_size * N_;
    batch_stride_C = M_ * N_;
  }

  /// Destructs the GEMM testbed
  ~GemmTestbed() {
    #if CUTLASS_ENABLE_CUBLAS
    if (status != CUBLAS_STATUS_NOT_INITIALIZED) {
      status = cublasDestroy(handle);
    }
    #endif
  }

  /// Returns true if the last CUBLAS call returned successfully
  bool good() const { return status == CUBLAS_STATUS_SUCCESS; }

  /// Returns a pointer to the A operand
  typename HostMatrixA::DeviceType* ptr_A() const { return A.device_data(); }

  /// Stride of A matrix
  int lda() const { return A.leading_dim(); }

  /// Returns a pointer to the B operand
  typename HostMatrixB::DeviceType* ptr_B() const { return B.device_data(); }

  /// Stride of B matrix
  int ldb() const { return B.leading_dim(); }

  /// Returns a pointer to the initial state of the result tensor in device memory
  typename HostMatrixC::DeviceType* ptr_C_initial() const { return C_initial.device_data(); }

  /// Returns a pointer to the result tensor in device memory
  typename HostMatrixC::DeviceType* ptr_computed() const { return computed.device_data(); }

  /// Returns a pointer to the result tensor in device memory
  typename HostMatrixC::DeviceType* ptr_cublas() const { return ref_cublas.device_data(); }

  /// Stride of C matrix
  int ldc() const {
    //return std::max(C_initial.stride(HostTensorC::Dim_H), C_initial.stride(HostTensorC::Dim_W));
    return C_initial.leading_dim();
  }

  /// Returns the number of flops implied by the computation (1 multiply-accumulate = 2 flops)
  uint64_t flops() const {
    if (partitionK_count == 1) {
      return uint64_t(batch_count) * uint64_t(M()) * uint64_t(N()) * uint64_t(K()) * 2ULL;
    }
    else {
      int partitionK_size = K() / partitionK_count;
      return (uint64_t(partitionK_count - 1) * uint64_t(batch_count) * uint64_t(M()) * uint64_t(N()) * uint64_t(partitionK_size) * 2ULL)
        + (uint64_t(batch_count) * uint64_t(M()) * uint64_t(N()) * uint64_t(K() - partitionK_size * (partitionK_count - 1)) * 2ULL);
    }
  }

  /// Computes the speed of the computation in GFLOPs/s
  double GFLOPs_per_sec(double runtime_ms) const { return double(flops()) / runtime_ms / 1.0e6; }

  /// Matrix layout of A
  cublasOperation_t layout_a() const { return layout_A; }

  /// Matrix layout of B
  cublasOperation_t layout_b() const { return layout_B; }

  /// Number of rows of problem, per batch; assumptions made here that we concat C by adding columns
  int M() const {
    return problem_size.m();
  }

  /// Number of columns of problem, per batch; assumptions made here that we concat C by adding
  /// columns
  int N() const {
    return problem_size.n();
  }

  /// Number of columns of problem, per batch; assumptions made here that we concat A by adding
  /// columns
  int K() const {
    return problem_size.k();
  }

  /// Number of batches
  int get_batch_count() const {
    return problem_size.batch();
  }

  ///
  long long int get_batch_stride_A() const { return batch_stride_A; }

  ///
  long long int get_batch_stride_B() const { return batch_stride_B; }

  ///
  long long int get_batch_stride_C() const { return batch_stride_C; }

  ///

  /// Initializes data, randomly
  void initialize(int seed = -1) {

    // Initialize the source matrix with a uniform distribution
    cutlass::Distribution dist;
    dist.set_uniform(-8, 8);

    cutlass::reference::host::TensorInitialize(A.host_view(), seed, dist);
    cutlass::reference::host::TensorInitialize(B.host_view(), seed + 11, dist);
    cutlass::reference::host::TensorInitialize(C_initial.host_view(), seed + 13, dist);

    A.sync_device();
    B.sync_device();
    C_initial.sync_device();

    computed.fill(0);
  }

  /// Initializes binary data
  void initialize_binary(int seed = -1) {
    //A.fill_random(RandomBitGenerator<AType>(seed));
    //B.fill_random(RandomBitGenerator<BType>(seed + 11));
    //C_initial.fill_random(RandomGenerator<CType>(seed + 13));
    A.fill_sequential();
    B.fill_sequential();
    C_initial.fill(0);
  }

  /// Initializes integer data (sequential for now)
  void initialize_integer(int seed =-1) {
    A.fill_sequential();
    B.fill_sequential();
    C_initial.fill(0);
  }

  /// Computes the matrix product on the host
  void compute_host() {
    ref_host.fill(C_initial);
    cutlass::reference::host::Gemm(problem_size, alpha, A.host_ref(), B.host_ref(), beta, ref_host.host_ref(), Accumulator(0));
  }

  /// Compute the matrix product using the device-side reference
  void compute_device_reference() {
    ref_device.fill(C_initial);
    cutlass::reference::device::Gemm(
      problem_size,
      cutlass::TypeTraits<Scalar>::to_device(alpha),
      A.device_ref(),
      B.device_ref(),
      cutlass::TypeTraits<Scalar>::to_device(beta),
      ref_device.device_ref(),
      cutlass::TypeTraits<Accumulator>::to_device(0)
    );
  }

  /// Excutes an equivalent GEMM using cuBLAS
  bool execute_cublas() {
    #if CUTLASS_ENABLE_CUBLAS
    if (partitionK_count == 1) {
        if (batch_count == 1) {
          status = cublasGemmEx(handle,
            layout_a(),
            layout_b(),
            M(),
            N(),
            K(),
            &alpha,
            ptr_A(),
            cutlass::TypeTraits<AType>::cublas_type,
            lda(),
            ptr_B(),
            cutlass::TypeTraits<BType>::cublas_type,
            ldb(),
            &beta,
            ref_cublas.device_data(),
            cutlass::TypeTraits<CType>::cublas_type,
            ldc(),
            cutlass::TypeTraits<Accumulator>::cublas_type,
            algorithm);

          return status == CUBLAS_STATUS_SUCCESS;
        }
        else {
          // call strided batched gemm
          status = cublasGemmStridedBatchedTemplate(handle,
            layout_a(),
            layout_b(),
            M(),
            N(),
            K(),
            &alpha,
            ptr_A(),
            lda(),
            batch_stride_A,
            ptr_B(),
            ldb(),
            batch_stride_B,
            &beta,
            ref_cublas.device_data(),
            ldc(),
            batch_stride_C,
            batch_count);

          return status == CUBLAS_STATUS_SUCCESS;
        }
    }
    else {
      assert(batch_count == 1);
      //the last batch is of a different K
      //first call strided batched gemm

      int partitionK_size = K() / partitionK_count;
      partitionK_size = partitionK_size - (partitionK_size % partitionK_multiple);
      //int lastK_size = (K() % partitionK_size) + partitionK_size;
      int lastK_size = K() - partitionK_size * (partitionK_count - 1);
      status = cublasGemmStridedBatchedTemplate(handle,
        layout_a(),
        layout_b(),
        M(),
        N(),
        partitionK_size,
        &alpha,
        ptr_A(),
        lda(),
        batch_stride_A,
        ptr_B(),
        ldb(),
        batch_stride_B,
        &beta,
        ref_cublas.device_data(),
        ldc(),
        batch_stride_C,
        partitionK_count - 1);
      //then call gemm for the last batch
      status = cublasGemmEx(handle,
        layout_a(),
        layout_b(),
        M(),
        N(),
        lastK_size,
        &alpha,
        ptr_A() + (partitionK_count - 1) * batch_stride_A,
        cutlass::TypeTraits<AType>::cublas_type,
        lda(),
        ptr_B() + (partitionK_count - 1) * batch_stride_B,
        cutlass::TypeTraits<BType>::cublas_type,
        ldb(),
        &beta,
        ref_cublas.device_data() + (partitionK_count - 1) * batch_stride_C,
        cutlass::TypeTraits<CType>::cublas_type,
        ldc(),
        cutlass::TypeTraits<Accumulator>::cublas_type,
        algorithm);
      return status == CUBLAS_STATUS_SUCCESS;

    }
    #else
    return false;
    #endif
  }

  /// Helper function to use cublasGemmStridedBatched
  cublasStatus_t cublasGemmStridedBatchedTemplate(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int M,
                                                  int N,
                                                  int K,
                                                  const Scalar *alpha,
                                                  const typename HostMatrixA::DeviceType *ptr_A,
                                                  int lda,
                                                  long long int stride_A,
                                                  const typename HostMatrixB::DeviceType *ptr_B,
                                                  int ldb,
                                                  long long int stride_B,
                                                  const Scalar *beta,
                                                  typename HostMatrixC::DeviceType *ptr_C,
                                                  int ldc,
                                                  long long int stride_C,
                                                  int batchCount) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }


  /// Computes the matrix product using cuBLAS
  void compute_cublas() {
    ref_cublas.fill(C_initial);

    if (!execute_cublas()) {
      throw std::runtime_error("compute_cublas() failed");
    }
  }

  //
  // Compute the GEMM yourself
  //

  /// Names a probelm based on data type and problem size
  std::string workspace_name() const {
    std::stringstream ss;
    ss << "gemm_" << (layout_a() == CUBLAS_OP_N ? "n" : "t")
       << (layout_b() == CUBLAS_OP_N ? "n" : "t") << "_" << typeid(AType).name() << "_"
       << typeid(BType).name() << "_" << typeid(CType).name() << "_" << typeid(Accumulator).name()
       << "_" << typeid(Scalar).name() << "_" << M() << "x" << N() << "x" << K();
    //make sure there is no space in the ss
    std::string thisString = ss.str();
    std::replace(thisString.begin(), thisString.end(), ' ', '_');
    std::replace(thisString.begin(), thisString.end(), ':', '_');
    return thisString;
  }

  /// Writes the workspace to an ostream
  std::ostream& write(std::ostream& out) const {
    out << "A = " << A << "\nB = " << B << "\nC_initial = " << C_initial
        << "\nref_host = " << ref_host << "\nref_cublas = " << ref_cublas
        << "\ncomputed = " << computed << std::endl;

    return out;
  }

  /// Outputs each mismatching element
  std::ostream& write_errors(std::ostream& out,
                             HostMatrixC const& experimental,
                             HostMatrixC const& ref) const {
    PrintErrors printer(out, ref, experimental);

    computed.visit(printer);

    return out;
  }

  /// Sync's all input tensors to device
  void sync_device() {
    A.sync_device();
    B.sync_device();
    C_initial.sync_device();

    ref_host.fill(C_initial);
    ref_cublas.fill(C_initial);
    computed.fill(C_initial);

    ref_cublas.sync_device();
    computed.sync_device();
  }

  /// Sync's all output tensors to host
  void sync_host() {
    computed.sync_host();
    ref_cublas.sync_host();
  }

  /// Saves the workspace to files
  void save_workspace(HostMatrixC const& experimental,
                      HostMatrixC const& ref) {
    std::string name = workspace_name();

    std::string results_name = name + "_results.txt";
    std::string errors_name = name + "_errors.txt";

    std::ofstream results(results_name.c_str());
    std::ofstream errors(errors_name.c_str());

    write(results);
    write_errors(errors, experimental, ref);
  }

  /// Verifies the contents of C equal the host-side reference
  bool verify_with_host(bool save_on_error = true, bool always_print = false) {
    compute_host();
    computed.sync_host();

    bool passed = computed.bit_equals(ref_host);

    if ((!passed && save_on_error) || always_print) {
      save_workspace(computed, ref_host);
    }
    return passed;
  }

  /// Verifies the contents of computed equal cuBLAS
  bool verify_with_cublas(bool save_on_error = true, bool always_print = false) {

    bool passed = false;

    #if CUTLASS_ENABLE_CUBLAS
    compute_cublas();

    ref_cublas.sync_host();
    computed.sync_host();

    passed = computed.bit_equals(ref_cublas);

    if ((!passed && save_on_error) || always_print) {
      save_workspace(computed, ref_cublas);
    }

    #endif
    return passed;
  }

  /// Verifies the host computation with cuBLAS
  bool verify_host_with_cublas(bool save_on_error = true, bool always_print = false) {

    bool passed = false;

    #if CUTLASS_ENABLE_CUBLAS

    compute_host();
    compute_cublas();
    ref_cublas.sync_host();

    passed = ref_host.bit_equals(ref_cublas);

    if ((!passed && save_on_error) || always_print) {
      save_workspace(ref_host, ref_cublas);
    }

    #endif

    return passed;
  }

  /// Verifies the reference implementation with cuBLAS
  bool verify_reference_with_cublas(bool save_on_error = true, bool always_print = false) {

    bool passed = false;

    #if CUTLASS_ENABLE_CUBLAS
    compute_device_reference();
    ref_device.sync_host();

    compute_cublas();
    ref_cublas.sync_host();

    passed = ref_device.bit_equals(ref_cublas);

    if ((!passed && save_on_error) || always_print) {
      save_workspace(ref_device, ref_cublas);
    }
    #endif

    return passed;
  }

  /// Verifies with host-side and device-side computations
  bool verify_with_all() {
    bool passed = true;

    computed.sync_host();

    // verify on host
    passed = (passed && verify_with_host());

    #if CUTLASS_ENABLE_CUBLAS
    // verify with cublas
    passed = (passed && verify_with_cublas());
    #endif

    return passed;
  }

  bool has_cublas_support() const {
    #if CUTLASS_ENABLE_CUBLAS
    return cutlass::platform::is_same<Accumulator, Scalar>::value;
    #else
    return false;
    #endif
  }
};

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//
//specialization for cublasGemmStridedBatchedTemplate
template<> inline cublasStatus_t GemmTestbed<float, float, float, float, float>::cublasGemmStridedBatchedTemplate(cublasHandle_t handle,
                                                                                                    cublasOperation_t transa,
                                                                                                    cublasOperation_t transb,
                                                                                                    int M,
                                                                                                    int N,
                                                                                                    int K,
                                                                                                    const float *alpha,
                                                                                                    const float *ptr_A,
                                                                                                    int lda,
                                                                                                    long long int stride_A,
                                                                                                    const float *ptr_B,
                                                                                                    int ldb,
                                                                                                    long long int stride_B,
                                                                                                    const float *beta,
                                                                                                    float *ptr_C,
                                                                                                    int ldc,
                                                                                                    long long int stride_C,
                                                                                                    int batchCount) {
  #if CUTLASS_ENABLE_CUBLAS
  return cublasSgemmStridedBatched(handle,
    transa,
    transb,
    M, N, K,
    alpha,
    ptr_A,
    lda,
    stride_A,
    ptr_B,
    ldb,
    stride_B,
    beta,
    ptr_C,
    ldc,
    stride_C,
    batchCount);
  #else
  return CUBLAS_STATUS_NOT_SUPPORTED;
  #endif
}

template<> inline cublasStatus_t GemmTestbed<double, double, double, double, double>::cublasGemmStridedBatchedTemplate(cublasHandle_t handle,
                                                                                                                cublasOperation_t transa,
                                                                                                                cublasOperation_t transb,
                                                                                                                int M,
                                                                                                                int N,
                                                                                                                int K,
                                                                                                                const double *alpha,
                                                                                                                const double *ptr_A,
                                                                                                                int lda,
                                                                                                                long long int stride_A,
                                                                                                                const double *ptr_B,
                                                                                                                int ldb,
                                                                                                                long long int stride_B,
                                                                                                                const double *beta,
                                                                                                                double *ptr_C,
                                                                                                                int ldc,
                                                                                                                long long int stride_C,
                                                                                                                int batchCount) {
  #if CUTLASS_ENABLE_CUBLAS
  return cublasDgemmStridedBatched(handle,
    transa,
    transb,
    M, N, K,
    alpha,
    ptr_A,
    lda,
    stride_A,
    ptr_B,
    ldb,
    stride_B,
    beta,
    ptr_C,
    ldc,
    stride_C,
    batchCount);
  #else
  return CUBLAS_STATUS_NOT_SUPPORTED;
  #endif
}

template<> inline cublasStatus_t GemmTestbed<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t>::cublasGemmStridedBatchedTemplate(cublasHandle_t handle,
                                                                                                      cublasOperation_t transa,
                                                                                                      cublasOperation_t transb,
                                                                                                      int M,
                                                                                                      int N,
                                                                                                      int K,
                                                                                                      const cutlass::half_t *alpha,
                                                                                                      const half *ptr_A,
                                                                                                      int lda,
                                                                                                      long long int stride_A,
                                                                                                      const half *ptr_B,
                                                                                                      int ldb,
                                                                                                      long long int stride_B,
                                                                                                      const cutlass::half_t *beta,
                                                                                                      half *ptr_C,
                                                                                                      int ldc,
                                                                                                      long long int stride_C,
                                                                                                      int batchCount) {
  #if CUTLASS_ENABLE_CUBLAS
  half temp_alpha = alpha->operator half();
  half temp_beta = beta->operator half();
  return cublasHgemmStridedBatched(handle,
    transa,
    transb,
    M, N, K,
    &temp_alpha,
    ptr_A,
    lda,
    stride_A,
    ptr_B,
    ldb,
    stride_B,
    &temp_beta,
    ptr_C,
    ldc,
    stride_C,
    batchCount);
  #else
  return CUBLAS_STATUS_NOT_SUPPORTED;
  #endif
}

template<> inline cublasStatus_t GemmTestbed<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, float>::cublasGemmStridedBatchedTemplate(cublasHandle_t handle,
                                                                                                      cublasOperation_t transa,
                                                                                                      cublasOperation_t transb,
                                                                                                      int M,
                                                                                                      int N,
                                                                                                      int K,
                                                                                                      const float *alpha,
                                                                                                      const half *ptr_A,
                                                                                                      int lda,
                                                                                                      long long int stride_A,
                                                                                                      const half *ptr_B,
                                                                                                      int ldb,
                                                                                                      long long int stride_B,
                                                                                                      const float *beta,
                                                                                                      half *ptr_C,
                                                                                                      int ldc,
                                                                                                      long long int stride_C,
                                                                                                      int batchCount) {
  #if CUTLASS_ENABLE_CUBLAS
  return cublasGemmStridedBatchedEx(handle,
    transa,
    transb,
    M, N, K,
    alpha,
    ptr_A,
    cutlass::TypeTraits<cutlass::half_t>::cublas_type,
    lda,
    stride_A,
    ptr_B,
    cutlass::TypeTraits<cutlass::half_t>::cublas_type,
    ldb,
    stride_B,
    beta,
    ptr_C,
    cutlass::TypeTraits<cutlass::half_t>::cublas_type,
    ldc,
    stride_C,
    batchCount,
    cutlass::TypeTraits<float>::cublas_type,
    CUBLAS_GEMM_DEFAULT);
  #else
  return CUBLAS_STATUS_NOT_SUPPORTED;
  #endif
}
}  // namespace test
