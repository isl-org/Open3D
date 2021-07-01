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

// Standard Library includes
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

// CUDA includes
#include <cublas_v2.h>
#include <curand_kernel.h>

// Cutlass includes
#include "tools/test/perf/gemm/cublas_dispatch.h"
#include "tools/test/perf/performance_result.h"
#include "tools/test/perf/testbench_options.h"
#include "tools/util/device_memory.h"
#include "tools/util/host_matrix.h"
#include "tools/util/reference/device/tensor_elementwise.h"
#include "tools/util/tensor_view_io.h"
#include "tools/util/type_traits.h"

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

  template <typename T>
  struct ElementCount {
    static int const kValue = 1;
  };

  template <typename T, int Elements>
  struct ElementCount<cutlass::Vector<T, Elements> > {
    static int const kValue = Elements * ElementCount<T>::kValue;
  };

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Host-side implementation of performance testbed
template <typename AType, typename BType, typename CType, typename Accumulator, typename Scalar>
class GemmTestbed {
 public:
  /// Type used for device-side allocations
  typedef typename cutlass::TypeTraits<AType>::device_type ADeviceType;
  typedef typename cutlass::TypeTraits<BType>::device_type BDeviceType;
  typedef typename cutlass::TypeTraits<CType>::device_type CDeviceType;
  typedef typename cutlass::TypeTraits<Accumulator>::device_type AccumulatorDeviceType;
  typedef typename cutlass::TypeTraits<Scalar>::device_type ScalarDeviceType;

  /// Dispatch object to cuBLAS GEMM
  typedef CublasGemmDispatch<AType, BType, CType, Accumulator, Scalar> CublasDispatch;
  typedef CublasBatchedStridedGemmDispatch<AType, BType, CType, Accumulator, Scalar> CublasBatchedStridedGemmDispatch;

  //
  // Type definitions
  //

  /// Host tensor for operand A
  typedef cutlass::device_memory::allocation<ADeviceType> TensorA;

  /// Host tensor for operand B
  typedef cutlass::device_memory::allocation<BDeviceType> TensorB;

  /// Host tensor for operand C
  typedef cutlass::device_memory::allocation<CDeviceType> TensorC;

 private:
  //
  // Data members
  //

  InitialDistribution initial_distribution;

  /// Status
  cublasStatus_t status;

  /// cuBLAS handle
  cublasHandle_t handle;

  /// GEMM problem
  GemmProblem problem;

  /// A matrix operand
  TensorA A;

  /// B matrix operand
  TensorB B;

  /// C matrix operand
  TensorC C_initial;

  /// Reference result
  TensorC reference;

  /// Experimental result
  TensorC experimental;

 private:
  //
  // Methods
  //

  /// Helper to resize a matrix with a given size and layout if needed
  template <typename T>
  static void resize_device_allocation(cutlass::device_memory::allocation<T> &tensor,
                                       cutlass::Distribution const &dist,
                                       int64_t seed,
                                       int rows,
                                       int columns,
                                       cutlass::MatrixLayout::Kind layout,
                                       int ldm = 0) {
    if (!ldm) {
      ldm = (layout == cutlass::MatrixLayout::kColumnMajor ? rows : columns);
    }

    size_t capacity = ldm * (layout == cutlass::MatrixLayout::kColumnMajor ? columns : rows);

    if (capacity > tensor.capacity) {
      tensor.reset(cutlass::device_memory::allocate<T>(capacity), capacity);

      int c_dim = (layout == cutlass::MatrixLayout::kColumnMajor ? rows : columns);
      int s_dim = (layout == cutlass::MatrixLayout::kColumnMajor ? columns : rows);

      cutlass::TensorView<T, 2> view(
        tensor.get(), 
        cutlass::make_Coord(ldm, 1), 
        cutlass::make_Coord(s_dim, c_dim));

      cutlass::reference::device::TensorInitialize(view, seed, dist);
    }
  }

  /// Resizes each tensor
  void resize_helper(GemmProblem const &problem) {

      resize_device_allocation(A,
        initial_distribution.dist_A,
        initial_distribution.seed,
        problem.m,
        problem.k * problem.batch_count,
        problem.layout_A);


    resize_device_allocation(
        B,
        initial_distribution.dist_B,
        initial_distribution.seed + 17,  // compute distinct value from initial seed
        problem.k * problem.batch_count,
        problem.n,
        problem.layout_B);

    resize_device_allocation(
        C_initial,
        initial_distribution.dist_C,
        initial_distribution.seed + 101,  // compute distinct value from initial seed
        problem.m,
        problem.n * problem.batch_count,
        cutlass::MatrixLayout::kColumnMajor);

    resize_device_allocation(reference,
                             cutlass::Distribution(),
                             0,
                             problem.m,
                             problem.n * problem.batch_count,
                             cutlass::MatrixLayout::kColumnMajor);

    resize_device_allocation(experimental,
                             cutlass::Distribution(),
                             0,
                             problem.m,
                             problem.n * problem.batch_count,
                             cutlass::MatrixLayout::kColumnMajor);
  }

  /// Functor to print errors
  struct PrintErrors {
    /// Equivalently sized integer type
    typedef typename cutlass::TypeTraits<CType>::integer_type integer_t;

    /// Performance testbench defined for a TensorView of rank-2 contiguous matrices
    typedef cutlass::TensorView<CType, 2, cutlass::MatrixLayout::ContiguousLayout> MatrixView;

    /// Output stream to write to
    std::ostream &out;

    /// Reference tensor view
    MatrixView const &reference;

    /// Computed tensor view
    MatrixView const &experimental;

    /// Errors greater than or this amount result in printing
    integer_t ulps_threshold;

    ///
    PrintErrors(std::ostream &_out,
                MatrixView const &_reference,
                MatrixView const &_experimental,
                integer_t _ulps_threshold = 1)
        : out(_out),
          reference(_reference),
          experimental(_experimental),
          ulps_threshold(_ulps_threshold) {}

    /// Compares one element
    void operator()(CType const &element, typename MatrixView::TensorCoord coord) {
      CType exp = experimental.at(coord);
      CType ref = reference.at(coord);

      int64_t int_exp = 0;
      int64_t int_ref = 0;

      *reinterpret_cast<CType *>(&int_exp) = exp;
      *reinterpret_cast<CType *>(&int_ref) = ref;

      integer_t ulps = integer_t(int_exp - int_ref);

      if (std::abs(ulps) >= ulps_threshold) {
        // width in hexadecimal digits of value
        int const width = sizeof(integer_t) * 2;

        double relative = double(exp) - double(ref);
        if (ref != CType(0)) {
          relative /= double(ref);
        }

        out << "[" << coord << "] expected: " << ref << " (0x" << std::hex << std::setw(width)
            << std::setfill('0') << integer_t(int_ref) << std::dec << ")"
            << ",  got: " << exp << " (0x" << std::hex << std::setw(width) << std::setfill('0')
            << integer_t(int_exp) << std::dec << ")"
            << "  relative error: " << relative << ", ulps: " << ulps << "\n";
      }
    }
  };

 public:
  /// Resizes tensors to accommodate the given problem
  void resize(GemmProblem const &_problem) {
    problem = _problem;

    try {
      resize_helper(problem);
    } catch (...) {
      // If out of memory, clear each allocation then allocate again
      A.reset();
      B.reset();
      C_initial.reset();
      reference.reset();
      experimental.reset();

      resize_helper(problem);
    }
  }

  /// Constructs a basic workspace
  GemmTestbed(InitialDistribution const &_dist = InitialDistribution())
      : initial_distribution(_dist) {
    #if CUTLASS_ENABLE_CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw cutlass::cuda_exception("Failed to create CUBLAS handle");
    }
    #else
    status = CUBLAS_STATUS_NOT_INITIALIZED;
    #endif
  }

  /// Constructs a workspace for verifying GEMM, assumes
  /// dense packing.
  GemmTestbed(GemmProblem const &_problem,
              cublasGemmAlgo_t algorithm_ = CUBLAS_GEMM_DEFAULT,
              InitialDistribution const &_dist = InitialDistribution())
      : problem(_problem), initial_distribution(_dist) {
    #if CUTLASS_ENABLE_CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw cutlass::cuda_exception("Failed to create CUBLAS handle");
    }
    #else
    status = CUBLAS_STATUS_NOT_INITIALIZED;
    #endif

    resize(problem);
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

  /// Rows of GEMM problem
  int M() const { return problem.m; }

  /// Columns of GEMM problem
  int N() const { return problem.n; }

  /// Inner dimension of GEMM problem
  int K() const { return problem.k; }

  /// batch count
  int batch_count() const { return problem.batch_count; }

  /// Returns a pointer to the A operand
  ADeviceType *ptr_A() const { return A.get(); }

  /// Leading dimension of A
  int lda() const { return problem.lda(); }

  ///
  long long int batch_stride_a() const{ return problem.batch_stride_a(); }

  /// Returns a pointer to the B operand
  BDeviceType *ptr_B() const { return B.get(); }

  /// Leading dimension of B
  int ldb() const { return problem.ldb(); }

  ///
  long long int batch_stride_b() const{ return problem.batch_stride_b(); }

  /// Returns a pointer to the initial state of the result tensor in device memory
  CDeviceType *ptr_C_initial() const { return C_initial.get(); }

  /// Leading dimension of C
  int ldc() const { return problem.ldc(); }

  ///
  long long int batch_stride_c() const { return problem.batch_stride_c(); }

  /// Returns a pointer to the result tensor in device memory
  CDeviceType *ptr_experimental() const { return experimental.get(); }

  /// Returns a pointer to the result tensor in device memory
  CDeviceType *ptr_reference() const { return reference.get(); }

  /// Returns the number of flops implied by the computation (1 multiply-accumulate = 2 flops)
  uint64_t flops() const {
    return uint64_t(problem.batch_count) * uint64_t(problem.m) * uint64_t(problem.n) * uint64_t(problem.k) * detail::ElementCount<AType>::kValue * 2ULL;
  }

  /// Computes the speed of the computation in GFLOPs/s
  double GFLOPs_per_sec(double runtime_ms) const { return double(flops()) / runtime_ms / 1.0e6; }

  /// Matrix layout of A
  cutlass::MatrixLayout::Kind layout_a() const { return problem.layout_A; }

  /// Matrix layout of B
  cutlass::MatrixLayout::Kind layout_b() const { return problem.layout_B; }

  /// Returns alpha scalar
  Scalar alpha() const { return Scalar(problem.alpha); }

  /// Returns alpha scalar
  Scalar beta() const { return Scalar(problem.beta); }

  /// Initializes C matrix by copying from C_initial
  void prepare_gemm(CDeviceType *target) {
    size_t count = ldc() * problem.n;
    cutlass::device_memory::copy_device_to_device(target, ptr_C_initial(), count);
  }

  /// Initializes output matrix of cublas
  void prepare_cublas() { prepare_gemm(ptr_reference()); }

  /// Initializes output matrix of cublas
  void prepare_experimental() { prepare_gemm(ptr_experimental()); }

  /// Launches the cuBLAS GEMM - does not initialize output matrix
  cublasStatus_t launch_cublas(cublasGemmAlgo_t algo) {
    #if CUTLASS_ENABLE_CUBLAS
    if (problem.batch_count == 1) {
      CublasDispatch dispatch;

      Scalar alpha(Scalar(problem.alpha));
      Scalar beta(Scalar(problem.beta));

      status = dispatch(handle,
        problem.layout_A,
        problem.layout_B,
        problem.m,
        problem.n,
        problem.k,
        alpha,
        ptr_A(),
        lda(),
        ptr_B(),
        ldb(),
        beta,
        ptr_reference(),
        ldc(),
        algo);

      return status;
    }
    else {
      // call batched strided cublas
      CublasBatchedStridedGemmDispatch dispatch;

      Scalar alpha(Scalar(problem.alpha));
      Scalar beta(Scalar(problem.beta));

      status = dispatch(handle,
        problem.layout_A,
        problem.layout_B,
        problem.m,
        problem.n,
        problem.k,
        alpha,
        ptr_A(),
        lda(),
        batch_stride_a(),
        ptr_B(),
        ldb(),
        batch_stride_b(),
        beta,
        ptr_reference(),
        ldc(),
        batch_stride_c(),
        batch_count(),
        algo);

      return status;
    }
    #else
    return CUBLAS_STATUS_NOT_SUPPORTED;
    #endif
  }

  /// Verifies the 'test' tensor with 'ref'
  bool verify(TensorC const &test, TensorC const &ref) {

    return cutlass::reference::device::TensorEquals(
      cutlass::TensorView<CDeviceType, 2>(
        test.get(), 
        cutlass::make_Coord(problem.m, 1),
        cutlass::make_Coord(problem.n, problem.m)),
      cutlass::TensorView<CDeviceType, 2>(
        ref.get(), 
        cutlass::make_Coord(problem.m, 1), 
        cutlass::make_Coord(problem.n, problem.m))
    );
  }

  /// Computes the reference output
  void compute_reference(cublasGemmAlgo_t algorithm) {
    prepare_cublas();
    launch_cublas(algorithm);
  }

  /// Helper to verify with reference
  bool verify_with_reference() { return verify(experimental, reference); }

  /// Writes the problem to an ostream in human-readable form
  void write_problem(std::ostream &results_output, std::ostream &errors_output) {
    cutlass::HostMatrix<AType> host_A;
    cutlass::HostMatrix<BType> host_B;
    cutlass::HostMatrix<CType> host_C;
    cutlass::HostMatrix<CType> host_D;
    cutlass::HostMatrix<CType> host_Ref;

    host_A.resize_matrix(M(), K(), layout_a());
    host_B.resize_matrix(K(), N(), layout_b());
    host_C.resize_matrix(M(), N(), cutlass::MatrixLayout::kColumnMajor);
    host_D.resize_matrix(M(), N(), cutlass::MatrixLayout::kColumnMajor);
    host_Ref.resize_matrix(M(), N(), cutlass::MatrixLayout::kColumnMajor);

    // copy from device allocations
    host_A.copy_to_host(ptr_A());
    host_B.copy_to_host(ptr_B());
    host_C.copy_to_host(ptr_C_initial());
    host_D.copy_to_host(ptr_experimental());
    host_Ref.copy_to_host(ptr_reference());

    // write out human readable
    results_output << "A =\n"
                   << host_A << "\n"
                   << "B =\n"
                   << host_B << "\n"
                   << "C = \n"
                   << host_C << "\n"
                   << "Ref =\n"
                   << host_Ref << "\n"
                   << "Experimental =\n"
                   << host_D << "\n";

    // write out list of errors
    PrintErrors printer(errors_output, host_Ref, host_D);

    host_D.visit(printer);
  }
};

}  // namespace perf
