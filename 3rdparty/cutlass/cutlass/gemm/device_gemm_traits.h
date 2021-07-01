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
#include <assert.h>
#include "cutlass/gemm/device_gemm.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/gemm/gemm_desc.h"
#include "tools/util/type_traits.h"
#include <iostream>

namespace cutlass {
namespace gemm {

template <
  /// The Tratis for the first kernel
  typename GemmTraits_,
  /// The Traits for the second kernel
  typename ReductionTraits_
>
struct SplitkPIGemmTraits {
  typedef GemmTraits_ GemmTraits;
  typedef ReductionTraits_ ReductionTraits;
  typedef SplitkPIGemmTraits<GemmTraits_, ReductionTraits_> This_;
  typedef typename cutlass::gemm::DeviceGemm<This_> KernelClass;

  ///
  typedef typename GemmTraits::Index Index;
  ///
  typedef typename ReductionTraits::ScalarAlphaBeta Scalar;
  ///
  typedef typename GemmTraits::ScalarA ScalarA;
  ///
  typedef typename GemmTraits::ScalarB ScalarB;
  ///
  typedef typename GemmTraits::ScalarD ScalarAccum;
  ///
  typedef typename ReductionTraits::ScalarC ScalarC;
  ///
  typedef typename ReductionTraits::ScalarD ScalarD;
  /// The layout of A. can be deduced from the layout set in batched gemm
  static MatrixLayout::Kind const kLayoutA = GemmTraits::kLayoutA;
  /// The layout of B. can be deduced from the layout set in batched gemm
  static MatrixLayout::Kind const kLayoutB = GemmTraits::kLayoutB;

  struct Params {
    /// The dimensions of the GEMM in K, N, M order
    GemmCoord problem_size;

    /// Check if params are init
    bool problem_size_initialized;
    /// The pointer to workspace memory
    ScalarAccum *workspace_ptr;
    ///
    size_t workspace_size;
    /// The Params for the first kernel
    typename GemmTraits::Params GemmParams;
    /// The Params for the second kernel
    typename ReductionTraits::Params ReductionParams;

    /// ctor
    Params() :
      workspace_size(0),
      problem_size_initialized(false) {}
    /// ctor
    Params(Index m_,
           Index n_,
           Index k_
      ):
        problem_size(k_, n_, m_, 1),
        workspace_size(0),
        problem_size_initialized(true) {

    }

    /// init problem is needed if using default ctor
    void init_problem(Index m_,
                     Index n_,
                     Index k_){
      problem_size = GemmCoord(k_, n_, m_, 1);
      problem_size_initialized = true;
    }

    int initialize(Scalar alpha_,
                   ScalarA const* d_a_,
                   Index lda_,
                   ScalarB const* d_b_,
                   Index ldb_,
                   Scalar beta_,
                   ScalarC const* d_c_,
                   Index ldc_,
                   ScalarD* d_d_,
                   Index ldd_,
                   ScalarAccum *workspace_ptr_,
                   Index partitionK_multiple = 1) {

      workspace_ptr = workspace_ptr_;

      //call GemmTraits (first kernel) param
      //for the first kernel A is A, B is B, C and D are workspace
      //alpha is one, beta is zero, partitionK_count is reductionTraits::reductionSize
      typename cutlass::gemm::GemmDesc<typename GemmTraits::ScalarA,
        typename GemmTraits::ScalarB,
        typename GemmTraits::ScalarC,
        typename GemmTraits::ScalarD,
        typename GemmTraits::Epilogue::Scalar>
        desc(
          problem_size,
          typename cutlass::TypeTraits<typename GemmTraits::Epilogue::Scalar>::host_type(1.0f), /*alpha*/
          TensorRef<typename GemmTraits::ScalarA const, 2>(d_a_, lda_),
          TensorRef<typename GemmTraits::ScalarB const, 2>(d_b_, ldb_),
          typename cutlass::TypeTraits<typename GemmTraits::Epilogue::Scalar>::host_type(0.0f), /*beta*/
          TensorRef<typename GemmTraits::ScalarC const, 2>(workspace_ptr, problem_size.m()), /*m = ldc, workspace is not transposed and is packed*/
          TensorRef<typename GemmTraits::ScalarD, 2>(workspace_ptr, problem_size.m()) /*m = ldd, workspace is not transposed and is packed*/
        );
      GemmParams.initialize(desc, ReductionTraits::ReductionSize, partitionK_multiple);
     

      //call batched reduction (second kernel) param
      ReductionParams.initialize(problem_size.m(), /*m*/
        problem_size.n(), /*n*/
        alpha_, /*alpha*/
        beta_, /*beta*/
        problem_size.n() * problem_size.m() /*reduction_stride*/,
        workspace_ptr,
        problem_size.m(),
        d_c_,
        ldc_,
        d_d_,
        ldd_);

      return 0;
    }

    // workspace will be used to store D (output) from the first gemm kernel (not D of the entire gemm)
    // note typedef typename GemmTraits::ScalarD ScalarAccum;
    // workspace of size of M * N * Reduction
    size_t required_workspace_memory_in_byte(){
      assert(problem_size_initialized == true);
      workspace_size = static_cast<size_t>(problem_size.n()) * 
                       static_cast<size_t>(problem_size.m()) * 
                       static_cast<size_t>(ReductionTraits::ReductionSize) * 
                       sizeof(ScalarAccum);
      return workspace_size;
    }


  };

};

} // namespace device_gemm
} // namespace cutalss
