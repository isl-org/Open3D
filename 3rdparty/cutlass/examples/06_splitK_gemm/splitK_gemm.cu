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

#include <iostream>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device_gemm.h"
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/reduction/batched_reduction_traits.h"
#include "cutlass/gemm/device_gemm_traits.h"
#pragma warning( disable : 4503)
/*
This example demonstrates how to use cutlass to compute sgemm with splitK
splitK is useful for gemm with small M and N and reasonably large K.
Because the sizes of M and N are small, the number of threadblocks we can launch is often limited and 
results in under utilization of the hardware. 
splitK allows us to divide a gemm across K dimension by first launching a partitionedK gemm (very similar to batched gemm), 
storing the intermediate result in workspace and then launching a second reduction kernel.
Thus, as demonstrated by function cutlass_splitK_sgemm_nn(), the users need to create two traits, one for the partitionedK gemm,
and one for the reduction. The users are also responsible for allocating and releasing the workspace memory. The size of the workspace
memory can be queried by calling required_workspace_memory_in_byte().
*/

template<int splits_count>
cudaError_t cutlass_splitK_sgemm_nn(float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float alpha,
  float beta,
  int m, 
  int n,
  int k) {
  cudaError_t result = cudaSuccess;

  // create cutlass gemm traits for the first kernel
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor, /*the layout of A*/
    cutlass::MatrixLayout::kColumnMajor, /*the layout of B*/
    cutlass::Shape<8, 128, 128> > /*the tile for each threadblock*/
    SgemmTraits;

  // create cutlass batched reduction traits for the second kernel
  // for reduction D = alpha * Reduction(A) + beta * C
  typedef cutlass::reduction::BatchedReductionTraits<float, /*the scalar type of A in reduction, not to be confused with A in GEMM*/
    float, /*the scalar type of C in reduction, not to be confused with C in GEMM*/
    float, /*the scalar type of D in reduction, not to be confused with D in GEMM*/
    float, /*the scalar type of alpha and beta in reduction*/
    float, /*the scalar type of accumulation in reduction*/
    splits_count /*reduction workload*/
    > 
    BatchedReductionTraits;

  // create a device gemm that packages gemm traits and batched reduction traits
  typedef cutlass::gemm::SplitkPIGemmTraits<SgemmTraits, BatchedReductionTraits> deviceGemmTraits;

  // kernel class
  typedef typename deviceGemmTraits::KernelClass deviceGemm;

  // Params ctor requires M, N, K sizes
  typename deviceGemm::Params deviceGemmParams(m, n, k);

  // query if workspace is needed. the workspace size is sizeof(accumulateType) * M * N * splits_count
  size_t workspace_size = deviceGemmParams.required_workspace_memory_in_byte();

  // allocate workspace memory
  float *workspace_ptr;
  result = cudaMalloc(&workspace_ptr, workspace_size);
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }

  // finish init Params
  deviceGemmParams.initialize(alpha, /*alpha*/
    A,                               /*A*/
    lda,                             /*lda*/
    B,                               /*B*/
    ldb,                             /*ldb*/
    beta,                            /*beta*/
    C,                               /*C*/
    ldc,                             /*ldc*/
    C,                               /*D, can point to the same memory with C*/
    ldc,                             /*ldc*/
    workspace_ptr                    /*ptr to workspace*/
  );

  // launch the kernel
  deviceGemm::launch(deviceGemmParams);
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "launch result = " << result << std::endl;
    cudaFree(workspace_ptr);
    return result;
  }

  // release the workspace memory
  result = cudaFree(workspace_ptr);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
  }

  return cudaGetLastError();
}

template<typename T> 
cudaError_t sgemm_nn_reference(std::vector<T> const &A,
  int lda,
  std::vector<T> const &B, 
  int ldb,
  std::vector<T> &C, 
  int ldc,
  T alpha,
  T beta,
  int m,
  int n,
  int k) {
  /*
  sgemm
  */
  
  cudaError_t result = cudaSuccess;
  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int m_idx = 0; m_idx < m; m_idx++) {
      T accum = beta * C[n_idx * ldc + m_idx];
      for (int k_idx = 0; k_idx < k; k_idx++) {
        accum += alpha
          * A[k_idx * lda + m_idx]
          * B[n_idx * ldb + k_idx];
      }
      C[n_idx * ldc + m_idx] = accum;
    }
  }

  return result;
}

int main() {
  int const m = 128;
  int const n = 128;
  int const k = 4096;
  //splits_count should be known at compile time
  int const splits_count = 80;

  // A, B are non-transpose, column major
  int const lda = m;
  int const ldb = k;
  int const ldc = m;

  int const count_A = lda * k;
  int const count_B = ldb * n;
  int const count_C = ldc * n;

  // alpha and beta
  float alpha = 1.0f;
  float beta = 2.0f;

  cudaError_t result = cudaSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);

  // allocate the device memory
  float *A;
  float *B;
  float *C;

  result = cudaMalloc(&A, count_A * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&B, count_B * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&C, count_C * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }

  // fill A
  for (int col_idx = 0; col_idx < k; col_idx++) {
    for (int row_idx = 0; row_idx < m; row_idx++) {
      host_A[row_idx + col_idx * lda] = static_cast<float>((row_idx + col_idx) % 10);
    }
  }
  
  // fill B
  for (int col_idx = 0; col_idx < n; col_idx++) {
    for (int row_idx = 0; row_idx < k; row_idx++) {
      host_B[row_idx + col_idx * ldb] = static_cast<float>((row_idx - col_idx) % 5);
    }
  }

  // fill C
  for (int col_idx = 0; col_idx < n; col_idx++) {
    for (int row_idx = 0; row_idx < m; row_idx++) {
      host_C[row_idx + col_idx * ldc] = 1.f;
    }
  }

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  // copy host memory to device
  result = cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  // run cutlass
  result = cutlass_splitK_sgemm_nn<splits_count>(A, lda, B, ldb, C, ldc, alpha, beta, m, n, k);
  if (result != cudaSuccess)
    return result;

  // copy device memory to host
  result = cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  //compare with reference code
  result = sgemm_nn_reference(ref_A, lda, ref_B, ldb, ref_C, ldc, alpha, beta, m, n, k);
  if (result != 0)
    return result;

  if (ref_C != result_C) {
    std::cout << "CUTLASS splitK gemm does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  // free memory
  result = cudaFree(A);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(B);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(C);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }


  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}
