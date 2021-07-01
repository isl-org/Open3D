# Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include_directories(
  .
)

set(CUTLASS_PERF_TEST_HEADERS
  testbench_output.h
  performance_result.h
  gemm/cublas_dispatch.h
  gemm/cutlass_dispatch.h
  gemm/cutlass_dispatch_splitK_PI.h
  gemm/gemm_perf_testbed.h
  gemm/gemm_profiler.h
  gemm/cutlass_volta884_dispatch.h
  gemm/cutlass_volta884_dispatch_splitK_PI.h
)

set(CUTLASS_PERF_TEST_SOURCES
  cutlass_perf_test.cu
  gemm/sgemm.cu
  gemm/sgemm_splitK.cu
  gemm/dgemm.cu
  gemm/hgemm.cu
  gemm/igemm.cu
  gemm/igemm_splitK.cu
  gemm/wmma_gemm.cu
  gemm/volta884_gemm.cu                       
  gemm/volta884_gemm_splitK.cu
  gemm/volta884_gemm_cta_rasterization_tn.cu
  gemm/volta884_gemm_cta_rasterization_tt.cu
  gemm/volta884_gemm_cta_rasterization_nn.cu
  gemm/volta884_gemm_cta_rasterization_nt.cu
  gemm/wmma_binary_gemm.cu                         
  gemm/wmma_integer_gemm.cu                        
)

source_group("Source\ Files" FILES ${CUTLASS_PERF_TEST_SOURCES})

if(NOT CUTLASS_NATIVE_CUDA)
  # cuda_add_executable does not take interface include directories into account
  # Let's fetch them and pass them to CUDA.
  get_target_property(CUTLASS_INCLUDES CUTLASS INTERFACE_INCLUDE_DIRECTORIES)
  include_directories("${CUTLASS_INCLUDES}")
endif()

cutlass_add_executable(
    cutlass_perf_test
    ${CUTLASS_PERF_TEST_SOURCES}
    ${CUTLASS_PERF_TEST_HEADERS}
)

if(CUTLASS_ENABLE_CUBLAS)
  target_link_libraries(cutlass_perf_test ${CUBLAS_LIBRARY})
endif()

