/***************************************************************************************************
* Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#define NVRTC_GET_TYPE_NAME 1

#include "cutlass/cutlass.h"

#include <nvrtc.h>
#include "tools/nvrtc/cutlass/nvrtc/environment.h"
#include <string>
#include <sstream>

static inline bool check_nvrtc_error(nvrtcResult error) {
  if (error != NVRTC_SUCCESS) {
    std::cerr << "failed to compile ";
    return false;
  }
  return true;
}

/// @param gemm_traits Must be the source string to generate GemmTraits_
template <typename GemmTraits_>
static __host__ void run_gemm_nvrtc(
    std::string const &gemm_traits,
    int m,
    int n,
    int k,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  typename Gemm::Params params;

  test::GemmTestbed<
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
      typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      >
      testbed(m,
              n,
              k,
              test::convert(GemmTraits_::kLayoutA),
              test::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  int currentDevice;
  cudaGetDevice(&currentDevice);

  // generate the architecture string for the nvrtc conmpiler
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, currentDevice);
  std::stringstream arch;
  arch << "-arch=compute_" << deviceProperties.major << deviceProperties.minor;

  // Instantiate gemm_kernel
  nvrtcResult result_nvrtc;
  nvrtcProgram program;
  static char const *src =
      "#include \"cutlass/gemm/gemm.h\"\n"
      "#include \"cutlass/gemm/sgemm_traits.h\"\n"
      "#include \"cutlass/gemm/dgemm_traits.h\"\n"
      "#include \"cutlass/gemm/igemm_traits.h\"\n"
#if defined(CUTLASS_NVRTC_HAS_FP16)
      "#include \"cutlass/gemm/hgemm_traits.h\"\n"
      "#include \"cutlass/gemm/wmma_gemm_traits.h\"\n"
#endif
      ;

  std::string type_name;
  nvrtcGetTypeName<GemmTraits_>(&type_name);

  result_nvrtc = nvrtcCreateProgram(&program,
                                    src,
                                    NULL,
                                    (int)cutlass::nvrtc::kCutlassHeaderCount,
                                    cutlass::nvrtc::kCutlassHeaders,
                                    cutlass::nvrtc::kCutlassHeaderNames);
  check_nvrtc_error(result_nvrtc);

  std::string gemm_kernel_instantiation =
      "cutlass::gemm::gemm_kernel<cutlass::gemm::Gemm< " + type_name + " >::KernelClass >";
  nvrtcAddNameExpression(program, gemm_kernel_instantiation.c_str());

  // generate option list to genereate kernel for the underlying GPU
  std::vector<std::string> options;
  std::vector<const char*> c_options;

  options.push_back(arch.str());

  // convert option list into a c-string list for the nvrtc interface
  for (std::vector<std::string>::const_iterator i = options.begin(); i != options.end(); ++i) {
      c_options.push_back(i->c_str());
  }

  // compile
  result_nvrtc = nvrtcCompileProgram(program, int(c_options.size()), c_options.data());
  if (result_nvrtc != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(program, &logSize);
    std::vector<char> log(logSize);
    nvrtcGetProgramLog(program, log.data());
    std::cout << "Compile log:" << std::endl << log.data() << std::endl;
  }
  if (!check_nvrtc_error(result_nvrtc)) {
    ASSERT_TRUE(false);
  }

  // The lowered name is the name of the template instantiation in the generated PTX code.
  char const *temp_gemm_kernel_lowered_name;
  nvrtcGetLoweredName(program, gemm_kernel_instantiation.c_str(), &temp_gemm_kernel_lowered_name);
  if (!check_nvrtc_error(result_nvrtc)) {
    ASSERT_TRUE(false);
  }
  // the ponter we got from nvrtcGetLoweredName is valid only as long as the program is valid. create a copy.
  std::string gemm_kernel_lowered_name(temp_gemm_kernel_lowered_name);

  // Query the size of the genereated PTX so that we can allocate storage and retrieve it afterwards
  size_t ptx_size;
  result_nvrtc = nvrtcGetPTXSize(program, &ptx_size);
  if (!check_nvrtc_error(result_nvrtc)) {
    ASSERT_TRUE(false);
  }

  std::vector<char> ptx(ptx_size);
  result_nvrtc = nvrtcGetPTX(program, ptx.data());
  if (!check_nvrtc_error(result_nvrtc)) {
      std::cerr << "failed to get ptx" << std::endl;
    ASSERT_TRUE(false);
  }

  // we do not need the nvrtc program anymore
  nvrtcDestroyProgram(&program);

  // Now load the module
  CUmodule module;
  CUresult result_cuda;

  result_cuda = cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
  if (result_cuda != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result_cuda, &msg);
    std::cerr << "\ncuModuleLoadDataEx error: failed with error " << msg << std::endl;
    ASSERT_TRUE(false);
  }

  // and retrieve the function
  CUfunction kernel;
  result_cuda = cuModuleGetFunction(&kernel, module, gemm_kernel_lowered_name.c_str());
  if (result_cuda != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result_cuda, &msg);
    std::cerr << "\ncuModuleGetFunction error: failed with error " << msg << std::endl;
    ASSERT_TRUE(false);
  }

  testbed.initialize();

  if (testbed.has_cublas_support()) {
    EXPECT_TRUE(testbed.verify_host_with_cublas());
  }

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
                    testbed.alpha,
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    testbed.beta,
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.ptr_computed(),
                    testbed.ldc());

  Gemm::launch(kernel, params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << std::endl;

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  } else {
    ASSERT_TRUE(testbed.verify_with_host());
  }

  result_cuda = cuModuleUnload(module);
  if (result_cuda != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result_cuda, &msg);
    std::cerr << "\ncuModuleUnload error: failed with error " << msg << std::endl;
    ASSERT_TRUE(false);
  }
}
