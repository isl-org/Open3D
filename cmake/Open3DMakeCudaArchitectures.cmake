# open3d_make_cuda_architectures(cuda_archs)
#
# Sets up CUDA architectures based on the following precedence rules
# and stores them into the <cuda_archs> variable.
#   1. All common architectures if BUILD_COMMON_CUDA_ARCHS=ON
#   2. User-defined architectures
#   3. Architectures detected on the current machine
#   4. CMake's default architectures
function(open3d_make_cuda_architectures cuda_archs)
    unset(${cuda_archs})

    find_package(CUDAToolkit REQUIRED)

    if(BUILD_COMMON_CUDA_ARCHS)
        # Build with all supported architectures for previous 2 generations and
        # M0 (minor=0) architectures for previous generations (including
        # deprecated). Note that cubin for M0 runs on GPUs with architecture Mx.
        # This is a tradeoff between binary size / build time and runtime on
        # older architectures. See:
        # https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#building-for-maximum-compatibility
        # https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html#application-compatibility-on-ampere
        # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
            set(${cuda_archs} 60-real 70-real 72-real 75-real 80-real 86)
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            set(${cuda_archs} 60-real 70-real 72-real 75-real 80)
        else()
            set(${cuda_archs} 30-real 50-real 60-real 70-real 72-real 75)
        endif()
    else()
        if(CMAKE_CUDA_ARCHITECTURES)
            set(${cuda_archs} ${CMAKE_CUDA_ARCHITECTURES})
            message(STATUS "Building with user-provided architectures")
        else()
            file(WRITE
                "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                "
                #include <stdio.h>
                #include <cuda_runtime_api.h>
                int main() {
                    int n;
                    if (cudaGetDeviceCount(&n) == cudaSuccess) {
                        for (int i = 0; i < n; ++i) {
                            int major, minor;
                            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                                    i) == cudaSuccess &&
                                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                                    i) == cudaSuccess) {
                                if (i > 0) {
                                    printf(\";\");
                                }
                                printf(\"%d%d-real\", major, minor);
                            }
                        }
                    }
                    return 0;
                }
                ")

            try_run(
                DETECTION_RETURN_VALUE DETECTION_COMPILED
                "${CMAKE_CURRENT_BINARY_DIR}"
                "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                LINK_LIBRARIES CUDA::cudart
                RUN_OUTPUT_VARIABLE DETECTED_ARCHITECTURES)

            if(DETECTED_ARCHITECTURES)
                message(STATUS "Building with detected architectures")
                set(${cuda_archs} ${DETECTED_ARCHITECTURES})
            else()
                message(STATUS "Failed to detect architectures. Falling back to CMake's default architectures")
            endif()
        endif()
    endif()

    set(${cuda_archs} ${${cuda_archs}} PARENT_SCOPE)

endfunction()
