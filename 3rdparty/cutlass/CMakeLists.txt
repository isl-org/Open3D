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

cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)

set(CUTLASS_LANGUAGES CXX)

if( CUDA_COMPILER STREQUAL "clang" )
# CMake 3.9.0 has native support for CUDA without the need of the CUDA package. Use it!
elseif(WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.9.0")
  list(APPEND CUTLASS_LANGUAGES CUDA)
  set(CUTLASS_NATIVE_CUDA TRUE)

  macro(cutlass_add_executable)
    add_executable(${ARGN})
  endmacro()
else()
  # FindCUDA fails to detect VS 2017 due to a changed directory format of the toolkits.
  # For this configuration we need CMake >= 3.9.0 to use the native CUDA support.
  if (WIN32 AND MSVC_VERSION GREATER 1800)
    message(SEND_ERROR "Please upgrade CMake to version >= 3.9.0 to support Visual Studio 2017 or higher")
    cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
  endif()

  # Fall back to the FindCUDA version to create an executable with CUDA files
  macro(cutlass_add_executable)
    cuda_add_executable(${ARGN})
  endmacro()
endif()

project(CUTLASS ${CUTLASS_LANGUAGES})

if( CUDA_COMPILER STREQUAL "clang" )
  if( NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
    message(FATAL_ERROR "C++ compiler must be Clang. Currently it's ${CMAKE_CXX_COMPILER_ID}" )
  endif()
  string(APPEND CLANG_FLAGS " --std=c++11")
  string(APPEND CLANG_FLAGS " --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
  string(APPEND CLANG_FLAGS " -mllvm -pragma-unroll-threshold=100000")
  string(APPEND CLANG_FLAGS " -mllvm -unroll-threshold=5000")
  string(APPEND CLANG_FLAGS " -Wno-unused-command-line-argument")
  # needed for libcublasLt.so in case it's installed in the same location as libcudart.so
  # dynamic linker can find it if linker sets RPATH (forced by --disable-new-tags)
  # Otherwise linker uses RUNPATH and that does not propagate to loaded libs.
  string(APPEND CLANG_FLAGS " -Wl,--disable-new-dtags")

  link_libraries(${CUDA_CUDART_LIBRARY})
  # Treat CUDA files as C++ files
  macro(cutlass_add_executable)
    foreach(File ${ARGN})
      if(${File} MATCHES ".*\.cu$")
        set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
      endif()
    endforeach()
    add_executable(${ARGN})
  endmacro()
endif()

# check if the configuration is supported
if( NOT CMAKE_SIZEOF_VOID_P EQUAL 8 )
    message(FATAL_ERROR "CUTLASS requires a 64-bit compiler!")
endif()

find_package(CUDA REQUIRED)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
# Some platforms (e.g. Visual Studio) don't add the CUDA include directories to the system include
# paths by default, so we add it explicitly here.

find_package(Doxygen QUIET)

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

#
# Conditionally enable cuBLAS
#
set(CUTLASS_ENABLE_CUBLAS ON CACHE BOOL "Enable CUTLASS Tests to build with cuBLAS library.")

if(CUTLASS_ENABLE_CUBLAS)

  find_library(CUBLAS_LIBRARY cublas HINTS
                                   ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                                   ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
endif()


# By default we want to build in Release mode to ensure that we're getting best performance
if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  # We do support Debug or Release builds
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

if(WIN32)
  # On Windows we link against the shared (DLL) runtime. Change gtest settings to match this.
  set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib" FORCE)
endif()

if (WIN32)
  # Enable more warnings and treat as errors
  string(APPEND NVCC_FLAGS " -Xcompiler /W3 -Xcompiler /WX")

  # Disable warning on Unicode characters
  string(APPEND NVCC_FLAGS " -Xcompiler /wd4819")

  # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
  string(APPEND NVCC_FLAGS " -Xcompiler /fp:strict")

  # Verbose option
  if (${CUTLASS_NVCC_VERBOSE})
    string(APPEND NVCC_FLAGS " -v")
  endif()
endif(WIN32)

set(CUTLASS_NVCC_ARCHS_DEFAULT "")
if(NOT CUDA_VERSION VERSION_LESS 7.5)
  list(APPEND CUTLASS_NVCC_ARCHS_DEFAULT 50)
endif()
if(NOT CUDA_VERSION VERSION_LESS 8.0)
  list(APPEND CUTLASS_NVCC_ARCHS_DEFAULT 60 61)
endif()
if(NOT CUDA_VERSION VERSION_LESS 9.0)
  list(APPEND CUTLASS_NVCC_ARCHS_DEFAULT 70)
endif()
if(NOT CUDA_VERSION VERSION_LESS 9.2)
  list(APPEND CUTLASS_NVCC_ARCHS_DEFAULT 72)
endif()
if(NOT CUDA_VERSION VERSION_LESS 10.0)
  list(APPEND CUTLASS_NVCC_ARCHS_DEFAULT 75)
endif()
set(CUTLASS_NVCC_ARCHS ${CUTLASS_NVCC_ARCHS_DEFAULT} CACHE STRING "The SM architectures to build code for.")

set(CUTLASS_NVCC_EMBED_CUBIN ON CACHE BOOL "Embed compiled CUDA kernel binaries into executables.")
set(CUTLASS_NVCC_EMBED_PTX ON CACHE BOOL "Embed compiled PTX into executables.")
set(CUTLASS_NVCC_KEEP OFF CACHE BOOL "Keep intermediate files generated by NVCC.")

# CUDA 10.1 introduces "mma" in PTX performing collective matrix multiply operations.
if (CUDA_VERSION VERSION_LESS 10.1)
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT OFF)
else()
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT ON)
endif()

set(CUTLASS_ENABLE_TENSOR_CORE_MMA ${CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT} CACHE BOOL
  "Enable PTX mma instruction for collective matrix multiply operations.")

set(CUTLASS_EXHAUSTIVE_PERFORMANCE_TEST ${CUTLASS_EXHAUSTIVE_PERFORMANCE_TEST} CACHE BOOL 
  "Enable more kernels instantiated in the perf suite. This might result in longer compiler time. ")

#
# NOTE: running with asan and CUDA requires the following environment variable:
#
#  ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0
#
# without the above environment setting, an error like the following may be generated:
#
#  *** Error: Could not detect active GPU device ID [out of memory]
#  ...
#  ==9149==ERROR: LeakSanitizer: detected memory leaks
#  ...
#
if(ENABLE_ASAN)  # https://github.com/google/sanitizers/wiki/AddressSanitizer
  string(APPEND NVCC_FLAGS " --compiler-options -fsanitize=address --compiler-options -fno-omit-frame-pointer")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
endif()

###################################################################################################
#
# Configure CUDA build options
#
###################################################################################################

# Set NVCC arguments
foreach(ARCH ${CUTLASS_NVCC_ARCHS})
  string(APPEND CLANG_FLAGS " --cuda-gpu-arch=sm_${ARCH}")
  if(CUTLASS_NVCC_EMBED_CUBIN)
    string(APPEND NVCC_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endif()
  if(CUTLASS_NVCC_EMBED_PTX)
    string(APPEND NVCC_FLAGS " -gencode arch=compute_${ARCH},code=compute_${ARCH}")
  endif()
endforeach()

if(CUTLASS_NVCC_EMBED_PTX)
  string(APPEND CLANG_FLAGS " --cuda-include-ptx=all")
endif()

if (CUTLASS_ENABLE_TENSOR_CORE_MMA)
  string(APPEND COMMON_FLAGS " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1")
endif()

if (CUTLASS_ENABLE_CUBLAS)
  string(APPEND COMMON_FLAGS " -DCUTLASS_ENABLE_CUBLAS=1")
endif()

if (CUTLASS_EXHAUSTIVE_PERFORMANCE_TEST)
    add_definitions(-DEXHAUSTIVE_PROF)
endif()

if (CUTLASS_NVCC_KEEP)
    string(APPEND NVCC_FLAGS " -keep")
    string(APPEND CLANG_FLAGS " -save-temps=obj")
endif()

if (WIN32 AND CUTLASS_NATIVE_CUDA)
  string(APPEND NVCC_FLAGS_RELEASE " -lineinfo")
else()
  string(APPEND NVCC_FLAGS " -lineinfo")
endif()
string(APPEND CLANG_FLAGS " -gmlt")

if (UNIX)
  string(APPEND NVCC_FLAGS " -Xcompiler -Wconversion")
endif()

string(APPEND COMMON_FLAGS_DEBUG " -g")
string(APPEND COMMON_FLAGS_RELWITHDEBINFO " -O3")
string(APPEND COMMON_FLAGS_RELEASE " -O3")

# define NDEBUG for release mode to disable assertions
string(APPEND NVCC_FLAGS_RELEASE " -DNDEBUG")

if( CUDA_COMPILER STREQUAL "clang" )
  set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} ${CLANG_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS_DEBUG}")
elseif (CUTLASS_NATIVE_CUDA)
  set(CMAKE_CUDA_FLAGS "${COMMON_FLAGS} ${NVCC_FLAGS}")
  set(CMAKE_CUDA_FLAGS_RELEASE "${COMMON_FLAGS_RELEASE}")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_CUDA_FLAGS_DEBUG "${COMMON_FLAGS_DEBUG}")
else()
  set(CUDA_NVCC_FLAGS "${COMMON_FLAGS} ${NVCC_FLAGS}")
  set(CUDA_NVCC_FLAGS_DEBUG ${COMMON_FLAGS_DEBUG})
  set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${COMMON_FLAGS_RELWITHDEBINFO})
  set(CUDA_NVCC_FLAGS_RELEASE ${COMMON_FLAGS_RELEASE})
endif()

#
# The following items should eventually be pushed into cutlass/CMakeLists.txt
#

# GLOB for CUTLASS header files. Should we use a static list instead?
file(GLOB CUTLASS_GEMM RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cutlass/gemm/*.h)
file(GLOB CUTLASS_UTIL RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cutlass/util/*.h)
file(GLOB CUTLASS_DEVICE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cutlass/device/*.h)
file(GLOB CUTLASS_CORE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cutlass/*.h)
file(GLOB CUTLASS_REDUCTION RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cutlass/reduction/*.h )
file(GLOB CUTLASS_LAYOUT_THREAD RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} cutlass/layout/thread/*.h)

###################################################################################################
#
# Define build targets
#
###################################################################################################

source_group("cutlass\\gemm" FILES ${CUTLASS_GEMM})
source_group("cutlass\\util" FILES ${CUTLASS_UTIL})
source_group("cutlass\\device" FILES ${CUTLASS_DEVICE})
source_group("cutlass\\reduction" FILES ${CUTLASS_REDUCTION})
source_group("cutlass\\layout\\thread" FILES ${CUTLASS_LAYOUT_THREAD})
source_group("cutlass" FILES ${CUTLASS_CORE})

add_library(CUTLASS INTERFACE)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Special policy introduced in CMake 3.13
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

target_sources(CUTLASS INTERFACE
  ${CUTLASS_GEMM}
  ${CUTLASS_UTIL}
  ${CUTLASS_DEVICE}
  ${CUTLASS_CORE}
  ${CUTLASS_REDUCTION}
  ${CUTLASS_LAYOUT_THREAD}
)

target_include_directories(CUTLASS INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Create a custom target to ensure that the CUTLASS sources are visible in an IDE
add_custom_target(cutlass_ide SOURCES
  ${CUTLASS_GEMM}
  ${CUTLASS_UTIL}
  ${CUTLASS_DEVICE}
  ${CUTLASS_CORE}
  ${CUTLASS_REDUCTION}
  ${CUTLASS_LAYOUT_THREAD}
)
# Doxygen is available. Generate documentation
if (DOXYGEN_FOUND)
    # DOT is available. Enable graph generation in the documentation
    if (DOXYGEN_DOT_EXECUTABLE)
        set(CUTLASS_ENABLE_DOXYGEN_DOT ON CACHE BOOL "Use dot to generate graphs in the doxygen documentation.")
    else()
        set(CUTLASS_ENABLE_DOXYGEN_DOT OFF CACHE BOOL "Use dot to generate graphs in the doxygen documentation." FORCE)
    endif()

    if (CUTLASS_ENABLE_DOXYGEN_DOT)
        set(HAVE_DOT "YES")
    else()
        set(HAVE_DOT "NO")
    endif()

    # Add custom target for Doxygen.
    add_custom_target(cutlass_docs ${CMAKE_COMMAND} -E env
        "DOT_PATH=${DOXYGEN_DOT_EXECUTABLE}"
        "HAVE_DOT=${HAVE_DOT}"
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
endif()

add_subdirectory(tools)
add_subdirectory(examples)
