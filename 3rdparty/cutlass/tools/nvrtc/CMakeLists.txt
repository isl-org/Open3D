# Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

option(CUTLASS_NVRTC_ENABLE "Enable NVRTC support" OFF)

set(CUTLASS_NVRTC_HAS_CUDA_FP16 FALSE)

if (CUTLASS_NVRTC_ENABLE)
  # CUTLASS NVRTC target
  macro(add_nvrtc_headers BASE_DIR FILES)
    foreach(CUTLASS_FILE ${FILES})
      set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/nvrtc/${CUTLASS_FILE}")

      string(REPLACE "/" "_" VARIABLE_NAME ${CUTLASS_FILE})
      string(REPLACE "." "_" VARIABLE_NAME ${VARIABLE_NAME})

      add_custom_command(OUTPUT ${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND}
        -DFILE_IN="${BASE_DIR}/${CUTLASS_FILE}"
        -DFILE_OUT="${OUTPUT_FILE}"
        -DVARIABLE_NAME="${VARIABLE_NAME}"
        -P ${CMAKE_SOURCE_DIR}/CMake/bin2hex.cmake
        DEPENDS ${BASE_DIR}/${CUTLASS_FILE}
      )

      list(APPEND GENERATED_HEADER_FILES "${OUTPUT_FILE}")

      string(APPEND NVRTC_INCLUDES_HEADERS "#include <${OUTPUT_FILE}>\n")
      string(APPEND NVRTC_INCLUDES_STRINGS "  ${VARIABLE_NAME},\n")
      string(APPEND NVRTC_INCLUDES_NAMES "  \"${CUTLASS_FILE}\",\n")
    endforeach()
  endmacro()


  string(APPEND NVRTC_INCLUDES_STRINGS "char const *kCutlassHeaders[] = {\n")
  string(APPEND NVRTC_INCLUDES_NAMES "char const *kCutlassHeaderNames[] = {\n")

  add_nvrtc_headers(${CMAKE_SOURCE_DIR} "${CUTLASS_CORE};${CUTLASS_GEMM};${CUTLASS_UTIL};${CUTLASS_DEVICE};${CUTLASS_ARCH};${CUTLASS_LAYOUT_THREAD}")
  message("${CMAKE_CURRENT_SOURCE_DIR}/")
  add_nvrtc_headers("${CMAKE_CURRENT_SOURCE_DIR}/stdlib" "assert.h;stdint.h")
  if(CUTLASS_NVRTC_HAS_CUDA_FP16)
    add_nvrtc_headers("${CMAKE_CURRENT_SOURCE_DIR}/stdlib" "cuda_fp16.h;cuda_fp16.hpp")
  endif()

  string(APPEND NVRTC_INCLUDES_STRINGS "};\n")
  string(APPEND NVRTC_INCLUDES_NAMES "};\n")

  string(APPEND NVRTC_INCLUDES_STRINGS "const size_t kCutlassHeaderCount = sizeof(kCutlassHeaders) / sizeof(*kCutlassHeaders);\n")

  file(WRITE "${CMAKE_BINARY_DIR}/cutlass/nvrtc/environment.cpp"
    "#include <cutlass/nvrtc/environment.h>\n"
    "${NVRTC_INCLUDES_HEADERS}"
    "\n"
    "namespace cutlass {\n"
    "namespace nvrtc {\n"
    "\n"
    "${NVRTC_INCLUDES_STRINGS}"
    "\n"
    "${NVRTC_INCLUDES_NAMES}"
    "\n"
    "} // namespace nvrtc\n"
    "} // namespace cutlass\n"
  )

  set(GENERATED_SOURCE_FILES "${CMAKE_BINARY_DIR}/cutlass/nvrtc/environment.cpp")

  source_group("Generated\\Header Files" FILES ${GENERATED_HEADER_FILES})
  source_group("Generated\\Source Files" FILES ${GENERATED_SOURCE_FILES})

  # THere is no NVRTC library in FindCUDA. Use cublas as base.
  string(REPLACE "cublas" "nvrtc" CUDA_nvrtc_LIBRARY ${CUDA_cublas_LIBRARY})

  add_library(cutlass_nvrtc STATIC
    cutlass/nvrtc/environment.h
    ${GENERATED_SOURCE_FILES}
    ${GENERATED_HEADER_FILES}
  )

  target_include_directories(cutlass_nvrtc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_include_directories(cutlass_nvrtc PRIVATE ${CMAKE_BINARY_DIR})
  target_link_libraries(cutlass_nvrtc ${CUDA_nvrtc_LIBRARY})
endif()
