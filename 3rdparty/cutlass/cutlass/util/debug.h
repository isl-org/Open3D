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

/**
 * \file
 * \brief Debugging and logging functionality
 */

#include <stdio.h>

namespace cutlass {

/******************************************************************************
 * Debug and logging macros
 ******************************************************************************/

/**
 * Formats and prints the given message to stdout
 */
#if !defined(CUDA_LOG)
#if !defined(__CUDA_ARCH__)
#define CUDA_LOG(format, ...) printf(format, __VA_ARGS__)
#else
#define CUDA_LOG(format, ...)                              \
  printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, \
         blockIdx.x,                                       \
         blockIdx.y,                                       \
         blockIdx.z,                                       \
         threadIdx.x,                                      \
         threadIdx.y,                                      \
         threadIdx.z,                                      \
         __VA_ARGS__);
#endif
#endif

/**
 * Formats and prints the given message to stdout only if DEBUG is defined
 */
#if !defined(CUDA_LOG_DEBUG)
#ifdef DEBUG
#define CUDA_LOG_DEBUG(format, ...) CUDA_LOG(format, __VA_ARGS__)
#else
#define CUDA_LOG_DEBUG(format, ...)
#endif
#endif

/**
 * \brief The corresponding error message is printed to \p stderr (or \p stdout in device code)
 * along with the supplied source context.
 *
 * \return The CUDA error.
 */
__host__ CUTLASS_DEVICE cudaError_t cuda_perror_impl(cudaError_t error,
                                                     const char* filename,
                                                     int line) {
  (void)filename;
  (void)line;
  if (error) {
#if !defined(__CUDA_ARCH__)
    fprintf(
        stderr, "CUDA error %d [%s, %d]: %s\n", error, filename, line, cudaGetErrorString(error));
    fflush(stderr);
#else
    printf("CUDA error %d [%s, %d]\n", error, filename, line);
#endif
  }
  return error;
}

/**
 * \brief Perror macro
 */
#ifndef CUDA_PERROR
#define CUDA_PERROR(e) cuda_perror_impl((cudaError_t)(e), __FILE__, __LINE__)
#endif

/**
 * \brief Perror macro with exit
 */
#ifndef CUDA_PERROR_EXIT
#define CUDA_PERROR_EXIT(e)                                     \
  if (cuda_perror_impl((cudaError_t)(e), __FILE__, __LINE__)) { \
    exit(1);                                                    \
  }
#endif

/**
 * \brief Perror macro only if DEBUG is defined
 */
#ifndef CUDA_PERROR_DEBUG
#ifdef DEBUG
#define CUDA_PERROR_DEBUG(e) CUDA_PERROR(e)
#else
#define CUDA_PERROR_DEBUG(e) (e)
#endif
#endif

}  // namespace cutlass
