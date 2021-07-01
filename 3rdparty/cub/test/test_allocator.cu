/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Test evaluation for caching allocator of device memory
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>]"
            "[--bytes=<timing bytes>]"
            "[--i=<timing iterations>]"
            "\n", argv[0]);
        exit(0);
    }

#if (CUB_PTX_ARCH == 0)

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get number of GPUs and current GPU
    int num_gpus;
    int initial_gpu;
    int timing_iterations           = 10000;
    int timing_bytes                = 1024 * 1024;

    if (CubDebug(cudaGetDeviceCount(&num_gpus))) exit(1);
    if (CubDebug(cudaGetDevice(&initial_gpu))) exit(1);
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("bytes", timing_bytes);

    // Create default allocator (caches up to 6MB in device allocations per GPU)
    CachingDeviceAllocator allocator;
    allocator.debug = true;

    printf("Running single-gpu tests...\n"); fflush(stdout);

    //
    // Test0
    //

    // Create a new stream
    cudaStream_t other_stream;
    CubDebugExit(cudaStreamCreate(&other_stream));

    // Allocate 999 bytes on the current gpu in stream0
    char *d_999B_stream0_a;
    char *d_999B_stream0_b;
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));

    // Run some big kernel in stream 0
    EmptyKernel<void><<<32000, 512, 1024 * 8, 0>>>();

    // Free d_999B_stream0_a
    CubDebugExit(allocator.DeviceFree(d_999B_stream0_a));

    // Allocate another 999 bytes in stream 0
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have no cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Run some big kernel in stream 0
    EmptyKernel<void><<<32000, 512, 1024 * 8, 0>>>();

    // Free d_999B_stream0_b
    CubDebugExit(allocator.DeviceFree(d_999B_stream0_b));

    // Allocate 999 bytes on the current gpu in other_stream
    char *d_999B_stream_other_a;
    char *d_999B_stream_other_b;
    allocator.DeviceAllocate((void **) &d_999B_stream_other_a, 999, other_stream);

    // Check that that we have 1 live blocks on the initial GPU (that we allocated a new one because d_999B_stream0_b is only available for stream 0 until it becomes idle)
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have one cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    // Run some big kernel in other_stream
    EmptyKernel<void><<<32000, 512, 1024 * 8, other_stream>>>();

    // Free d_999B_stream_other
    CubDebugExit(allocator.DeviceFree(d_999B_stream_other_a));

    // Check that we can now use both allocations in stream 0 after synchronizing the device
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 2);

    // Check that that we have no cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Free d_999B_stream0_a and d_999B_stream0_b
    CubDebugExit(allocator.DeviceFree(d_999B_stream0_a));
    CubDebugExit(allocator.DeviceFree(d_999B_stream0_b));

    // Check that we can now use both allocations in other_stream
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream_other_a, 999, other_stream));
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream_other_b, 999, other_stream));

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 2);

    // Check that that we have no cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Run some big kernel in other_stream
    EmptyKernel<void><<<32000, 512, 1024 * 8, other_stream>>>();

    // Free d_999B_stream_other_a and d_999B_stream_other_b
    CubDebugExit(allocator.DeviceFree(d_999B_stream_other_a));
    CubDebugExit(allocator.DeviceFree(d_999B_stream_other_b));

    // Check that we can now use both allocations in stream 0 after synchronizing the device and destroying the other stream
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(cudaStreamDestroy(other_stream));
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));
    CubDebugExit(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 2);

    // Check that that we have no cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Free d_999B_stream0_a and d_999B_stream0_b
    CubDebugExit(allocator.DeviceFree(d_999B_stream0_a));
    CubDebugExit(allocator.DeviceFree(d_999B_stream0_b));

    // Free all cached
    CubDebugExit(allocator.FreeAllCached());

    //
    // Test1
    //

    // Allocate 5 bytes on the current gpu
    char *d_5B;
    CubDebugExit(allocator.DeviceAllocate((void **) &d_5B, 5));

    // Check that that we have zero free bytes cached on the initial GPU
    AssertEquals(allocator.cached_bytes[initial_gpu].free, 0);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    //
    // Test2
    //

    // Allocate 4096 bytes on the current gpu
    char *d_4096B;
    CubDebugExit(allocator.DeviceAllocate((void **) &d_4096B, 4096));

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 2);

    //
    // Test3
    //

    // DeviceFree d_5B
    CubDebugExit(allocator.DeviceFree(d_5B));

    // Check that that we have min_bin_bytes free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test4
    //

    // DeviceFree d_4096B
    CubDebugExit(allocator.DeviceFree(d_4096B));

    // Check that that we have the 4096 + min_bin free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes + 4096);

    // Check that that we have 0 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 0);

    // Check that that we have 2 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 2);

    //
    // Test5
    //

    // Allocate 768 bytes on the current gpu
    char *d_768B;
    CubDebugExit(allocator.DeviceAllocate((void **) &d_768B, 768));

    // Check that that we have the min_bin free bytes cached on the initial gpu (4096 was reused)
    AssertEquals(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test6
    //

    // Allocate max_cached_bytes on the current gpu
    char *d_max_cached;
    CubDebugExit(allocator.DeviceAllocate((void **) &d_max_cached, allocator.max_cached_bytes));

    // DeviceFree d_max_cached
    CubDebugExit(allocator.DeviceFree(d_max_cached));

    // Check that that we have the min_bin free bytes cached on the initial gpu (max cached was not returned because we went over)
    AssertEquals(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we still have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test7
    //

    // Free all cached blocks on all GPUs
    CubDebugExit(allocator.FreeAllCached());

    // Check that that we have 0 bytes cached on the initial GPU
    AssertEquals(allocator.cached_bytes[initial_gpu].free, 0);

    // Check that that we have 0 cached blocks across all GPUs
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Check that that still we have 1 live block across all GPUs
    AssertEquals(allocator.live_blocks.size(), 1);

    //
    // Test8
    //

    // Allocate max cached bytes + 1 on the current gpu
    char *d_max_cached_plus;
    CubDebugExit(allocator.DeviceAllocate((void **) &d_max_cached_plus, allocator.max_cached_bytes + 1));

    // DeviceFree max cached bytes
    CubDebugExit(allocator.DeviceFree(d_max_cached_plus));

    // DeviceFree d_768B
    CubDebugExit(allocator.DeviceFree(d_768B));

    unsigned int power;
    size_t rounded_bytes;
    allocator.NearestPowerOf(power, rounded_bytes, allocator.bin_growth, 768);

    // Check that that we have 4096 free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu].free, rounded_bytes);

    // Check that that we have 1 cached blocks across all GPUs
    AssertEquals(allocator.cached_blocks.size(), 1);

    // Check that that still we have 0 live block across all GPUs
    AssertEquals(allocator.live_blocks.size(), 0);

#ifndef CUB_CDP
    // BUG: find out why these tests fail when one GPU is CDP compliant and the other is not

    if (num_gpus > 1)
    {
        printf("\nRunning multi-gpu tests...\n"); fflush(stdout);

        //
        // Test9
        //

        // Allocate 768 bytes on the next gpu
        int next_gpu = (initial_gpu + 1) % num_gpus;
        char *d_768B_2;
        CubDebugExit(allocator.DeviceAllocate(next_gpu, (void **) &d_768B_2, 768));

        // DeviceFree d_768B on the next gpu
        CubDebugExit(allocator.DeviceFree(next_gpu, d_768B_2));

        // Re-allocate 768 bytes on the next gpu
        CubDebugExit(allocator.DeviceAllocate(next_gpu, (void **) &d_768B_2, 768));

        // Re-free d_768B on the next gpu
        CubDebugExit(allocator.DeviceFree(next_gpu, d_768B_2));

        // Check that that we have 4096 free bytes cached on the initial gpu
        AssertEquals(allocator.cached_bytes[initial_gpu].free, rounded_bytes);

        // Check that that we have 4096 free bytes cached on the second gpu
        AssertEquals(allocator.cached_bytes[next_gpu].free, rounded_bytes);

        // Check that that we have 2 cached blocks across all GPUs
        AssertEquals(allocator.cached_blocks.size(), 2);

        // Check that that still we have 0 live block across all GPUs
        AssertEquals(allocator.live_blocks.size(), 0);
    }
#endif  // CUB_CDP

    //
    // Performance
    //

    printf("\nCPU Performance (%d timing iterations, %d bytes):\n", timing_iterations, timing_bytes);
    fflush(stdout); fflush(stderr);

    // CPU performance comparisons vs cached.  Allocate and free a 1MB block 2000 times
    CpuTimer    cpu_timer;
    char        *d_1024MB                       = NULL;
    allocator.debug                             = false;

    // Prime the caching allocator and the kernel
    CubDebugExit(allocator.DeviceAllocate((void **) &d_1024MB, timing_bytes));
    CubDebugExit(allocator.DeviceFree(d_1024MB));
    cub::EmptyKernel<void><<<1, 32>>>();

    // CUDA
    cpu_timer.Start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        CubDebugExit(cudaMalloc((void **) &d_1024MB, timing_bytes));
        CubDebugExit(cudaFree(d_1024MB));
    }
    cpu_timer.Stop();
    float cuda_malloc_elapsed_millis = cpu_timer.ElapsedMillis();

    // CUB
    cpu_timer.Start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        CubDebugExit(allocator.DeviceAllocate((void **) &d_1024MB, timing_bytes));
        CubDebugExit(allocator.DeviceFree(d_1024MB));
    }
    cpu_timer.Stop();
    float cub_calloc_elapsed_millis = cpu_timer.ElapsedMillis();

    printf("\t CUB CachingDeviceAllocator allocation CPU speedup: %.2f (avg cudaMalloc %.4f ms vs. avg DeviceAllocate %.4f ms)\n",
        cuda_malloc_elapsed_millis / cub_calloc_elapsed_millis,
        cuda_malloc_elapsed_millis / timing_iterations,
        cub_calloc_elapsed_millis / timing_iterations);

    // GPU performance comparisons.  Allocate and free a 1MB block 2000 times
    GpuTimer gpu_timer;

    printf("\nGPU Performance (%d timing iterations, %d bytes):\n", timing_iterations, timing_bytes);
    fflush(stdout); fflush(stderr);

    // Kernel-only
    gpu_timer.Start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        cub::EmptyKernel<void><<<1, 32>>>();
    }
    gpu_timer.Stop();
    float cuda_empty_elapsed_millis = gpu_timer.ElapsedMillis();

    // CUDA
    gpu_timer.Start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        CubDebugExit(cudaMalloc((void **) &d_1024MB, timing_bytes));
        cub::EmptyKernel<void><<<1, 32>>>();
        CubDebugExit(cudaFree(d_1024MB));
    }
    gpu_timer.Stop();
    cuda_malloc_elapsed_millis = gpu_timer.ElapsedMillis() - cuda_empty_elapsed_millis;

    // CUB
    gpu_timer.Start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        CubDebugExit(allocator.DeviceAllocate((void **) &d_1024MB, timing_bytes));
        cub::EmptyKernel<void><<<1, 32>>>();
        CubDebugExit(allocator.DeviceFree(d_1024MB));
    }
    gpu_timer.Stop();
    cub_calloc_elapsed_millis = gpu_timer.ElapsedMillis() - cuda_empty_elapsed_millis;

    printf("\t CUB CachingDeviceAllocator allocation GPU speedup: %.2f (avg cudaMalloc %.4f ms vs. avg DeviceAllocate %.4f ms)\n",
        cuda_malloc_elapsed_millis / cub_calloc_elapsed_millis,
        cuda_malloc_elapsed_millis / timing_iterations,
        cub_calloc_elapsed_millis / timing_iterations);


#endif

    printf("Success\n");

    return 0;
}

