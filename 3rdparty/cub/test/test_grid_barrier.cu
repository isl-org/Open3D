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
 * Test evaluation for software global barrier throughput
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/grid/grid_barrier.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Kernel that iterates through the specified number of software global barriers
 */
__global__ void Kernel(
    GridBarrier global_barrier,
    int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        global_barrier.Sync();
    }
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;

    // Defaults
    int iterations = 10000;
    int block_size = 128;
    int grid_size = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);

    // Get args
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("grid-size", grid_size);
    args.GetCmdLineArgument("block-size", block_size);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>]"
            "[--i=<iterations>]"
            "[--grid-size<grid-size>]"
            "[--block-size<block-size>]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get device SM version
    int sm_version;
    CubDebugExit(SmVersion(sm_version, device_ordinal));

    // Get SM properties
    int sm_count, max_block_threads, max_sm_occupancy;
    CubDebugExit(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
    CubDebugExit(cudaDeviceGetAttribute(&max_block_threads, cudaDevAttrMaxThreadsPerBlock, device_ordinal));
    CubDebugExit(MaxSmOccupancy(max_sm_occupancy, EmptyKernel<void>, 32));

    // Compute grid size and occupancy
    int occupancy = CUB_MIN((max_block_threads / block_size), max_sm_occupancy);

    if (grid_size == -1)
    {
        grid_size = occupancy * sm_count;
    }
    else
    {
        occupancy = grid_size / sm_count;
    }

    printf("Initializing software global barrier for Kernel<<<%d,%d>>> with %d occupancy\n",
        grid_size, block_size, occupancy);
    fflush(stdout);

    // Init global barrier
    GridBarrierLifetime global_barrier;
    global_barrier.Setup(grid_size);

    // Time kernel
    GpuTimer gpu_timer;
    gpu_timer.Start();
    Kernel<<<grid_size, block_size>>>(global_barrier, iterations);
    gpu_timer.Stop();

    retval = CubDebug(cudaThreadSynchronize());

    // Output timing results
    float avg_elapsed = gpu_timer.ElapsedMillis() / float(iterations);
    printf("%d iterations, %f total elapsed millis, %f avg elapsed millis\n",
        iterations,
        gpu_timer.ElapsedMillis(),
        avg_elapsed);

    return retval;
}
