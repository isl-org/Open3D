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
 * Simple demonstration of cub::BlockScan
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_block_scan.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <stdio.h>
#include <iostream>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

/// Verbose output
bool g_verbose = false;

/// Timing iterations
int g_timing_iterations = 100;

/// Default grid size
int g_grid_size = 1;



//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for performing a block-wide exclusive prefix sum over integers
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockScanAlgorithm      ALGORITHM>
__global__ void BlockPrefixSumKernel(
    int         *d_in,          // Tile of input
    int         *d_out,         // Tile of output
    clock_t     *d_elapsed)     // Elapsed cycle count of block scan
{
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

    // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

    // Specialize BlockScan type for our thread block
    typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;

    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage    load;
        typename BlockStoreT::TempStorage   store;
        typename BlockScanT::TempStorage    scan;
    } temp_storage;

    // Per-thread tile data
    int data[ITEMS_PER_THREAD];

    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(d_in, data);

    // Barrier for smem reuse
    __syncthreads();

    // Start cycle timer
    clock_t start = clock();

    // Compute exclusive prefix sum
    int aggregate;
    BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

    // Stop cycle timer
    clock_t stop = clock();

    // Barrier for smem reuse
    __syncthreads();

    // Store items from a blocked arrangement
    BlockStoreT(temp_storage.store).Store(d_out, data);

    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
        d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
    }
}



//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/**
 * Initialize exclusive prefix sum problem (and solution).
 * Returns the aggregate
 */
int Initialize(
    int *h_in,
    int *h_reference,
    int num_items)
{
    int inclusive = 0;

    for (int i = 0; i < num_items; ++i)
    {
        h_in[i] = i % 17;

        h_reference[i] = inclusive;
        inclusive += h_in[i];
    }

    return inclusive;
}


/**
 * Test thread block scan
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockScanAlgorithm  ALGORITHM>
void Test()
{
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate host arrays
    int *h_in           = new int[TILE_SIZE];
    int *h_reference    = new int[TILE_SIZE];
    int *h_gpu          = new int[TILE_SIZE + 1];

    // Initialize problem and reference output on host
    int h_aggregate = Initialize(h_in, h_reference, TILE_SIZE);

    // Initialize device arrays
    int *d_in           = NULL;
    int *d_out          = NULL;
    clock_t *d_elapsed  = NULL;
    cudaMalloc((void**)&d_in,          sizeof(int) * TILE_SIZE);
    cudaMalloc((void**)&d_out,         sizeof(int) * (TILE_SIZE + 1));
    cudaMalloc((void**)&d_elapsed,     sizeof(clock_t));

    // Display input problem data
    if (g_verbose)
    {
        printf("Input data: ");
        for (int i = 0; i < TILE_SIZE; i++)
            printf("%d, ", h_in[i]);
        printf("\n\n");
    }

    // Kernel props
    int max_sm_occupancy;
    CubDebugExit(MaxSmOccupancy(max_sm_occupancy, BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>, BLOCK_THREADS));

    // Copy problem to device
    cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

    printf("BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n",
        (ALGORITHM == BLOCK_SCAN_RAKING) ? "BLOCK_SCAN_RAKING" : (ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE) ? "BLOCK_SCAN_RAKING_MEMOIZE" : "BLOCK_SCAN_WARP_SCANS",
        TILE_SIZE, g_timing_iterations, g_grid_size, BLOCK_THREADS, ITEMS_PER_THREAD, max_sm_occupancy);

    // Run aggregate/prefix kernel
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<g_grid_size, BLOCK_THREADS>>>(
        d_in,
        d_out,
        d_elapsed);

    // Check results
    printf("\tOutput items: ");
    int compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check total aggregate
    printf("\tAggregate: ");
    compare = CompareDeviceResults(&h_aggregate, d_out + TILE_SIZE, 1, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Run this several times and average the performance results
    GpuTimer    timer;
    float       elapsed_millis          = 0.0;
    clock_t     elapsed_clocks          = 0;

    for (int i = 0; i < g_timing_iterations; ++i)
    {
        // Copy problem to device
        cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

        timer.Start();

        // Run aggregate/prefix kernel
        BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<g_grid_size, BLOCK_THREADS>>>(
            d_in,
            d_out,
            d_elapsed);

        timer.Stop();
        elapsed_millis += timer.ElapsedMillis();

        // Copy clocks from device
        clock_t clocks;
        CubDebugExit(cudaMemcpy(&clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost));
        elapsed_clocks += clocks;

    }

    // Check for kernel errors and STDIO from the kernel, if any
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Display timing results
    float avg_millis            = elapsed_millis / g_timing_iterations;
    float avg_items_per_sec     = float(TILE_SIZE * g_grid_size) / avg_millis / 1000.0f;
    float avg_clocks            = float(elapsed_clocks) / g_timing_iterations;
    float avg_clocks_per_item   = avg_clocks / TILE_SIZE;

    printf("\tAverage BlockScan::Sum clocks: %.3f\n", avg_clocks);
    printf("\tAverage BlockScan::Sum clocks per item: %.3f\n", avg_clocks_per_item);
    printf("\tAverage kernel millis: %.4f\n", avg_millis);
    printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_gpu) delete[] h_gpu;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_elapsed) cudaFree(d_elapsed);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("grid-size", g_grid_size);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--i=<timing iterations (default:%d)>]"
            "[--grid-size=<grid size (default:%d)>]"
            "[--v] "
            "\n", argv[0], g_timing_iterations, g_grid_size);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Run tests
    Test<1024, 1, BLOCK_SCAN_RAKING>();
    Test<512, 2, BLOCK_SCAN_RAKING>();
    Test<256, 4, BLOCK_SCAN_RAKING>();
    Test<128, 8, BLOCK_SCAN_RAKING>();
    Test<64, 16, BLOCK_SCAN_RAKING>();
    Test<32, 32, BLOCK_SCAN_RAKING>();

    printf("-------------\n");

    Test<1024, 1, BLOCK_SCAN_RAKING_MEMOIZE>();
    Test<512, 2, BLOCK_SCAN_RAKING_MEMOIZE>();
    Test<256, 4, BLOCK_SCAN_RAKING_MEMOIZE>();
    Test<128, 8, BLOCK_SCAN_RAKING_MEMOIZE>();
    Test<64, 16, BLOCK_SCAN_RAKING_MEMOIZE>();
    Test<32, 32, BLOCK_SCAN_RAKING_MEMOIZE>();

    printf("-------------\n");

    Test<1024, 1, BLOCK_SCAN_WARP_SCANS>();
    Test<512, 2, BLOCK_SCAN_WARP_SCANS>();
    Test<256, 4, BLOCK_SCAN_WARP_SCANS>();
    Test<128, 8, BLOCK_SCAN_WARP_SCANS>();
    Test<64, 16, BLOCK_SCAN_WARP_SCANS>();
    Test<32, 32, BLOCK_SCAN_WARP_SCANS>();


    return 0;
}

