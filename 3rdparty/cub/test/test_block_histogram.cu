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
 * Test of BlockHistogram utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <string>
#include <typeinfo>

#include <cub/block/block_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * BlockHistogram test kernel.
 */
template <
    int                     BINS,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockHistogramAlgorithm ALGORITHM,
    typename                T,
    typename                HistoCounter>
__global__ void BlockHistogramKernel(
    T                       *d_samples,
    HistoCounter            *d_histogram)
{
    // Parameterize BlockHistogram type for our thread block
    typedef BlockHistogram<T, BLOCK_THREADS, ITEMS_PER_THREAD, BINS, ALGORITHM> BlockHistogram;

    // Allocate temp storage in shared memory
    __shared__ typename BlockHistogram::TempStorage temp_storage;

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_samples, data);

    // Test histo (writing directly to histogram buffer in global)
    BlockHistogram(temp_storage).Histogram(data, d_histogram);
}


/**
 * Initialize problem (and solution)
 */
template <
    int             BINS,
    typename        SampleT>
void Initialize(
    GenMode         gen_mode,
    SampleT         *h_samples,
    int             *h_histograms_linear,
    int             num_samples)
{
    // Init bins
    for (int bin = 0; bin < BINS; ++bin)
    {
        h_histograms_linear[bin] = 0;
    }

    if (g_verbose) printf("Samples: \n");

    // Initialize interleaved channel samples and histogram them correspondingly
    for (int i = 0; i < num_samples; ++i)
    {
        InitValue(gen_mode, h_samples[i], i);
        h_samples[i] %= BINS;

        if (g_verbose) std::cout << CoutCast(h_samples[i]) << ", ";

        h_histograms_linear[h_samples[i]]++;
    }

    if (g_verbose) printf("\n\n");
}


/**
 * Test BlockHistogram
 */
template <
    typename                    SampleT,
    int                         BINS,
    int                         BLOCK_THREADS,
    int                         ITEMS_PER_THREAD,
    BlockHistogramAlgorithm     ALGORITHM>
void Test(
    GenMode                     gen_mode)
{
    int num_samples = BLOCK_THREADS * ITEMS_PER_THREAD;

    printf("cub::BlockHistogram %s %d %s samples (%dB), %d bins, %d threads, gen-mode %s\n",
        (ALGORITHM == BLOCK_HISTO_SORT) ? "BLOCK_HISTO_SORT" : "BLOCK_HISTO_ATOMIC",
        num_samples,
        typeid(SampleT).name(),
        (int) sizeof(SampleT),
        BINS,
        BLOCK_THREADS,
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    SampleT         *h_samples          = new SampleT[num_samples];
    int   *h_reference = new int[BINS];

    // Initialize problem
    Initialize<BINS>(gen_mode, h_samples, h_reference, num_samples);

    // Allocate problem device arrays
    SampleT         *d_samples = NULL;
    int             *d_histogram = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples,             sizeof(SampleT) * num_samples));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram,   sizeof(int) * BINS));

    // Initialize/clear device arrays
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * num_samples, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_histogram, 0, sizeof(int) * BINS));

    // Run kernel
    BlockHistogramKernel<BINS, BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<1, BLOCK_THREADS>>>(
        d_samples,
        d_histogram);

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults((int*) h_reference, d_histogram, BINS, g_verbose, g_verbose);
    printf("\t%s\n\n", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (h_samples) delete[] h_samples;
    if (h_reference) delete[] h_reference;
    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_histogram) CubDebugExit(g_allocator.DeviceFree(d_histogram));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test different sample distributions
 */
template <
    typename                    SampleT,
    int                         BINS,
    int                         BLOCK_THREADS,
    int                         ITEMS_PER_THREAD,
    BlockHistogramAlgorithm     ALGORITHM>
void Test()
{
    Test<SampleT, BINS, BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>(UNIFORM);
    Test<SampleT, BINS, BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>(INTEGER_SEED);
    Test<SampleT, BINS, BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>(RANDOM);
}


/**
 * Test different ALGORITHM
 */
template <
    typename                    SampleT,
    int                         BINS,
    int                         BLOCK_THREADS,
    int                         ITEMS_PER_THREAD>
void Test()
{
    Test<SampleT, BINS, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_HISTO_SORT>();
    Test<SampleT, BINS, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_HISTO_ATOMIC>();
}


/**
 * Test different ITEMS_PER_THREAD
 */
template <
    typename                    SampleT,
    int                         BINS,
    int                         BLOCK_THREADS>
void Test()
{
    Test<SampleT, BINS, BLOCK_THREADS, 1>();
    Test<SampleT, BINS, BLOCK_THREADS, 5>();
}


/**
 * Test different BLOCK_THREADS
 */
template <
    typename                    SampleT,
    int                         BINS>
void Test()
{
    Test<SampleT, BINS, 32>();
    Test<SampleT, BINS, 96>();
    Test<SampleT, BINS, 128>();
}





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
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<total input samples across all channels> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

#ifdef QUICK_TEST

    // Compile/run quick tests
    Test<unsigned char, 256, 128, 4, BLOCK_HISTO_SORT>(RANDOM);
    Test<unsigned char, 256, 128, 4, BLOCK_HISTO_ATOMIC>(RANDOM);

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        Test<unsigned char, 32>();
        Test<unsigned char, 256>();
        Test<unsigned short, 1024>();
    }

#endif

    return 0;
}



