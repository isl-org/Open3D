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
 * Test of BlockLoad and BlockStore utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;
CachingDeviceAllocator  g_allocator(true);


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/**
 * Test load/store kernel.
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    typename            InputIteratorT,
    typename            OutputIteratorT>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    InputIteratorT    d_in,
    OutputIteratorT    d_out_unguarded,
    OutputIteratorT    d_out_guarded,
    int               num_items)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    // Threadblock load/store abstraction types
    typedef BlockLoad<InputT, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM> BlockLoad;
    typedef BlockStore<OutputT, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_ALGORITHM> BlockStore;

    // Shared memory type for this thread block
    union TempStorage
    {
        typename BlockLoad::TempStorage     load;
        typename BlockStore::TempStorage    store;
    };

    // Allocate temp storage in shared memory
    __shared__ TempStorage temp_storage;

    // Threadblock work bounds
    int block_offset = blockIdx.x * TILE_SIZE;
    int guarded_elements = num_items - block_offset;

    // Tile of items
    OutputT data[ITEMS_PER_THREAD];

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_unguarded + block_offset, data);

    __syncthreads();

    // reset data
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        data[ITEM] = OutputT();

    __syncthreads();

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data, guarded_elements);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_guarded + block_offset, data, guarded_elements);
}


//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------


/**
 * Test load/store variants
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    typename            InputIteratorT,
    typename            OutputIteratorT>
void TestKernel(
    T                   *h_in,
    InputIteratorT      d_in,
    OutputIteratorT      d_out_unguarded_itr,
    OutputIteratorT      d_out_guarded_itr,
    T                   *d_out_unguarded_ptr,
    T                   *d_out_guarded_ptr,
    int                 grid_size,
    int                 guarded_elements)
{
    int compare;

    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;

    // Test with discard output iterator
    typedef typename std::iterator_traits<InputIteratorT>::difference_type OffsetT;
    DiscardOutputIterator<OffsetT> discard_itr;

    Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM>
        <<<grid_size, BLOCK_THREADS>>>(
            d_in,
            discard_itr,
            discard_itr,
            guarded_elements);

    // Test with regular output iterator
    Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM>
        <<<grid_size, BLOCK_THREADS>>>(
            d_in,
            d_out_unguarded_itr,
            d_out_guarded_itr,
            guarded_elements);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    compare = CompareDeviceResults(h_in, d_out_guarded_ptr, guarded_elements, g_verbose, g_verbose);
    printf("\tGuarded: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check results
    compare = CompareDeviceResults(h_in, d_out_unguarded_ptr, unguarded_elements, g_verbose, g_verbose);
    printf("\tUnguarded: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);
}


/**
 * Test native pointer.  Specialized for sufficient resources
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM>
void TestNative(
    int                 grid_size,
    float               fraction_valid,
    Int2Type<true>      sufficient_resources)
{
    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;
    int guarded_elements = int(fraction_valid * float(unguarded_elements));

    // Allocate host arrays
    T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

    // Allocate device arrays
    T *d_in = NULL;
    T *d_out_unguarded = NULL;
    T *d_out_guarded = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_guarded, sizeof(T) * guarded_elements));
    CubDebugExit(cudaMemset(d_out_unguarded, 0, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMemset(d_out_guarded, 0, sizeof(T) * guarded_elements));

    // Initialize problem on host and device
    for (int i = 0; i < unguarded_elements; ++i)
    {
        InitValue(INTEGER_SEED, h_in[i], i);
    }
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * unguarded_elements, cudaMemcpyHostToDevice));

    printf("TestNative "
        "grid_size(%d) "
        "guarded_elements(%d) "
        "unguarded_elements(%d) "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "LOAD_ALGORITHM(%d) "
        "STORE_ALGORITHM(%d) "
        "sizeof(T)(%d)\n",
            grid_size, guarded_elements, unguarded_elements, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, (int) sizeof(T));

    TestKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM>(
        h_in,
        (T const *) d_in,   // Test const
        d_out_unguarded,
        d_out_guarded,
        d_out_unguarded,
        d_out_guarded,
        grid_size,
        guarded_elements);

    // Cleanup
    if (h_in) free(h_in);
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out_unguarded) CubDebugExit(g_allocator.DeviceFree(d_out_unguarded));
    if (d_out_guarded) CubDebugExit(g_allocator.DeviceFree(d_out_guarded));
}


/**
 * Test native pointer.  Specialized for insufficient resources
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM>
void TestNative(
    int                 grid_size,
    float               fraction_valid,
    Int2Type<false>      sufficient_resources)
{}


/**
 * Test iterator.  Specialized for sufficient resources.
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    CacheLoadModifier   LOAD_MODIFIER,
    CacheStoreModifier  STORE_MODIFIER>
void TestIterator(
    int                 grid_size,
    float               fraction_valid,
    Int2Type<true>      sufficient_resources)
{
    int unguarded_elements = grid_size * BLOCK_THREADS * ITEMS_PER_THREAD;
    int guarded_elements = int(fraction_valid * float(unguarded_elements));

    // Allocate host arrays
    T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

    // Allocate device arrays
    T *d_in = NULL;
    T *d_out_unguarded = NULL;
    T *d_out_guarded = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_guarded, sizeof(T) * guarded_elements));
    CubDebugExit(cudaMemset(d_out_unguarded, 0, sizeof(T) * unguarded_elements));
    CubDebugExit(cudaMemset(d_out_guarded, 0, sizeof(T) * guarded_elements));

    // Initialize problem on host and device
    for (int i = 0; i < unguarded_elements; ++i)
    {
        InitValue(INTEGER_SEED, h_in[i], i);
    }
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * unguarded_elements, cudaMemcpyHostToDevice));

    printf("TestIterator "
        "grid_size(%d) "
        "guarded_elements(%d) "
        "unguarded_elements(%d) "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "LOAD_ALGORITHM(%d) "
        "STORE_ALGORITHM(%d) "
        "LOAD_MODIFIER(%d) "
        "STORE_MODIFIER(%d) "
        "sizeof(T)(%d)\n",
            grid_size, guarded_elements, unguarded_elements, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

    TestKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM>(
        h_in,
        CacheModifiedInputIterator<LOAD_MODIFIER, T>(d_in),
        CacheModifiedOutputIterator<STORE_MODIFIER, T>(d_out_unguarded),
        CacheModifiedOutputIterator<STORE_MODIFIER, T>(d_out_guarded),
        d_out_unguarded,
        d_out_guarded,
        grid_size,
        guarded_elements);

    // Cleanup
    if (h_in) free(h_in);
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out_unguarded) CubDebugExit(g_allocator.DeviceFree(d_out_unguarded));
    if (d_out_guarded) CubDebugExit(g_allocator.DeviceFree(d_out_guarded));
}

/**
 * Test iterator.  Specialized for insufficient resources.
 */
template <
    typename            T,
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  LOAD_ALGORITHM,
    BlockStoreAlgorithm STORE_ALGORITHM,
    CacheLoadModifier   LOAD_MODIFIER,
    CacheStoreModifier  STORE_MODIFIER>
void TestIterator(
    int                 grid_size,
    float               fraction_valid,
    Int2Type<false>     sufficient_resources)
{}


/**
 * Evaluate different pointer access types
 */
template <
    typename                T,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockLoadAlgorithm      LOAD_ALGORITHM,
    BlockStoreAlgorithm     STORE_ALGORITHM>
void TestPointerType(
    int             grid_size,
    float           fraction_valid)
{
    // Threadblock load/store abstraction types
    typedef BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM> BlockLoad;
    typedef BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_ALGORITHM> BlockStore;

#if defined(SM100) || defined(SM110) || defined(SM130)
    static const bool sufficient_load_smem  = sizeof(typename BlockLoad::TempStorage)   <= 1024 * 16;
    static const bool sufficient_store_smem = sizeof(typename BlockStore::TempStorage)  <= 1024 * 16;
    static const bool sufficient_threads    = BLOCK_THREADS <= 512;
#else
    static const bool sufficient_load_smem  = sizeof(typename BlockLoad::TempStorage)   <= 1024 * 48;
    static const bool sufficient_store_smem = sizeof(typename BlockStore::TempStorage)  <= 1024 * 48;
    static const bool sufficient_threads    = BLOCK_THREADS <= 1024;
#endif

    static const bool sufficient_resources  = sufficient_load_smem && sufficient_store_smem && sufficient_threads;

    TestNative<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM>(grid_size, fraction_valid, Int2Type<sufficient_resources>());
    TestIterator<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, LOAD_DEFAULT, STORE_DEFAULT>(grid_size, fraction_valid, Int2Type<sufficient_resources>());
}


/**
 * Evaluate different time-slicing strategies
 */
template <
    typename                T,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockLoadAlgorithm      LOAD_ALGORITHM,
    BlockStoreAlgorithm     STORE_ALGORITHM>
void TestSlicedStrategy(
    int             grid_size,
    float           fraction_valid)
{
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, true>(grid_size, fraction_valid);
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM, STORE_ALGORITHM, false>(grid_size, fraction_valid);
}



/**
 * Evaluate different load/store strategies (specialized for block sizes that are not a multiple of 32)
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD>
void TestStrategy(
    int             grid_size,
    float           fraction_valid,
    Int2Type<false> is_warp_multiple)
{
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_DIRECT, BLOCK_STORE_DIRECT>(grid_size, fraction_valid);
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE, BLOCK_STORE_TRANSPOSE>(grid_size, fraction_valid);
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE, BLOCK_STORE_VECTORIZE>(grid_size, fraction_valid);
}


/**
 * Evaluate different load/store strategies (specialized for block sizes that are a multiple of 32)
 */
template <
    typename        T,
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD>
void TestStrategy(
    int             grid_size,
    float           fraction_valid,
    Int2Type<true>  is_warp_multiple)
{
    TestStrategy<T, BLOCK_THREADS, ITEMS_PER_THREAD>(grid_size, fraction_valid, Int2Type<false>());
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE>(grid_size, fraction_valid);
    TestPointerType<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED, BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED>(grid_size, fraction_valid);
}


/**
 * Evaluate different register blocking
 */
template <
    typename T,
    int BLOCK_THREADS>
void TestItemsPerThread(
    int grid_size,
    float fraction_valid)
{
    Int2Type<BLOCK_THREADS % 32 == 0> is_warp_multiple;

    TestStrategy<T, BLOCK_THREADS, 1>(grid_size, fraction_valid, is_warp_multiple);
    TestStrategy<T, BLOCK_THREADS, 3>(grid_size, fraction_valid, is_warp_multiple);
    TestStrategy<T, BLOCK_THREADS, 4>(grid_size, fraction_valid, is_warp_multiple);
    TestStrategy<T, BLOCK_THREADS, 11>(grid_size, fraction_valid, is_warp_multiple);
}


/**
 * Evaluate different thread block sizes
 */
template <typename T>
void TestThreads(
    int grid_size,
    float fraction_valid)
{
    TestItemsPerThread<T, 15>(grid_size, fraction_valid);
    TestItemsPerThread<T, 32>(grid_size, fraction_valid);
    TestItemsPerThread<T, 72>(grid_size, fraction_valid);
    TestItemsPerThread<T, 96>(grid_size, fraction_valid);
    TestItemsPerThread<T, 128>(grid_size, fraction_valid);
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef QUICK_TEST

    // Compile/run quick tests
    TestNative<     int, 64, 2, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE>(1, 0.8f, Int2Type<true>());
    TestIterator<   int, 64, 2, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE, LOAD_DEFAULT, STORE_DEFAULT>(1, 0.8f, Int2Type<true>());

#else

    // Compile/run thorough tests
    TestThreads<char>(2, 0.8f);
    TestThreads<int>(2, 0.8f);
    TestThreads<long>(2, 0.8f);
    TestThreads<long2>(2, 0.8f);

    if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
        TestThreads<double2>(2, 0.8f);
    TestThreads<TestFoo>(2, 0.8f);
    TestThreads<TestBar>(2, 0.8f);

#endif

    return 0;
}



