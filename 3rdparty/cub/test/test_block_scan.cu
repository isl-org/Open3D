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
 * Test of BlockScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <limits>
#include <typeinfo>

#include <cub/block/block_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"


using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose       = false;
int                     g_repeat        = 0;
CachingDeviceAllocator  g_allocator(true);


/**
 * Primitive variant to test
 */
enum TestMode
{
    BASIC,
    AGGREGATE,
    PREFIX,
};


/**
 * Scan mode to test
 */
enum ScanMode
{
    EXCLUSIVE,
    INCLUSIVE
};


/**
 * \brief WrapperFunctor (for precluding test-specialized dispatch to *Sum variants)
 */
template<typename OpT>
struct WrapperFunctor
{
    OpT op;

    WrapperFunctor(OpT op) : op(op) {}

    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return op(a, b);
    }
};


/**
 * Stateful prefix functor
 */
template <
    typename T,
    typename ScanOpT>
struct BlockPrefixCallbackOp
{
    int     linear_tid;
    T       prefix;
    ScanOpT  scan_op;

    __device__ __forceinline__
    BlockPrefixCallbackOp(int linear_tid, T prefix, ScanOpT scan_op) :
        linear_tid(linear_tid),
        prefix(prefix),
        scan_op(scan_op)
    {}

    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        // For testing purposes
        T retval = (linear_tid == 0) ? prefix  : T();
        prefix = scan_op(prefix, block_aggregate);
        return retval;
    }
};


//---------------------------------------------------------------------
// Exclusive scan
//---------------------------------------------------------------------

/// Exclusive scan (BASIC, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data[0], data[0], initial_value, scan_op);
}

/// Exclusive scan (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, int ITEMS_PER_THREAD, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data, data, initial_value, scan_op);
}

/// Exclusive scan (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data[0], data[0], initial_value, scan_op, block_aggregate);
}

/// Exclusive scan (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, int ITEMS_PER_THREAD, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data, data, initial_value, scan_op, block_aggregate);
}

/// Exclusive scan (PREFIX, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data[0], data[0], scan_op, prefix_op);
}

/// Exclusive scan (PREFIX, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, int ITEMS_PER_THREAD, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data, data, scan_op, prefix_op);
}


//---------------------------------------------------------------------
// Exclusive sum
//---------------------------------------------------------------------

/// Exclusive sum (BASIC, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data[0], data[0]);
}

/// Exclusive sum (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data, data);
}

/// Exclusive sum (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data[0], data[0], block_aggregate);
}

/// Exclusive sum (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data, data, block_aggregate);
}

/// Exclusive sum (PREFIX, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data[0], data[0], prefix_op);
}

/// Exclusive sum (PREFIX, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data, data, prefix_op);
}


//---------------------------------------------------------------------
// Inclusive scan
//---------------------------------------------------------------------

/// Inclusive scan (BASIC, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.InclusiveScan(data[0], data[0], scan_op);
}

/// Inclusive scan (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, int ITEMS_PER_THREAD, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.InclusiveScan(data, data, scan_op);
}

/// Inclusive scan (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.InclusiveScan(data[0], data[0], scan_op, block_aggregate);
}

/// Inclusive scan (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, int ITEMS_PER_THREAD, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.InclusiveScan(data, data, scan_op, block_aggregate);
}

/// Inclusive scan (PREFIX, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.InclusiveScan(data[0], data[0], scan_op, prefix_op);
}

/// Inclusive scan (PREFIX, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, int ITEMS_PER_THREAD, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.InclusiveScan(data, data, scan_op, prefix_op);
}


//---------------------------------------------------------------------
// Inclusive sum
//---------------------------------------------------------------------

/// Inclusive sum (BASIC, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, Int2Type<true> is_primitive)
{
    block_scan.InclusiveSum(data[0], data[0]);
}

/// Inclusive sum (BASIC, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, Int2Type<true> is_primitive)
{
    block_scan.InclusiveSum(data, data);
}

/// Inclusive sum (AGGREGATE, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, Int2Type<true> is_primitive)
{
    block_scan.InclusiveSum(data[0], data[0], block_aggregate);
}

/// Inclusive sum (AGGREGATE, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<AGGREGATE> test_mode, Int2Type<true> is_primitive)
{
    block_scan.InclusiveSum(data, data, block_aggregate);
}

/// Inclusive sum (PREFIX, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, Int2Type<true> is_primitive)
{
    block_scan.InclusiveSum(data[0], data[0], prefix_op);
}

/// Inclusive sum (PREFIX, ITEMS_PER_THREAD)
template <typename BlockScanT, typename T, typename PrefixCallbackOp, int ITEMS_PER_THREAD>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[ITEMS_PER_THREAD], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<INCLUSIVE> scan_mode, Int2Type<PREFIX> test_mode, Int2Type<true> is_primitive)
{
    block_scan.InclusiveSum(data, data, prefix_op);
}



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * BlockScan test kernel.
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            T,
    typename            ScanOpT>
__launch_bounds__ (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)
__global__ void BlockScanKernel(
    T                   *d_in,
    T                   *d_out,
    T                   *d_aggregate,
    ScanOpT              scan_op,
    T                   initial_value,
    clock_t             *d_elapsed)
{
    const int BLOCK_THREADS     = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    const int TILE_SIZE         = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Parameterize BlockScan type for our thread block
    typedef BlockScan<T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockScanT;

    // Allocate temp storage in shared memory
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int linear_tid = RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    LoadDirectBlocked(linear_tid, d_in, data);

    __threadfence_block();      // workaround to prevent clock hoisting
    clock_t start = clock();
    __threadfence_block();      // workaround to prevent clock hoisting

    // Test scan
    T                                   block_aggregate;
    BlockScanT                          block_scan(temp_storage);
    BlockPrefixCallbackOp<T, ScanOpT>   prefix_op(linear_tid, initial_value, scan_op);

    DeviceTest(block_scan, data, initial_value, scan_op, block_aggregate, prefix_op,
        Int2Type<SCAN_MODE>(), Int2Type<TEST_MODE>(), Int2Type<Traits<T>::PRIMITIVE>());

    // Stop cycle timer
    __threadfence_block();      // workaround to prevent clock hoisting
    clock_t stop = clock();
    __threadfence_block();      // workaround to prevent clock hoisting

    // Store output
    StoreDirectBlocked(linear_tid, d_out, data);

    // Store block_aggregate
    if (TEST_MODE != BASIC)
        d_aggregate[linear_tid] = block_aggregate;

    // Store prefix
    if (TEST_MODE == PREFIX)
    {
        if (linear_tid == 0)
            d_out[TILE_SIZE] = prefix_op.prefix;
    }

    // Store time
    if (linear_tid == 0)
        *d_elapsed = (start > stop) ? start - stop : stop - start;
}



//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize exclusive-scan problem (and solution)
 */
template <typename T, typename ScanOpT>
T Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           *h_reference,
    int         num_items,
    ScanOpT     scan_op,
    T           initial_value,
    Int2Type<EXCLUSIVE>)
{
    InitValue(gen_mode, h_in[0], 0);

    T block_aggregate   = h_in[0];
    h_reference[0]      = initial_value;
    T inclusive         = scan_op(initial_value, h_in[0]);

    for (int i = 1; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        h_reference[i] = inclusive;
        inclusive = scan_op(inclusive, h_in[i]);
        block_aggregate = scan_op(block_aggregate, h_in[i]);
    }

    return block_aggregate;
}


/**
 * Initialize inclusive-scan problem (and solution)
 */
template <typename T, typename ScanOpT>
T Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           *h_reference,
    int         num_items,
    ScanOpT      scan_op,
    T           initial_value,
    Int2Type<INCLUSIVE>)
{
    InitValue(gen_mode, h_in[0], 0);

    T block_aggregate   = h_in[0];
    T inclusive         = scan_op(initial_value, h_in[0]);
    h_reference[0]      = inclusive;

    for (int i = 1; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        inclusive = scan_op(inclusive, h_in[i]);
        block_aggregate = scan_op(block_aggregate, h_in[i]);
        h_reference[i] = inclusive;
    }

    return block_aggregate;
}


/**
 * Test thread block scan.  (Specialized for sufficient resources)
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOpT,
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOpT             scan_op,
    T                   initial_value,
    Int2Type<true>      sufficient_resources)
{
    const int BLOCK_THREADS     = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    const int TILE_SIZE         = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate host arrays
    T *h_in = new T[TILE_SIZE];
    T *h_reference = new T[TILE_SIZE];
    T *h_aggregate = new T[BLOCK_THREADS];

    // Initialize problem
    T block_aggregate = Initialize(
        gen_mode,
        h_in,
        h_reference,
        TILE_SIZE,
        scan_op,
        initial_value,
        Int2Type<SCAN_MODE>());

    // Test reference block_aggregate is returned in all threads
    for (int i = 0; i < BLOCK_THREADS; ++i)
    {
        h_aggregate[i] = block_aggregate;
    }

    // Run kernel
    printf("Test-mode %d, gen-mode %d, policy %d, %s %s BlockScan, %d (%d,%d,%d) thread block threads, %d items per thread, %d tile size, %s (%d bytes) elements:\n",
        TEST_MODE, gen_mode, ALGORITHM,
        (SCAN_MODE == INCLUSIVE) ? "Inclusive" : "Exclusive", typeid(ScanOpT).name(),
        BLOCK_THREADS, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z,
        ITEMS_PER_THREAD,  TILE_SIZE,
        typeid(T).name(), (int) sizeof(T));
    fflush(stdout);

    // Initialize/clear device arrays
    T       *d_in = NULL;
    T       *d_out = NULL;
    T       *d_aggregate = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(unsigned long long)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * (TILE_SIZE + 2)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_aggregate, sizeof(T) * BLOCK_THREADS));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * (TILE_SIZE + 1)));
    CubDebugExit(cudaMemset(d_aggregate, 0, sizeof(T) * BLOCK_THREADS));

    // Display input problem data
    if (g_verbose)
    {
        printf("Input data: ");
        for (int i = 0; i < TILE_SIZE; i++)
        {
            std::cout << CoutCast(h_in[i]) << ", ";
        }
        printf("\n\n");
    }

    // Run block_aggregate/prefix kernel
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
    BlockScanKernel<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, ALGORITHM><<<1, block_dims>>>(
        d_in,
        d_out,
        d_aggregate,
        scan_op,
        initial_value,
        d_elapsed);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tScan results: ");
    int compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    if (TEST_MODE == AGGREGATE)
    {
        // Copy out and display block_aggregate
        printf("\tScan block aggregate: ");
        compare = CompareDeviceResults(h_aggregate, d_aggregate, BLOCK_THREADS, g_verbose, g_verbose);
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);
    }

    if (TEST_MODE == PREFIX)
    {
        // Copy out and display updated prefix
        printf("\tScan running total: ");
        T running_total = scan_op(initial_value, block_aggregate);
        compare = CompareDeviceResults(&running_total, d_out + TILE_SIZE, 1, g_verbose, g_verbose);
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);
    }

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_aggregate) delete[] h_aggregate;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_aggregate) CubDebugExit(g_allocator.DeviceFree(d_aggregate));
    if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}


/**
 * Test thread block scan.  (Specialized for insufficient resources)
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOpT,
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOpT             scan_op,
    T                   initial_value,
    Int2Type<false>     sufficient_resources)
{}


/**
 * Test thread block scan.
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOpT,
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOpT             scan_op,
    T                   initial_value)
{
    // Check size of smem storage for the target arch to make sure it will fit
    typedef BlockScan<T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockScanT;

    enum
    {
#if defined(SM100) || defined(SM110) || defined(SM130)
        sufficient_smem         = (sizeof(typename BlockScanT::TempStorage)     <= 16 * 1024),
        sufficient_threads      = ((BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)    <= 512),
#else
        sufficient_smem         = (sizeof(typename BlockScanT::TempStorage)     <= 16 * 1024),
        sufficient_threads      = ((BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)    <= 1024),
#endif

#if defined(_WIN32) || defined(_WIN64)
        // Accommodate ptxas crash bug (access violation) on Windows
        special_skip            = ((TEST_ARCH <= 130) && (Equals<T, TestBar>::VALUE) && (BLOCK_DIM_Z > 1)),
#else
        special_skip            = false,
#endif
        sufficient_resources    = (sufficient_smem && sufficient_threads && !special_skip),
    };

    Test<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, ALGORITHM>(
        gen_mode, scan_op, initial_value, Int2Type<sufficient_resources>());
}



/**
 * Run test for different thread block dimensions
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOpT,
    typename            T>
void Test(
    GenMode     gen_mode,
    ScanOpT     scan_op,
    T           initial_value)
{
    Test<BLOCK_THREADS, 1, 1, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, ALGORITHM>(gen_mode, scan_op, initial_value);
    Test<BLOCK_THREADS, 2, 2, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, ALGORITHM>(gen_mode, scan_op, initial_value);
}


/**
 * Run test for different policy types
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    ScanMode    SCAN_MODE,
    TestMode    TEST_MODE,
    typename    ScanOpT,
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOpT     scan_op,
    T           initial_value)
{
#ifdef TEST_RAKING
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, BLOCK_SCAN_RAKING>(gen_mode, scan_op, initial_value);
#endif
#ifdef TEST_RAKING_MEMOIZE
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, BLOCK_SCAN_RAKING_MEMOIZE>(gen_mode, scan_op, initial_value);
#endif
#ifdef TEST_WARP_SCANS
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, BLOCK_SCAN_WARP_SCANS>(gen_mode, scan_op, initial_value);
#endif
}


/**
 * Run tests for different primitive variants
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    ScanOpT,
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOpT     scan_op,
    T           identity,
    T           initial_value)
{
    // Exclusive (use identity as initial value because it will dispatch to *Sum variants that don't take initial values)
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, EXCLUSIVE, BASIC>(gen_mode, scan_op, identity);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, EXCLUSIVE, AGGREGATE>(gen_mode, scan_op, identity);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, EXCLUSIVE, PREFIX>(gen_mode, scan_op, identity);

    // Exclusive (non-specialized, so we can use initial-value)
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, EXCLUSIVE, BASIC>(gen_mode, WrapperFunctor<ScanOpT>(scan_op), initial_value);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, EXCLUSIVE, AGGREGATE>(gen_mode, WrapperFunctor<ScanOpT>(scan_op), initial_value);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, EXCLUSIVE, PREFIX>(gen_mode, WrapperFunctor<ScanOpT>(scan_op), initial_value);

    // Inclusive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, INCLUSIVE, BASIC>(gen_mode, scan_op, identity);      // This scan doesn't take an initial value
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, INCLUSIVE, AGGREGATE>(gen_mode, scan_op, identity);  // This scan doesn't take an initial value
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, INCLUSIVE, PREFIX>(gen_mode, scan_op, initial_value);
}


/**
 * Run tests for different problem-generation options
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    ScanOpT,
    typename    T>
void Test(
    ScanOpT     scan_op,
    T           identity,
    T           initial_value)
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(UNIFORM, scan_op, identity, initial_value);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(INTEGER_SEED, scan_op, identity, initial_value);

    // Don't test randomly-generated floats b/c of stability
    if (Traits<T>::CATEGORY != FLOATING_POINT)
        Test<BLOCK_THREADS, ITEMS_PER_THREAD>(RANDOM, scan_op, identity, initial_value);
}


/**
 * Run tests for different data types and scan ops
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
void Test()
{
    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    // primitive
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned char) 0, (unsigned char) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned short) 0, (unsigned short) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned int) 0, (unsigned int) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (unsigned long long) 0, (unsigned long long) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), (float) 0, (float) 99);

    // primitive (alternative scan op)
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<char>::min(), (char) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<short>::min(), (short) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<int>::min(), (int) 99);
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<long long>::min(), (long long) 99);

    if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
        Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Max(), std::numeric_limits<double>::max() * -1, (double) 99);

    // vec-1
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_uchar1(0), make_uchar1(17));

    // vec-2
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_uchar2(0, 0), make_uchar2(17, 21));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_ushort2(0, 0), make_ushort2(17, 21));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_uint2(0, 0), make_uint2(17, 21));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_ulonglong2(0, 0), make_ulonglong2(17, 21));

    // vec-4
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_char4(0, 0, 0, 0), make_char4(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_short4(0, 0, 0, 0), make_short4(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_int4(0, 0, 0, 0), make_int4(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), make_longlong4(0, 0, 0, 0), make_longlong4(17, 21, 32, 85));

    // complex
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), TestFoo::MakeTestFoo(0, 0, 0, 0), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<BLOCK_THREADS, ITEMS_PER_THREAD>(Sum(), TestBar(0, 0), TestBar(17, 21));

}


/**
 * Run tests for different items per thread
 */
template <int BLOCK_THREADS>
void Test()
{
    Test<BLOCK_THREADS, 1>();
    Test<BLOCK_THREADS, 2>();
    Test<BLOCK_THREADS, 9>();
}



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
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

#ifdef QUICK_TEST

    Test<128, 1, 1, 1, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), int(0));

    // Compile/run quick tests
    Test<128, 1, 1, 4, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), int(0));
    Test<128, 1, 1, 4, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_RAKING>(UNIFORM, Sum(), int(0));
    Test<128, 1, 1, 4, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_RAKING_MEMOIZE>(UNIFORM, Sum(), int(0));

    Test<128, 1, 1, 2, INCLUSIVE, PREFIX, BLOCK_SCAN_RAKING>(INTEGER_SEED, Sum(), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<128, 1, 1, 1, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), make_longlong4(17, 21, 32, 85));


#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Run tests for different thread block sizes
        Test<17>();
        Test<32>();
        Test<62>();
        Test<65>();
//            Test<96>();             // TODO: file bug for UNREACHABLE error for Test<96, 9, BASIC, BLOCK_SCAN_RAKING>(UNIFORM, Sum(), NullType(), make_ulonglong2(17, 21));
        Test<128>();
    }

#endif

    return 0;
}




