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
 * Test of WarpScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <cub/warp/warp_scan.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

static const int        NUM_WARPS       = 2;


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

//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/// Exclusive scan basic
template <typename WarpScanT, typename T, typename ScanOpT, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    T                               &initial_value,
    ScanOpT                         &scan_op,
    T                               &aggregate,
    Int2Type<BASIC>                 test_mode,
    IsPrimitiveT                    is_primitive)
{
    // Test basic warp scan
    warp_scan.ExclusiveScan(data, data, initial_value, scan_op);
}

/// Exclusive scan aggregate
template <
    typename    WarpScanT,
    typename    T,
    typename    ScanOpT,
    typename    IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    T                               &initial_value,
    ScanOpT                         &scan_op,
    T                               &aggregate,
    Int2Type<AGGREGATE>             test_mode,
    IsPrimitiveT                    is_primitive)
{
    // Test with cumulative aggregate
    warp_scan.ExclusiveScan(data, data, initial_value, scan_op, aggregate);
}


/// Exclusive sum basic
template <
    typename    WarpScanT,
    typename    T>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    T                               &initial_value,
    Sum                             &scan_op,
    T                               &aggregate,
    Int2Type<BASIC>                 test_mode,
    Int2Type<true>                  is_primitive)
{
    // Test basic warp scan
    warp_scan.ExclusiveSum(data, data);
}


/// Exclusive sum aggregate
template <
    typename    WarpScanT,
    typename    T>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    T                               &initial_value,
    Sum                             &scan_op,
    T                               &aggregate,
    Int2Type<AGGREGATE>             test_mode,
    Int2Type<true>                  is_primitive)
{
    // Test with cumulative aggregate
    warp_scan.ExclusiveSum(data, data, aggregate);
}


/// Inclusive scan basic
template <
    typename    WarpScanT,
    typename    T,
    typename    ScanOpT,
    typename    IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    NullType                        &initial_value,
    ScanOpT                         &scan_op,
    T                               &aggregate,
    Int2Type<BASIC>                 test_mode,
    IsPrimitiveT                    is_primitive)
{
    // Test basic warp scan
    warp_scan.InclusiveScan(data, data, scan_op);
}

/// Inclusive scan aggregate
template <
    typename    WarpScanT,
    typename    T,
    typename    ScanOpT,
    typename    IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    NullType                        &initial_value,
    ScanOpT                         &scan_op,
    T                               &aggregate,
    Int2Type<AGGREGATE>             test_mode,
    IsPrimitiveT                    is_primitive)
{
    // Test with cumulative aggregate
    warp_scan.InclusiveScan(data, data, scan_op, aggregate);
}

/// Inclusive sum basic
template <
    typename    WarpScanT,
    typename    T,
    typename    InitialValueT>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    NullType                        &initial_value,
    Sum                             &scan_op,
    T                               &aggregate,
    Int2Type<BASIC>                 test_mode,
    Int2Type<true>                  is_primitive)
{
    // Test basic warp scan
    warp_scan.InclusiveSum(data, data);
}

/// Inclusive sum aggregate
template <
    typename    WarpScanT,
    typename    T,
    typename    InitialValueT>
__device__ __forceinline__ void DeviceTest(
    WarpScanT                       &warp_scan,
    T                               &data,
    NullType                        &initial_value,
    Sum                             &scan_op,
    T                               &aggregate,
    Int2Type<AGGREGATE>             test_mode,
    Int2Type<true>                  is_primitive)
{
    // Test with cumulative aggregate
    warp_scan.InclusiveSum(data, data, aggregate);
}


/**
 * WarpScan test kernel
 */
template <
    int         LOGICAL_WARP_THREADS,
    TestMode    TEST_MODE,
    typename    T,
    typename    ScanOpT,
    typename    InitialValueT>
__global__ void WarpScanKernel(
    T               *d_in,
    T               *d_out,
    T               *d_aggregate,
    ScanOpT         scan_op,
    InitialValueT   initial_value,
    clock_t         *d_elapsed)
{
    // Cooperative warp-scan utility type (1 warp)
    typedef WarpScan<T, LOGICAL_WARP_THREADS> WarpScanT;

    // Allocate temp storage in shared memory
    __shared__ typename WarpScanT::TempStorage temp_storage[NUM_WARPS];

    // Get warp index
    int warp_id = threadIdx.x / LOGICAL_WARP_THREADS;

    // Per-thread tile data
    T data = d_in[threadIdx.x];

    // Start cycle timer
    __threadfence_block();      // workaround to prevent clock hoisting
    clock_t start = clock();
    __threadfence_block();      // workaround to prevent clock hoisting

    T aggregate;

    // Test scan
    WarpScanT warp_scan(temp_storage[warp_id]);
    DeviceTest(
        warp_scan,
        data,
        initial_value,
        scan_op,
        aggregate,
        Int2Type<TEST_MODE>(),
        Int2Type<Traits<T>::PRIMITIVE>());

    // Stop cycle timer
    __threadfence_block();      // workaround to prevent clock hoisting
    clock_t stop = clock();
    __threadfence_block();      // workaround to prevent clock hoisting

    // Store data
    d_out[threadIdx.x] = data;

    if (TEST_MODE != BASIC)
    {
        // Store aggregate
        d_aggregate[threadIdx.x] = aggregate;
    }

    // Store time
    if (threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
    }
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize exclusive-scan problem (and solution)
 */
template <
    typename        T,
    typename        ScanOpT>
void Initialize(
    GenMode         gen_mode,
    T               *h_in,
    T               *h_reference,
    int             logical_warp_items,
    ScanOpT         scan_op,
    T               initial_value,
    T               warp_aggregates[NUM_WARPS])
{
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        int base_idx = (w * logical_warp_items);
        int i = base_idx;

        InitValue(gen_mode, h_in[i], i);

        T warp_aggregate   = h_in[i];
        h_reference[i]      = initial_value;
        T inclusive         = scan_op(initial_value, h_in[i]);

        for (i = i + 1; i < base_idx + logical_warp_items; ++i)
        {
            InitValue(gen_mode, h_in[i], i);
            h_reference[i] = inclusive;
            inclusive = scan_op(inclusive, h_in[i]);
            warp_aggregate = scan_op(warp_aggregate, h_in[i]);
        }

        warp_aggregates[w] = warp_aggregate;
    }

}


/**
 * Initialize inclusive-scan problem (and solution)
 */
template <
    typename    T,
    typename    ScanOpT>
void Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           *h_reference,
    int         logical_warp_items,
    ScanOpT     scan_op,
    NullType,
    T           warp_aggregates[NUM_WARPS])
{
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        int base_idx = (w * logical_warp_items);
        int i = base_idx;

        InitValue(gen_mode, h_in[i], i);

        T warp_aggregate    = h_in[i];
        T inclusive         = h_in[i];
        h_reference[i]      = inclusive;

        for (i = i + 1; i < base_idx + logical_warp_items; ++i)
        {
            InitValue(gen_mode, h_in[i], i);
            inclusive = scan_op(inclusive, h_in[i]);
            warp_aggregate = scan_op(warp_aggregate, h_in[i]);
            h_reference[i] = inclusive;
        }

        warp_aggregates[w] = warp_aggregate;
    }
}


/**
 * Test warp scan
 */
template <
    int             LOGICAL_WARP_THREADS,
    TestMode        TEST_MODE,
    typename        T,
    typename        ScanOpT,
    typename        InitialValueT>        // NullType implies inclusive-scan, otherwise inclusive scan
void Test(
    GenMode         gen_mode,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    enum {
        TOTAL_ITEMS = LOGICAL_WARP_THREADS * NUM_WARPS,
    };

    // Allocate host arrays
    T *h_in = new T[TOTAL_ITEMS];
    T *h_reference = new T[TOTAL_ITEMS];
    T *h_aggregate = new T[TOTAL_ITEMS];

    // Initialize problem
    T aggregates[NUM_WARPS];

    Initialize(
        gen_mode,
        h_in,
        h_reference,
        LOGICAL_WARP_THREADS,
        scan_op,
        initial_value,
        aggregates);

    if (g_verbose)
    {
        printf("Input: \n");
        DisplayResults(h_in, TOTAL_ITEMS);
        printf("\n");
    }

    for (int w = 0; w < NUM_WARPS; ++w)
    {
        for (int i = 0; i < LOGICAL_WARP_THREADS; ++i)
        {
            h_aggregate[(w * LOGICAL_WARP_THREADS) + i] = aggregates[w];
        }
    }

    // Initialize/clear device arrays
    T *d_in = NULL;
    T *d_out = NULL;
    T *d_aggregate = NULL;
    clock_t *d_elapsed = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * TOTAL_ITEMS));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * (TOTAL_ITEMS + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_aggregate, sizeof(T) * TOTAL_ITEMS));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TOTAL_ITEMS, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * (TOTAL_ITEMS + 1)));
    CubDebugExit(cudaMemset(d_aggregate, 0, sizeof(T) * TOTAL_ITEMS));

    // Run kernel
    printf("Test-mode %d (%s), gen-mode %d (%s), %s warpscan, %d warp threads, %s (%d bytes) elements:\n",
        TEST_MODE, typeid(TEST_MODE).name(),
        gen_mode, typeid(gen_mode).name(),
        (Equals<InitialValueT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        LOGICAL_WARP_THREADS,
        typeid(T).name(),
        (int) sizeof(T));
    fflush(stdout);

    // Run aggregate/prefix kernel
    WarpScanKernel<LOGICAL_WARP_THREADS, TEST_MODE><<<1, TOTAL_ITEMS>>>(
        d_in,
        d_out,
        d_aggregate,
        scan_op,
        initial_value,
        d_elapsed);

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Copy out and display results
    printf("\tScan results: ");
    int compare = CompareDeviceResults(h_reference, d_out, TOTAL_ITEMS, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Copy out and display aggregate
    if (TEST_MODE == AGGREGATE)
    {
        printf("\tScan aggregate: ");
        compare = CompareDeviceResults(h_aggregate, d_aggregate, TOTAL_ITEMS, g_verbose, g_verbose);
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);
    }

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
 * Run battery of tests for different primitive variants
 */
template <
    int         LOGICAL_WARP_THREADS,
    typename    ScanOpT,
    typename    T>
void Test(
    GenMode     gen_mode,
    ScanOpT     scan_op,
    T           initial_value)
{
    // Exclusive
    Test<LOGICAL_WARP_THREADS, BASIC, T>(gen_mode, scan_op, T());
    Test<LOGICAL_WARP_THREADS, AGGREGATE, T>(gen_mode, scan_op, T());

    // Exclusive (non-specialized, so we can use initial-value)
    Test<LOGICAL_WARP_THREADS, BASIC, T>(gen_mode, WrapperFunctor<ScanOpT>(scan_op), initial_value);
    Test<LOGICAL_WARP_THREADS, AGGREGATE, T>(gen_mode, WrapperFunctor<ScanOpT>(scan_op), initial_value);

    // Inclusive
    Test<LOGICAL_WARP_THREADS, BASIC, T>(gen_mode, scan_op, NullType());
    Test<LOGICAL_WARP_THREADS, AGGREGATE, T>(gen_mode, scan_op, NullType());
}


/**
 * Run battery of tests for different data types and scan ops
 */
template <int LOGICAL_WARP_THREADS>
void Test(GenMode gen_mode)
{
    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    // primitive
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (char) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (short) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (int) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (long) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (long long) 99);
    if (gen_mode != RANDOM) {
        // Only test numerically stable inputs
        Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (float) 99);
        if (ptx_version > 100)
            Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), (double) 99);
    }

    // primitive (alternative scan op)
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned char) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned short) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned int) 99);
    Test<LOGICAL_WARP_THREADS>(gen_mode, Max(), (unsigned long long) 99);

    // vec-2
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_uchar2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_ushort2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_uint2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_ulong2(17, 21));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_ulonglong2(17, 21));
    if (gen_mode != RANDOM) {
        // Only test numerically stable inputs
        Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_float2(17, 21));
        if (ptx_version > 100)
            Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_double2(17, 21));
    }

    // vec-4
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_char4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_short4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_int4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_long4(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_longlong4(17, 21, 32, 85));
    if (gen_mode != RANDOM) {
        // Only test numerically stable inputs
        Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_float4(17, 21, 32, 85));
        if (ptx_version > 100)
            Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), make_double4(17, 21, 32, 85));
    }

    // complex
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<LOGICAL_WARP_THREADS>(gen_mode, Sum(), TestBar(17, 21));

}


/**
 * Run battery of tests for different problem generation options
 */
template <int LOGICAL_WARP_THREADS>
void Test()
{
    Test<LOGICAL_WARP_THREADS>(UNIFORM);
    Test<LOGICAL_WARP_THREADS>(INTEGER_SEED);
    Test<LOGICAL_WARP_THREADS>(RANDOM);
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

    // Compile/run quick tests
    Test<32, AGGREGATE, int>(UNIFORM, Sum(), (int) 0);
    Test<32, AGGREGATE, float>(UNIFORM, Sum(), (float) 0);
    Test<32, AGGREGATE, long long>(UNIFORM, Sum(), (long long) 0);
    Test<32, AGGREGATE, double>(UNIFORM, Sum(), (double) 0);

    typedef KeyValuePair<int, float> T;
    cub::Sum sum_op;
    Test<32, AGGREGATE, T>(UNIFORM, ReduceBySegmentOp<cub::Sum>(sum_op), T());

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test logical warp sizes
        Test<32>();
        Test<16>();
        Test<9>();
        Test<2>();
    }

#endif

    return 0;
}




