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
 * Test of DeviceReduce::ReduceByKey utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include <cub/util_allocator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/thread/thread_operators.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    THRUST,     // Thrust method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


//---------------------------------------------------------------------
// Dispatch to different CUB entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to reduce-by-key entrypoint
 */
template <
    typename                    KeyInputIteratorT,
    typename                    KeyOutputIteratorT,
    typename                    ValueInputIteratorT,
    typename                    ValueOutputIteratorT,
    typename                    NumRunsIteratorT,
    typename                    EqualityOpT,
    typename                    ReductionOpT,
    typename                    OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    KeyInputIteratorT           d_keys_in,
    KeyOutputIteratorT          d_keys_out,
    ValueInputIteratorT         d_values_in,
    ValueOutputIteratorT        d_values_out,
    NumRunsIteratorT            d_num_runs,
    EqualityOpT                  equality_op,
    ReductionOpT                 reduction_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceReduce::ReduceByKey(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            d_num_runs,
            reduction_op,
            num_items,
            stream,
            debug_synchronous);
    }
    return error;
}


//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to reduce-by-key entrypoint
 */
template <
    typename                    KeyInputIteratorT,
    typename                    KeyOutputIteratorT,
    typename                    ValueInputIteratorT,
    typename                    ValueOutputIteratorT,
    typename                    NumRunsIteratorT,
    typename                    EqualityOpT,
    typename                    ReductionOpT,
    typename                    OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    KeyInputIteratorT           d_keys_in,
    KeyOutputIteratorT          d_keys_out,
    ValueInputIteratorT         d_values_in,
    ValueOutputIteratorT        d_values_out,
    NumRunsIteratorT            d_num_runs,
    EqualityOpT                 equality_op,
    ReductionOpT                reduction_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // The input keys type
    typedef typename std::iterator_traits<KeyInputIteratorT>::value_type KeyInputT;

    // The output keys type
    typedef typename If<(Equals<typename std::iterator_traits<KeyOutputIteratorT>::value_type, void>::VALUE),   // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<KeyInputIteratorT>::value_type,                                           // ... then the input iterator's value type,
        typename std::iterator_traits<KeyOutputIteratorT>::value_type>::Type KeyOutputT;                        // ... else the output iterator's value type

    // The input values type
    typedef typename std::iterator_traits<ValueInputIteratorT>::value_type ValueInputT;

    // The output values type
    typedef typename If<(Equals<typename std::iterator_traits<ValueOutputIteratorT>::value_type, void>::VALUE), // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<ValueInputIteratorT>::value_type,                                         // ... then the input iterator's value type,
        typename std::iterator_traits<ValueOutputIteratorT>::value_type>::Type ValueOuputT;                     // ... else the output iterator's value type

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<KeyInputT> d_keys_in_wrapper(d_keys_in);
        thrust::device_ptr<KeyOutputT> d_keys_out_wrapper(d_keys_out);

        thrust::device_ptr<ValueInputT> d_values_in_wrapper(d_values_in);
        thrust::device_ptr<ValueOuputT> d_values_out_wrapper(d_values_out);

        thrust::pair<thrust::device_ptr<KeyOutputT>, thrust::device_ptr<ValueOuputT> > d_out_ends;

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_ends = thrust::reduce_by_key(
                d_keys_in_wrapper,
                d_keys_in_wrapper + num_items,
                d_values_in_wrapper,
                d_keys_out_wrapper,
                d_values_out_wrapper);
        }

        OffsetT num_segments = OffsetT(d_out_ends.first - d_keys_out_wrapper);
        CubDebugExit(cudaMemcpy(d_num_runs, &num_segments, sizeof(OffsetT), cudaMemcpyHostToDevice));

    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceSelect
 */
template <
    typename                    KeyInputIteratorT,
    typename                    KeyOutputIteratorT,
    typename                    ValueInputIteratorT,
    typename                    ValueOutputIteratorT,
    typename                    NumRunsIteratorT,
    typename                    EqualityOpT,
    typename                    ReductionOpT,
    typename                    OffsetT>
__global__ void CnpDispatchKernel(
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      temp_storage_bytes,
    KeyInputIteratorT           d_keys_in,
    KeyOutputIteratorT          d_keys_out,
    ValueInputIteratorT         d_values_in,
    ValueOutputIteratorT        d_values_out,
    NumRunsIteratorT            d_num_runs,
    EqualityOpT                 equality_op,
    ReductionOpT                reduction_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs, equality_op, reduction_op, num_items, 0, debug_synchronous);

    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <
    typename                    KeyInputIteratorT,
    typename                    KeyOutputIteratorT,
    typename                    ValueInputIteratorT,
    typename                    ValueOutputIteratorT,
    typename                    NumRunsIteratorT,
    typename                    EqualityOpT,
    typename                    ReductionOpT,
    typename                    OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CDP>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    KeyInputIteratorT           d_keys_in,
    KeyOutputIteratorT          d_keys_out,
    ValueInputIteratorT         d_values_in,
    ValueOutputIteratorT        d_values_out,
    NumRunsIteratorT            d_num_runs,
    EqualityOpT                 equality_op,
    ReductionOpT                reduction_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs, equality_op, reduction_op, num_items, 0, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}



//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Initialize problem
 */
template <typename T>
void Initialize(
    int         entropy_reduction,
    T           *h_in,
    int         num_items,
    int         max_segment)
{
    unsigned int max_int = (unsigned int) -1;

    int key = 0;
    int i = 0;
    while (i < num_items)
    {
        // Select number of repeating occurrences

        int repeat;

        if (max_segment < 0)
        {
            repeat = num_items;
        }
        else if (max_segment < 2)
        {
            repeat = 1;
        }
        else
        {
            RandomBits(repeat, entropy_reduction);
            repeat = (int) ((double(repeat) * double(max_segment)) / double(max_int));
            repeat = CUB_MAX(1, repeat);
        }

        int j = i;
        while (j < CUB_MIN(i + repeat, num_items))
        {
            InitValue(INTEGER_SEED, h_in[j], key);
            j++;
        }

        i = j;
        key++;
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/**
 * Solve problem.  Returns total number of segments identified
 */
template <
    typename        KeyInputIteratorT,
    typename        ValueInputIteratorT,
    typename        KeyT,
    typename        ValueT,
    typename        EqualityOpT,
    typename        ReductionOpT>
int Solve(
    KeyInputIteratorT       h_keys_in,
    KeyT                    *h_keys_reference,
    ValueInputIteratorT     h_values_in,
    ValueT                  *h_values_reference,
    EqualityOpT             equality_op,
    ReductionOpT            reduction_op,
    int                     num_items)
{
    // First item
    KeyT previous        = h_keys_in[0];
    ValueT aggregate     = h_values_in[0];
    int num_segments    = 0;

    // Subsequent items
    for (int i = 1; i < num_items; ++i)
    {
        if (!equality_op(previous, h_keys_in[i]))
        {
            h_keys_reference[num_segments] = previous;
            h_values_reference[num_segments] = aggregate;
            num_segments++;
            aggregate = h_values_in[i];
        }
        else
        {
            aggregate = reduction_op(aggregate, h_values_in[i]);
        }
        previous = h_keys_in[i];
    }

    h_keys_reference[num_segments] = previous;
    h_values_reference[num_segments] = aggregate;
    num_segments++;

    return num_segments;
}



/**
 * Test DeviceSelect for a given problem input
 */
template <
    Backend             BACKEND,
    typename            DeviceKeyInputIteratorT,
    typename            DeviceValueInputIteratorT,
    typename            KeyT,
    typename            ValueT,
    typename            EqualityOpT,
    typename            ReductionOpT>
void Test(
    DeviceKeyInputIteratorT     d_keys_in,
    DeviceValueInputIteratorT   d_values_in,
    KeyT*                       h_keys_reference,
    ValueT*                     h_values_reference,
    EqualityOpT                 equality_op,
    ReductionOpT                reduction_op,
    int                         num_segments,
    int                         num_items)
{
    // Allocate device output arrays and number of segments
    KeyT*   d_keys_out             = NULL;
    ValueT* d_values_out           = NULL;
    int*    d_num_runs         = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_out, sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_out, sizeof(ValueT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_runs, sizeof(int)));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs, equality_op, reduction_op, num_items, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output arrays
    CubDebugExit(cudaMemset(d_keys_out, 0, sizeof(KeyT) * num_items));
    CubDebugExit(cudaMemset(d_values_out, 0, sizeof(ValueT) * num_items));
    CubDebugExit(cudaMemset(d_num_runs, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs, equality_op, reduction_op, num_items, 0, true));

    // Check for correctness (and display results, if specified)
    int compare1 = CompareDeviceResults(h_keys_reference, d_keys_out, num_segments, true, g_verbose);
    printf("\t Keys %s ", compare1 ? "FAIL" : "PASS");

    int compare2 = CompareDeviceResults(h_values_reference, d_values_out, num_segments, true, g_verbose);
    printf("\t Values %s ", compare2 ? "FAIL" : "PASS");

    int compare3 = CompareDeviceResults(&num_segments, d_num_runs, 1, true, g_verbose);
    printf("\t Count %s ", compare3 ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs, equality_op, reduction_op, num_items, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float   avg_millis  = elapsed_millis / g_timing_iterations;
        float   giga_rate   = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        int     bytes_moved = ((num_items + num_segments) * sizeof(KeyT)) + ((num_items + num_segments) * sizeof(ValueT));
        float   giga_bandwidth  = float(bytes_moved) / avg_millis / 1000.0f / 1000.0f;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, giga_rate, giga_bandwidth);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_keys_out) CubDebugExit(g_allocator.DeviceFree(d_keys_out));
    if (d_values_out) CubDebugExit(g_allocator.DeviceFree(d_values_out));
    if (d_num_runs) CubDebugExit(g_allocator.DeviceFree(d_num_runs));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare1 | compare2 | compare3);
}


/**
 * Test DeviceSelect on pointer type
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT,
    typename        ReductionOpT>
void TestPointer(
    int             num_items,
    int             entropy_reduction,
    int             max_segment,
    ReductionOpT    reduction_op)
{
    // Allocate host arrays
    KeyT* h_keys_in        = new KeyT[num_items];
    KeyT* h_keys_reference = new KeyT[num_items];

    ValueT* h_values_in        = new ValueT[num_items];
    ValueT* h_values_reference = new ValueT[num_items];

    for (int i = 0; i < num_items; ++i)
        InitValue(INTEGER_SEED, h_values_in[i], 1);

    // Initialize problem and solution
    Equality equality_op;
    Initialize(entropy_reduction, h_keys_in, num_items, max_segment);
    int num_segments = Solve(h_keys_in, h_keys_reference, h_values_in, h_values_reference, equality_op, reduction_op, num_items);

    printf("\nPointer %s cub::DeviceReduce::ReduceByKey %s reduction of %d items, %d segments (avg run length %.3f), {%s,%s} key value pairs, max_segment %d, entropy_reduction %d\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<ReductionOpT, Sum>::VALUE) ? "Sum" : "Max",
        num_items, num_segments, float(num_items) / num_segments,
        typeid(KeyT).name(), typeid(ValueT).name(),
        max_segment, entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    KeyT     *d_keys_in = NULL;
    ValueT   *d_values_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_in, sizeof(ValueT) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values_in, h_values_in, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_keys_in, d_values_in, h_keys_reference, h_values_reference, equality_op, reduction_op, num_segments, num_items);

    // Cleanup
    if (h_keys_in) delete[] h_keys_in;
    if (h_values_in) delete[] h_values_in;
    if (h_keys_reference) delete[] h_keys_reference;
    if (h_values_reference) delete[] h_values_reference;
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    if (d_values_in) CubDebugExit(g_allocator.DeviceFree(d_values_in));
}


/**
 * Test on iterator type
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT,
    typename        ReductionOpT>
void TestIterator(
    int             num_items,
    int             entropy_reduction,
    int             max_segment,
    ReductionOpT    reduction_op)
{
    // Allocate host arrays
    KeyT* h_keys_in        = new KeyT[num_items];
    KeyT* h_keys_reference = new KeyT[num_items];

    ValueT one_val;
    InitValue(INTEGER_SEED, one_val, 1);
    ConstantInputIterator<ValueT, int> h_values_in(one_val);
    ValueT* h_values_reference = new ValueT[num_items];

    // Initialize problem and solution
    Equality equality_op;
    Initialize(entropy_reduction, h_keys_in, num_items, max_segment);
    int num_segments = Solve(h_keys_in, h_keys_reference, h_values_in, h_values_reference, equality_op, reduction_op, num_items);

    printf("\nIterator %s cub::DeviceReduce::ReduceByKey %s reduction of %d items, %d segments (avg run length %.3f), {%s,%s} key value pairs, max_segment %d, entropy_reduction %d\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<ReductionOpT, Sum>::VALUE) ? "Sum" : "Max",
        num_items, num_segments, float(num_items) / num_segments,
        typeid(KeyT).name(), typeid(ValueT).name(),
        max_segment, entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    KeyT     *d_keys_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(KeyT) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_keys_in, h_values_in, h_keys_reference, h_values_reference, equality_op, reduction_op, num_segments, num_items);

    // Cleanup
    if (h_keys_in) delete[] h_keys_in;
    if (h_keys_reference) delete[] h_keys_reference;
    if (h_values_reference) delete[] h_values_reference;
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT,
    typename        ReductionOpT>
void Test(
    int             num_items,
    ReductionOpT    reduction_op,
    int             max_segment)
{
    // 0 key-bit entropy reduction rounds
    TestPointer<BACKEND, KeyT, ValueT>(num_items, 0, max_segment, reduction_op);

    if (max_segment > 1)
    {
        // 2 key-bit entropy reduction rounds
        TestPointer<BACKEND, KeyT, ValueT>(num_items, 2, max_segment, reduction_op);

        // 7 key-bit entropy reduction rounds
        TestPointer<BACKEND, KeyT, ValueT>(num_items, 7, max_segment, reduction_op);
    }
}


/**
 * Test different avg segment lengths modes
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT,
    typename        ReductionOpT>
void Test(
    int             num_items,
    ReductionOpT    reduction_op)
{
    Test<BACKEND, KeyT, ValueT>(num_items, reduction_op, -1);
    Test<BACKEND, KeyT, ValueT>(num_items, reduction_op, 1);

    // Evaluate different max-segment lengths
    for (int max_segment = 3; max_segment < CUB_MIN(num_items, (unsigned short) -1); max_segment *= 11)
    {
        Test<BACKEND, KeyT, ValueT>(num_items, reduction_op, max_segment);
    }
}



/**
 * Test different dispatch
 */
template <
    typename        KeyT,
    typename        ValueT,
    typename        ReductionOpT>
void TestDispatch(
    int             num_items,
    ReductionOpT    reduction_op)
{
    Test<CUB, KeyT, ValueT>(num_items, reduction_op);
#ifdef CUB_CDP
    Test<CDP, KeyT, ValueT>(num_items, reduction_op);
#endif
}


/**
 * Test different input sizes
 */
template <
    typename        KeyT,
    typename        ValueT,
    typename        ReductionOpT>
void TestSize(
    int             num_items,
    ReductionOpT    reduction_op)
{
    if (num_items < 0)
    {
        TestDispatch<KeyT, ValueT>(1,        reduction_op);
        TestDispatch<KeyT, ValueT>(100,      reduction_op);
        TestDispatch<KeyT, ValueT>(10000,    reduction_op);
        TestDispatch<KeyT, ValueT>(1000000,  reduction_op);
    }
    else
    {
        TestDispatch<KeyT, ValueT>(num_items, reduction_op);
    }

}


template <
    typename        KeyT,
    typename        ValueT>
void TestOp(
    int             num_items)
{
    TestSize<KeyT, ValueT>(num_items, cub::Sum());
    TestSize<KeyT, ValueT>(num_items, cub::Max());
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items           = -1;
    int entropy_reduction   = 0;
    int maxseg              = 1000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("maxseg", maxseg);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--maxseg=<max segment length>]"
            "[--entropy=<segment length bit entropy reduction rounds>]"
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    printf("\n");

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef QUICKER_TEST

    // Compile/run basic CUB test
    if (num_items < 0) num_items = 32000000;

    TestPointer<CUB, int, double>(num_items, entropy_reduction, maxseg, cub::Sum());
    TestPointer<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum());
    TestIterator<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum());

#elif defined(QUICK_TEST)

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    printf("---- RLE int ---- \n");
    TestIterator<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum());

    printf("---- RLE long long ---- \n");
    TestIterator<CUB, long long, int>(num_items, entropy_reduction, maxseg, cub::Sum());

    printf("---- int ---- \n");
    TestPointer<CUB, int, int>(num_items, entropy_reduction, maxseg, cub::Sum());
    TestPointer<THRUST, int, int>(num_items, entropy_reduction, maxseg, cub::Sum());

    printf("---- float ---- \n");
    TestPointer<CUB, int, float>(num_items, entropy_reduction, maxseg, cub::Sum());
    TestPointer<THRUST, int, float>(num_items, entropy_reduction, maxseg, cub::Sum());

    if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
    {
        printf("---- double ---- \n");
        TestPointer<CUB, int, double>(num_items, entropy_reduction, maxseg, cub::Sum());
        TestPointer<THRUST, int, double>(num_items, entropy_reduction, maxseg, cub::Sum());
    }

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {

        // Test different input types
        TestOp<int, char>(num_items);
        TestOp<int, short>(num_items);
        TestOp<int, int>(num_items);
        TestOp<int, long>(num_items);
        TestOp<int, long long>(num_items);
        TestOp<int, float>(num_items);
        if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
            TestOp<int, double>(num_items);

        TestOp<int, uchar2>(num_items);
        TestOp<int, uint2>(num_items);
        TestOp<int, uint3>(num_items);
        TestOp<int, uint4>(num_items);
        TestOp<int, ulonglong4>(num_items);
        TestOp<int, TestFoo>(num_items);
        TestOp<int, TestBar>(num_items);

        TestOp<char, int>(num_items);
        TestOp<long long, int>(num_items);
        TestOp<TestFoo, int>(num_items);
        TestOp<TestBar, int>(num_items);

    }

#endif

    return 0;
}



