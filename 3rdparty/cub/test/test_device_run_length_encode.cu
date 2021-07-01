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
 * Test of DeviceReduce::RunLengthEncode utilities
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

// Operation types
enum RleMethod
{
    RLE,                // Run length encode
    NON_TRIVIAL,
    CSR,
};


//---------------------------------------------------------------------
// Dispatch to different CUB entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to run-length encode entrypoint
 */
template <
    typename                    InputIteratorT,
    typename                    UniqueOutputIteratorT,
    typename                    OffsetsOutputIteratorT,
    typename                    LengthsOutputIteratorT,
    typename                    NumRunsIterator,
    typename                    OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<RLE>               method,
    Int2Type<CUB>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void*               d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIteratorT              d_in,
    UniqueOutputIteratorT       d_unique_out,
    OffsetsOutputIteratorT      d_offsets_out,
    LengthsOutputIteratorT      d_lengths_out,
    NumRunsIterator             d_num_runs,
    cub::Equality               equality_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceRunLengthEncode::Encode(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_unique_out,
            d_lengths_out,
            d_num_runs,
            num_items,
            stream,
            debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to non-trivial runs entrypoint
 */
template <
    typename                    InputIteratorT,
    typename                    UniqueOutputIteratorT,
    typename                    OffsetsOutputIteratorT,
    typename                    LengthsOutputIteratorT,
    typename                    NumRunsIterator,
    typename                    OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<NON_TRIVIAL>       method,
    Int2Type<CUB>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void*               d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIteratorT              d_in,
    UniqueOutputIteratorT       d_unique_out,
    OffsetsOutputIteratorT      d_offsets_out,
    LengthsOutputIteratorT      d_lengths_out,
    NumRunsIterator             d_num_runs,
    cub::Equality               equality_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceRunLengthEncode::NonTrivialRuns(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_offsets_out,
            d_lengths_out,
            d_num_runs,
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
 * Dispatch to run-length encode entrypoint
 */
template <
    typename                    InputIteratorT,
    typename                    UniqueOutputIteratorT,
    typename                    OffsetsOutputIteratorT,
    typename                    LengthsOutputIteratorT,
    typename                    NumRunsIterator,
    typename                    OffsetT>
cudaError_t Dispatch(
    Int2Type<RLE>               method,
    Int2Type<THRUST>            dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIteratorT              d_in,
    UniqueOutputIteratorT       d_unique_out,
    OffsetsOutputIteratorT      d_offsets_out,
    LengthsOutputIteratorT      d_lengths_out,
    NumRunsIterator             d_num_runs,
    cub::Equality               equality_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<UniqueOutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                                // ... then the input iterator's value type,
        typename std::iterator_traits<UniqueOutputIteratorT>::value_type>::Type UniqueT;                          // ... else the output iterator's value type

    // The lengths output value type
    typedef typename If<(Equals<typename std::iterator_traits<LengthsOutputIteratorT>::value_type, void>::VALUE),   // LengthT =  (if output iterator's value type is void) ?
        OffsetT,                                                                                                    // ... then the OffsetT type,
        typename std::iterator_traits<LengthsOutputIteratorT>::value_type>::Type LengthT;                           // ... else the output iterator's value type

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<InputT>      d_in_wrapper(d_in);
        thrust::device_ptr<UniqueT>     d_unique_out_wrapper(d_unique_out);
        thrust::device_ptr<LengthT>     d_lengths_out_wrapper(d_lengths_out);

        thrust::pair<thrust::device_ptr<UniqueT>, thrust::device_ptr<LengthT> > d_out_ends;

        LengthT one_val;
        InitValue(INTEGER_SEED, one_val, 1);
        thrust::constant_iterator<LengthT> constant_one(one_val);

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_ends = thrust::reduce_by_key(
                d_in_wrapper,
                d_in_wrapper + num_items,
                constant_one,
                d_unique_out_wrapper,
                d_lengths_out_wrapper);
        }

        OffsetT num_runs = OffsetT(d_out_ends.first - d_unique_out_wrapper);
        CubDebugExit(cudaMemcpy(d_num_runs, &num_runs, sizeof(OffsetT), cudaMemcpyHostToDevice));
    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceRunLengthEncode
 */
template <
    int                         RLE_METHOD,
    typename                    InputIteratorT,
    typename                    UniqueOutputIteratorT,
    typename                    OffsetsOutputIteratorT,
    typename                    LengthsOutputIteratorT,
    typename                    NumRunsIterator,
    typename                    EqualityOp,
    typename                    OffsetT>
__global__ void CnpDispatchKernel(
    Int2Type<RLE_METHOD>            method,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void*               d_temp_storage,
    size_t                      temp_storage_bytes,
    InputIteratorT              d_in,
    UniqueOutputIteratorT       d_unique_out,
    OffsetsOutputIteratorT      d_offsets_out,
    LengthsOutputIteratorT      d_lengths_out,
    NumRunsIterator             d_num_runs,
    cub::Equality               equality_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(method, Int2Type<CUB>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_offsets_out, d_lengths_out, d_num_runs, equality_op, num_items, 0, debug_synchronous);

    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <
    int                         RLE_METHOD,
    typename                    InputIteratorT,
    typename                    UniqueOutputIteratorT,
    typename                    OffsetsOutputIteratorT,
    typename                    LengthsOutputIteratorT,
    typename                    NumRunsIterator,
    typename                    EqualityOp,
    typename                    OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<RLE_METHOD>        method,
    Int2Type<CDP>               dispatch_to,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void*               d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIteratorT              d_in,
    UniqueOutputIteratorT       d_unique_out,
    OffsetsOutputIteratorT      d_offsets_out,
    LengthsOutputIteratorT      d_lengths_out,
    NumRunsIterator             d_num_runs,
    EqualityOp                  equality_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(method, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_offsets_out, d_lengths_out, d_num_runs, equality_op, num_items, 0, debug_synchronous);

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
        // Select number of repeating occurrences for the current run
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
    RleMethod       RLE_METHOD,
    typename        InputIteratorT,
    typename        T,
    typename        OffsetT,
    typename        LengthT,
    typename        EqualityOp>
int Solve(
    InputIteratorT  h_in,
    T               *h_unique_reference,
    OffsetT         *h_offsets_reference,
    LengthT         *h_lengths_reference,
    EqualityOp      equality_op,
    int             num_items)
{
    if (num_items == 0) 
        return 0;

    // First item
    T       previous        = h_in[0];
    LengthT  length          = 1;
    int     num_runs        = 0;
    int     run_begin       = 0;

    // Subsequent items
    for (int i = 1; i < num_items; ++i)
    {
        if (!equality_op(previous, h_in[i]))
        {
            if ((RLE_METHOD != NON_TRIVIAL) || (length > 1))
            {
                h_unique_reference[num_runs]      = previous;
                h_offsets_reference[num_runs]     = run_begin;
                h_lengths_reference[num_runs]     = length;
                num_runs++;
            }
            length = 1;
            run_begin = i;
        }
        else
        {
            length++;
        }
        previous = h_in[i];
    }

    if ((RLE_METHOD != NON_TRIVIAL) || (length > 1))
    {
        h_unique_reference[num_runs]    = previous;
        h_offsets_reference[num_runs]   = run_begin;
        h_lengths_reference[num_runs]   = length;
        num_runs++;
    }

    return num_runs;
}



/**
 * Test DeviceRunLengthEncode for a given problem input
 */
template <
    RleMethod           RLE_METHOD,
    Backend             BACKEND,
    typename            DeviceInputIteratorT,
    typename            T,
    typename            OffsetT,
    typename            LengthT,
    typename            EqualityOp>
void Test(
    DeviceInputIteratorT d_in,
    T                   *h_unique_reference,
    OffsetT             *h_offsets_reference,
    LengthT             *h_lengths_reference,
    EqualityOp          equality_op,
    int                 num_runs,
    int                 num_items)
{
    // Allocate device output arrays and number of segments
    T*          d_unique_out       = NULL;
    LengthT*    d_offsets_out      = NULL;
    OffsetT*    d_lengths_out      = NULL;
    int*        d_num_runs         = NULL;

    if (RLE_METHOD == RLE)
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_unique_out, sizeof(T) * num_items));
    if (RLE_METHOD == NON_TRIVIAL)
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_offsets_out, sizeof(OffsetT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_lengths_out, sizeof(LengthT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_runs, sizeof(int)));

    // Allocate CDP device arrays
    size_t*          d_temp_storage_bytes = NULL;
    cudaError_t*     d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void*           d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<RLE_METHOD>(), Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_offsets_out, d_lengths_out, d_num_runs, equality_op, num_items, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output arrays
    if (RLE_METHOD == RLE)
        CubDebugExit(cudaMemset(d_unique_out,   0, sizeof(T) * num_items));
    if (RLE_METHOD == NON_TRIVIAL)
        CubDebugExit(cudaMemset(d_offsets_out,  0, sizeof(OffsetT) * num_items));
    CubDebugExit(cudaMemset(d_lengths_out,  0, sizeof(LengthT) * num_items));
    CubDebugExit(cudaMemset(d_num_runs,     0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<RLE_METHOD>(), Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_offsets_out, d_lengths_out, d_num_runs, equality_op, num_items, 0, true));

    // Check for correctness (and display results, if specified)
    int compare0 = 0;
    int compare1 = 0;
    int compare2 = 0;
    int compare3 = 0;

    if (RLE_METHOD == RLE)
    {
        compare0 = CompareDeviceResults(h_unique_reference, d_unique_out, num_runs, true, g_verbose);
        printf("\t Keys %s\n", compare0 ? "FAIL" : "PASS");
    }

    if (RLE_METHOD != RLE)
    {
        compare1 = CompareDeviceResults(h_offsets_reference, d_offsets_out, num_runs, true, g_verbose);
        printf("\t Offsets %s\n", compare1 ? "FAIL" : "PASS");
    }

    if (RLE_METHOD != CSR)
    {
        compare2 = CompareDeviceResults(h_lengths_reference, d_lengths_out, num_runs, true, g_verbose);
        printf("\t Lengths %s\n", compare2 ? "FAIL" : "PASS");
    }

    compare3 = CompareDeviceResults(&num_runs, d_num_runs, 1, true, g_verbose);
    printf("\t Count %s\n", compare3 ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<RLE_METHOD>(), Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_offsets_out, d_lengths_out, d_num_runs, equality_op, num_items, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        int bytes_moved = (num_items * sizeof(T)) + (num_runs * (sizeof(OffsetT) + sizeof(LengthT)));
        float giga_bandwidth = float(bytes_moved) / avg_millis / 1000.0f / 1000.0f;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, giga_rate, giga_bandwidth);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_unique_out) CubDebugExit(g_allocator.DeviceFree(d_unique_out));
    if (d_offsets_out) CubDebugExit(g_allocator.DeviceFree(d_offsets_out));
    if (d_lengths_out) CubDebugExit(g_allocator.DeviceFree(d_lengths_out));
    if (d_num_runs) CubDebugExit(g_allocator.DeviceFree(d_num_runs));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare0 | compare1 | compare2 | compare3);
}


/**
 * Test DeviceRunLengthEncode on pointer type
 */
template <
    RleMethod       RLE_METHOD,
    Backend         BACKEND,
    typename        T,
    typename        OffsetT,
    typename        LengthT>
void TestPointer(
    int             num_items,
    int             entropy_reduction,
    int             max_segment)
{
    // Allocate host arrays
    T*      h_in                    = new T[num_items];
    T*      h_unique_reference      = new T[num_items];
    OffsetT* h_offsets_reference     = new OffsetT[num_items];
    LengthT* h_lengths_reference     = new LengthT[num_items];

    for (int i = 0; i < num_items; ++i)
        InitValue(INTEGER_SEED, h_offsets_reference[i], 1);

    // Initialize problem and solution
    Equality equality_op;
    Initialize(entropy_reduction, h_in, num_items, max_segment);

    int num_runs = Solve<RLE_METHOD>(h_in, h_unique_reference, h_offsets_reference, h_lengths_reference, equality_op, num_items);

    printf("\nPointer %s cub::%s on %d items, %d segments (avg run length %.3f), {%s key, %s offset, %s length}, max_segment %d, entropy_reduction %d\n",
        (RLE_METHOD == RLE) ? "DeviceReduce::RunLengthEncode" : (RLE_METHOD == NON_TRIVIAL) ? "DeviceRunLengthEncode::NonTrivialRuns" : "Other",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_runs, float(num_items) / num_runs,
        typeid(T).name(), typeid(OffsetT).name(), typeid(LengthT).name(),
        max_segment, entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    T* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<RLE_METHOD, BACKEND>(d_in, h_unique_reference, h_offsets_reference, h_lengths_reference, equality_op, num_runs, num_items);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_unique_reference) delete[] h_unique_reference;
    if (h_offsets_reference) delete[] h_offsets_reference;
    if (h_lengths_reference) delete[] h_lengths_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test on iterator type
 */
template <
    RleMethod       RLE_METHOD,
    Backend         BACKEND,
    typename        T,
    typename        OffsetT,
    typename        LengthT>
void TestIterator(
    int             num_items,
    Int2Type<true>  is_primitive)
{
    // Allocate host arrays
    T* h_unique_reference       = new T[num_items];
    OffsetT* h_offsets_reference = new OffsetT[num_items];
    LengthT* h_lengths_reference = new LengthT[num_items];

    T one_val;
    InitValue(INTEGER_SEED, one_val, 1);
    ConstantInputIterator<T, int> h_in(one_val);

    // Initialize problem and solution
    Equality equality_op;
    int num_runs = Solve<RLE_METHOD>(h_in, h_unique_reference, h_offsets_reference, h_lengths_reference, equality_op, num_items);

    printf("\nIterator %s cub::%s on %d items, %d segments (avg run length %.3f), {%s key, %s offset, %s length}\n",
        (RLE_METHOD == RLE) ? "DeviceReduce::RunLengthEncode" : (RLE_METHOD == NON_TRIVIAL) ? "DeviceRunLengthEncode::NonTrivialRuns" : "Other",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_runs, float(num_items) / num_runs,
        typeid(T).name(), typeid(OffsetT).name(), typeid(LengthT).name());
    fflush(stdout);

    // Run Test
    Test<RLE_METHOD, BACKEND>(h_in, h_unique_reference, h_offsets_reference, h_lengths_reference, equality_op, num_runs, num_items);

    // Cleanup
    if (h_unique_reference) delete[] h_unique_reference;
    if (h_offsets_reference) delete[] h_offsets_reference;
    if (h_lengths_reference) delete[] h_lengths_reference;
}


template <
    RleMethod       RLE_METHOD,
    Backend         BACKEND,
    typename        T,
    typename        OffsetT,
    typename        LengthT>
void TestIterator(
    int             num_items,
    Int2Type<false> is_primitive)
{}


/**
 * Test different gen modes
 */
template <
    RleMethod       RLE_METHOD,
    Backend         BACKEND,
    typename        T,
    typename        OffsetT,
    typename        LengthT>
void Test(
    int             num_items)
{
    // Test iterator (one run)
    TestIterator<RLE_METHOD, BACKEND, T, OffsetT, LengthT>(num_items, Int2Type<Traits<T>::PRIMITIVE>());

    // num_items runs
    TestPointer<RLE_METHOD, BACKEND, T, OffsetT, LengthT>(num_items, 0, 1);

    // Evaluate different run lengths
    for (int max_segment = 3; max_segment < CUB_MIN(num_items, (unsigned short) -1); max_segment *= 3)
    {
        // Uniform selection run length
        TestPointer<RLE_METHOD, BACKEND, T, OffsetT, LengthT>(num_items, 0, max_segment);

        // Reduced-entropy run length
        TestPointer<RLE_METHOD, BACKEND, T, OffsetT, LengthT>(num_items, 4, max_segment);
    }
}


/**
 * Test different dispatch
 */
template <
    typename        T,
    typename        OffsetT,
    typename        LengthT>
void TestDispatch(
    int             num_items)
{
    Test<RLE,           CUB, T, OffsetT, LengthT>(num_items);
    Test<NON_TRIVIAL,   CUB, T, OffsetT, LengthT>(num_items);

#ifdef CUB_CDP
    Test<RLE,           CDP, T, OffsetT, LengthT>(num_items);
    Test<NON_TRIVIAL,   CDP, T, OffsetT, LengthT>(num_items);
#endif
}


/**
 * Test different input sizes
 */
template <
    typename        T,
    typename        OffsetT,
    typename        LengthT>
void TestSize(
    int             num_items)
{
    if (num_items < 0)
    {
        TestDispatch<T, OffsetT, LengthT>(0);
        TestDispatch<T, OffsetT, LengthT>(1);
        TestDispatch<T, OffsetT, LengthT>(100);
        TestDispatch<T, OffsetT, LengthT>(10000);
        TestDispatch<T, OffsetT, LengthT>(1000000);

        // Randomly select problem size between 1:10,000,000
        unsigned int max_int = (unsigned int) -1;
        for (int i = 0; i < 10; ++i)
        {
            unsigned int num_items;
            RandomBits(num_items);
            num_items = (unsigned int) ((double(num_items) * double(10000000)) / double(max_int));
            num_items = CUB_MAX(1, num_items);
            TestDispatch<T, OffsetT, LengthT>(num_items);
        }
    }
    else
    {
        TestDispatch<T, OffsetT, LengthT>(num_items);
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
    int num_items           = -1;
    int entropy_reduction   = 0;
    int max_segment              = 1000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("maxseg", max_segment);
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

    TestPointer<RLE,            CUB, int, int, int>(    num_items, entropy_reduction, max_segment);
    TestPointer<NON_TRIVIAL,    CUB, int, int, int>(    num_items, entropy_reduction, max_segment);
    TestIterator<RLE,           CUB, float, int, int>(  num_items, Int2Type<Traits<float>::PRIMITIVE>());


#elif defined(QUICK_TEST)

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    TestPointer<RLE,            CUB, int, int, int>(    num_items, entropy_reduction, max_segment);
    TestPointer<RLE,            THRUST, int, int, int>(    num_items, entropy_reduction, max_segment);

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input types
        TestSize<char,          int, int>(num_items);
        TestSize<short,         int, int>(num_items);
        TestSize<int,           int, int>(num_items);
        TestSize<long,          int, int>(num_items);
        TestSize<long long,     int, int>(num_items);
        TestSize<float,         int, int>(num_items);
        TestSize<double,        int, int>(num_items);

        TestSize<uchar2,        int, int>(num_items);
        TestSize<uint2,         int, int>(num_items);
        TestSize<uint3,         int, int>(num_items);
        TestSize<uint4,         int, int>(num_items);
        TestSize<ulonglong4,    int, int>(num_items);
        TestSize<TestFoo,       int, int>(num_items);
        TestSize<TestBar,       int, int>(num_items);
    }

#endif

    return 0;
}



