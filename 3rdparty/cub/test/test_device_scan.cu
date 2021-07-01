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
 * Test of DeviceScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cub/util_allocator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/device/device_scan.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
double                  g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    THRUST,     // Thrust method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
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
// Dispatch to different CUB DeviceScan entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, initial_value, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to exclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename InitialValueT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    Int2Type<true>      is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    Sum                 scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to inclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    NullType            initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to inclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    Int2Type<true>      is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    Sum                 scan_op,
    NullType            initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
    }
    return error;
}

//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<InputT> d_in_wrapper(d_in);
        thrust::device_ptr<OutputT> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::exclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper, initial_value, scan_op);
        }
    }

    return cudaSuccess;
}


/**
 * Dispatch to exclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename InitialValueT, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    Int2Type<true>      is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    Sum                 scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<InputT> d_in_wrapper(d_in);
        thrust::device_ptr<OutputT> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::exclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper);
        }
    }

    return cudaSuccess;
}


/**
 * Dispatch to inclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    NullType            initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<InputT> d_in_wrapper(d_in);
        thrust::device_ptr<OutputT> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::inclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper, scan_op);
        }
    }

    return cudaSuccess;
}


/**
 * Dispatch to inclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<THRUST>    dispatch_to,
    Int2Type<true>      is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    Sum                 scan_op,
    NullType            initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<InputT> d_in_wrapper(d_in);
        thrust::device_ptr<OutputT> d_out_wrapper(d_out);
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            thrust::inclusive_scan(d_in_wrapper, d_in_wrapper + num_items, d_out_wrapper);
        }
    }

    return cudaSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceScan
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT>
__global__ void CnpDispatchKernel(
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t              temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(
        Int2Type<CUB>(),
        is_primitive,
        timing_timing_iterations,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items,
        0,
        debug_synchronous);

    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT>
cudaError_t Dispatch(
    Int2Type<CDP>       dispatch_to,
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(
        is_primitive,
        timing_timing_iterations,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items,
        debug_synchronous);

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
    GenMode      gen_mode,
    T            *h_in,
    int          num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}

/**
 * Solve exclusive-scan problem
 */
template <
    typename        InputIteratorT,
    typename        OutputT,
    typename        ScanOpT>
void Solve(
    InputIteratorT  h_in,
    OutputT         *h_reference,
    int             num_items,
    ScanOpT         scan_op,
    OutputT         initial_value)
{
    if (num_items > 0)
    {
        OutputT val         = h_in[0];
        h_reference[0]      = initial_value;
        OutputT inclusive   = scan_op(initial_value, val);

        for (int i = 1; i < num_items; ++i)
        {
            val = h_in[i];
            h_reference[i] = inclusive;
            inclusive = scan_op(inclusive, val);
        }
    }
}


/**
 * Solve inclusive-scan problem
 */
template <
    typename        InputIteratorT,
    typename        OutputT,
    typename        ScanOpT>
void Solve(
    InputIteratorT  h_in,
    OutputT         *h_reference,
    int             num_items,
    ScanOpT         scan_op,
    NullType)
{
    if (num_items > 0)
    {
        OutputT inclusive   = h_in[0];
        h_reference[0]      = inclusive;

        for (int i = 1; i < num_items; ++i)
        {
            OutputT val = h_in[i];
            inclusive = scan_op(inclusive, val);
            h_reference[i] = inclusive;
        }
    }
}


/**
 * Test DeviceScan for a given problem input
 */
template <
    Backend             BACKEND,
    typename            DeviceInputIteratorT,
    typename            OutputT,
    typename            ScanOpT,
    typename            InitialValueT>
void Test(
    DeviceInputIteratorT    d_in,
    OutputT                 *h_reference,
    int                     num_items,
    ScanOpT                 scan_op,
    InitialValueT           initial_value)
{
    typedef typename std::iterator_traits<DeviceInputIteratorT>::value_type InputT;

    // Allocate device output array
    OutputT *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(OutputT) * num_items));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,   sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(
        Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        1,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items,
        0,
        true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output array
    CubDebugExit(cudaMemset(d_out, 0, sizeof(OutputT) * num_items));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(
        Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        1,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items,
        0,
        true));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        g_timing_iterations,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items,
        0,
        false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = giga_rate * (sizeof(InputT) + sizeof(OutputT));
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak", avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }

    printf("\n\n");

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test DeviceScan on pointer type
 */
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void TestPointer(
    int             num_items,
    GenMode         gen_mode,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    printf("\nPointer %s %s cub::DeviceScan::%s %d items, %s->%s (%d->%d bytes) , gen-mode %s\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<InitialValueT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        (Equals<ScanOpT, Sum>::VALUE) ? "Sum" : "Scan",
        num_items,
        typeid(InputT).name(), typeid(OutputT).name(), (int) sizeof(InputT), (int) sizeof(OutputT),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    InputT*     h_in        = new InputT[num_items];
    OutputT*    h_reference = new OutputT[num_items];

    // Initialize problem and solution
    Initialize(gen_mode, h_in, num_items);
    Solve(h_in, h_reference, num_items, scan_op, initial_value);

    // Allocate problem device arrays
    InputT *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(InputT) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(InputT) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_in, h_reference, num_items, scan_op, initial_value);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceScan on iterator type
 */
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void TestIterator(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    printf("\nIterator %s %s cub::DeviceScan::%s %d items, %s->%s (%d->%d bytes)\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (Equals<InitialValueT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
        (Equals<ScanOpT, Sum>::VALUE) ? "Sum" : "Scan",
        num_items,
        typeid(InputT).name(), typeid(OutputT).name(), (int) sizeof(InputT), (int) sizeof(OutputT));
    fflush(stdout);

    // Use a constant iterator as the input
    InputT val = InputT();
    ConstantInputIterator<InputT, int> h_in(val);

    // Allocate host arrays
    OutputT*  h_reference = new OutputT[num_items];

    // Initialize problem and solution
    Solve(h_in, h_reference, num_items, scan_op, initial_value);

    // Run Test
    Test<BACKEND>(h_in, h_reference, num_items, scan_op, initial_value);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void Test(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    TestPointer<BACKEND, InputT, OutputT>(  num_items, UNIFORM, scan_op, initial_value);
    TestPointer<BACKEND, InputT, OutputT>(  num_items, RANDOM,  scan_op, initial_value);
    TestIterator<BACKEND, InputT, OutputT>( num_items, scan_op, initial_value);
}


/**
 * Test different dispatch
 */
template <
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void Test(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    Test<CUB, InputT, OutputT>(num_items, scan_op, initial_value);
#ifdef CUB_CDP
    Test<CDP, InputT, OutputT>(num_items, scan_op, initial_value);
#endif
}


/**
 * Test different operators
 */
template <typename InputT, typename OutputT>
void TestOp(
    int             num_items,
    OutputT         identity,
    OutputT         initial_value)
{
    // Exclusive (use identity as initial value because it will dispatch to *Sum variants that don't take initial values)
    Test<InputT, OutputT>(num_items, cub::Sum(), identity);
    Test<InputT, OutputT>(num_items, cub::Max(), identity);

    // Exclusive (non-specialized, so we can test initial-value)
    Test<InputT, OutputT>(num_items, WrapperFunctor<cub::Sum>(cub::Sum()), initial_value);
    Test<InputT, OutputT>(num_items, WrapperFunctor<cub::Max>(cub::Max()), initial_value);

    // Inclusive (no initial value)
    Test<InputT, OutputT>(num_items, cub::Sum(), NullType());
    Test<InputT, OutputT>(num_items, cub::Max(), NullType());
}


/**
 * Test different input sizes
 */
template <
    typename InputT,
    typename OutputT>
void TestSize(
    int     num_items,
    OutputT identity,
    OutputT initial_value)
{
    if (num_items < 0)
    {
        TestOp<InputT>(0,        identity, initial_value);
        TestOp<InputT>(1,        identity, initial_value);
        TestOp<InputT>(100,      identity, initial_value);
        TestOp<InputT>(10000,    identity, initial_value);
        TestOp<InputT>(1000000,  identity, initial_value);

        // Randomly select problem size between 1:10,000,000
        unsigned int max_int = (unsigned int) -1;
        for (int i = 0; i < 10; ++i)
        {
            unsigned int num_items;
            RandomBits(num_items);
            num_items = (unsigned int) ((double(num_items) * double(10000000)) / double(max_int));
            num_items = CUB_MAX(1, num_items);
            TestOp<InputT>(num_items,  identity, initial_value);
        }
    }
    else
    {
        TestOp<InputT>(num_items, identity, initial_value);
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
    int num_items = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

#ifdef QUICKER_TEST

    // Compile/run basic CUB test
    if (num_items < 0) num_items = 32000000;

    TestPointer<CUB, char, int>(         num_items    , UNIFORM, Sum(), (int) (0));
    TestPointer<CUB, int, int>(         num_items    , UNIFORM, Sum(), (int) (0));

#elif defined(QUICK_TEST)

    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get device SM version
    int sm_version;
    CubDebugExit(SmVersion(sm_version, device_ordinal));

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    TestPointer<CUB, char, char>(        num_items * ((sm_version <= 130) ? 1 : 4), UNIFORM, Sum(), char(0));
    TestPointer<THRUST, char, char>(     num_items * ((sm_version <= 130) ? 1 : 4), UNIFORM, Sum(), char(0));

    printf("----------------------------\n");
    TestPointer<CUB, short, short>(       num_items * ((sm_version <= 130) ? 1 : 2), UNIFORM, Sum(), short(0));
    TestPointer<THRUST, short, short>(    num_items * ((sm_version <= 130) ? 1 : 2), UNIFORM, Sum(), short(0));

    printf("----------------------------\n");
    TestPointer<CUB, int, int>(         num_items    , UNIFORM, Sum(), (int) (0));
    TestPointer<THRUST, int, int>(      num_items    , UNIFORM, Sum(), (int) (0));

    printf("----------------------------\n");
    TestPointer<CUB, long long, long long>(   num_items / 2, UNIFORM, Sum(), (long long) (0));
    TestPointer<THRUST, long long, long long>(num_items / 2, UNIFORM, Sum(), (long long) (0));

    printf("----------------------------\n");
    TestPointer<CUB, TestBar, TestBar>(     num_items / 4, UNIFORM, Sum(), TestBar());
    TestPointer<THRUST, TestBar, TestBar>(  num_items / 4, UNIFORM, Sum(), TestBar());

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input+output data types
        TestSize<unsigned char>(num_items,      (int) 0, (int) 99);

        // Test same intput+output data types
        TestSize<unsigned char>(num_items,      (unsigned char) 0,      (unsigned char) 99);
        TestSize<char>(num_items,               (char) 0,               (char) 99);
        TestSize<unsigned short>(num_items,     (unsigned short) 0,     (unsigned short)99);
        TestSize<unsigned int>(num_items,       (unsigned int) 0,       (unsigned int) 99);
        TestSize<unsigned long long>(num_items, (unsigned long long) 0, (unsigned long long) 99);

        TestSize<uchar2>(num_items,     make_uchar2(0, 0),              make_uchar2(17, 21));
        TestSize<char2>(num_items,      make_char2(0, 0),               make_char2(17, 21));
        TestSize<ushort2>(num_items,    make_ushort2(0, 0),             make_ushort2(17, 21));
        TestSize<uint2>(num_items,      make_uint2(0, 0),               make_uint2(17, 21));
        TestSize<ulonglong2>(num_items, make_ulonglong2(0, 0),          make_ulonglong2(17, 21));
        TestSize<uchar4>(num_items,     make_uchar4(0, 0, 0, 0),        make_uchar4(17, 21, 32, 85));
        TestSize<char4>(num_items,      make_char4(0, 0, 0, 0),         make_char4(17, 21, 32, 85));

        TestSize<ushort4>(num_items,    make_ushort4(0, 0, 0, 0),       make_ushort4(17, 21, 32, 85));
        TestSize<uint4>(num_items,      make_uint4(0, 0, 0, 0),         make_uint4(17, 21, 32, 85));
        TestSize<ulonglong4>(num_items, make_ulonglong4(0, 0, 0, 0),    make_ulonglong4(17, 21, 32, 85));

        TestSize<TestFoo>(num_items,
            TestFoo::MakeTestFoo(0, 0, 0, 0),
            TestFoo::MakeTestFoo(1ll << 63, 1 << 31, short(1 << 15), char(1 << 7)));

        TestSize<TestBar>(num_items,
            TestBar(0, 0),
            TestBar(1ll << 63, 1 << 31));
    }

#endif

    return 0;
}



