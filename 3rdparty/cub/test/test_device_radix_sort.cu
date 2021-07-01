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
 * Test of DeviceRadixSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>
#include <typeinfo>

#if (__CUDACC_VER_MAJOR__ >= 9)
    #include <cuda_fp16.h>
#endif

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "test_util.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reverse.h>

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
    CUB,                        // CUB method (allows overwriting of input)
    CUB_NO_OVERWRITE,           // CUB method (disallows overwriting of input)

    CUB_SEGMENTED,              // CUB method (allows overwriting of input)
    CUB_SEGMENTED_NO_OVERWRITE, // CUB method (disallows overwriting of input)

    THRUST,                     // Thrust method
    CDP,                        // GPU-based (dynamic parallelism) dispatch to CUB method
};


//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>         is_descending,
    Int2Type<CUB>           dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        num_items, begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_NO_OVERWRITE sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>             is_descending,
    Int2Type<CUB_NO_OVERWRITE>  dispatch_to,
    int                         *d_selector,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        num_items, begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

/**
 * Dispatch to CUB sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>          is_descending,
    Int2Type<CUB>           dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        num_items, begin_bit, end_bit, stream, debug_synchronous);
}


/**
 * Dispatch to CUB_NO_OVERWRITE sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>              is_descending,
    Int2Type<CUB_NO_OVERWRITE>  dispatch_to,
    int                         *d_selector,
    size_t                      *d_temp_storage_bytes,
    cudaError_t                 *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        num_items, begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}

//---------------------------------------------------------------------
// Dispatch to different DeviceRadixSort entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB_SEGMENTED sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>         is_descending,
    Int2Type<CUB_SEGMENTED> dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
        begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_SEGMENTED_NO_OVERWRITE sorting entrypoint (specialized for ascending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<false>                         is_descending,
    Int2Type<CUB_SEGMENTED_NO_OVERWRITE>    dispatch_to,
    int                                     *d_selector,
    size_t                                  *d_temp_storage_bytes,
    cudaError_t                             *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
        begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}


/**
 * Dispatch to CUB_SEGMENTED sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>          is_descending,
    Int2Type<CUB_SEGMENTED> dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    return DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
        begin_bit, end_bit, stream, debug_synchronous);
}

/**
 * Dispatch to CUB_SEGMENTED_NO_OVERWRITE sorting entrypoint (specialized for descending)
 */
template <typename KeyT, typename ValueT>
CUB_RUNTIME_FUNCTION
__forceinline__
cudaError_t Dispatch(
    Int2Type<true>                          is_descending,
    Int2Type<CUB_SEGMENTED_NO_OVERWRITE>    dispatch_to,
    int                                     *d_selector,
    size_t                                  *d_temp_storage_bytes,
    cudaError_t                             *d_cdp_error,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    KeyT      const *const_keys_itr     = d_keys.Current();
    ValueT    const *const_values_itr   = d_values.Current();

    cudaError_t retval = DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        const_keys_itr, d_keys.Alternate(), const_values_itr, d_values.Alternate(),
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
        begin_bit, end_bit, stream, debug_synchronous);

    d_keys.selector ^= 1;
    d_values.selector ^= 1;
    return retval;
}


//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch keys-only to Thrust sorting entrypoint
 */
template <int IS_DESCENDING, typename KeyT>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING> is_descending,
    Int2Type<THRUST>        dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<NullType>  &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<KeyT> d_keys_wrapper(d_keys.Current());

        if (IS_DESCENDING) thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
        thrust::sort(d_keys_wrapper, d_keys_wrapper + num_items);
        if (IS_DESCENDING) thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
    }

    return cudaSuccess;
}


/**
 * Dispatch key-value pairs to Thrust sorting entrypoint
 */
template <int IS_DESCENDING, typename KeyT, typename ValueT>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING> is_descending,
    Int2Type<THRUST>        dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<KeyT>     d_keys_wrapper(d_keys.Current());
        thrust::device_ptr<ValueT>   d_values_wrapper(d_values.Current());

        if (IS_DESCENDING) {
            thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            thrust::reverse(d_values_wrapper, d_values_wrapper + num_items);
        }

        thrust::sort_by_key(d_keys_wrapper, d_keys_wrapper + num_items, d_values_wrapper);

        if (IS_DESCENDING) {
            thrust::reverse(d_keys_wrapper, d_keys_wrapper + num_items);
            thrust::reverse(d_values_wrapper, d_values_wrapper + num_items);
        }
    }

    return cudaSuccess;
}


//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceRadixSort
 */
template <int IS_DESCENDING, typename KeyT, typename ValueT>
__global__ void CnpDispatchKernel(
    Int2Type<IS_DESCENDING> is_descending,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  temp_storage_bytes,
    DoubleBuffer<KeyT>      d_keys,
    DoubleBuffer<ValueT>    d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    bool                    debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error            = cudaErrorNotSupported;
#else
    *d_cdp_error            = Dispatch(
                                is_descending, Int2Type<CUB>(), d_selector, d_temp_storage_bytes, d_cdp_error,
                                d_temp_storage, temp_storage_bytes, d_keys, d_values,
                                num_items, num_segments, d_segment_offsets,
                                begin_bit, end_bit, 0, debug_synchronous);
    *d_temp_storage_bytes   = temp_storage_bytes;
    *d_selector             = d_keys.selector;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <int IS_DESCENDING, typename KeyT, typename ValueT>
cudaError_t Dispatch(
    Int2Type<IS_DESCENDING> is_descending,
    Int2Type<CDP>           dispatch_to,
    int                     *d_selector,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                    *d_temp_storage,
    size_t                  &temp_storage_bytes,
    DoubleBuffer<KeyT>      &d_keys,
    DoubleBuffer<ValueT>    &d_values,
    int                     num_items,
    int                     num_segments,
    const int               *d_segment_offsets,
    int                     begin_bit,
    int                     end_bit,
    cudaStream_t            stream,
    bool                    debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<<<1,1>>>(
        is_descending, d_selector, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, d_segment_offsets,
        begin_bit, end_bit, debug_synchronous);

    // Copy out selector
    CubDebugExit(cudaMemcpy(&d_keys.selector, d_selector, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    d_values.selector = d_keys.selector;

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}



//---------------------------------------------------------------------
// Problem generation
//---------------------------------------------------------------------


/**
 * Simple key-value pairing
 */
template <
    typename KeyT,
    typename ValueT,
    bool IS_FLOAT = (Traits<KeyT>::CATEGORY == FLOATING_POINT)>
struct Pair
{
    KeyT     key;
    ValueT   value;

    bool operator<(const Pair &b) const
    {
        return (key < b.key);
    }
};


/**
 * Simple key-value pairing (specialized for bool types)
 */
template <typename ValueT>
struct Pair<bool, ValueT, false>
{
    bool     key;
    ValueT   value;

    bool operator<(const Pair &b) const
    {
        return (!key && b.key);
    }
};


/**
 * Simple key-value pairing (specialized for floating point types)
 */
template <typename KeyT, typename ValueT>
struct Pair<KeyT, ValueT, true>
{
    KeyT     key;
    ValueT   value;

    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;

        if (key > b.key)
            return false;

        // KeyT in unsigned bits
        typedef typename Traits<KeyT>::UnsignedBits UnsignedBits;

        // Return true if key is negative zero and b.key is positive zero
        UnsignedBits key_bits   = *reinterpret_cast<UnsignedBits*>(const_cast<KeyT*>(&key));
        UnsignedBits b_key_bits = *reinterpret_cast<UnsignedBits*>(const_cast<KeyT*>(&b.key));
        UnsignedBits HIGH_BIT   = Traits<KeyT>::HIGH_BIT;

        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};


/**
 * Initialize key data
 */
template <typename KeyT>
void InitializeKeyBits(
    GenMode         gen_mode,
    KeyT            *h_keys,
    int             num_items,
    int             entropy_reduction)
{
    for (int i = 0; i < num_items; ++i)
        InitValue(gen_mode, h_keys[i], i);
}


/**
 * Initialize solution
 */
template <bool IS_DESCENDING, typename KeyT>
void InitializeSolution(
    KeyT    *h_keys,
    int     num_items,
    int     num_segments,
    int     *h_segment_offsets,
    int     begin_bit,
    int     end_bit,
    int     *&h_reference_ranks,
    KeyT    *&h_reference_keys)
{
    typedef Pair<KeyT, int> PairT;

    PairT *h_pairs = new PairT[num_items];

    int num_bits = end_bit - begin_bit;
    for (int i = 0; i < num_items; ++i)
    {

        // Mask off unwanted portions
        if (num_bits < sizeof(KeyT) * 8)
        {
            unsigned long long base = 0;
            memcpy(&base, &h_keys[i], sizeof(KeyT));
            base &= ((1ull << num_bits) - 1) << begin_bit;
            memcpy(&h_pairs[i].key, &base, sizeof(KeyT));
        }
        else
        {
            h_pairs[i].key = h_keys[i];
        }

        h_pairs[i].value = i;
    }

    printf("\nSorting reference solution on CPU (%d segments)...", num_segments); fflush(stdout);

    for (int i = 0; i < num_segments; ++i)
    {
        if (IS_DESCENDING) std::reverse(h_pairs + h_segment_offsets[i], h_pairs + h_segment_offsets[i + 1]);
        std::stable_sort(               h_pairs + h_segment_offsets[i], h_pairs + h_segment_offsets[i + 1]);
        if (IS_DESCENDING) std::reverse(h_pairs + h_segment_offsets[i], h_pairs + h_segment_offsets[i + 1]);
    }

    printf(" Done.\n"); fflush(stdout);

    h_reference_ranks  = new int[num_items];
    h_reference_keys   = new KeyT[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        h_reference_ranks[i]    = h_pairs[i].value;
        h_reference_keys[i]     = h_keys[h_pairs[i].value];
    }

    if (h_pairs) delete[] h_pairs;
}


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Test DeviceRadixSort
 */
template <
    Backend     BACKEND,
    bool        IS_DESCENDING,
    typename    KeyT,
    typename    ValueT>
void Test(
    KeyT        *h_keys,
    ValueT      *h_values,
    int         num_items,
    int         num_segments,
    int         *h_segment_offsets,
    int         begin_bit,
    int         end_bit,
    KeyT        *h_reference_keys,
    ValueT      *h_reference_values)
{
    // Key alias type
#if (__CUDACC_VER_MAJOR__ >= 9)
    typedef typename If<Equals<KeyT, half_t>::VALUE, __half, KeyT>::Type KeyAliasT;
#else
    typedef KeyT KeyAliasT;
#endif

    const bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

    printf("%s %s cub::DeviceRadixSort %d items, %d segments, %d-byte keys (%s) %d-byte values (%s), descending %d, begin_bit %d, end_bit %d\n",
        (BACKEND == CUB_NO_OVERWRITE) ? "CUB_NO_OVERWRITE" : (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        (KEYS_ONLY) ? "keys-only" : "key-value",
        num_items, num_segments,
        (int) sizeof(KeyT), typeid(KeyT).name(), (KEYS_ONLY) ? 0 : (int) sizeof(ValueT), typeid(ValueT).name(),
        IS_DESCENDING, begin_bit, end_bit);
    fflush(stdout);

    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
    }

    // Allocate device arrays
    DoubleBuffer<KeyAliasT> d_keys;
    DoubleBuffer<ValueT>    d_values;
    int                     *d_selector;
    int                     *d_segment_offsets;
    size_t                  *d_temp_storage_bytes;
    cudaError_t             *d_cdp_error;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_selector, sizeof(int) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(int) * (num_segments + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes, sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error, sizeof(cudaError_t) * 1));
    if (!KEYS_ONLY)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(ValueT) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(ValueT) * num_items));
    }

    // Allocate temporary storage (and make it un-aligned)
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit(Dispatch(
        Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, d_segment_offsets,
        begin_bit, end_bit, 0, true));

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes + 1));
    void* mis_aligned_temp = static_cast<char*>(d_temp_storage) + 1;

    // Initialize/clear device arrays
    d_keys.selector = 0;
    CubDebugExit(cudaMemcpy(d_keys.d_buffers[0], h_keys, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_keys.d_buffers[1], 0, sizeof(KeyT) * num_items));
    if (!KEYS_ONLY)
    {
        d_values.selector = 0;
        CubDebugExit(cudaMemcpy(d_values.d_buffers[0], h_values, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_values.d_buffers[1], 0, sizeof(ValueT) * num_items));
    }
    CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(
        Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error,
        mis_aligned_temp, temp_storage_bytes, d_keys, d_values,
        num_items, num_segments, d_segment_offsets,
        begin_bit, end_bit, 0, true));

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    printf("Warmup done.  Checking results:\n"); fflush(stdout);
    int compare = CompareDeviceResults(h_reference_keys, reinterpret_cast<KeyT*>(d_keys.Current()), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s ", d_keys.selector, compare ? "FAIL" : "PASS"); fflush(stdout);
    if (!KEYS_ONLY)
    {
        int values_compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
        compare |= values_compare;
        printf("\t Compare values (selector %d): %s ", d_values.selector, values_compare ? "FAIL" : "PASS"); fflush(stdout);
    }
    if (BACKEND == CUB_NO_OVERWRITE)
    {
        // Check that input isn't overwritten
        int input_compare = CompareDeviceResults(h_keys, reinterpret_cast<KeyT*>(d_keys.d_buffers[0]), num_items, true, g_verbose);
        compare |= input_compare;
        printf("\t Compare input keys: %s ", input_compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Performance
    if (g_timing_iterations)
        printf("\nPerforming timing iterations:\n"); fflush(stdout);

    GpuTimer gpu_timer;
    float elapsed_millis = 0.0f;
    for (int i = 0; i < g_timing_iterations; ++i)
    {
        // Initialize/clear device arrays
        CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(KeyT) * num_items));
        if (!KEYS_ONLY)
        {
            CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice));
            CubDebugExit(cudaMemset(d_values.d_buffers[d_values.selector ^ 1], 0, sizeof(ValueT) * num_items));
        }

        gpu_timer.Start();
        CubDebugExit(Dispatch(
            Int2Type<IS_DESCENDING>(), Int2Type<BACKEND>(), d_selector, d_temp_storage_bytes, d_cdp_error,
            mis_aligned_temp, temp_storage_bytes, d_keys, d_values,
            num_items, num_segments, d_segment_offsets,
            begin_bit, end_bit, 0, false));
        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = (KEYS_ONLY) ?
            giga_rate * sizeof(KeyT) * 2 :
            giga_rate * (sizeof(KeyT) + sizeof(ValueT)) * 2;
        printf("\n%.3f elapsed ms, %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", elapsed_millis, avg_millis, giga_rate, giga_bandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
    if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
    if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
    if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_selector) CubDebugExit(g_allocator.DeviceFree(d_selector));
    if (d_segment_offsets) CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));

    // Correctness asserts
    AssertEquals(0, compare);
}


/**
 * Test backend
 */
template <bool IS_DESCENDING, typename KeyT, typename ValueT>
void TestBackend(
    KeyT    *h_keys,
    int     num_items,
    int     num_segments,
    int     *h_segment_offsets,
    int     begin_bit,
    int     end_bit,
    KeyT    *h_reference_keys,
    int     *h_reference_ranks)
{
    const bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

    ValueT *h_values             = NULL;
    ValueT *h_reference_values   = NULL;

    if (!KEYS_ONLY)
    {
        h_values            = new ValueT[num_items];
        h_reference_values  = new ValueT[num_items];

        for (int i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }

#ifdef SEGMENTED_SORT
    // Test multi-segment implementations
    Test<CUB_SEGMENTED, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    Test<CUB_SEGMENTED_NO_OVERWRITE, IS_DESCENDING>(  h_keys, h_values, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
#else   // SEGMENTED_SORT
    if (num_segments == 1)
    {
        // Test single-segment implementations
        Test<CUB, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
        Test<CUB_NO_OVERWRITE, IS_DESCENDING>(  h_keys, h_values, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    #ifdef CUB_CDP
        Test<CDP, IS_DESCENDING>(               h_keys, h_values, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_values);
    #endif
    }
#endif  // SEGMENTED_SORT

    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
}




/**
 * Test value type
 */
template <bool IS_DESCENDING, typename KeyT>
void TestValueTypes(
    KeyT    *h_keys,
    int     num_items,
    int     num_segments,
    int     *h_segment_offsets,
    int     begin_bit,
    int     end_bit)
{
    // Initialize the solution

    int *h_reference_ranks = NULL;
    KeyT *h_reference_keys = NULL;
    InitializeSolution<IS_DESCENDING>(h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    // Test keys-only
    TestBackend<IS_DESCENDING, KeyT, NullType>          (h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with 8b value
    TestBackend<IS_DESCENDING, KeyT, unsigned char>     (h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with 32b value
    TestBackend<IS_DESCENDING, KeyT, unsigned int>      (h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with 64b value
    TestBackend<IS_DESCENDING, KeyT, unsigned long long>(h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Test with non-trivially-constructable value
    TestBackend<IS_DESCENDING, KeyT, TestBar>           (h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit, h_reference_keys, h_reference_ranks);

    // Cleanup
    if (h_reference_ranks) delete[] h_reference_ranks;
    if (h_reference_keys) delete[] h_reference_keys;
}



/**
 * Test ascending/descending
 */
template <typename KeyT>
void TestDirection(
    KeyT    *h_keys,
    int     num_items,
    int     num_segments,
    int     *h_segment_offsets,
    int     begin_bit,
    int     end_bit)
{
    TestValueTypes<true>(h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit);
    TestValueTypes<false>(h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit);
}


/**
 * Test different bit ranges
 */
template <typename KeyT>
void TestBits(
    KeyT    *h_keys,
    int     num_items,
    int     num_segments,
    int     *h_segment_offsets)
{
    // Don't test partial-word sorting for boolean, fp, or signed types (the bit-flipping techniques get in the way)
    if ((Traits<KeyT>::CATEGORY == UNSIGNED_INTEGER) && (!Equals<KeyT, bool>::VALUE))
    {
        // Partial bits
        int begin_bit = 1;
        int end_bit = (sizeof(KeyT) * 8) - 1;
        printf("Testing key bits [%d,%d)\n", begin_bit, end_bit); fflush(stdout);
        TestDirection(h_keys, num_items, num_segments, h_segment_offsets, begin_bit, end_bit);

        // Across subword boundaries
        int mid_bit = sizeof(KeyT) * 4;
        printf("Testing key bits [%d,%d)\n", mid_bit - 1, mid_bit + 1); fflush(stdout);
        TestDirection(h_keys, num_items, num_segments, h_segment_offsets, mid_bit - 1, mid_bit + 1);
    }

    printf("Testing key bits [%d,%d)\n", 0, int(sizeof(KeyT)) * 8); fflush(stdout);
    TestDirection(h_keys, num_items, num_segments, h_segment_offsets, 0, sizeof(KeyT) * 8);
}


/**
 * Test different segment compositions
 */
template <typename KeyT>
void TestSegments(
    KeyT    *h_keys,
    int     num_items,
    int     max_segments)
{
    int *h_segment_offsets = new int[max_segments + 1];

#ifdef SEGMENTED_SORT
    for (int num_segments = max_segments; num_segments > 1; num_segments = (num_segments + 32 - 1) / 32)
    {
        if (num_items / num_segments < 128 * 1000) {
            // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
            InitializeSegments(num_items, num_segments, h_segment_offsets);
            TestBits(h_keys, num_items, num_segments, h_segment_offsets);
        }
    }
#else
    // Test single segment
    if (num_items < 128 * 1000) {
        // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
        InitializeSegments(num_items, 1, h_segment_offsets);
        TestBits(h_keys, num_items, 1, h_segment_offsets);
    }
#endif
    if (h_segment_offsets) delete[] h_segment_offsets;
}


/**
 * Test different (sub)lengths and number of segments
 */
template <typename KeyT>
void TestSizes(
    KeyT    *h_keys,
    int     max_items,
    int     max_segments)
{
    for (int num_items = max_items; num_items > 1; num_items = (num_items + 32 - 1) / 32)
    {
        TestSegments(h_keys, num_items, max_segments);
    }
    TestSegments(h_keys, 1, max_segments);
    TestSegments(h_keys, 0, max_segments);
}


/**
 * Test key sampling distributions
 */
template <typename KeyT>
void TestGen(
    int             max_items,
    int             max_segments)
{
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    if (max_items < 0)
        max_items = (ptx_version > 100) ? 9000003 : max_items = 5000003;

    if (max_segments < 0)
        max_segments = 5003;

    KeyT *h_keys = new KeyT[max_items];

    for (int entropy_reduction = 0; entropy_reduction <= 6; entropy_reduction += 3)
    {
        printf("\nTesting random %s keys with entropy reduction factor %d\n", typeid(KeyT).name(), entropy_reduction); fflush(stdout);
        InitializeKeyBits(RANDOM, h_keys, max_items, entropy_reduction);
        TestSizes(h_keys, max_items, max_segments);
    }

    printf("\nTesting uniform %s keys\n", typeid(KeyT).name()); fflush(stdout);
    InitializeKeyBits(UNIFORM, h_keys, max_items, 0);
    TestSizes(h_keys, max_items, max_segments);

    printf("\nTesting natural number %s keys\n", typeid(KeyT).name()); fflush(stdout);
    InitializeKeyBits(INTEGER_SEED, h_keys, max_items, 0);
    TestSizes(h_keys, max_items, max_segments);

    if (h_keys) delete[] h_keys;
}


//---------------------------------------------------------------------
// Simple test
//---------------------------------------------------------------------

template <
    Backend     BACKEND,
    typename    KeyT,
    typename    ValueT,
    bool        IS_DESCENDING>
void Test(
    int         num_items,
    int         num_segments,
    GenMode     gen_mode,
    int         entropy_reduction,
    int         begin_bit,
    int         end_bit)
{
    const bool KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

    KeyT    *h_keys             = new KeyT[num_items];
    int     *h_reference_ranks  = NULL;
    KeyT    *h_reference_keys   = NULL;
    ValueT  *h_values           = NULL;
    ValueT  *h_reference_values = NULL;
    int     *h_segment_offsets  = new int[num_segments + 1];

    if (end_bit < 0)
        end_bit = sizeof(KeyT) * 8;

    InitializeKeyBits(gen_mode, h_keys, num_items, entropy_reduction);
    InitializeSegments(num_items, num_segments, h_segment_offsets);
    InitializeSolution<IS_DESCENDING>(
        h_keys, num_items, num_segments, h_segment_offsets,
        begin_bit, end_bit, h_reference_ranks, h_reference_keys);

    if (!KEYS_ONLY)
    {
        h_values            = new ValueT[num_items];
        h_reference_values  = new ValueT[num_items];

        for (int i = 0; i < num_items; ++i)
        {
            InitValue(INTEGER_SEED, h_values[i], i);
            InitValue(INTEGER_SEED, h_reference_values[i], h_reference_ranks[i]);
        }
    }
    if (h_reference_ranks) delete[] h_reference_ranks;

    printf("\nTesting bits [%d,%d) of %s keys with gen-mode %d\n", begin_bit, end_bit, typeid(KeyT).name(), gen_mode); fflush(stdout);
    Test<BACKEND, IS_DESCENDING>(
        h_keys, h_values,
        num_items, num_segments, h_segment_offsets,
        begin_bit, end_bit, h_reference_keys, h_reference_values);

    if (h_keys)             delete[] h_keys;
    if (h_reference_keys)   delete[] h_reference_keys;
    if (h_values)           delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
    if (h_segment_offsets)  delete[] h_segment_offsets;
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int bits = -1;
    int num_items = -1;
    int num_segments = -1;
    int entropy_reduction = 0;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("s", num_segments);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("bits", bits);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--bits=<valid key bits>]"
            "[--n=<input items> "
            "[--s=<num segments> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--entropy=<entropy-reduction factor (default 0)>]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef QUICKER_TEST

    enum {
        IS_DESCENDING   = false
    };

    // Compile/run basic CUB test
    if (num_items < 0)      num_items       = 48000000;
    if (num_segments < 0)   num_segments    = 5000;

    Test<CUB,           unsigned char, NullType, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned char, unsigned int, IS_DESCENDING>(num_items, 1, RANDOM, entropy_reduction, 0, bits);

#if (__CUDACC_VER_MAJOR__ >= 9)
    Test<CUB,           half_t,       NullType, IS_DESCENDING>(       num_items, 1,               RANDOM, entropy_reduction, 0, bits);
#endif

    Test<CUB_SEGMENTED, unsigned int,       NullType, IS_DESCENDING>(       num_items, num_segments,    RANDOM, entropy_reduction, 0, bits);

    Test<CUB,           unsigned int,       NullType, IS_DESCENDING>(       num_items, 1,               RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned long long, NullType, IS_DESCENDING>(       num_items, 1,               RANDOM, entropy_reduction, 0, bits);

    Test<CUB,           unsigned int,       unsigned int, IS_DESCENDING>(   num_items, 1,               RANDOM, entropy_reduction, 0, bits);
    Test<CUB,           unsigned long long, unsigned int, IS_DESCENDING>(   num_items, 1,               RANDOM, entropy_reduction, 0, bits);

#elif defined(QUICK_TEST)

    // Compile/run quick tests
    if (num_items < 0)      num_items       = 48000000;
    if (num_segments < 0)   num_segments    = 5000;

    // Compare CUB and thrust on 32b keys-only
    Test<CUB, unsigned int, NullType, false> (                      num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned int, NullType, false> (                   num_items, 1, RANDOM, entropy_reduction, 0, bits);

    // Compare CUB and thrust on 64b keys-only
    Test<CUB, unsigned long long, NullType, false> (                num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned long long, NullType, false> (             num_items, 1, RANDOM, entropy_reduction, 0, bits);


    // Compare CUB and thrust on 32b key-value pairs
    Test<CUB, unsigned int, unsigned int, false> (                  num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned int, unsigned int, false> (               num_items, 1, RANDOM, entropy_reduction, 0, bits);

    // Compare CUB and thrust on 64b key-value pairs
    Test<CUB, unsigned long long, unsigned long long, false> (      num_items, 1, RANDOM, entropy_reduction, 0, bits);
    Test<THRUST, unsigned long long, unsigned long long, false> (   num_items, 1, RANDOM, entropy_reduction, 0, bits);


#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        TestGen<bool>                 (num_items, num_segments);

        TestGen<char>                 (num_items, num_segments);
        TestGen<signed char>          (num_items, num_segments);
        TestGen<unsigned char>        (num_items, num_segments);

        TestGen<short>                (num_items, num_segments);
        TestGen<unsigned short>       (num_items, num_segments);

        TestGen<int>                  (num_items, num_segments);
        TestGen<unsigned int>         (num_items, num_segments);

        TestGen<long>                 (num_items, num_segments);
        TestGen<unsigned long>        (num_items, num_segments);

        TestGen<long long>            (num_items, num_segments);
        TestGen<unsigned long long>   (num_items, num_segments);

#if (__CUDACC_VER_MAJOR__ >= 9)
        TestGen<half_t>                (num_items, num_segments);
#endif
        TestGen<float>                (num_items, num_segments);

        if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
            TestGen<double>           (num_items, num_segments);

    }

#endif

    return 0;
}

