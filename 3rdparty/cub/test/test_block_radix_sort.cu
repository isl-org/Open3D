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
 * Test of BlockRadixSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
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


/// Specialized descending, blocked -> blocked
template <int BLOCK_THREADS, typename BlockRadixSort, int ITEMS_PER_THREAD, typename Key, typename Value>
__device__ __forceinline__ void TestBlockSort(
    typename BlockRadixSort::TempStorage &temp_storage,
    Key                         (&keys)[ITEMS_PER_THREAD],
    Value                       (&values)[ITEMS_PER_THREAD],
    Key                         *d_keys,
    Value                       *d_values,
    int                         begin_bit,
    int                         end_bit,
    clock_t                     &stop,
    Int2Type<true>              is_descending,
    Int2Type<true>              is_blocked_output)
{
    BlockRadixSort(temp_storage).SortDescending(keys, values, begin_bit, end_bit);
    stop = clock();
    StoreDirectBlocked(threadIdx.x, d_keys, keys);
    StoreDirectBlocked(threadIdx.x, d_values, values);
}

/// Specialized descending, blocked -> striped
template <int BLOCK_THREADS, typename BlockRadixSort, int ITEMS_PER_THREAD, typename Key, typename Value>
__device__ __forceinline__ void TestBlockSort(
    typename BlockRadixSort::TempStorage &temp_storage,
    Key                         (&keys)[ITEMS_PER_THREAD],
    Value                       (&values)[ITEMS_PER_THREAD],
    Key                         *d_keys,
    Value                       *d_values,
    int                         begin_bit,
    int                         end_bit,
    clock_t                     &stop,
    Int2Type<true>              is_descending,
    Int2Type<false>             is_blocked_output)
{
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, values, begin_bit, end_bit);
    stop = clock();
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys, keys);
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values, values);
}

/// Specialized ascending, blocked -> blocked
template <int BLOCK_THREADS, typename BlockRadixSort, int ITEMS_PER_THREAD, typename Key, typename Value>
__device__ __forceinline__ void TestBlockSort(
    typename BlockRadixSort::TempStorage &temp_storage,
    Key                         (&keys)[ITEMS_PER_THREAD],
    Value                       (&values)[ITEMS_PER_THREAD],
    Key                         *d_keys,
    Value                       *d_values,
    int                         begin_bit,
    int                         end_bit,
    clock_t                     &stop,
    Int2Type<false>             is_descending,
    Int2Type<true>              is_blocked_output)
{
    BlockRadixSort(temp_storage).Sort(keys, values, begin_bit, end_bit);
    stop = clock();
    StoreDirectBlocked(threadIdx.x, d_keys, keys);
    StoreDirectBlocked(threadIdx.x, d_values, values);
}

/// Specialized ascending, blocked -> striped
template <int BLOCK_THREADS, typename BlockRadixSort, int ITEMS_PER_THREAD, typename Key, typename Value>
__device__ __forceinline__ void TestBlockSort(
    typename BlockRadixSort::TempStorage &temp_storage,
    Key                         (&keys)[ITEMS_PER_THREAD],
    Value                       (&values)[ITEMS_PER_THREAD],
    Key                         *d_keys,
    Value                       *d_values,
    int                         begin_bit,
    int                         end_bit,
    clock_t                     &stop,
    Int2Type<false>             is_descending,
    Int2Type<false>             is_blocked_output)
{
    BlockRadixSort(temp_storage).SortBlockedToStriped(keys, values, begin_bit, end_bit);
    stop = clock();
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys, keys);
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values, values);
}



/**
 * BlockRadixSort kernel
 */
template <
    int                 BLOCK_THREADS,
    int                 ITEMS_PER_THREAD,
    int                 RADIX_BITS,
    bool                MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm  INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig SMEM_CONFIG,
    int                 DESCENDING,
    int                 BLOCKED_OUTPUT,
    typename            Key,
    typename            Value>
__launch_bounds__ (BLOCK_THREADS, 1)
__global__ void Kernel(
    Key                         *d_keys,
    Value                       *d_values,
    int                         begin_bit,
    int                         end_bit,
    clock_t                     *d_elapsed)
{
    // Threadblock load/store abstraction types
    typedef BlockRadixSort<
            Key,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            Value,
            RADIX_BITS,
            MEMOIZE_OUTER_SCAN,
            INNER_SCAN_ALGORITHM,
            SMEM_CONFIG>
        BlockRadixSortT;

    // Allocate temp storage in shared memory
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

    // Items per thread
    Key     keys[ITEMS_PER_THREAD];
    Value   values[ITEMS_PER_THREAD];

    LoadDirectBlocked(threadIdx.x, d_keys, keys);
    LoadDirectBlocked(threadIdx.x, d_values, values);

    // Start cycle timer
    clock_t stop;
    clock_t start = clock();

    TestBlockSort<BLOCK_THREADS, BlockRadixSortT>(
        temp_storage, keys, values, d_keys, d_values, begin_bit, end_bit, stop, Int2Type<DESCENDING>(), Int2Type<BLOCKED_OUTPUT>());

    // Store time
    if (threadIdx.x == 0)
        *d_elapsed = (start > stop) ? start - stop : stop - start;
}



//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------


/**
 * Simple key-value pairing
 */
template <
    typename Key,
    typename Value,
    bool IS_FLOAT = (Traits<Key>::CATEGORY == FLOATING_POINT)>
struct Pair
{
    Key     key;
    Value   value;

    bool operator<(const Pair &b) const
    {
        return (key < b.key);
    }
};

/**
 * Simple key-value pairing (specialized for floating point types)
 */
template <typename Key, typename Value>
struct Pair<Key, Value, true>
{
    Key     key;
    Value   value;

    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;

        if (key > b.key)
            return false;

        // Key in unsigned bits
        typedef typename Traits<Key>::UnsignedBits UnsignedBits;

        // Return true if key is negative zero and b.key is positive zero
        UnsignedBits key_bits   = *reinterpret_cast<UnsignedBits*>(const_cast<Key*>(&key));
        UnsignedBits b_key_bits = *reinterpret_cast<UnsignedBits*>(const_cast<Key*>(&b.key));
        UnsignedBits HIGH_BIT   = Traits<Key>::HIGH_BIT;

        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};


/**
 * Initialize key-value sorting problem.
 */
template <bool DESCENDING, typename Key, typename Value>
void Initialize(
    GenMode         gen_mode,
    Key             *h_keys,
    Value           *h_values,
    Key             *h_reference_keys,
    Value           *h_reference_values,
    int             num_items,
    int             entropy_reduction,
    int             begin_bit,
    int             end_bit)
{
    Pair<Key, Value> *h_pairs = new Pair<Key, Value>[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_keys[i], i);

        RandomBits(h_values[i]);

        // Mask off unwanted portions
        int num_bits = end_bit - begin_bit;
        if ((begin_bit > 0) || (end_bit < sizeof(Key) * 8))
        {
            unsigned long long base = 0;
            memcpy(&base, &h_keys[i], sizeof(Key));
            base &= ((1ull << num_bits) - 1) << begin_bit;
            memcpy(&h_keys[i], &base, sizeof(Key));
        }

        h_pairs[i].key    = h_keys[i];
        h_pairs[i].value  = h_values[i];
    }

    if (DESCENDING) std::reverse(h_pairs, h_pairs + num_items);
    std::stable_sort(h_pairs, h_pairs + num_items);
    if (DESCENDING) std::reverse(h_pairs, h_pairs + num_items);

    for (int i = 0; i < num_items; ++i)
    {
        h_reference_keys[i]     = h_pairs[i].key;
        h_reference_values[i]   = h_pairs[i].value;
    }

    delete[] h_pairs;
}




/**
 * Test BlockRadixSort kernel
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    bool                    DESCENDING,
    bool                    BLOCKED_OUTPUT,
    typename                Key,
    typename                Value>
void TestDriver(
    GenMode                 gen_mode,
    int                     entropy_reduction,
    int                     begin_bit,
    int                     end_bit)
{
    enum
    {
        TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD,
        KEYS_ONLY = Equals<Value, NullType>::VALUE,
    };

    // Allocate host arrays
    Key     *h_keys             = new Key[TILE_SIZE];
    Key     *h_reference_keys   = new Key[TILE_SIZE];
    Value   *h_values           = new Value[TILE_SIZE];
    Value   *h_reference_values = new Value[TILE_SIZE];

    // Allocate device arrays
    Key     *d_keys     = NULL;
    Value   *d_values   = NULL;
    clock_t *d_elapsed  = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys, sizeof(Key) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values, sizeof(Value) * TILE_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));

    // Initialize problem and solution on host
    Initialize<DESCENDING>(gen_mode, h_keys, h_values, h_reference_keys, h_reference_values,
        TILE_SIZE, entropy_reduction, begin_bit, end_bit);

    // Copy problem to device
    CubDebugExit(cudaMemcpy(d_keys, h_keys, sizeof(Key) * TILE_SIZE, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(Value) * TILE_SIZE, cudaMemcpyHostToDevice));

    printf("%s "
        "BLOCK_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "RADIX_BITS(%d) "
        "MEMOIZE_OUTER_SCAN(%d) "
        "INNER_SCAN_ALGORITHM(%d) "
        "SMEM_CONFIG(%d) "
        "DESCENDING(%d) "
        "BLOCKED_OUTPUT(%d) "
        "sizeof(Key)(%d) "
        "sizeof(Value)(%d) "
        "gen_mode(%d), "
        "entropy_reduction(%d) "
        "begin_bit(%d) "
        "end_bit(%d), "
        "samples(%d)\n",
            ((KEYS_ONLY) ? "Keys-only" : "Key-value"),
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            RADIX_BITS,
            MEMOIZE_OUTER_SCAN,
            INNER_SCAN_ALGORITHM,
            SMEM_CONFIG,
            DESCENDING,
            BLOCKED_OUTPUT,
            (int) sizeof(Key),
            (int) sizeof(Value),
            gen_mode,
            entropy_reduction,
            begin_bit,
            end_bit,
            g_num_rand_samples);

    // Set shared memory config
    cudaDeviceSetSharedMemConfig(SMEM_CONFIG);

    // Run kernel
    Kernel<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, DESCENDING, BLOCKED_OUTPUT><<<1, BLOCK_THREADS>>>(
        d_keys, d_values, begin_bit, end_bit, d_elapsed);

    // Flush kernel output / errors
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Check keys results
    printf("\tKeys: ");
    int compare = CompareDeviceResults(h_reference_keys, d_keys, TILE_SIZE, g_verbose, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check value results
    if (!KEYS_ONLY)
    {
        printf("\tValues: ");
        int compare = CompareDeviceResults(h_reference_values, d_values, TILE_SIZE, g_verbose, g_verbose);
        printf("%s\n", compare ? "FAIL" : "PASS");
        AssertEquals(0, compare);
    }
    printf("\n");

    printf("\tElapsed clocks: ");
    DisplayDeviceResults(d_elapsed, 1);
    printf("\n");

    // Cleanup
    if (h_keys)             delete[] h_keys;
    if (h_reference_keys)   delete[] h_reference_keys;
    if (h_values)           delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
    if (d_keys)             CubDebugExit(g_allocator.DeviceFree(d_keys));
    if (d_values)           CubDebugExit(g_allocator.DeviceFree(d_values));
    if (d_elapsed)          CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}


/**
 * Test driver (valid tile size <= MAX_SMEM_BYTES)
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    bool                    DESCENDING,
    bool                    BLOCKED_OUTPUT,
    typename                Key,
    typename                Value>
void TestValid(Int2Type<true> fits_smem_capacity)
{
    // Iterate begin_bit
    for (int begin_bit = 0; begin_bit <= 1; begin_bit++)
    {
        // Iterate end bit
        for (int end_bit = begin_bit + 1; end_bit <= sizeof(Key) * 8; end_bit = end_bit * 2 + begin_bit)
        {
            // Uniform key distribution
            TestDriver<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, DESCENDING, BLOCKED_OUTPUT, Key, Value>(
                UNIFORM, 0, begin_bit, end_bit);

            // Sequential key distribution
            TestDriver<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, DESCENDING, BLOCKED_OUTPUT, Key, Value>(
                INTEGER_SEED, 0, begin_bit, end_bit);

            // Iterate random with entropy_reduction
            for (int entropy_reduction = 0; entropy_reduction <= 9; entropy_reduction += 3)
            {
                TestDriver<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, DESCENDING, BLOCKED_OUTPUT, Key, Value>(
                    RANDOM, entropy_reduction, begin_bit, end_bit);
            }
        }
    }
}


/**
 * Test driver (invalid tile size)
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    bool                    DESCENDING,
    bool                    BLOCKED_OUTPUT,
    typename                Key,
    typename                Value>
void TestValid(Int2Type<false> fits_smem_capacity)
{}


/**
 * Test ascending/descending and to-blocked/to-striped
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                Key,
    typename                Value>
void Test()
{
    // Check size of smem storage for the target arch to make sure it will fit
    typedef BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD, Value, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG> BlockRadixSortT;

#if defined(SM100) || defined(SM110) || defined(SM130)
    Int2Type<sizeof(typename BlockRadixSortT::TempStorage) <= 16 * 1024> fits_smem_capacity;
#else
    Int2Type<(sizeof(typename BlockRadixSortT::TempStorage) <= 48 * 1024)> fits_smem_capacity;
#endif

    // Sort-ascending, to-striped
    TestValid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, true, false, Key, Value>(fits_smem_capacity);

    // Sort-descending, to-blocked
    TestValid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, false, true, Key, Value>(fits_smem_capacity);

    // Not necessary
//    TestValid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, false, false, Key, Value>(fits_smem_capacity);
//    TestValid<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, SMEM_CONFIG, true, true, Key, Value>(fits_smem_capacity);
}


/**
 * Test value type and smem config
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    typename                Key>
void TestKeys()
{
    // Test keys-only sorting with both smem configs
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeFourByte, Key, NullType>();    // Keys-only (4-byte smem bank config)
#if !defined(SM100) && !defined(SM110) && !defined(SM130) && !defined(SM200)
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeEightByte, Key, NullType>();   // Keys-only (8-byte smem bank config)
#endif
}


/**
 * Test value type and smem config
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM,
    typename                Key>
void TestKeysAndPairs()
{
    // Test pairs sorting with only 4-byte configs
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeFourByte, Key, char>();        // With small-values
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeFourByte, Key, Key>();         // With same-values
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, cudaSharedMemBankSizeFourByte, Key, TestFoo>();     // With large values
}


/**
 * Test key type
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM>
void Test()
{
    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

#ifdef TEST_KEYS_ONLY

    // Test unsigned types with keys-only
    TestKeys<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, unsigned char>();
    TestKeys<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, unsigned short>();
    TestKeys<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, unsigned int>();
    TestKeys<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, unsigned long>();
    TestKeys<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, unsigned long long>();

#else

    // Test signed and fp types with paired values
    TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, char>();
    TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, short>();
    TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, int>();
    TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, long>();
    TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, long long>();
    TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, float>();
    if (ptx_version > 120)
    {
        // Don't check doubles on PTX120 or below because they're down-converted
        TestKeysAndPairs<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, INNER_SCAN_ALGORITHM, double>();
    }

#endif
}


/**
 * Test inner scan algorithm
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    bool                    MEMOIZE_OUTER_SCAN>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, BLOCK_SCAN_RAKING>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, MEMOIZE_OUTER_SCAN, BLOCK_SCAN_WARP_SCANS>();
}


/**
 * Test outer scan algorithm
 */
template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, true>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, RADIX_BITS, false>();
}


/**
 * Test radix bits
 */
template <
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
void Test()
{
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 1>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 2>();
    Test<BLOCK_THREADS, ITEMS_PER_THREAD, 5>();
}


/**
 * Test items per thread
 */
template <int BLOCK_THREADS>
void Test()
{
    Test<BLOCK_THREADS, 1>();
#if defined(SM100) || defined(SM110) || defined(SM130)
    // Open64 compiler can't handle the number of test cases
#else
    Test<BLOCK_THREADS, 4>();
#endif
    Test<BLOCK_THREADS, 11>();
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

#ifdef QUICK_TEST

    {
        typedef float T;
        TestDriver<32, 4, 4, true, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, false, false, T, NullType>(INTEGER_SEED, 0, 0, sizeof(T) * 8);
    }
/*
    // Compile/run quick tests
    typedef unsigned int T;
    TestDriver<64, 17, 4, true, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, false, false, T, NullType>(RANDOM, 0, 0, sizeof(T) * 8);
    TestDriver<96, 8, 4, true, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, false, false, T, NullType>(RANDOM, 0, 0, sizeof(T) * 8);
    TestDriver<128, 2, 4, true, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, false, false, T, NullType>(RANDOM, 0, 0, sizeof(T) * 8);
*/

#else

    // Compile/run thorough tests
    Test<32>();
    Test<64>();
    Test<160>();


#endif  // QUICK_TEST

    return 0;
}



