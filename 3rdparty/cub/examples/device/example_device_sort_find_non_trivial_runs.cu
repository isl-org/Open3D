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
 * Simple example of sorting a sequence of keys and values (each pair is a
 * randomly-selected int32 paired with its original offset in the unsorted sequence), and then
 * isolating all maximal, non-trivial (having length > 1) "runs" of duplicates.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_sort_find_non_trivial_runs.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include "../../test/test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Simple key-value pairing for using std::sort on key-value pairs.
 */
template <typename Key, typename Value>
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
 * Pair ostream operator
 */
template <typename Key, typename Value>
std::ostream& operator<<(std::ostream& os, const Pair<Key, Value>& val)
{
    os << '<' << val.key << ',' << val.value << '>';
    return os;
}


/**
 * Initialize problem
 */
template <typename Key, typename Value>
void Initialize(
    Key    *h_keys,
    Value  *h_values,
    int    num_items,
    int    max_key)
{
    float scale = float(max_key) / float(UINT_MAX);
    for (int i = 0; i < num_items; ++i)
    {
        Key sample;
        RandomBits(sample);
        h_keys[i] = (max_key == -1) ? i : (Key) (scale * sample);
        h_values[i] = i;
    }

    if (g_verbose)
    {
        printf("Keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");

        printf("Values:\n");
        DisplayResults(h_values, num_items);
        printf("\n\n");
    }
}


/**
 * Solve sorted non-trivial subrange problem.  Returns the number
 * of non-trivial runs found.
 */
template <typename Key, typename Value>
int Solve(
    Key     *h_keys,
    Value   *h_values,
    int     num_items,
    int     *h_offsets_reference,
    int     *h_lengths_reference)
{
    // Sort

    Pair<Key, Value> *h_pairs = new Pair<Key, Value>[num_items];
    for (int i = 0; i < num_items; ++i)
    {
        h_pairs[i].key    = h_keys[i];
        h_pairs[i].value  = h_values[i];
    }

    std::stable_sort(h_pairs, h_pairs + num_items);

    if (g_verbose)
    {
        printf("Sorted pairs:\n");
        DisplayResults(h_pairs, num_items);
        printf("\n\n");
    }

    // Find non-trivial runs

    Key     previous        = h_pairs[0].key;
    int     length          = 1;
    int     num_runs        = 0;
    int     run_begin       = 0;

    for (int i = 1; i < num_items; ++i)
    {
        if (previous != h_pairs[i].key)
        {
            if (length > 1)
            {
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
        previous = h_pairs[i].key;
    }

    if (length > 1)
    {
        h_offsets_reference[num_runs]   = run_begin;
        h_lengths_reference[num_runs]   = length;
        num_runs++;
    }

    delete[] h_pairs;

    return num_runs;
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    typedef unsigned int    Key;
    typedef int             Value;

    int timing_iterations   = 0;
    int num_items           = 40;
    Key max_key             = 20;       // Max item

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("maxkey", max_key);
    args.GetCmdLineArgument("i", timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--i=<timing iterations> "
            "[--n=<input items, default 40> "
            "[--maxkey=<max key, default 20 (use -1 to test only unique keys)>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Allocate host arrays (problem and reference solution)

    Key     *h_keys                 = new Key[num_items];
    Value   *h_values               = new Value[num_items];
    int     *h_offsets_reference    = new int[num_items];
    int     *h_lengths_reference    = new int[num_items];

    // Initialize key-value pairs and compute reference solution (sort them, and identify non-trivial runs)
    printf("Computing reference solution on CPU for %d items (max key %d)\n", num_items, max_key);
    fflush(stdout);

    Initialize(h_keys, h_values, num_items, max_key);
    int num_runs = Solve(h_keys, h_values, num_items, h_offsets_reference, h_lengths_reference);

    printf("%d non-trivial runs\n", num_runs);
    fflush(stdout);

    // Repeat for performance timing
    GpuTimer gpu_timer;
    GpuTimer gpu_rle_timer;
    float elapsed_millis = 0.0;
    float elapsed_rle_millis = 0.0;
    for (int i = 0; i <= timing_iterations; ++i)
    {

        // Allocate and initialize device arrays for sorting
        DoubleBuffer<Key>       d_keys;
        DoubleBuffer<Value>     d_values;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(Key) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(Key) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(Value) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(Value) * num_items));

        CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));

        // Start timer
        gpu_timer.Start();

        // Allocate temporary storage for sorting
        size_t  temp_storage_bytes  = 0;
        void    *d_temp_storage     = NULL;
        CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

        // Do the sort
        CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

        // Free unused buffers and sorting temporary storage
        if (d_keys.d_buffers[d_keys.selector ^ 1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[d_keys.selector ^ 1]));
        if (d_values.d_buffers[d_values.selector ^ 1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[d_values.selector ^ 1]));
        if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

        // Start timer
        gpu_rle_timer.Start();

        // Allocate device arrays for enumerating non-trivial runs
        int     *d_offests_out   = NULL;
        int     *d_lengths_out   = NULL;
        int     *d_num_runs      = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_offests_out, sizeof(int) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_lengths_out, sizeof(int) * num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_runs, sizeof(int) * 1));

        // Allocate temporary storage for isolating non-trivial runs
        d_temp_storage = NULL;
        CubDebugExit(DeviceRunLengthEncode::NonTrivialRuns(
            d_temp_storage,
            temp_storage_bytes,
            d_keys.d_buffers[d_keys.selector],
            d_offests_out,
            d_lengths_out,
            d_num_runs,
            num_items));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

        // Do the isolation
        CubDebugExit(DeviceRunLengthEncode::NonTrivialRuns(
            d_temp_storage,
            temp_storage_bytes,
            d_keys.d_buffers[d_keys.selector],
            d_offests_out,
            d_lengths_out,
            d_num_runs,
            num_items));

        // Free keys buffer
        if (d_keys.d_buffers[d_keys.selector]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[d_keys.selector]));

        //
        // Hypothetically do stuff with the original key-indices corresponding to non-trivial runs of identical keys
        //

        // Stop sort timer
        gpu_timer.Stop();
        gpu_rle_timer.Stop();

        if (i == 0)
        {
            // First iteration is a warmup: // Check for correctness (and display results, if specified)

            printf("\nRUN OFFSETS: \n");
            int compare = CompareDeviceResults(h_offsets_reference, d_offests_out, num_runs, true, g_verbose);
            printf("\t\t %s ", compare ? "FAIL" : "PASS");

            printf("\nRUN LENGTHS: \n");
            compare |= CompareDeviceResults(h_lengths_reference, d_lengths_out, num_runs, true, g_verbose);
            printf("\t\t %s ", compare ? "FAIL" : "PASS");

            printf("\nNUM RUNS: \n");
            compare |= CompareDeviceResults(&num_runs, d_num_runs, 1, true, g_verbose);
            printf("\t\t %s ", compare ? "FAIL" : "PASS");

            AssertEquals(0, compare);
        }
        else
        {
            elapsed_millis += gpu_timer.ElapsedMillis();
            elapsed_rle_millis += gpu_rle_timer.ElapsedMillis();
        }

        // GPU cleanup

        if (d_values.d_buffers[d_values.selector]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[d_values.selector]));
        if (d_offests_out) CubDebugExit(g_allocator.DeviceFree(d_offests_out));
        if (d_lengths_out) CubDebugExit(g_allocator.DeviceFree(d_lengths_out));
        if (d_num_runs) CubDebugExit(g_allocator.DeviceFree(d_num_runs));
        if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    }

    // Host cleanup
    if (h_keys) delete[] h_keys;
    if (h_values) delete[] h_values;
    if (h_offsets_reference) delete[] h_offsets_reference;
    if (h_lengths_reference) delete[] h_lengths_reference;

    printf("\n\n");

    if (timing_iterations > 0)
    {
        printf("%d timing iterations, average time to sort and isolate non-trivial duplicates: %.3f ms (%.3f ms spent in RLE isolation)\n",
            timing_iterations,
            elapsed_millis / timing_iterations,
            elapsed_rle_millis / timing_iterations);
    }

    return 0;
}



