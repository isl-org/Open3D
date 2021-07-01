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
 * Simple example of DeviceRadixSort::SortPairs().
 *
 * Sorts an array of float keys paired with a corresponding array of int values.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_radix_sort.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

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
 * Simple key-value pairing for floating point types.  Distinguishes
 * between positive and negative zero.
 */
struct Pair
{
    float   key;
    int     value;

    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;

        if (key > b.key)
            return false;

        // Return true if key is negative zero and b.key is positive zero
        unsigned int key_bits   = *reinterpret_cast<unsigned*>(const_cast<float*>(&key));
        unsigned int b_key_bits = *reinterpret_cast<unsigned*>(const_cast<float*>(&b.key));
        unsigned int HIGH_BIT   = 1u << 31;

        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};


/**
 * Initialize key-value sorting problem.
 */
void Initialize(
    float           *h_keys,
    int             *h_values,
    float           *h_reference_keys,
    int             *h_reference_values,
    int             num_items)
{
    Pair *h_pairs = new Pair[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        RandomBits(h_keys[i]);
        RandomBits(h_values[i]);
        h_pairs[i].key    = h_keys[i];
        h_pairs[i].value  = h_values[i];
    }

    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");

        printf("Input values:\n");
        DisplayResults(h_values, num_items);
        printf("\n\n");
    }

    std::stable_sort(h_pairs, h_pairs + num_items);

    for (int i = 0; i < num_items; ++i)
    {
        h_reference_keys[i]     = h_pairs[i].key;
        h_reference_values[i]   = h_pairs[i].value;
    }

    delete[] h_pairs;
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items = 150;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    printf("cub::DeviceRadixSort::SortPairs() %d items (%d-byte keys %d-byte values)\n",
        num_items, int(sizeof(float)), int(sizeof(int)));
    fflush(stdout);

    // Allocate host arrays
    float   *h_keys             = new float[num_items];
    float   *h_reference_keys   = new float[num_items];
    int     *h_values           = new int[num_items];
    int     *h_reference_values = new int[num_items];

    // Initialize problem and solution on host
    Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items);

    // Allocate device arrays
    DoubleBuffer<float> d_keys;
    DoubleBuffer<int>   d_values;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(int) * num_items));

    // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;

    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));

    // Run
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s\n", d_keys.selector, compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);
    compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
    printf("\t Compare values (selector %d): %s\n", d_values.selector, compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_keys) delete[] h_keys;
    if (h_reference_keys) delete[] h_reference_keys;
    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;

    if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
    if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
    if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
    if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    printf("\n\n");

    return 0;
}



