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
 * Simple example of DeviceSelect::Flagged().
 *
 * Selects flagged items from from a sequence of int keys using a
 * corresponding sequence of unsigned char flags.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_select_flagged.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>

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
 * Initialize problem, setting flags at distances of random length
 * chosen from [1..max_segment]
 */
void Initialize(
    int             *h_in,
    unsigned char   *h_flags,
    int             num_items,
    int             max_segment)
{
    unsigned short max_short = (unsigned short) -1;

    int key = 0;
    int i = 0;
    while (i < num_items)
    {
        // Select number of repeating occurrences
        unsigned short repeat;
        RandomBits(repeat);
        repeat = (unsigned short) ((float(repeat) * (float(max_segment) / float(max_short))));
        repeat = CUB_MAX(1, repeat);

        int j = i;
        while (j < CUB_MIN(i + repeat, num_items))
        {
            h_flags[j] = 0;
            h_in[j] = key;
            j++;
        }

        h_flags[i] = 1;
        i = j;
        key++;
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("Flags:\n");
        DisplayResults(h_flags, num_items);
        printf("\n\n");
    }
}


/**
 * Solve unique problem
 */
int Solve(
    int             *h_in,
    unsigned char   *h_flags,
    int             *h_reference,
    int             num_items)
{
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i)
    {
        if (h_flags[i])
        {
            h_reference[num_selected] = h_in[i];
            num_selected++;
        }
        else
        {
            h_reference[num_items - (i - num_selected) - 1] = h_in[i];
        }
    }

    return num_selected;
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items           = 150;
    int max_segment         = 40;       // Maximum segment length

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("maxseg", max_segment);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--device=<device-id>] "
            "[--maxseg=<max segment length>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Allocate host arrays
    int             *h_in        = new int[num_items];
    int             *h_reference = new int[num_items];
    unsigned char   *h_flags     = new unsigned char[num_items];

    // Initialize problem and solution
    Initialize(h_in, h_flags, num_items, max_segment);
    int num_selected = Solve(h_in, h_flags, h_reference, num_items);

    printf("cub::DeviceSelect::Flagged %d items, %d selected (avg distance %d), %d-byte elements\n",
        num_items, num_selected, (num_selected > 0) ? num_items / num_selected : 0, (int) sizeof(int));
    fflush(stdout);

    // Allocate problem device arrays
    int             *d_in = NULL;
    unsigned char   *d_flags = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_flags, sizeof(unsigned char) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_flags, h_flags, sizeof(unsigned char) * num_items, cudaMemcpyHostToDevice));

    // Allocate device output array and num selected
    int     *d_out            = NULL;
    int     *d_num_selected_out   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int)));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run
    CubDebugExit(DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, num_selected, true, g_verbose);
    printf("\t Data %s ", compare ? "FAIL" : "PASS");
    compare |= CompareDeviceResults(&num_selected, d_num_selected_out, 1, true, g_verbose);
    printf("\t Count %s ", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d_num_selected_out));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_flags) CubDebugExit(g_allocator.DeviceFree(d_flags));

    printf("\n\n");

    return 0;
}



