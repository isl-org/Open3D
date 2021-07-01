/******************************************************************************
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

#include <cub/device/device_histogram.cuh>

using namespace cub;

template <
    int         NUM_CHANNELS,
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_cub_histogram(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist, 
    bool is_warmup)
{
    enum {
        is_float = Equals<PixelType, float4>::VALUE,
    };

    typedef typename If<is_float, float, unsigned char>::Type    SampleT;    // Sample type
    typedef typename If<is_float, float, unsigned int>::Type     LevelT;     // Level type (uint32 for uchar)

    // Setup data structures
    unsigned int*       d_histogram[ACTIVE_CHANNELS];
    int                 num_levels[ACTIVE_CHANNELS];            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              lower_level[ACTIVE_CHANNELS];           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT              upper_level[ACTIVE_CHANNELS];           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
    {
        d_histogram[CHANNEL] = d_hist + (CHANNEL * NUM_BINS);
        num_levels[CHANNEL] = NUM_BINS + 1;
        lower_level[CHANNEL] = 0;
        upper_level[CHANNEL] = (is_float) ? 1 : 256;
    }

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    SampleT* d_image_samples = (SampleT*) d_image;

    // Get amount of temporary storage needed
    DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, ACTIVE_CHANNELS>(
        d_temp_storage,
        temp_storage_bytes,
        d_image_samples,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        width * height, 
        (cudaStream_t) 0,
        is_warmup);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    // Compute histogram
    DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, ACTIVE_CHANNELS>(
        d_temp_storage,
        temp_storage_bytes,
        d_image_samples,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        width * height, 
        (cudaStream_t) 0,
        is_warmup);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    cudaFree(d_temp_storage);

    return elapsed_millis;
}

