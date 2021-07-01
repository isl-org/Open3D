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

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>

#include "histogram/histogram_gmem_atomics.h"
#include "histogram/histogram_smem_atomics.h"
#include "histogram/histogram_cub.h"

#include <cub/util_allocator.cuh>
#include <test/test_util.h>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

bool                    g_verbose = false;  // Whether to display input/output to console
bool                    g_report = false;   // Whether to display a full report in CSV format
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

struct less_than_value
{
    inline bool operator()(
        const std::pair<std::string, double> &a,
        const std::pair<std::string, double> &b)
    {
        return a.second < b.second;
    }
};


//---------------------------------------------------------------------
// Targa (.tga) image file parsing
//---------------------------------------------------------------------

/**
 * TGA image header info
 */
struct TgaHeader
{
    char idlength;
    char colormaptype;
    char datatypecode;
    short colormaporigin;
    short colormaplength;
    char colormapdepth;
    short x_origin;
    short y_origin;
    short width;
    short height;
    char bitsperpixel;
    char imagedescriptor;

    void Parse (FILE *fptr)
    {
        idlength = fgetc(fptr);
        colormaptype = fgetc(fptr);
        datatypecode = fgetc(fptr);
        fread(&colormaporigin, 2, 1, fptr);
        fread(&colormaplength, 2, 1, fptr);
        colormapdepth = fgetc(fptr);
        fread(&x_origin, 2, 1, fptr);
        fread(&y_origin, 2, 1, fptr);
        fread(&width, 2, 1, fptr);
        fread(&height, 2, 1, fptr);
        bitsperpixel = fgetc(fptr);
        imagedescriptor = fgetc(fptr);
    }

    void Display (FILE *fptr)
    {
        fprintf(fptr, "ID length:           %d\n", idlength);
        fprintf(fptr, "Color map type:      %d\n", colormaptype);
        fprintf(fptr, "Image type:          %d\n", datatypecode);
        fprintf(fptr, "Color map offset:    %d\n", colormaporigin);
        fprintf(fptr, "Color map length:    %d\n", colormaplength);
        fprintf(fptr, "Color map depth:     %d\n", colormapdepth);
        fprintf(fptr, "X origin:            %d\n", x_origin);
        fprintf(fptr, "Y origin:            %d\n", y_origin);
        fprintf(fptr, "Width:               %d\n", width);
        fprintf(fptr, "Height:              %d\n", height);
        fprintf(fptr, "Bits per pixel:      %d\n", bitsperpixel);
        fprintf(fptr, "Descriptor:          %d\n", imagedescriptor);
    }
};


/**
 * Decode image byte data into pixel
 */
void ParseTgaPixel(uchar4 &pixel, unsigned char *tga_pixel, int bytes)
{
    if (bytes == 4)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = tga_pixel[3];
    }
    else if (bytes == 3)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = 0;
    }
    else if (bytes == 2)
    {
        pixel.x = (tga_pixel[1] & 0x7c) << 1;
        pixel.y = ((tga_pixel[1] & 0x03) << 6) | ((tga_pixel[0] & 0xe0) >> 2);
        pixel.z = (tga_pixel[0] & 0x1f) << 3;
        pixel.w = (tga_pixel[1] & 0x80);
    }
}


/**
 * Reads a .tga image file
 */
void ReadTga(uchar4* &pixels, int &width, int &height, const char *filename)
{
    // Open the file
    FILE *fptr;
    if ((fptr = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "File open failed\n");
        exit(-1);
    }

    // Parse header
    TgaHeader header;
    header.Parse(fptr);
//    header.Display(stdout);
    width = header.width;
    height = header.height;

    // Verify compatibility
    if (header.datatypecode != 2 && header.datatypecode != 10)
    {
        fprintf(stderr, "Can only handle image type 2 and 10\n");
        exit(-1);
    }
    if (header.bitsperpixel != 16 && header.bitsperpixel != 24 && header.bitsperpixel != 32)
    {
        fprintf(stderr, "Can only handle pixel depths of 16, 24, and 32\n");
        exit(-1);
    }
    if (header.colormaptype != 0 && header.colormaptype != 1)
    {
        fprintf(stderr, "Can only handle color map types of 0 and 1\n");
        exit(-1);
    }

    // Skip unnecessary header info
    int skip_bytes = header.idlength + (header.colormaptype * header.colormaplength);
    fseek(fptr, skip_bytes, SEEK_CUR);

    // Read the image
    int pixel_bytes = header.bitsperpixel / 8;

    // Allocate and initialize pixel data
    size_t image_bytes = width * height * sizeof(uchar4);
    if ((pixels == NULL) && ((pixels = (uchar4*) malloc(image_bytes)) == NULL))
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }
    memset(pixels, 0, image_bytes);

    // Parse pixels
    unsigned char   tga_pixel[5];
    int             current_pixel = 0;
    while (current_pixel < header.width * header.height)
    {
        if (header.datatypecode == 2)
        {
            // Uncompressed
            if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d  (uncompressed)\n", current_pixel);
                exit(-1);
            }
            ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
            current_pixel++;
        }
        else if (header.datatypecode == 10)
        {
            // Compressed
            if (fread(tga_pixel, 1, pixel_bytes + 1, fptr) != pixel_bytes + 1)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d (compressed)\n", current_pixel);
                exit(-1);
            }
            int run_length = tga_pixel[0] & 0x7f;
            ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
            current_pixel++;

            if (tga_pixel[0] & 0x80)
            {
                // RLE chunk
                for (int i = 0; i < run_length; i++)
                {
                    ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
                    current_pixel++;
                }
            }
            else
            {
                // Normal chunk
                for (int i = 0; i < run_length; i++)
                {
                    if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
                    {
                        fprintf(stderr, "Unexpected end of file at pixel %d (normal)\n", current_pixel);
                        exit(-1);
                    }
                    ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
                    current_pixel++;
                }
            }
        }
    }

    // Close file
    fclose(fptr);
}



//---------------------------------------------------------------------
// Random image generation
//---------------------------------------------------------------------

/**
 * Generate a random image with specified entropy
 */
void GenerateRandomImage(uchar4* &pixels, int width, int height, int entropy_reduction)
{
    int num_pixels = width * height;
    size_t image_bytes = num_pixels * sizeof(uchar4);
    if ((pixels == NULL) && ((pixels = (uchar4*) malloc(image_bytes)) == NULL))
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }

    for (int i = 0; i < num_pixels; ++i)
    {
        RandomBits(pixels[i].x, entropy_reduction);
        RandomBits(pixels[i].y, entropy_reduction);
        RandomBits(pixels[i].z, entropy_reduction);
        RandomBits(pixels[i].w, entropy_reduction);
    }
}



//---------------------------------------------------------------------
// Histogram verification
//---------------------------------------------------------------------

// Decode float4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    float* samples = reinterpret_cast<float*>(&pixel);

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
}

// Decode uchar4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}

// Decode uchar1 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    bins[0] = (unsigned int) pixel.x;
}


// Compute reference histogram.  Specialized for uchar4
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void HistogramGold(PixelType *image, int width, int height, unsigned int* hist)
{
    memset(hist, 0, ACTIVE_CHANNELS * NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            PixelType pixel = image[i + j * width];

            unsigned int bins[ACTIVE_CHANNELS];
            DecodePixelGold<NUM_BINS>(pixel, bins);

            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            {
                hist[(NUM_BINS * CHANNEL) + bins[CHANNEL]]++;
            }
        }
    }
}


//---------------------------------------------------------------------
// Test execution
//---------------------------------------------------------------------

/**
 * Run a specific histogram implementation
 */
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void RunTest(
    std::vector<std::pair<std::string, double> >&   timings,
    PixelType*                                      d_pixels,
    const int                                       width,
    const int                                       height,
    unsigned int *                                  d_hist,
    unsigned int *                                  h_hist,
    int                                             timing_iterations,
    const char *                                    long_name,
    const char *                                    short_name,
    double (*f)(PixelType*, int, int, unsigned int*, bool))
{
    if (!g_report) printf("%s ", long_name); fflush(stdout);

    // Run single test to verify (and code cache)
    (*f)(d_pixels, width, height, d_hist, !g_report);

    int compare = CompareDeviceResults(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS, true, g_verbose);
    if (!g_report) printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);

    double elapsed_ms = 0;
    for (int i = 0; i < timing_iterations; i++)
    {
        elapsed_ms += (*f)(d_pixels, width, height, d_hist, false);
    }
    double avg_us = (elapsed_ms / timing_iterations) * 1000;    // average in us
    timings.push_back(std::pair<std::string, double>(short_name, avg_us));

    if (!g_report)
    {
        printf("Avg time %.3f us (%d iterations)\n", avg_us, timing_iterations); fflush(stdout);
    }
    else
    {
        printf("%.3f, ", avg_us); fflush(stdout);
    }

    AssertEquals(0, compare);
}


/**
 * Evaluate corpus of histogram implementations
 */
template <
    int         NUM_CHANNELS,
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void TestMethods(
    PixelType*  h_pixels,
    int         height,
    int         width,
    int         timing_iterations,
    double      bandwidth_GBs)
{
    // Copy data to gpu
    PixelType* d_pixels;
    size_t pixel_bytes = width * height * sizeof(PixelType);
    CubDebugExit(g_allocator.DeviceAllocate((void**) &d_pixels, pixel_bytes));
    CubDebugExit(cudaMemcpy(d_pixels, h_pixels, pixel_bytes, cudaMemcpyHostToDevice));

    if (g_report) printf("%.3f, ", double(pixel_bytes) / bandwidth_GBs / 1000);

    // Allocate results arrays on cpu/gpu
    unsigned int *h_hist;
    unsigned int *d_hist;
    size_t histogram_bytes = NUM_BINS * ACTIVE_CHANNELS * sizeof(unsigned int);
    h_hist = (unsigned int *) malloc(histogram_bytes);
    g_allocator.DeviceAllocate((void **) &d_hist, histogram_bytes);

    // Compute reference cpu histogram
    HistogramGold<ACTIVE_CHANNELS, NUM_BINS>(h_pixels, width, height, h_hist);

    // Store timings
    std::vector<std::pair<std::string, double> > timings;

    // Run experiments
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "CUB", "CUB", run_cub_histogram<NUM_CHANNELS, ACTIVE_CHANNELS, NUM_BINS, PixelType>);
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Shared memory atomics", "smem atomics", run_smem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Global memory atomics", "gmem atomics", run_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);

    // Report timings
    if (!g_report)
    {
        std::sort(timings.begin(), timings.end(), less_than_value());
        printf("Timings (us):\n");
        for (int i = 0; i < timings.size(); i++)
        {
            double bandwidth = height * width * sizeof(PixelType) / timings[i].second / 1000;
            printf("\t %.3f %s (%.3f GB/s, %.3f%% peak)\n", timings[i].second, timings[i].first.c_str(), bandwidth, bandwidth / bandwidth_GBs * 100);
        }
        printf("\n");
    }

    // Free data
    CubDebugExit(g_allocator.DeviceFree(d_pixels));
    CubDebugExit(g_allocator.DeviceFree(d_hist));
    free(h_hist);
}


/**
 * Test different problem genres
 */
void TestGenres(
    uchar4*     uchar4_pixels,
    int         height,
    int         width,
    int         timing_iterations,
    double      bandwidth_GBs)
{
    int num_pixels = width * height;

    {
        if (!g_report) printf("1 channel uchar1 tests (256-bin):\n\n"); fflush(stdout);

        size_t      image_bytes     = num_pixels * sizeof(uchar1);
        uchar1*     uchar1_pixels   = (uchar1*) malloc(image_bytes);

        // Convert to 1-channel (averaging first 3 channels)
        for (int i = 0; i < num_pixels; ++i)
        {
            uchar1_pixels[i].x = (unsigned char)
                (((unsigned int) uchar4_pixels[i].x +
                  (unsigned int) uchar4_pixels[i].y +
                  (unsigned int) uchar4_pixels[i].z) / 3);
        }

        TestMethods<1, 1, 256>(uchar1_pixels, width, height, timing_iterations, bandwidth_GBs);
        free(uchar1_pixels);
        if (g_report) printf(", ");
    }

    {
        if (!g_report) printf("3/4 channel uchar4 tests (256-bin):\n\n"); fflush(stdout);
        TestMethods<4, 3, 256>(uchar4_pixels, width, height, timing_iterations, bandwidth_GBs);
        if (g_report) printf(", ");
    }

    {
        if (!g_report) printf("3/4 channel float4 tests (256-bin):\n\n"); fflush(stdout);
        size_t      image_bytes     = num_pixels * sizeof(float4);
        float4*     float4_pixels   = (float4*) malloc(image_bytes);

        // Convert to float4 with range [0.0, 1.0)
        for (int i = 0; i < num_pixels; ++i)
        {
            float4_pixels[i].x = float(uchar4_pixels[i].x) / 256;
            float4_pixels[i].y = float(uchar4_pixels[i].y) / 256;
            float4_pixels[i].z = float(uchar4_pixels[i].z) / 256;
            float4_pixels[i].w = float(uchar4_pixels[i].w) / 256;
        }
        TestMethods<4, 3, 256>(float4_pixels, width, height, timing_iterations, bandwidth_GBs);
        free(float4_pixels);
        if (g_report) printf("\n");
    }
}


/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--i=<timing iterations>] "
            "\n\t"
                "--file=<.tga filename> "
            "\n\t"
                "--entropy=<-1 (0%), 0 (100%), 1 (81%), 2 (54%), 3 (34%), 4 (20%), ..."
                "[--height=<default: 1080>] "
                "[--width=<default: 1920>] "
            "\n", argv[0]);
        exit(0);
    }

    std::string         filename;
    int                 timing_iterations   = 100;
    int                 entropy_reduction   = 0;
    int                 height              = 1080;
    int                 width               = 1920;

    g_verbose = args.CheckCmdLineFlag("v");
    g_report = args.CheckCmdLineFlag("report");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("file", filename);
    args.GetCmdLineArgument("height", height);
    args.GetCmdLineArgument("width", width);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get GPU device bandwidth (GB/s)
    int device_ordinal, bus_width, mem_clock_khz;
    CubDebugExit(cudaGetDevice(&device_ordinal));
    CubDebugExit(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device_ordinal));
    CubDebugExit(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_ordinal));
    double bandwidth_GBs = double(bus_width) * mem_clock_khz * 2 / 8 / 1000 / 1000;

    // Run test(s)
    uchar4* uchar4_pixels = NULL;
    if (!g_report)
    {
        if (!filename.empty())
        {
            // Parse targa file
            ReadTga(uchar4_pixels, width, height, filename.c_str());
            printf("File %s: width(%d) height(%d)\n\n", filename.c_str(), width, height); fflush(stdout);
        }
        else
        {
            // Generate image
            GenerateRandomImage(uchar4_pixels, width, height, entropy_reduction);
            printf("Random image: entropy-reduction(%d) width(%d) height(%d)\n\n", entropy_reduction, width, height); fflush(stdout);
        }

        TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
    }
    else
    {
        // Run test suite
        printf("Test, MIN, RLE CUB, SMEM, GMEM, , MIN, RLE_CUB, SMEM, GMEM, , MIN, RLE_CUB, SMEM, GMEM\n");

        // Entropy reduction tests
        for (entropy_reduction = 0; entropy_reduction < 5; ++entropy_reduction)
        {
            printf("entropy reduction %d, ", entropy_reduction);
            GenerateRandomImage(uchar4_pixels, width, height, entropy_reduction);
            TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        }
        printf("entropy reduction -1, ");
        GenerateRandomImage(uchar4_pixels, width, height, -1);
        TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        printf("\n");

        // File image tests
        std::vector<std::string> file_tests;
        file_tests.push_back("animals");
        file_tests.push_back("apples");
        file_tests.push_back("sunset");
        file_tests.push_back("cheetah");
        file_tests.push_back("nature");
        file_tests.push_back("operahouse");
        file_tests.push_back("austin");
        file_tests.push_back("cityscape");

        for (int i = 0; i < file_tests.size(); ++i)
        {
            printf("%s, ", file_tests[i].c_str());
            std::string filename = std::string("histogram/benchmark/") + file_tests[i] + ".tga";
            ReadTga(uchar4_pixels, width, height, filename.c_str());
            TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        }
    }

    free(uchar4_pixels);

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n\n");

    return 0;
}
