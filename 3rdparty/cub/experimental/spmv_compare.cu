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
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIAeBILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//---------------------------------------------------------------------
// SpMV comparison tool
//---------------------------------------------------------------------

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>

#include <cusparse.h>

#include "sparse_matrix.h"

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <test/test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet     = false;        // Whether to display stats in CSV format
bool                    g_verbose   = false;        // Whether to display output to console
bool                    g_verbose2  = false;        // Whether to display input to console
CachingDeviceAllocator  g_allocator(true);          // Caching allocator for device memory


//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
    typename ValueT,
    typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta)
{
    for (OffsetT row = 0; row < a.num_rows; ++row)
    {
        ValueT partial = beta * vector_y_in[row];
        for (
            OffsetT offset = a.row_offsets[row];
            offset < a.row_offsets[row + 1];
            ++offset)
        {
            partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}


//---------------------------------------------------------------------
// GPU I/O proxy
//---------------------------------------------------------------------

/**
 * Read every matrix nonzero value, read every corresponding vector value
 */
template <
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    typename    ValueT,
    typename    OffsetT,
    typename    VectorItr>
__launch_bounds__ (int(BLOCK_THREADS))
__global__ void NonZeroIoKernel(
    SpmvParams<ValueT, OffsetT> params,
    VectorItr                   d_vector_x)
{
    enum
    {
        TILE_ITEMS      = BLOCK_THREADS * ITEMS_PER_THREAD,
    };


    ValueT nonzero = 0.0;

    int tile_idx = blockIdx.x;

    OffsetT block_offset = tile_idx * TILE_ITEMS;

    OffsetT column_indices[ITEMS_PER_THREAD];
    ValueT values[ITEMS_PER_THREAD];

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
        OffsetT nonzero_idx = block_offset + (ITEM * BLOCK_THREADS) + threadIdx.x;

        OffsetT* ci = params.d_column_indices + nonzero_idx;
        ValueT*a = params.d_values + nonzero_idx;

        column_indices[ITEM]    = (nonzero_idx < params.num_nonzeros) ? *ci : 0;
        values[ITEM]            = (nonzero_idx < params.num_nonzeros) ? *a : 0.0;
    }

    __syncthreads();

    // Read vector
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
        ValueT vector_value    = ThreadLoad<LOAD_LDG>(params.d_vector_x + column_indices[ITEM]);
        nonzero                += vector_value * values[ITEM];
    }

    __syncthreads();

    if (block_offset < params.num_rows)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            OffsetT row_idx = block_offset + (ITEM * BLOCK_THREADS) + threadIdx.x;
            if (row_idx < params.num_rows)
            {
                OffsetT row_end_offset = ThreadLoad<LOAD_DEFAULT>(params.d_row_end_offsets + row_idx);

                if ((row_end_offset >= 0) && (nonzero == nonzero))
                    params.d_vector_y[row_idx] = nonzero;
            }
        }
    }

}


/**
 * Run GPU I/O proxy
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuCsrIoProxy(
    SpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations)
{
    enum {
        BLOCK_THREADS       = 128,
        ITEMS_PER_THREAD    = 7,
        TILE_SIZE           = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

//    size_t smem = 1024 * 16;
    size_t smem = 1024 * 0;

    unsigned int nonzero_blocks = (params.num_nonzeros + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int row_blocks = (params.num_rows + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int blocks = std::max(nonzero_blocks, row_blocks);

    typedef TexRefInputIterator<ValueT, 1234, int> TexItr;
    TexItr x_itr;
    CubDebugExit(x_itr.BindTexture(params.d_vector_x));

    // Get device ordinal
    int device_ordinal;
    CubDebugExit(cudaGetDevice(&device_ordinal));

    // Get device SM version
    int sm_version;
    CubDebugExit(SmVersion(sm_version, device_ordinal));

    void (*kernel)(SpmvParams<ValueT, OffsetT>, TexItr) = NonZeroIoKernel<BLOCK_THREADS, ITEMS_PER_THREAD>;


    int spmv_sm_occupancy;
    CubDebugExit(MaxSmOccupancy(spmv_sm_occupancy, kernel, BLOCK_THREADS, smem));

    if (!g_quiet)
        printf("NonZeroIoKernel<%d,%d><<<%d, %d>>>, sm occupancy %d\n", BLOCK_THREADS, ITEMS_PER_THREAD, blocks, BLOCK_THREADS, spmv_sm_occupancy);

    // Warmup
    NonZeroIoKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<blocks, BLOCK_THREADS, smem>>>(params, x_itr);

    // Check for failures
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(SyncStream(0));

    // Timing
    GpuTimer timer;
    float elapsed_millis = 0.0;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        NonZeroIoKernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<blocks, BLOCK_THREADS, smem>>>(params, x_itr);
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    CubDebugExit(x_itr.UnbindTexture());

    return elapsed_millis / timing_iterations;
}



//---------------------------------------------------------------------
// cuSparse HybMV
//---------------------------------------------------------------------

/**
 * Run cuSparse HYB SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float TestCusparseHybmv(
    float*                          vector_y_in,
    float*                          reference_vector_y_out,
    SpmvParams<float, OffsetT>&     params,
    int                             timing_iterations,
    cusparseHandle_t                cusparse)
{
    CpuTimer cpu_timer;
    cpu_timer.Start();

    // Construct Hyb matrix
    cusparseMatDescr_t mat_desc;
    cusparseHybMat_t hyb_desc;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&mat_desc));
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateHybMat(&hyb_desc));
    cusparseStatus_t status = cusparseScsr2hyb(
        cusparse,
        params.num_rows, params.num_cols,
        mat_desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        hyb_desc,
        0,
        CUSPARSE_HYB_PARTITION_AUTO);
    AssertEquals(CUSPARSE_STATUS_SUCCESS, status);

    cudaDeviceSynchronize();
    cpu_timer.Stop();
    float elapsed_millis = cpu_timer.ElapsedMillis();
    printf("HYB setup ms, %.5f, ", elapsed_millis);

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseShybmv(
        cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &params.alpha, mat_desc,
        hyb_desc,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    elapsed_millis    = 0.0;
    GpuTimer timer;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseShybmv(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &params.alpha, mat_desc,
            hyb_desc,
            params.d_vector_x, &params.beta, params.d_vector_y));
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    // Cleanup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyHybMat(hyb_desc));
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(mat_desc));

    return elapsed_millis / timing_iterations;
}


/**
 * Run cuSparse HYB SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float TestCusparseHybmv(
    double*                         vector_y_in,
    double*                         reference_vector_y_out,
    SpmvParams<double, OffsetT>&    params,
    int                             timing_iterations,
    cusparseHandle_t                cusparse)
{
    CpuTimer cpu_timer;
    cpu_timer.Start();

    // Construct Hyb matrix
    cusparseMatDescr_t mat_desc;
    cusparseHybMat_t hyb_desc;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&mat_desc));
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateHybMat(&hyb_desc));
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsr2hyb(
        cusparse,
        params.num_rows, params.num_cols,
        mat_desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        hyb_desc,
        0,
        CUSPARSE_HYB_PARTITION_AUTO));

    cudaDeviceSynchronize();
    cpu_timer.Stop();
    float elapsed_millis = cpu_timer.ElapsedMillis();
    printf("HYB setup ms, %.5f, ", elapsed_millis);

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDhybmv(
        cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &params.alpha, mat_desc,
        hyb_desc,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    elapsed_millis    = 0.0;
    GpuTimer timer;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDhybmv(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &params.alpha, mat_desc,
            hyb_desc,
            params.d_vector_x, &params.beta, params.d_vector_y));
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    // Cleanup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyHybMat(hyb_desc));
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(mat_desc));

    return elapsed_millis / timing_iterations;
}



//---------------------------------------------------------------------
// cuSparse CsrMV
//---------------------------------------------------------------------

/**
 * Run cuSparse SpMV (specialized for fp32)
 */
template <
    typename OffsetT>
float TestCusparseCsrmv(
    float*                          vector_y_in,
    float*                          reference_vector_y_out,
    SpmvParams<float, OffsetT>&     params,
    int                             timing_iterations,
    cusparseHandle_t                cusparse)
{
    cusparseMatDescr_t desc;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&desc));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_millis    = 0.0;
    GpuTimer timer;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseScsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(desc));
    return elapsed_millis / timing_iterations;
}


/**
 * Run cuSparse SpMV (specialized for fp64)
 */
template <
    typename OffsetT>
float TestCusparseCsrmv(
    double*                         vector_y_in,
    double*                         reference_vector_y_out,
    SpmvParams<double, OffsetT>&    params,
    int                             timing_iterations,
    cusparseHandle_t                cusparse)
{
    cusparseMatDescr_t desc;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreateMatDescr(&desc));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, &params.beta, params.d_vector_y));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    float elapsed_millis = 0.0;
    GpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDcsrmv(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            params.num_rows, params.num_cols, params.num_nonzeros, &params.alpha, desc,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, &params.beta, params.d_vector_y));

    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseDestroyMatDescr(desc));
    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// GPU Merge-based SpMV
//---------------------------------------------------------------------

/**
 * Run CUB SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
float TestGpuMergeCsrmv(
    ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    SpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations)
{
    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Get amount of temporary storage needed
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros,
// params.alpha, params.beta,
        (cudaStream_t) 0, false));

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(ValueT) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros, 
// params.alpha, params.beta,
        (cudaStream_t) 0, !g_quiet));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
    }

    // Timing
    GpuTimer timer;
    float elapsed_millis = 0.0;

    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        CubDebugExit(DeviceSpmv::CsrMV(
            d_temp_storage, temp_storage_bytes,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, params.d_vector_y,
            params.num_rows, params.num_cols, params.num_nonzeros, 
// params.alpha, params.beta,
            (cudaStream_t) 0, false));
    }
    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    return elapsed_millis / timing_iterations;
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Display perf
 */
template <typename ValueT, typename OffsetT>
void DisplayPerf(
    float                           device_giga_bandwidth,
    double                          avg_millis,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_millis / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_millis / 1.0e6;

    if (!g_quiet)
        printf("fp%d: %.4f avg ms, %.5f gflops, %.3lf effective GB/s (%.2f%% peak)\n",
            sizeof(ValueT) * 8,
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / device_giga_bandwidth * 100);
    else
        printf("%.5f, %.6f, %.3lf, %.2f%%, ",
            avg_millis,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / device_giga_bandwidth * 100);

    fflush(stdout);
}



/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTest(
    bool                        rcm_relabel,
    ValueT                      alpha,
    ValueT                      beta,
    CooMatrix<ValueT, OffsetT>& coo_matrix,
    int                         timing_iterations,
    CommandLineArgs&            args)
{
    // Adaptive timing iterations: run 16 billion nonzeros through
    if (timing_iterations == -1)
        timing_iterations = std::min(50000ull, std::max(100ull, ((16ull << 30) / coo_matrix.num_nonzeros)));

    if (!g_quiet)
        printf("\t%d timing iterations\n", timing_iterations);

    // Convert to CSR
    CsrMatrix<ValueT, OffsetT> csr_matrix;
    csr_matrix.FromCoo(coo_matrix);
    if (!args.CheckCmdLineFlag("csrmv"))
        coo_matrix.Clear();

    // Relabel
    if (rcm_relabel)
    {
        if (!g_quiet)
        {
            csr_matrix.Stats().Display();
            printf("\n");
            csr_matrix.DisplayHistogram();
            printf("\n");
            if (g_verbose2)
                csr_matrix.Display();
            printf("\n");
        }

        RcmRelabel(csr_matrix, !g_quiet);

        if (!g_quiet) printf("\n");
    }

    // Display matrix info
    csr_matrix.Stats().Display(!g_quiet);
    if (!g_quiet)
    {
        printf("\n");
        csr_matrix.DisplayHistogram();
        printf("\n");
        if (g_verbose2)
            csr_matrix.Display();
        printf("\n");
    }
    fflush(stdout);

    // Allocate input and output vectors
    ValueT* vector_x        = new ValueT[csr_matrix.num_cols];
    ValueT* vector_y_in     = new ValueT[csr_matrix.num_rows];
    ValueT* vector_y_out    = new ValueT[csr_matrix.num_rows];

    for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    float avg_millis;

    if (g_quiet) {
        printf("%s, %s, ", args.deviceProp.name, (sizeof(ValueT) > 4) ? "fp64" : "fp32"); fflush(stdout);
    }

    // Get GPU device bandwidth (GB/s)
    float device_giga_bandwidth = args.device_giga_bandwidth;

    // Allocate and initialize GPU problem
    SpmvParams<ValueT, OffsetT> params;

    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_values,          sizeof(ValueT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets, sizeof(OffsetT) * (csr_matrix.num_rows + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_x,        sizeof(ValueT) * csr_matrix.num_cols));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_y,        sizeof(ValueT) * csr_matrix.num_rows));
    params.num_rows         = csr_matrix.num_rows;
    params.num_cols         = csr_matrix.num_cols;
    params.num_nonzeros     = csr_matrix.num_nonzeros;
    params.alpha            = alpha;
    params.beta             = beta;

    CubDebugExit(cudaMemcpy(params.d_values,            csr_matrix.values,          sizeof(ValueT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(params.d_row_end_offsets,   csr_matrix.row_offsets,     sizeof(OffsetT) * (csr_matrix.num_rows + 1), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(params.d_column_indices,    csr_matrix.column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(params.d_vector_x,          vector_x,                   sizeof(ValueT) * csr_matrix.num_cols, cudaMemcpyHostToDevice));

    if (!g_quiet) printf("\n\n");
    printf("GPU CSR I/O Prox, "); fflush(stdout);
    avg_millis = TestGpuCsrIoProxy(params, timing_iterations);
    DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);

    if (args.CheckCmdLineFlag("csrmv"))
    {
        if (!g_quiet) printf("\n\n");
        printf("CUB, "); fflush(stdout);
        avg_millis = TestGpuMergeCsrmv(vector_y_in, vector_y_out, params, timing_iterations);
        DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);
    }

    // Initialize cuSparse
    cusparseHandle_t cusparse;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreate(&cusparse));

    if (args.CheckCmdLineFlag("csrmv"))
    {
        if (!g_quiet) printf("\n\n");
        printf("Cusparse CsrMV, "); fflush(stdout);
        avg_millis = TestCusparseCsrmv(vector_y_in, vector_y_out, params, timing_iterations, cusparse);
        DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);
    }

    if (args.CheckCmdLineFlag("hybmv"))
    {
        if (!g_quiet) printf("\n\n");
        printf("Cusparse HybMV, "); fflush(stdout);

        avg_millis = TestCusparseHybmv(vector_y_in, vector_y_out, params, timing_iterations, cusparse);
        DisplayPerf(device_giga_bandwidth, avg_millis, csr_matrix);
    }


    // Cleanup
    if (params.d_values)            CubDebugExit(g_allocator.DeviceFree(params.d_values));
    if (params.d_row_end_offsets)   CubDebugExit(g_allocator.DeviceFree(params.d_row_end_offsets));
    if (params.d_column_indices)    CubDebugExit(g_allocator.DeviceFree(params.d_column_indices));
    if (params.d_vector_x)          CubDebugExit(g_allocator.DeviceFree(params.d_vector_x));
    if (params.d_vector_y)          CubDebugExit(g_allocator.DeviceFree(params.d_vector_y));

    if (vector_x)                   delete[] vector_x;
    if (vector_y_in)                delete[] vector_y_in;
    if (vector_y_out)               delete[] vector_y_out;
}

/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    bool                rcm_relabel,
    ValueT              alpha,
    ValueT              beta,
    const std::string&  mtx_filename,
    int                 grid2d,
    int                 grid3d,
    int                 wheel,
    int                 dense,
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        printf("%s, ", mtx_filename.c_str()); fflush(stdout);
        coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);

        if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
        {
            if (!g_quiet) printf("Trivial dataset\n");
            exit(0);
        }
    }
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        printf("grid2d_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        printf("grid3d_%d, ", grid3d); fflush(stdout);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        printf("wheel_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        // Generate dense graph
        OffsetT size = 1 << 24; // 16M nnz
        args.GetCmdLineArgument("size", size);

        OffsetT rows = size / dense;
        printf("dense_%d_x_%d, ", rows, dense); fflush(stdout);
        coo_matrix.InitDense(rows, dense);
    }
    else
    {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }

    RunTest(
        rcm_relabel,
        alpha,
        beta,
        coo_matrix,
        timing_iterations,
        args);
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
            "[--csrmv | --hybmv | --bsrmv ] "
            "[--device=<device-id>] "
            "[--quiet] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp64] "
            "[--rcm] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n\t"
                "--dense=<cols>"
            "\n\t"
                "--grid2d=<width>"
            "\n\t"
                "--grid3d=<width>"
            "\n\t"
                "--wheel=<spokes>"
            "\n", argv[0]);
        exit(0);
    }

    bool                fp64;
    bool                rcm_relabel;
    std::string         mtx_filename;
    int                 grid2d              = -1;
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 dense               = -1;
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp64 = args.CheckCmdLineFlag("fp64");
    rcm_relabel = args.CheckCmdLineFlag("rcm");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("wheel", wheel);
    args.GetCmdLineArgument("dense", dense);
    args.GetCmdLineArgument("alpha", alpha);
    args.GetCmdLineArgument("beta", beta);

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Run test(s)
    if (fp64)
    {
        RunTests<double, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }
    else
    {
        RunTests<float, int>(rcm_relabel, alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");

    return 0;
}
