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
 * An implementation of COO SpMV using prefix scan to implement a
 * reduce-value-by-row strategy
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>

#include <cub/cub.cuh>

#include "coo_graph.cuh"
#include "../test/test_util.h"

using namespace cub;
using namespace std;


/******************************************************************************
 * Globals, constants, and typedefs
 ******************************************************************************/

typedef int         VertexId;   // uint32s as vertex ids
typedef double      Value;      // double-precision floating point values

bool                    g_verbose       = false;
int                     g_timing_iterations    = 1;
CachingDeviceAllocator  g_allocator;


/******************************************************************************
 * Texture referencing
 ******************************************************************************/

/**
 * Templated texture reference type for multiplicand vector
 */
template <typename Value>
struct TexVector
{
    // Texture type to actually use (e.g., because CUDA doesn't load doubles as texture items)
    typedef typename If<(Equals<Value, double>::VALUE), uint2, Value>::Type CastType;

    // Texture reference type
    typedef texture<CastType, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind textures
     */
    static void BindTexture(void *d_in, int elements)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<CastType>();
        if (d_in)
        {
            size_t offset;
            size_t bytes = sizeof(CastType) * elements;
            CubDebugExit(cudaBindTexture(&offset, ref, d_in, tex_desc, bytes));
        }
    }

    /**
     * Unbind textures
     */
    static void UnbindTexture()
    {
        CubDebugExit(cudaUnbindTexture(ref));
    }

    /**
     * Load
     */
    static __device__ __forceinline__ Value Load(int offset)
    {
        Value output;
        reinterpret_cast<typename TexVector<Value>::CastType &>(output) = tex1Dfetch(TexVector<Value>::ref, offset);
        return output;
    }
};

// Texture reference definitions
template <typename Value>
typename TexVector<Value>::TexRef TexVector<Value>::ref = 0;


/******************************************************************************
 * Utility types
 ******************************************************************************/


/**
 * A partial dot-product sum paired with a corresponding row-id
 */
template <typename VertexId, typename Value>
struct PartialProduct
{
    VertexId    row;            /// Row-id
    Value       partial;        /// PartialProduct sum
};


/**
 * A partial dot-product sum paired with a corresponding row-id (specialized for double-int pairings)
 */
template <>
struct PartialProduct<int, double>
{
    long long   row;            /// Row-id
    double      partial;        /// PartialProduct sum
};


/**
 * Reduce-value-by-row scan operator
 */
struct ReduceByKeyOp
{
    template <typename PartialProduct>
    __device__ __forceinline__ PartialProduct operator()(
        const PartialProduct &first,
        const PartialProduct &second)
    {
        PartialProduct retval;

        retval.partial = (second.row != first.row) ?
                second.partial :
                first.partial + second.partial;

        retval.row = second.row;
        return retval;
    }
};


/**
 * Stateful block-wide prefix operator for BlockScan
 */
template <typename PartialProduct>
struct BlockPrefixCallbackOp
{
    // Running block-wide prefix
    PartialProduct running_prefix;

    /**
     * Returns the block-wide running_prefix in thread-0
     */
    __device__ __forceinline__ PartialProduct operator()(
        const PartialProduct &block_aggregate)              ///< The aggregate sum of the BlockScan inputs
    {
        ReduceByKeyOp scan_op;

        PartialProduct retval = running_prefix;
        running_prefix = scan_op(running_prefix, block_aggregate);
        return retval;
    }
};


/**
 * Operator for detecting discontinuities in a list of row identifiers.
 */
struct NewRowOp
{
    /// Returns true if row_b is the start of a new row
    template <typename VertexId>
    __device__ __forceinline__ bool operator()(
        const VertexId& row_a,
        const VertexId& row_b)
    {
        return (row_a != row_b);
    }
};



/******************************************************************************
 * Persistent thread block types
 ******************************************************************************/

/**
 * SpMV thread block abstraction for processing a contiguous segment of
 * sparse COO tiles.
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
struct PersistentBlockSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef int HeadFlag;

    // Partial dot product type
    typedef PartialProduct<VertexId, Value> PartialProduct;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockExchange type for exchanging rows between warp-striped -> blocked arrangements
    typedef BlockExchange<VertexId, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeRows;

    // Parameterized BlockExchange type for exchanging values between warp-striped -> blocked arrangements
    typedef BlockExchange<Value, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeValues;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS> BlockDiscontinuity;

    // Shared memory type for this thread block
    struct TempStorage
    {
        union
        {
            typename BlockExchangeRows::TempStorage         exchange_rows;      // Smem needed for BlockExchangeRows
            typename BlockExchangeValues::TempStorage       exchange_values;    // Smem needed for BlockExchangeValues
            struct
            {
                typename BlockScan::TempStorage             scan;               // Smem needed for BlockScan
                typename BlockDiscontinuity::TempStorage    discontinuity;      // Smem needed for BlockDiscontinuity
            };
        };

        VertexId        first_block_row;    ///< The first row-ID seen by this thread block
        VertexId        last_block_row;     ///< The last row-ID seen by this thread block
        Value           first_product;      ///< The first dot-product written by this thread block
    };

    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixCallbackOp<PartialProduct>   prefix_op;
    VertexId                        *d_rows;
    VertexId                        *d_columns;
    Value                           *d_values;
    Value                           *d_vector;
    Value                           *d_result;
    PartialProduct                  *d_block_partials;
    int                             block_offset;
    int                             block_end;


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    PersistentBlockSpmv(
        TempStorage                 &temp_storage,
        VertexId                    *d_rows,
        VertexId                    *d_columns,
        Value                       *d_values,
        Value                       *d_vector,
        Value                       *d_result,
        PartialProduct              *d_block_partials,
        int                         block_offset,
        int                         block_end)
    :
        temp_storage(temp_storage),
        d_rows(d_rows),
        d_columns(d_columns),
        d_values(d_values),
        d_vector(d_vector),
        d_result(d_result),
        d_block_partials(d_block_partials),
        block_offset(block_offset),
        block_end(block_end)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            VertexId first_block_row            = d_rows[block_offset];
            VertexId last_block_row             = d_rows[block_end - 1];

            temp_storage.first_block_row        = first_block_row;
            temp_storage.last_block_row         = last_block_row;
            temp_storage.first_product          = Value(0);

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = Value(0);
        }

        __syncthreads();
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        VertexId        columns[ITEMS_PER_THREAD];
        VertexId        rows[ITEMS_PER_THREAD];
        Value           values[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a thread block-striped tile of A (sparse row-ids, column-ids, and values)
        if (FULL_TILE)
        {
            // Unguarded loads
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_columns + block_offset, columns);
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_values + block_offset, values);
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_rows + block_offset, rows);
        }
        else
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_columns + block_offset, columns, guarded_items, VertexId(0));
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_values + block_offset, values, guarded_items, Value(0));
            LoadDirectWarpStriped<LOAD_DEFAULT>(threadIdx.x, d_rows + block_offset, rows, guarded_items, temp_storage.last_block_row);
        }

        // Load the referenced values from x and compute the dot product partials sums
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
#if CUB_PTX_ARCH >= 350
            values[ITEM] *= ThreadLoad<LOAD_LDG>(d_vector + columns[ITEM]);
#else
            values[ITEM] *= TexVector<Value>::Load(columns[ITEM]);
#endif
        }

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeValues(temp_storage.exchange_values).WarpStripedToBlocked(values);

        __syncthreads();

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeRows(temp_storage.exchange_rows).WarpStripedToBlocked(rows);

        // Barrier for smem reuse and coherence
        __syncthreads();

        // FlagT row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            head_flags,                     // (Out) Head flags
            rows,                           // Original row ids
            NewRowOp(),                     // Functor for detecting start of new rows
            prefix_op.running_prefix.row);  // Last row ID from previous tile to compare with first row ID in this tile

        // Assemble partial product structures
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            partial_sums[ITEM].partial = values[ITEM];
            partial_sums[ITEM].row = rows[ITEM];
        }

        // Reduce reduce-value-by-row across partial_sums using exclusive prefix scan
        PartialProduct block_aggregate;
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
            ReduceByKeyOp(),                // Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Scatter an accumulated dot product if it is the head of a valid row
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;

                // Save off the first partial product that this thread block will scatter
                if (partial_sums[ITEM].row == temp_storage.first_block_row)
                {
                    temp_storage.first_product = partial_sums[ITEM].partial;
                }
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessTiles()
    {
        // Process full tiles
        while (block_offset <= block_end - TILE_ITEMS)
        {
            ProcessTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Process the last, partially-full tile (if present)
        int guarded_items = block_end - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, guarded_items);
        }

        if (threadIdx.x == 0)
        {
            if (gridDim.x == 1)
            {
                // Scatter the final aggregate (this kernel contains only 1 thread block)
                d_result[prefix_op.running_prefix.row] = prefix_op.running_prefix.partial;
            }
            else
            {
                // Write the first and last partial products from this thread block so
                // that they can be subsequently "fixed up" in the next kernel.

                PartialProduct first_product;
                first_product.row       = temp_storage.first_block_row;
                first_product.partial   = temp_storage.first_product;

                d_block_partials[blockIdx.x * 2]          = first_product;
                d_block_partials[(blockIdx.x * 2) + 1]    = prefix_op.running_prefix;
            }
        }
    }
};


/**
 * Threadblock abstraction for "fixing up" an array of interblock SpMV partial products.
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
struct FinalizeSpmvBlock
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef int HeadFlag;

    // Partial dot product type
    typedef PartialProduct<VertexId, Value> PartialProduct;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<HeadFlag, BLOCK_THREADS> BlockDiscontinuity;

    // Shared memory type for this thread block
    struct TempStorage
    {
        typename BlockScan::TempStorage           scan;               // Smem needed for reduce-value-by-row scan
        typename BlockDiscontinuity::TempStorage  discontinuity;      // Smem needed for head-flagging

        VertexId last_block_row;
    };


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixCallbackOp<PartialProduct>   prefix_op;
    Value                           *d_result;
    PartialProduct                  *d_block_partials;
    int                             num_partials;


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    FinalizeSpmvBlock(
        TempStorage                 &temp_storage,
        Value                       *d_result,
        PartialProduct              *d_block_partials,
        int                         num_partials)
    :
        temp_storage(temp_storage),
        d_result(d_result),
        d_block_partials(d_block_partials),
        num_partials(num_partials)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            VertexId first_block_row            = d_block_partials[0].row;
            VertexId last_block_row             = d_block_partials[num_partials - 1].row;
            temp_storage.last_block_row         = last_block_row;

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = Value(0);
        }

        __syncthreads();
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    template <bool FULL_TILE>
    __device__ __forceinline__
    void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        VertexId        rows[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a tile of block partials from previous kernel
        if (FULL_TILE)
        {
            // Full tile
#if CUB_PTX_ARCH >= 350
            LoadDirectBlocked<LOAD_LDG>(threadIdx.x, d_block_partials + block_offset, partial_sums);
#else
            LoadDirectBlocked(threadIdx.x, d_block_partials + block_offset, partial_sums);
#endif
        }
        else
        {
            // Partial tile (extend zero-valued coordinates of the last partial-product for out-of-bounds items)
            PartialProduct default_sum;
            default_sum.row = temp_storage.last_block_row;
            default_sum.partial = Value(0);

#if CUB_PTX_ARCH >= 350
            LoadDirectBlocked<LOAD_LDG>(threadIdx.x, d_block_partials + block_offset, partial_sums, guarded_items, default_sum);
#else
            LoadDirectBlocked(threadIdx.x, d_block_partials + block_offset, partial_sums, guarded_items, default_sum);
#endif
        }

        // Copy out row IDs for row-head flagging
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = partial_sums[ITEM].row;
        }

        // FlagT row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            rows,                           // Original row ids
            head_flags,                     // (Out) Head flags
            NewRowOp(),                     // Functor for detecting start of new rows
            prefix_op.running_prefix.row);   // Last row ID from previous tile to compare with first row ID in this tile

        // Reduce reduce-value-by-row across partial_sums using exclusive prefix scan
        PartialProduct block_aggregate;
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
            ReduceByKeyOp(),                // Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

        // Scatter an accumulated dot product if it is the head of a valid row
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessTiles()
    {
        // Process full tiles
        int block_offset = 0;
        while (block_offset <= num_partials - TILE_ITEMS)
        {
            ProcessTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Process final partial tile (if present)
        int guarded_items = num_partials - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, guarded_items);
        }

        // Scatter the final aggregate (this kernel contains only 1 thread block)
        if (threadIdx.x == 0)
        {
            d_result[prefix_op.running_prefix.row] = prefix_op.running_prefix.partial;
        }
    }
};


/******************************************************************************
 * Kernel entrypoints
 ******************************************************************************/



/**
 * SpMV kernel whose thread blocks each process a contiguous segment of sparse COO tiles.
 */
template <
    int                             BLOCK_THREADS,
    int                             ITEMS_PER_THREAD,
    typename                        VertexId,
    typename                        Value>
__launch_bounds__ (BLOCK_THREADS)
__global__ void CooKernel(
    GridEvenShare<int>              even_share,
    PartialProduct<VertexId, Value> *d_block_partials,
    VertexId                        *d_rows,
    VertexId                        *d_columns,
    Value                           *d_values,
    Value                           *d_vector,
    Value                           *d_result)
{
    // Specialize SpMV thread block abstraction type
    typedef PersistentBlockSpmv<BLOCK_THREADS, ITEMS_PER_THREAD, VertexId, Value> PersistentBlockSpmv;

    // Shared memory allocation
    __shared__ typename PersistentBlockSpmv::TempStorage temp_storage;

    // Initialize thread block even-share to tell us where to start and stop our tile-processing
    even_share.BlockInit();

    // Construct persistent thread block
    PersistentBlockSpmv persistent_block(
        temp_storage,
        d_rows,
        d_columns,
        d_values,
        d_vector,
        d_result,
        d_block_partials,
        even_share.block_offset,
        even_share.block_end);

    // Process input tiles
    persistent_block.ProcessTiles();
}


/**
 * Kernel for "fixing up" an array of interblock SpMV partial products.
 */
template <
    int                             BLOCK_THREADS,
    int                             ITEMS_PER_THREAD,
    typename                        VertexId,
    typename                        Value>
__launch_bounds__ (BLOCK_THREADS,  1)
__global__ void CooFinalizeKernel(
    PartialProduct<VertexId, Value> *d_block_partials,
    int                             num_partials,
    Value                           *d_result)
{
    // Specialize "fix-up" thread block abstraction type
    typedef FinalizeSpmvBlock<BLOCK_THREADS, ITEMS_PER_THREAD, VertexId, Value> FinalizeSpmvBlock;

    // Shared memory allocation
    __shared__ typename FinalizeSpmvBlock::TempStorage temp_storage;

    // Construct persistent thread block
    FinalizeSpmvBlock persistent_block(temp_storage, d_result, d_block_partials, num_partials);

    // Process input tiles
    persistent_block.ProcessTiles();
}



//---------------------------------------------------------------------
// Host subroutines
//---------------------------------------------------------------------


/**
 * Simple test of device
 */
template <
    int                         COO_BLOCK_THREADS,
    int                         COO_ITEMS_PER_THREAD,
    int                         COO_SUBSCRIPTION_FACTOR,
    int                         FINALIZE_BLOCK_THREADS,
    int                         FINALIZE_ITEMS_PER_THREAD,
    typename                    VertexId,
    typename                    Value>
void TestDevice(
    CooGraph<VertexId, Value>&  coo_graph,
    Value*                      h_vector,
    Value*                      h_reference)
{
    typedef PartialProduct<VertexId, Value> PartialProduct;

    const int COO_TILE_SIZE = COO_BLOCK_THREADS * COO_ITEMS_PER_THREAD;

    // SOA device storage
    VertexId        *d_rows;             // SOA graph row coordinates
    VertexId        *d_columns;          // SOA graph col coordinates
    Value           *d_values;           // SOA graph values
    Value           *d_vector;           // Vector multiplicand
    Value           *d_result;           // Output row
    PartialProduct  *d_block_partials;   // Temporary storage for communicating dot product partials between thread blocks

    // Create SOA version of coo_graph on host
    int             num_edges   = coo_graph.coo_tuples.size();
    VertexId        *h_rows     = new VertexId[num_edges];
    VertexId        *h_columns  = new VertexId[num_edges];
    Value           *h_values   = new Value[num_edges];
    for (int i = 0; i < num_edges; i++)
    {
        h_rows[i]       = coo_graph.coo_tuples[i].row;
        h_columns[i]    = coo_graph.coo_tuples[i].col;
        h_values[i]     = coo_graph.coo_tuples[i].val;
    }

    // Get CUDA properties
    Device device_props;
    CubDebugExit(device_props.Init());

    // Determine launch configuration from kernel properties
    int coo_sm_occupancy;
    CubDebugExit(device_props.MaxSmOccupancy(
        coo_sm_occupancy,
        CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, VertexId, Value>,
        COO_BLOCK_THREADS));
    int max_coo_grid_size   = device_props.sm_count * coo_sm_occupancy * COO_SUBSCRIPTION_FACTOR;

    // Construct an even-share work distribution
    GridEvenShare<int> even_share(num_edges, max_coo_grid_size, COO_TILE_SIZE);
    int coo_grid_size  = even_share.grid_size;
    int num_partials   = coo_grid_size * 2;

    // Allocate COO device arrays
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_rows,            sizeof(VertexId) * num_edges));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_columns,         sizeof(VertexId) * num_edges));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values,          sizeof(Value) * num_edges));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_vector,          sizeof(Value) * coo_graph.col_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_result,          sizeof(Value) * coo_graph.row_dim));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_block_partials,  sizeof(PartialProduct) * num_partials));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows,     h_rows,     sizeof(VertexId) * num_edges,       cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns,  h_columns,  sizeof(VertexId) * num_edges,       cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values,   h_values,   sizeof(Value) * num_edges,          cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector,   h_vector,   sizeof(Value) * coo_graph.col_dim,  cudaMemcpyHostToDevice));

    // Bind textures
    TexVector<Value>::BindTexture(d_vector, coo_graph.col_dim);

    // Print debug info
    printf("CooKernel<%d, %d><<<%d, %d>>>(...), Max SM occupancy: %d\n",
        COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD, coo_grid_size, COO_BLOCK_THREADS, coo_sm_occupancy);
    if (coo_grid_size > 1)
    {
        printf("CooFinalizeKernel<<<1, %d>>>(...)\n", FINALIZE_BLOCK_THREADS);
    }
    fflush(stdout);

    CubDebugExit(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    // Run kernel (always run one iteration without timing)
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i <= g_timing_iterations; i++)
    {
        gpu_timer.Start();

        // Initialize output
        CubDebugExit(cudaMemset(d_result, 0, coo_graph.row_dim * sizeof(Value)));

        // Run the COO kernel
        CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD><<<coo_grid_size, COO_BLOCK_THREADS>>>(
            even_share,
            d_block_partials,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result);

        if (coo_grid_size > 1)
        {
            // Run the COO finalize kernel
            CooFinalizeKernel<FINALIZE_BLOCK_THREADS, FINALIZE_ITEMS_PER_THREAD><<<1, FINALIZE_BLOCK_THREADS>>>(
                d_block_partials,
                num_partials,
                d_result);
        }

        gpu_timer.Stop();

        if (i > 0)
            elapsed_millis += gpu_timer.ElapsedMillis();
    }

    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());
    fflush(stdout);

    // Display timing
    if (g_timing_iterations > 0)
    {
        float avg_elapsed = elapsed_millis / g_timing_iterations;
        int total_bytes = ((sizeof(VertexId) + sizeof(VertexId)) * 2 * num_edges) + (sizeof(Value) * coo_graph.row_dim);
        printf("%d iterations, average elapsed (%.3f ms), utilized bandwidth (%.3f GB/s), GFLOPS(%.3f)\n",
            g_timing_iterations,
            avg_elapsed,
            total_bytes / avg_elapsed / 1000.0 / 1000.0,
            num_edges * 2 / avg_elapsed / 1000.0 / 1000.0);
    }

    // Check results
    int compare = CompareDeviceResults(h_reference, d_result, coo_graph.row_dim, true, g_verbose);
    printf("%s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    TexVector<Value>::UnbindTexture();
    CubDebugExit(g_allocator.DeviceFree(d_block_partials));
    CubDebugExit(g_allocator.DeviceFree(d_rows));
    CubDebugExit(g_allocator.DeviceFree(d_columns));
    CubDebugExit(g_allocator.DeviceFree(d_values));
    CubDebugExit(g_allocator.DeviceFree(d_vector));
    CubDebugExit(g_allocator.DeviceFree(d_result));
    delete[] h_rows;
    delete[] h_columns;
    delete[] h_values;
}


/**
 * Compute reference answer on CPU
 */
template <typename VertexId, typename Value>
void ComputeReference(
    CooGraph<VertexId, Value>&  coo_graph,
    Value*                      h_vector,
    Value*                      h_reference)
{
    for (VertexId i = 0; i < coo_graph.row_dim; i++)
    {
        h_reference[i] = 0.0;
    }

    for (VertexId i = 0; i < coo_graph.coo_tuples.size(); i++)
    {
        h_reference[coo_graph.coo_tuples[i].row] +=
            coo_graph.coo_tuples[i].val *
            h_vector[coo_graph.coo_tuples[i].col];
    }
}


/**
 * Assign arbitrary values to vector items
 */
template <typename Value>
void AssignVectorValues(Value *vector, int col_dim)
{
    for (int i = 0; i < col_dim; i++)
    {
        vector[i] = 1.0;
    }
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", g_timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s\n [--device=<device-id>] [--v] [--iterations=<test iterations>] [--grid-size=<grid-size>]\n"
            "\t--type=wheel --spokes=<spokes>\n"
            "\t--type=grid2d --width=<width> [--no-self-loops]\n"
            "\t--type=grid3d --width=<width> [--no-self-loops]\n"
            "\t--type=market --file=<file>\n"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get graph type
    string type;
    args.GetCmdLineArgument("type", type);

    // Generate graph structure

    CpuTimer timer;
    timer.Start();
    CooGraph<VertexId, Value> coo_graph;
    if (type == string("grid2d"))
    {
        VertexId width;
        args.GetCmdLineArgument("width", width);
        bool self_loops = !args.CheckCmdLineFlag("no-self-loops");
        printf("Generating %s grid2d width(%d)... ", (self_loops) ? "5-pt" : "4-pt", width); fflush(stdout);
        if (coo_graph.InitGrid2d(width, self_loops)) exit(1);
    } else if (type == string("grid3d"))
    {
        VertexId width;
        args.GetCmdLineArgument("width", width);
        bool self_loops = !args.CheckCmdLineFlag("no-self-loops");
        printf("Generating %s grid3d width(%d)... ", (self_loops) ? "7-pt" : "6-pt", width); fflush(stdout);
        if (coo_graph.InitGrid3d(width, self_loops)) exit(1);
    }
    else if (type == string("wheel"))
    {
        VertexId spokes;
        args.GetCmdLineArgument("spokes", spokes);
        printf("Generating wheel spokes(%d)... ", spokes); fflush(stdout);
        if (coo_graph.InitWheel(spokes)) exit(1);
    }
    else if (type == string("market"))
    {
        string filename;
        args.GetCmdLineArgument("file", filename);
        printf("Generating MARKET for %s... ", filename.c_str()); fflush(stdout);
        if (coo_graph.InitMarket(filename)) exit(1);
    }
    else
    {
        printf("Unsupported graph type\n");
        exit(1);
    }
    timer.Stop();
    printf("Done (%.3fs). %d non-zeros, %d rows, %d columns\n",
        timer.ElapsedMillis() / 1000.0,
        coo_graph.coo_tuples.size(),
        coo_graph.row_dim,
        coo_graph.col_dim);
    fflush(stdout);

    if (g_verbose)
    {
        cout << coo_graph << "\n";
    }

    // Create vector
    Value *h_vector = new Value[coo_graph.col_dim];
    AssignVectorValues(h_vector, coo_graph.col_dim);
    if (g_verbose)
    {
        printf("Vector[%d]: ", coo_graph.col_dim);
        DisplayResults(h_vector, coo_graph.col_dim);
        printf("\n\n");
    }

    // Compute reference answer
    Value *h_reference = new Value[coo_graph.row_dim];
    ComputeReference(coo_graph, h_vector, h_reference);
    if (g_verbose)
    {
        printf("Results[%d]: ", coo_graph.row_dim);
        DisplayResults(h_reference, coo_graph.row_dim);
        printf("\n\n");
    }

    // Parameterization for SM35
    enum
    {
        COO_BLOCK_THREADS           = 64,
        COO_ITEMS_PER_THREAD        = 10,
        COO_SUBSCRIPTION_FACTOR     = 4,
        FINALIZE_BLOCK_THREADS      = 256,
        FINALIZE_ITEMS_PER_THREAD   = 4,
    };

    // Run GPU version
    TestDevice<
        COO_BLOCK_THREADS,
        COO_ITEMS_PER_THREAD,
        COO_SUBSCRIPTION_FACTOR,
        FINALIZE_BLOCK_THREADS,
        FINALIZE_ITEMS_PER_THREAD>(coo_graph, h_vector, h_reference);

    // Cleanup
    delete[] h_vector;
    delete[] h_reference;

    return 0;
}



