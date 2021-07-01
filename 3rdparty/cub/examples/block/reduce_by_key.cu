

#include <cub/cub.cuh>


template <
    int         BLOCK_THREADS,          ///< Number of CTA threads
    typename    KeyT,                   ///< Key type
    typename    ValueT>                 ///< Value type
__global__ void Kernel()
{
    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef cub::KeyValuePair<int, ValueT> OffsetValuePairT;

    // Reduce-value-by-segment scan operator
    typedef cub::ReduceBySegmentOp<cub::Sum> ReduceBySegmentOpT;

    // Parameterized BlockDiscontinuity type for setting head flags
    typedef cub::BlockDiscontinuity<
            KeyT,
            BLOCK_THREADS>
        BlockDiscontinuityKeysT;

    // Parameterized BlockScan type
    typedef cub::BlockScan<
            OffsetValuePairT,
            BLOCK_THREADS,
            cub::BLOCK_SCAN_WARP_SCANS>
        BlockScanT;

    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockScanT::TempStorage                scan;           // Scan storage
        typename BlockDiscontinuityKeysT::TempStorage   discontinuity;  // Discontinuity storage
    } temp_storage;


    // Read data (each thread gets 3 items each, every 9 items is a segment)
    KeyT    my_keys[3]      = {threadIdx.x / 3, threadIdx.x / 3, threadIdx.x / 3};
    ValueT  my_values[3]    = {1, 1, 1};

    // Set head segment head flags
    int     my_flags[3];
    BlockDiscontinuityKeysT(temp_storage.discontinuity).FlagHeads(
        my_flags,
        my_keys,
        cub::Inequality());

    __syncthreads();






}
