/**********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011  Andreas Muetzel (amuetzel@uni-koblenz.de). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/
#ifndef FLANN_UTIL_CUDA_RESULTSET_H
#define FLANN_UTIL_CUDA_RESULTSET_H

#include <flann/util/cuda/heap.h>
#include <limits>

__device__ __forceinline__
float infinity()
{
       return __int_as_float(0x7f800000);
}

#ifndef INFINITY
#define INFINITY infinity()
#endif

namespace flann
{
namespace cuda
{
//! result set for the 1nn search. Doesn't do any global memory accesses on its own,
template< typename DistanceType >
struct SingleResultSet
{
    int bestIndex;
    DistanceType bestDist;
    const DistanceType epsError;

    __device__
    SingleResultSet( DistanceType eps ) : bestIndex(-1),bestDist(INFINITY), epsError(eps){ }

    __device__
    inline float
    worstDist()
    {
        return bestDist;
    }

    __device__
    inline void
    insert(int index, DistanceType dist)
    {
        if( dist <= bestDist ) {
            bestIndex=index;
            bestDist=dist;
        }
    }

    DistanceType* resultDist;
    int* resultIndex;

    __device__
    inline void
    setResultLocation( DistanceType* dists, int* index, int thread, int stride )
    {
        resultDist=dists+thread*stride;
        resultIndex=index+thread*stride;
        if( stride != 1 ) {
            for( int i=1; i<stride; i++ ) {
                resultDist[i]=INFINITY;
                resultIndex[i]=-1;
            }
        }
    }

    __device__
    inline void
    finish()
    {
        resultDist[0]=bestDist;
        resultIndex[0]=bestIndex;
    }
};

template< typename DistanceType >
struct GreaterThan
{
    __device__
    bool operator()(DistanceType a, DistanceType b)
    {
        return a>b;
    }
};


// using this and the template uses 2 or 3 registers more than the direct implementation in the kNearestKernel, but
// there is no speed difference.
// Setting useHeap as a template parameter leads to a whole lot of things being
// optimized away by nvcc.
// Register counts are the same as when removing not-needed variables in explicit specializations
// and the "if( useHeap )" branches are eliminated at compile time.
// The downside of this: a bit more complex kernel launch code.
template< typename DistanceType, bool useHeap >
struct KnnResultSet
{
    int foundNeighbors;
    DistanceType largestHeapDist;
    int maxDistIndex;
    const int k;
    const bool sorted;
    const DistanceType epsError;


    __device__
    KnnResultSet(int knn, bool sortResults, DistanceType eps) : foundNeighbors(0),largestHeapDist(INFINITY),k(knn), sorted(sortResults), epsError(eps){ }

    //          __host__ __device__
    //          KnnResultSet(const KnnResultSet& o):foundNeighbors(o.foundNeighbors),largestHeapDist(o.largestHeapDist),k(o.k){ }

    __device__
    inline DistanceType
    worstDist()
    {
        return largestHeapDist;
    }

    __device__
    inline void
    insert(int index, DistanceType dist)
    {
        if( foundNeighbors<k ) {
            resultDist[foundNeighbors]=dist;
            resultIndex[foundNeighbors]=index;
            if( foundNeighbors==k-1) {
                if( useHeap ) {
                    flann::cuda::heap::make_heap(resultDist,resultIndex,k,GreaterThan<DistanceType>());
                    largestHeapDist=resultDist[0];
                }
                else {
                    findLargestDistIndex();
                }

            }
            foundNeighbors++;
        }
        else if( dist < largestHeapDist ) {
            if( useHeap ) {
                resultDist[0]=dist;
                resultIndex[0]=index;
                flann::cuda::heap::sift_down(resultDist,resultIndex,0,k,GreaterThan<DistanceType>());
                largestHeapDist=resultDist[0];
            }
            else {
                resultDist[maxDistIndex]=dist;
                resultIndex[maxDistIndex]=index;
                findLargestDistIndex();
            }

        }
    }

    __device__
    void
    findLargestDistIndex( )
    {
        largestHeapDist=resultDist[0];
        maxDistIndex=0;
        for( int i=1; i<k; i++ )
            if( resultDist[i] > largestHeapDist ) {
                maxDistIndex=i;
                largestHeapDist=resultDist[i];
            }
    }

    float* resultDist;
    int* resultIndex;

    __device__
    inline void
    setResultLocation( DistanceType* dists, int* index, int thread, int stride )
    {
        resultDist=dists+stride*thread;
        resultIndex=index+stride*thread;
        for( int i=0; i<stride; i++ ) {
            resultDist[i]=INFINITY;
            resultIndex[i]=-1;
            //                  resultIndex[tid+i*blockDim.x]=-1;
            //                  resultDist[tid+i*blockDim.x]=INFINITY;
        }
    }

    __host__ __device__
    inline void
    finish()
    {
        if( sorted ) {
            if( !useHeap ) flann::cuda::heap::make_heap(resultDist,resultIndex,k,GreaterThan<DistanceType>());
            for( int i=k-1; i>0; i-- ) {
                flann::cuda::swap( resultDist[0], resultDist[i] );
                flann::cuda::swap( resultIndex[0], resultIndex[i] );
                flann::cuda::heap::sift_down( resultDist,resultIndex, 0, i, GreaterThan<DistanceType>() );
            }
        }
    }
};

template <typename DistanceType>
struct CountingRadiusResultSet
{
    int count_;
    DistanceType radius_sq_;
    int max_neighbors_;

    __device__
    CountingRadiusResultSet(DistanceType radius, int max_neighbors) : count_(0),radius_sq_(radius), max_neighbors_(max_neighbors){ }

    __device__
    inline DistanceType
    worstDist()
    {
        return radius_sq_;
    }

    __device__
    inline void
    insert(int index, float dist)
    {
        if( dist < radius_sq_ ) {
            count_++;
        }
    }

    int* resultIndex;

    __device__
    inline void
    setResultLocation( DistanceType* /*dists*/, int* count, int thread, int stride )
    {
        resultIndex=count+thread*stride;
    }

    __device__
    inline void
    finish()
    {
        if(( max_neighbors_<=0) ||( count_<=max_neighbors_) ) resultIndex[0]=count_;
        else resultIndex[0]=max_neighbors_;
    }
};

template<typename DistanceType, bool useHeap>
struct RadiusKnnResultSet
{
    int foundNeighbors;
    DistanceType largestHeapDist;
    int maxDistElem;
    const int k;
    const bool sorted;
    const DistanceType radius_sq_;
    int* segment_starts_;
    //          int count_;


    __device__
    RadiusKnnResultSet(DistanceType radius, int knn, int* segment_starts, bool sortResults) : foundNeighbors(0),largestHeapDist(radius),k(knn), sorted(sortResults), radius_sq_(radius),segment_starts_(segment_starts) { }

    //          __host__ __device__
    //          KnnResultSet(const KnnResultSet& o):foundNeighbors(o.foundNeighbors),largestHeapDist(o.largestHeapDist),k(o.k){ }

    __device__
    inline DistanceType
    worstDist()
    {
        return largestHeapDist;
    }

    __device__
    inline void
    insert(int index, DistanceType dist)
    {
        if( dist < radius_sq_ ) {
            if( foundNeighbors<k ) {
                resultDist[foundNeighbors]=dist;
                resultIndex[foundNeighbors]=index;
                if(( foundNeighbors==k-1) && useHeap) {
                    if( useHeap ) {
                        flann::cuda::heap::make_heap(resultDist,resultIndex,k,GreaterThan<DistanceType>());
                        largestHeapDist=resultDist[0];
                    }
                    else {
                        findLargestDistIndex();
                    }
                }
                foundNeighbors++;

            }
            else if( dist < largestHeapDist ) {
                if( useHeap ) {
                    resultDist[0]=dist;
                    resultIndex[0]=index;
                    flann::cuda::heap::sift_down(resultDist,resultIndex,0,k,GreaterThan<DistanceType>());
                    largestHeapDist=resultDist[0];
                }
                else {
                    resultDist[maxDistElem]=dist;
                    resultIndex[maxDistElem]=index;
                    findLargestDistIndex();
                }
            }
        }
    }

    __device__
    void
    findLargestDistIndex( )
    {
        largestHeapDist=resultDist[0];
        maxDistElem=0;
        for( int i=1; i<k; i++ )
            if( resultDist[i] > largestHeapDist ) {
                maxDistElem=i;
                largestHeapDist=resultDist[i];
            }
    }


    DistanceType* resultDist;
    int* resultIndex;

    __device__
    inline void
    setResultLocation( DistanceType* dists, int* index, int thread, int /*stride*/ )
    {
        resultDist=dists+segment_starts_[thread];
        resultIndex=index+segment_starts_[thread];
    }

    __device__
    inline void
    finish()
    {
        if( sorted ) {
            if( !useHeap ) flann::cuda::heap::make_heap(resultDist,resultIndex,k,GreaterThan<DistanceType>());
            for( int i=foundNeighbors-1; i>0; i-- ) {
                flann::cuda::swap( resultDist[0], resultDist[i] );
                flann::cuda::swap( resultIndex[0], resultIndex[i] );
                flann::cuda::heap::sift_down( resultDist,resultIndex, 0, i, GreaterThan<DistanceType>() );
            }
        }
    }
};

// Difference to RadiusKnnResultSet: Works like KnnResultSet, doesn't pack the results densely (as the RadiusResultSet does)
template <typename DistanceType, bool useHeap>
struct KnnRadiusResultSet
{
    int foundNeighbors;
    DistanceType largestHeapDist;
    int maxDistIndex;
    const int k;
    const bool sorted;
    const DistanceType epsError;
    const DistanceType radius_sq;


    __device__
    KnnRadiusResultSet(int knn, bool sortResults, DistanceType eps, DistanceType radius) : foundNeighbors(0),largestHeapDist(radius),k(knn), sorted(sortResults), epsError(eps),radius_sq(radius){ }

    //          __host__ __device__
    //          KnnResultSet(const KnnResultSet& o):foundNeighbors(o.foundNeighbors),largestHeapDist(o.largestHeapDist),k(o.k){ }

    __device__
    inline DistanceType
    worstDist()
    {
        return largestHeapDist;
    }

    __device__
    inline void
    insert(int index, DistanceType dist)
    {
        if( dist < largestHeapDist ) {
            if( foundNeighbors<k ) {
                resultDist[foundNeighbors]=dist;
                resultIndex[foundNeighbors]=index;
                if( foundNeighbors==k-1 ) {
                    if( useHeap ) {
                        flann::cuda::heap::make_heap(resultDist,resultIndex,k,GreaterThan<DistanceType>());
                        largestHeapDist=resultDist[0];
                    }
                    else {
                        findLargestDistIndex();
                    }
                }
                foundNeighbors++;
            }
            else { //if( dist < largestHeapDist )
                if( useHeap ) {
                    resultDist[0]=dist;
                    resultIndex[0]=index;
                    flann::cuda::heap::sift_down(resultDist,resultIndex,0,k,GreaterThan<DistanceType>());
                    largestHeapDist=resultDist[0];
                }
                else {
                    resultDist[maxDistIndex]=dist;
                    resultIndex[maxDistIndex]=index;
                    findLargestDistIndex();
                }
            }
        }
    }
    __device__
    void
    findLargestDistIndex( )
    {
        largestHeapDist=resultDist[0];
        maxDistIndex=0;
        for( int i=1; i<k; i++ )
            if( resultDist[i] > largestHeapDist ) {
                maxDistIndex=i;
                largestHeapDist=resultDist[i];
            }
    }

    DistanceType* resultDist;
    int* resultIndex;

    __device__
    inline void
    setResultLocation( DistanceType* dists, int* index, int thread, int stride )
    {
        resultDist=dists+stride*thread;
        resultIndex=index+stride*thread;
        for( int i=0; i<stride; i++ ) {
            resultDist[i]=INFINITY;
            resultIndex[i]=-1;
            //                  resultIndex[tid+i*blockDim.x]=-1;
            //                  resultDist[tid+i*blockDim.x]=INFINITY;
        }
    }

    __device__
    inline void
    finish()
    {
        if( sorted ) {
            if( !useHeap ) flann::cuda::heap::make_heap(resultDist,resultIndex,k,GreaterThan<DistanceType>());
            for( int i=k-1; i>0; i-- ) {
                flann::cuda::swap( resultDist[0], resultDist[i] );
                flann::cuda::swap( resultIndex[0], resultIndex[i] );
                flann::cuda::heap::sift_down( resultDist,resultIndex, 0, i, GreaterThan<DistanceType>() );
            }
        }
    }
};

//! fills the radius output buffer.
//! IMPORTANT ASSERTION: ASSUMES THAT THERE IS ENOUGH SPACE FOR EVERY NEIGHBOR! IF THIS ISN'T
//! TRUE, USE KnnRadiusResultSet! (Otherwise, the neighbors of one element might overflow into the next element, or past the buffer.)
template< typename DistanceType >
struct RadiusResultSet
{
    DistanceType radius_sq_;
    int* segment_starts_;
    int count_;
    bool sorted_;

    __device__
    RadiusResultSet(DistanceType radius, int* segment_starts, bool sorted) : radius_sq_(radius), segment_starts_(segment_starts), count_(0), sorted_(sorted){ }

    __device__
    inline DistanceType
    worstDist()
    {
        return radius_sq_;
    }

    __device__
    inline void
    insert(int index, DistanceType dist)
    {
        if( dist < radius_sq_ ) {
            resultIndex[count_]=index;
            resultDist[count_]=dist;
            count_++;
        }
    }

    int* resultIndex;
    DistanceType* resultDist;

    __device__
    inline void
    setResultLocation( DistanceType* dists, int* index, int thread, int /*stride*/ )
    {
        resultIndex=index+segment_starts_[thread];
        resultDist=dists+segment_starts_[thread];
    }

    __device__
    inline void
    finish()
    {
        if( sorted_ ) {
            flann::cuda::heap::make_heap( resultDist,resultIndex, count_, GreaterThan<DistanceType>() );
            for( int i=count_-1; i>0; i-- ) {
                flann::cuda::swap( resultDist[0], resultDist[i] );
                flann::cuda::swap( resultIndex[0], resultIndex[i] );
                flann::cuda::heap::sift_down( resultDist,resultIndex, 0, i, GreaterThan<DistanceType>() );
            }
        }
    }
};
}
}

#endif
