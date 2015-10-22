#ifndef FLANN_UTIL_CUDA_HEAP_H
#define FLANN_UTIL_CUDA_HEAP_H

/*
    Copyright (c) 2011, Andreas Mützel <andreas.muetzel@gmx.net>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Andreas Mützel <andreas.muetzel@gmx.net> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Andreas Mützel <andreas.muetzel@gmx.net> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace flann
{
namespace cuda
{
template <class T>
__device__ __host__ void swap( T& x, T& y )
{
    T t=x;
    x=y;
    y=t;
}

namespace heap
{

//! moves an element down the heap until all children are smaller than the elemnent
//! if c is a less-than comparator, it do this until all children are larger
template <class GreaterThan, class RandomAccessIterator>
__host__ __device__ void
sift_down( RandomAccessIterator array, size_t begin, size_t length, GreaterThan c = GreaterThan() )
{

    while( 2*begin+1 < length ) {
        size_t left = 2*begin+1;
        size_t right = 2*begin+2;
        size_t largest=begin;
        if((left < length)&& c(array[left], array[largest]) ) largest=left;

        if((right < length)&& c(array[right], array[largest]) ) largest=right;

        if( largest != begin ) {
            cuda::swap( array[begin], array[largest] );
            begin=largest;
        }
        else return;
    }
}

//! creates a max-heap in the array beginning at begin of length "length"
//! if c is a less-than comparator, it will create a min-heap
template <class GreaterThan, class RandomAccessIterator>
__host__ __device__ void
make_heap( RandomAccessIterator begin, size_t length, GreaterThan c = GreaterThan() )
{
    int i=length/2-1;
    while( i>=0 ) {
        sift_down( begin, i, length, c );
        i--;
    }
}


//! verifies if the array is a max-heap
//! if c is a less-than comparator, it will verify if it is a min-heap
template <class GreaterThan, class RandomAccessIterator>
__host__ __device__ bool
is_heap( RandomAccessIterator begin, size_t length, GreaterThan c = GreaterThan() )
{
    for( unsigned i=0; i<length; i++ ) {
        if((2*i+1 < length)&& c(begin[2*i+1],begin[i]) ) return false;
        if((2*i+2 < length)&& c(begin[2*i+2],begin[i]) ) return false;
    }
    return true;
}


//! moves an element down the heap until all children are smaller than the elemnent
//! if c is a less-than comparator, it do this until all children are larger
template <class GreaterThan, class RandomAccessIterator, class RandomAccessIterator2>
__host__ __device__ void
sift_down( RandomAccessIterator key, RandomAccessIterator2 value, size_t begin, size_t length, GreaterThan c = GreaterThan() )
{

    while( 2*begin+1 < length ) {
        size_t left = 2*begin+1;
        size_t right = 2*begin+2;
        size_t largest=begin;
        if((left < length)&& c(key[left], key[largest]) ) largest=left;

        if((right < length)&& c(key[right], key[largest]) ) largest=right;

        if( largest != begin ) {
            cuda::swap( key[begin], key[largest] );
            cuda::swap( value[begin], value[largest] );
            begin=largest;
        }
        else return;
    }
}

//! creates a max-heap in the array beginning at begin of length "length"
//! if c is a less-than comparator, it will create a min-heap
template <class GreaterThan, class RandomAccessIterator, class RandomAccessIterator2>
__host__ __device__ void
make_heap( RandomAccessIterator key,  RandomAccessIterator2 value, size_t length, GreaterThan c = GreaterThan() )
{
    int i=length/2-1;
    while( i>=0 ) {
        sift_down( key, value, i, length, c );
        i--;
    }
}

}

}
}

#endif