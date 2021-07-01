<hr>
<h3>About CUB</h3>

Current release: v1.8.0 (02/16/2018)

We recommend the [CUB Project Website](http://nvlabs.github.com/cub) for further information and examples.

CUB provides state-of-the-art, reusable software components for every layer 
of the CUDA programming model:
- [<b><em>Device-wide primitives</em></b>] (https://nvlabs.github.com/cub/group___device_module.html) 
  - Sort, prefix scan, reduction, histogram, etc.  
  - Compatible with CUDA dynamic parallelism
- [<b><em>Block-wide "collective" primitives</em></b>] (https://nvlabs.github.com/cub/group___block_module.html)
  - I/O, sort, prefix scan, reduction, histogram, etc.  
  - Compatible with arbitrary thread block sizes and types 
- [<b><em>Warp-wide "collective" primitives</em></b>] (https://nvlabs.github.com/cub/group___warp_module.html)
  - Warp-wide prefix scan, reduction, etc.
  - Safe and architecture-specific
- [<b><em>Thread and resource utilities</em></b>](https://nvlabs.github.com/cub/group___thread_module.html)
  - PTX intrinsics, device reflection, texture-caching iterators, caching memory allocators, etc. 

![Orientation of collective primitives within the CUDA software stack](http://nvlabs.github.com/cub/cub_overview.png)

<br><hr>
<h3>A Simple Example</h3>

```C++
#include <cub/cub.cuh>
 
// Block-sorting CUDA kernel
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
     using namespace cub;

     // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads 
     // owning 16 integer items each
     typedef BlockRadixSort<int, 128, 16>                     BlockRadixSort;
     typedef BlockLoad<int, 128, 16, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     typedef BlockStore<int, 128, 16, BLOCK_STORE_TRANSPOSE> BlockStore;
 
     // Allocate shared memory
     __shared__ union {
         typename BlockRadixSort::TempStorage  sort;
         typename BlockLoad::TempStorage       load; 
         typename BlockStore::TempStorage      store; 
     } temp_storage; 

     int block_offset = blockIdx.x * (128 * 16);	  // OffsetT for this block's ment

     // Obtain a segment of 2048 consecutive keys that are blocked across threads
     int thread_keys[16];
     BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
     __syncthreads();

     // Collectively sort the keys
     BlockRadixSort(temp_storage.sort).Sort(thread_keys);
     __syncthreads();

     // Store the sorted segment 
     BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}
```

Each thread block uses cub::BlockRadixSort to collectively sort 
its own input segment.  The class is specialized by the 
data type being sorted, by the number of threads per block, by the number of 
keys per thread, and implicitly by the targeted compilation architecture.  

The cub::BlockLoad and cub::BlockStore classes are similarly specialized.    
Furthermore, to provide coalesced accesses to device memory, these primitives are 
configured to access memory using a striped access pattern (where consecutive threads 
simultaneously access consecutive items) and then <em>transpose</em> the keys into 
a [<em>blocked arrangement</em>](index.html#sec4sec3) of elements across threads. 

Once specialized, these classes expose opaque \p TempStorage member types.  
The thread block uses these storage types to statically allocate the union of 
shared memory needed by the thread block.  (Alternatively these storage types 
could be aliased to global memory allocations).

<br><hr>
<h3>Stable Releases</h3>

CUB releases are labeled using version identifiers having three fields: 
*epoch.feature.update*.  The *epoch* field corresponds to support for
a major change in the CUDA programming model.  The *feature* field 
corresponds to a stable set of features, functionality, and interface.  The
*update* field corresponds to a bug-fix or performance update for that
feature set.  At the moment, we do not publicly provide non-stable releases 
such as development snapshots, beta releases or rolling releases.  (Feel free
to contact us if you would like such things.)  See the 
[CUB Project Website](http://nvlabs.github.com/cub) for more information.

<br><hr>
<h3>Contributors</h3>

CUB is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).  The primary contributor is [Duane Merrill](http://github.com/dumerrill).

<br><hr>
<h3>Open Source License</h3>

CUB is available under the "New BSD" open-source license:

```
Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
