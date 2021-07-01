![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete CUDA GEMM decomposition")

# CUTLASS

This document is intended to accompany the CUTLASS source code, to describe the interaction between
CUTLASS core components, and to identify their role in implementing GEMM computations efficiently in CUDA.

1. [Design Patterns](#S-design-patterns)
2. [General Matrix Multiply](#S-general-matrix-multiply)
3. [Core Components](#S-core-components)
4. [Utilities](#S-utilities)
5. [Optimization Strategies](#S-optimization-strategies)

# <a name="S-design-patterns"></a> 1. Design Patterns

CUTLASS strives to achieve the highest performance possible on NVIDIA GPUs while also offering a
flexible composition that can be easily applied to solve new problems related to Deep Learning and
linear algebra. Though we intend to make CUTLASS as simple and straightforward as possible, given
a tradeoff between simplicity and performance, CUTLASS chooses performance. Consequently, several
design patterns are necessary to yield a composable structure while also satisfying these performance
objectives. This section is intended to provide more detail.

* [Sequencing and Nesting](#S-patterns-sequencing-nesting)
* [Tiles and Iterators](#S-patterns-tiles-iterators)
* [Host-side Params](#S-patterns-host-side-params)
* [Composable Shared Memory](#S-patterns-composable-shared-memory)

## <a name="S-patterns-sequencing-nesting"></a> Sequencing and Nesting of Collective Primitives

CUTLASS embodies a design paradigm exemplified by the [CUB library](https://nvlabs.github.io/cub/) for expressing collective operations. Objects expose an interface for a problem that is then decomposed into concurrent subtasks executed by cooperating threadblocks, warps, and threads. For example, a grid-level object may be constructed with base pointers to the start of a GEMM operation, add a threadblock-dependent offset to partition the problem, and then compute a per-threadblock GEMM. This in turn performs some operations as a collection of cooperating threads, while it may partition other parts of the task into warp-level subtasks.

## <a name="S-patterns-tiles-iterators"></a> Tiles and Iterators

Efficient dense linear algebra computations emphasize data movement to match the execution of mathematical operators to the flow of data. Consequently, CUTLASS defines a rich set of primitives for partitioning a tile of data among participating threads, warps, and threadblocks. CUTLASS applies the familiar iterator design pattern to provide an abstraction layer to (1.) access these tile objects and (2.) traverse a sequence of objects embedded in a higher level data structure. These subpartitions are typically defined by compile-time constants
specifying element type, size, and data layout. CUTLASS refers to subpartitions as _tiles_.

_Iterators_ are familiar design patterns in C++ that provide an abstraction for accessing individual
elements in memory as well as traversing over a collection. GEMM kernels in CUTLASS depend on accessing
a sequence of tiles from global memory, from shared memory, and in registers. Consequently, _tile iterators_
are prevalent throughout the CUTLASS implementation.

The canonical CUTLASS tile iterator template is defined in [cutlass/tile_iterator.h](cutlass/tile_iterator.h).

## <a name="S-patterns-host-side-params"></a> Host-side Params structure

Several CUTLASS template classes exhibit a pattern in which problem-specific internal state is known at kernel launch time and remains invariant throughout the execution of a kernel. For example, tile iterators compute several offsets based on the strides of the input tensor that is added to an internal pointer when loading the elements of a tile. These are computed from the tensor stride and never updated; the per-thread internal state consists only of the internal global memory pointer.

CUTLASS can take advantage of this CUDA grid-invariant property by constructing the object in host code and passing a composed parameters structure to the kernel. This confers two benefits: (1.) invariant state is held in constant memory, and (2.) there is no overhead to compute the initial state by each thread.

The design pattern in CUTLASS is for classes with nontrivial constructors to define `struct Params` as an inner class which contains grid-invariant state. These should define a constructor and an `initialize()` method. The `Params` structure should also include a data member corresponding to each data member in the parent class, so these too can be properly constructed in host code. The parent class should define a constructor which accepts `Params const &` as its first argument.

For example, `cutlass::gemm::Gemm<>` should define `struct cutlass::gemm::Gemm::Params`. The latter should define data members for each data member in `cutlass::gemm::Gemm<>`.


## <a name="S-patterns-composable-shared-memory"></a> Composable shared memory allocation

Shared memory requires explicit effort by the programmer to allocate and de-allocate. CUTLASS follows the paradigm introduced by [CUB](https://nvlabs.github.io/cub/) to define composed structures for storing data intended to be held in shared memory. Any object requiring shared memory storage for itself or its data members should define a child structure called SharedStorage. This holds data needed by the class and also instantiates SharedStorage objects for each data member.

To be consistent, this pattern defines a convention in which classes define internal shared memory storage requirements. Classes should consider all SharedStorage structures to be opaque other than their own child class. When the lifetimes of child objects are known to be non-overlapping, unions may be used to alias multiple SharedStorage objects to the same shared memory region and reduce overall SMEM capacity.

## <a name="S-patterns-loop-unrolling"></a> Loop Unrolling

CUTLASS requires tiles of data to be stored in registers for high-bandwidth access. Simultaneously, high-throughput math instructions
must be issued concurrently with memory instructions to hide latency with relatively few concurrent threads. These objectives are
achieved by unrolling loops whose iteration counts are known at compile time.

Consequently, most loops within the CUTLASS GEMM implementation are specified by constant values and template arguments. The CUDA compiler
is able to unroll the loop bodies, map array elements to registers, and construct an efficient instruction schedule.

## <a name="S-patterns-loop-unrolling"></a> Templates

CUDA C++ templates and modern generic programming techniques enable CUTLASS device code to span a large design space.

This design space includes:
* Mixed precision arithmetic and data storage
* Kernels specialized for layout and problem size
* Support for kernel fusion

Moreover, templates provided a structured approach to collecting compile-time constants such as tile dimensions. These
must be template arguments to target static array allocation and take advantage of loop unrolling, constant folding,
and function inlining.

# <a name="S-general-matrix-multiply"></a> 2. General Matrix Multiply

The following figure illustrates the hierarchical GEMM computation embodied by CUTLASS. Each stage depicts a nested level of tiling which corresponds to a layer of concurrency within the CUDA execution model and to a level within the memory hierarchy, becoming increasingly finer moving left to right.

![ALT](/media/images/gemm-structural-components.png "CUTLASS GEMM Structural Components")

## Threadblock-level GEMM

The CUTLASS GEMM kernel partitions the _C_ matrix into a 2D tiling of threadblocks.
Each threadblock computes a matrix product whose outer dimensions _M_ and _N_ are compile-time constants. The
GEMM's _K_ dimension is partitioned into tiles and iterated over by the GEMM _mainloop_. The shape of the matrix
multiply operation performed by each iteration of the mainloop is referred to as _OutputTile_.

The threadblock loads a sequence of tiles from global memory and stores this data to shared memory. The iterative
access and traversal of tiles in global memory are performed by a _TileLoadIterator_, and storing to a circular
buffer in shared memory is performed by a _GlobalLoadIterator_.

**[Global Load Stream](cutlass/gemm/gemm_global_stream.h)** manages loading of the threadblock-scope multiplicands to the GEMM kernel. It owns an iterator into global memory for loading tiles of data, a TensorAllocation in shared memory to hold the resulting tile, and an iterator for writing the tile into this allocation. A transformer exists to optionally transform the data as it is loaded which may of use to perform type conversion or, in the case of int8 GEMM, transpose 4x4 tiles held in registers.

The Global Load Stream template contains members defined by the following templates:

* [GemmGlobalIteratorAb](cutlass/gemm/gemm_global_tile.h)
* [Transformer](cutlass/convert.h)
* [GemmSharedStoreTileAb](cutlass/gemm/gemm_shared_tile.h)

## Warp-level GEMM

The threadblock's _OutputTile_ is partitioned among the warps, and each computes a warp-level matrix product.
Data is loaded from shared memory into registers, and math instructions are dispatched to CUDA Cores or Tensor Cores.

[**Shared Load Stream**](cutlass/gemm/gemm_shared_stream.h) manages loading of warp-level multiplicands from shared memory into registers. This owns an iterator for fetching data and the destination fragments for holding the results.

* [GemmSharedLoadTile{A,B}](cutlass/gemm/gemm_shared_tile.h)

**Matrix Multiply** computes a matrix product operation on data held in registers. Specializations exist for thread-level instructions such as single-precision fused multiply-add as well as warp-level matrix operations targeting TensorCores.

* [WMMA Multiply Add](cutlass/gemm/wmma_gemm_multiply_add.h)

## Thread-level GEMM

SGEMM, IGEMM, HGEMM, and DGEMM are computed by SIMT math instructions issued by thread-level matrix multiply
procedures.

* [ThreadMultiplyAdd](cutlass/gemm/thread_multiply_add.h)
* [IGEMM specialization](cutlass/gemm/igemm_multiply_add.h)
* [HGEMM specialization](cutlass/gemm/hgemm_multiply_add.h)

## Epilogue

The [**epilogue**](cutlass/gemm/gemm_epilogue.h) iteratively selects a subset of accumulator elements held by a warp, writes them to shared memory, and loads them by different threads such that a threadblock-scoped tile store operation will make contiguous, striped accesses to global memory. Thus, the flow of data utilizes the following components:

1. [Transformer](cutlass/convert.h) for converting the data types of accumulator elements
2. [GemmSharedStoreTileD](cutlass/gemm/gemm_shared_tile.h) to store to shared memory specialized to the accumulator layout.
3. [GemmSharedLoadTileD](cutlass/gemm/gemm_shared_tile.h) to load the data from shared memory.
4. [GemmGlobalIteratorC](cutlass/gemm/gemm_global_tile.h) to load a tile from global memory.
5. A [functor](cutlass/gemm/linear_scaling.h) to compute an element-wise operation on the matrix product and source data (such as alpha*AB+beta*C).
6. [GemmGlobalIteratorD](cutlass/gemm/gemm_global_tile.h) to write the output to global memory.

## GEMM Traits

[**cutlass::gemm::GemmTraits**](cutlass/gemm/gemm_traits.h) collects the structural properties of a complete GEMM computation into a single template class. As a result, the Traits classes encapsulate the the iterators and transformers for all supported GEMM operands and layouts. Low-level details needed by Traits (such as scalar types for operands, thread-block tile size, number of scalar elements per memory access within each phase, number of stages in shared memory, as well as other implementation-specific properties of the GEMM computation) are specified in class [**cutlass::gemm::GemmConfig**](cutlass/gemm/gemm_config.h).


# <a name="S-core-components"></a> 3. Core Components

CUTLASS GEMM kernels are implemented by a set of Core components for interacting with mathematical tensor and matrix
objects as well as constructing efficient CUDA kernels.

* [Tensor views](#S-core-tensor-views)
* [Shape](#S-core-shape)
* [Tile structure](#S-core-tile-structure)
* [Fragment](#S-core-fragment)
* [Predicate vector](#S-core-predicate-vector)

## <a name="S-core-tensor-views"></a> Tensor View

Matrices and tensors are typically represented as n-D arrays held in linear memory with a single base pointer and a stride vector. Element _i_ of the stride vector indicates the offset in linear memory between consecutive elements in dimension i. Consequently, the linear offset for an arbitrary element specified as an n-tuple may be computed as the dot product of the coordinate and the stride vector.

CUTLASS provides abstractions for interacting with multidimension tensors in device memory.
Consequently, we define a hierarchy of pointer-like types for referencing tensors.

`T *` - raw pointer to elements of type T

`cutlass::TensorRef<T, Rank>`  - reference to a tensor of elements of type T and given rank. Includes a mapping function and associated stride vector for accessing elements in linear memory.

`cutlass::TensorView<T, Rank>` - extends `TensorRef<>` by adding bounds information. This is a complete mathematical object which may be used as the argument to CUTLASS functions.

The above provide an identity maping of a logical index space to linear memory. An element
at logical coordinate X has an offset computed as follows:
```
offset = dot(X, stride)
```
where `dot()` computes the inner product of X and a vector of "strides."

CUTLASS 1.1 introduces a mapping function and an additional "storage rank" to offer a flexible way to
map the logical index space of the tensor to memory. The mapping function maps a coordinate
of rank _R_ to an index space of rank _S_. The linear offset is computed as:
```
offset = dot( MapFunc(X), stride )
```
where stride is a vector of rank _S_.

CUTLASS kernels make extensive use of vectorization of memory accesses for efficiency and
correctness. Consequently, we enforce a constraint on the strides used by mapping functions
such that:

1. The "fastest-changing" stride is always 1 thereby mandating that consecutive elements in
   that rank are consecutive in linear memory.

2. The fastest changing rank is always last in the stride vector and not explicitly stored.

Thus, the stride vector used by mapping functions has length of one fewer than the rank of the
storage tensor. These constraints are consistent with the BLAS interface of passing matrices as
a tuple consisting of a pointer and a "leading dimension." In fact, these are rank=2 tensors
whose fastest changing dimension is 1, and only the strided dimension is explicitly represented.

A typical mapping function might simply map the rows and columns of a matrix, a rank=2 tensor,
to linear memory such that (1.) elements in the same column are consecutive in memory
(column-major), or (2.) elements in the same row are consecutive (row-major). These can be
accomplished by two different mapping functions whose stride vector is length=2. The first
element is the "leading dimension."

The requirement that the fastest-changing stride always be of unit size need not be a limitation.
To implement "sparse" computations or matrix operations in which matrix elements have arbitrary
stride along both row and column, define a mapping function whose storage rank is 3. This permits
two elements of the stride vector to have a non-unit value.

`cutlass::TensorView<>` extends this concept by including a size vector to specify the bounds of
the index space. The value of each coordinate in the size vector defines the half-open range of
indices whose smallest value is zero.

## <a name="S-core-shape"></a> Shape

To avoid complicated template metaprogramming, CUTLASS targets fixed compile-time tile sizes specified
by a four-dimensional template `cutlass::Shape<>`. This defines the following dimensions, mirroring
the NHWC tensor format used for convolution in Deep Learning frameworks.

- `D`: depth of tensor
- `H`: first strided dimension
- `W`: contiguous sequence of tensor elements
- `C`: number of channels, usually used for vectorized access

Template specializations of `Shape` appear as arguments to numerous dependent template classes which
must specify compile-time constant tile sizes.

## <a name="S-core-tile-structure"></a> Tile Structure

Tiled structures express an arrangement of data in memory as well as a logical mapping of concurrent CUDA
threads to the problem space. For example, the CUTLASS GEMM

Tiled structures can be defined using the `cutlass::TileTraits<>` concept which defines the following
members. Collectively, these members offer a flexible way to define a 4-D subpartition of an integer
lattice, partition its elements among a collection of threads, and map each unique thread ID to a unique
offset.

- _Tile_ (concept `Shape<>`) - describes the dimensions of the tile in terms of scalar elements
- _Delta_ (concept `Shape<>`) - describes the distance along each logical dimension between items
- _Iterations_ (concept `Shape<>`) - describes the number of items along each logical dimension
- _ThreadOffset_ (concept _functor_) - implements `Coord<4> operator()() const` to determine a thread's
  initial offset in the logical 4-D coordinate space

The following figure illustrates the CUTLASS tile structure. The overall shape, 16-by-16, is partitioned into
vectors of length two among 32 threads. The elements stored by thread 9 are highlighted.

<img src="/media/images/cutlass-tile-structure.png" alt="CUTLASS tile structure" width="30%" />

The `cutlass::TileTraits<>` definition that describes this arrangement may be defined as follows:

```
struct ExampleTileTraits {

  /// Overall shape of tile
  typedef Shape<1, 16, 16, 1> Tile;

  /// Distance along each dimension of accesses
  typedef Shape<1, 4, 1, 1> Delta;

  /// Number of memory accesses performed by each thread
  typedef Shape<1, 4, 1, 1> Iterations;

  /// Offset function - maps each thread to a unique starting offset within the 4D tile
  struct ThreadOffset {

    CUTLASS_DEVICE Coord<4> operator()() const {

      typdef Shape<1, 16, 8, 2> Vectorized;

      return make_Coord(
        0,                              // depth "D" dimension
        threadIdx.x / Vectorized::kW,   // horisontal "H" dimension - first strided dimension
        threadIdx.x % Vectorized::kW,   // vertical "W" dimension - contiguous dimension
        0
      );
    }
  };
};
```

## <a name="S-core-tile-iterator"></a> Tile Iterator

The iterator design pattern provides an abstraction for accessing the items in a collection in sequence. Basic
operators defined by iterators consist of accessing an item - either a load or store - followed by traversal to
the next item in sequence.

<img src="/media/images/cutlass-tile-iteration.png" alt="CUTLASS tile access and traversal" width="50%" />

To offer a generic solution that spans numerous data types and layouts, CUTLASS defines the _TileIterator_ concept.
This concept provides access to a sequence of _tiles_ embedded in a tensor in addressable memory.

The canonical CUTLASS tile iterator template is defined in [cutlass/tile_iterator.h](cutlass/tile_iterator.h).

## <a name="S-core-fragment"></a> Fragment

A fragment is analogous to `std::array<>` in that it is a constant-sized array of elements. Typically backed by storage in the SM's register file, CUTLASS `Fragment<>` objects are used to store tiles. For threadblock- and warp-scope operations, the contents of these tiles are distributed across the partipcipating threads. In such cases, a thread's `Fragment<>` contains the part of the tile held by that thread.

## <a name="S-core-predicate-vector"></a> Predicate Vector

SIMT architectures utilize predicated execution in place of control flow when conditional code sequences are fairly short, on the order of a few machine instructions. While CUDA C++ does not include constructs at the language level for predication, PTX makes this explicit, and compilation to SASS is assumed to aggressively utilize predication. Typical applications are to initialize a sequence of bits used to mask memory operations and use these bits as predicates guarding memory load and store instructions.

CUTLASS provides `PredicateVector` defined in [cutlass/predicate_vector.h](cutlass/predicate_vector.h) to manage a statically-sized bit vector, store them into general purpose registers, and efficiently access them in sequence. By storing four predicates per byte in hardware registers, the CUDA compiler is able to issue specialized instructions to achieve very efficient unpacking.


# <a name="S-utilities"></a> 4. Utilities

CUTLASS implements efficient matrix multiply computations on GPUs. It is accompanied by an extensive utility
framework offering features such as:

* [cutlass::half_t](tools/util/half.h) - a host-side half-precision type
* Components for allocating and initializing [host-side and device-side tensors](tools/util/host_tensor.h) usable by CUTLASS
* Reference implementations of [GEMM](tools/util/reference/host/gemm.h) and [element-wise operations](tools/util/reference/host/tensor_elementwise.h)


# <a name="S-optimization-strategies"></a>5. Optimization Strategies

This section describes several strategies taken to increase performance beyond what is achievable with
a basic implementation of the hierarchical GEMM structure.


## Threadblock Rasterization

To maximize reuse of data held in the last level cache, CUTLASS defines several functions to
affect the mapping of threadblocks to logical partitions of the GEMM problem. These map
consecutively launched threadblocks to packed two-dimensional regions of the partitioned GEMM
problem to increase the probability that these will access the same tiles of global memory at
approximately the same time.

Several functions are defined in [cutlass/gemm/threadblock_swizzle.h](cutlass/gemm/threadblock_swizzle.h).


## Parallel Reductions across GEMM _K_

Matrix product computations expose parallelism among _O(MN)_ independent inner product
computations. For sufficiently large problem sizes, a GEMM kernel in CUTLASS may approach
the theoretical maximum computational throughput. For small problems, however, there are
too few threadblocks to efficiently occupy the entire GPU.

As a recourse, parallelizing the reduction performed during the inner product computation
enables more threadblocks to execute concurrently while still taking advantage of the throughput
benefits of large threadblock-level GEMM tiles.

CUTLASS implements parallel reductions across threadblocks by partitioning the GEMM _K_ dimension
and launching an additional set of threadblocks for each partition. Consequently, we refer to
this strategy within CUTLASS as "parallel reduction splitK." The "parallel reduction splitK" in cutlass requires the execution of 2 kernels. The first one is called partitionedK GEMM. The second one is called batched reduction.

The partitionedK GEMM is very similar to one flavor of batched strided GEMM. Instead of requiring users to specify the problem size of each batch, partitionedK GEMM asks for the overall problem size and the number of partition that will be applied along K dimension for operand A and B. For example, parameters of m=128, n=128, k=4096 and partition=16 will result in 16 batched strided GEMMs with each batch of m=128, n=128, k=256. PartitionedK also allows scenario where k is not divisible by partition count. For example, parameters of m=128, n=128, k=4096 and partition=20 will result in 20 batched strided GEMMs with the first 19 batches of m=128, n=128, k=4096/20=204 and the last batch of m=128, n=128, k=220.

The batched reduction kernel will further perform reduction along the K-dimension. Thus, the input of the batched reduction kernel is the output (C) of partitionedK GEMM. An workspace memory is managed by the users to store this intermediate results.

An example of splitK usage can be found [here](examples/06_splitK_gemm/splitK_gemm.cu).


# Copyright

Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.

```
  Redistribution and use in source and binary forms, with or without modification, are permitted
  provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright notice, this list of
        conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright notice, this list of
        conditions and the following disclaimer in the documentation and/or other materials
        provided with the distribution.
      * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
        to endorse or promote products derived from this software without specific prior written
        permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
