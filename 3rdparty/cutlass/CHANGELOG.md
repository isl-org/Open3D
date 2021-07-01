# NVIDIA CUTLASS Changelog

## [1.3.2](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.2) (2019-07-09)
 * Performance improvement for Volta Tensor Cores TN and TT layouts.

## [1.3.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.1) (2019-04-09)
 * Corrected NVRTC unit tests.

## [1.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.0) (2019-03-20)
 * Efficient GEMM kernel targeting Volta Tensor Cores via `mma.sync` instruction added in CUDA 10.1.

## [1.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.2.0) (2018-10-26)
 * Parallelized reductions across threadblocks ("Split-K")
   * Improved IGEMM performance
 * Batched strided WMMA GEMMs

## [1.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.1.0) (2018-09-19)
  * Turing Features
    * WMMA GEMM targeting TensorCores - INT8, INT4, 1-bit
  * Batched Strided GEMM
  * Threadblock rasterization strategies
    * Improved performance for adverse problem sizes and data layouts
  * Extended CUTLASS Core comonents
    * Tensor views support arbitrary matrix and tensor layouts
    * Zip iterators for structuring multiple data streams
  * Enhanced CUTLASS utilities
    * Reference code for tensor operations in host and device code
    * Added HostMatrix<> for simplified matrix creation
  * Examples
    * Basic GEMM, tensor views, CUTLASS utilities, batched GEMM, WMMA GEMM

## [1.0.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.0.1) (2018-06-11)

  * Intra-threadblock reduction added for small threadblock tile sizes
    * sgemm_64x128x16, sgemm_128x128x16, sgemm_128x64x16, sgemm_128x32x16, sgemm_64x64x16, sgemm_64x32x16
    * igemm_32x32x128
  * GEMM _K_ residue handled during prologue prior to mainloop
  * Replaced Google Test copy with submodule. Use `git submodule init --recursive --update`

## [1.0.0](https://github.com/NVIDIA/cutlass/commit/2028ebe120aab22bfd0b2baf8902d4c9627eb33f) (2018-05-16)

  * Substantial rewrite to accommodate new architecture
  * Kernels: SGEMM, DGEMM, IGEMM, HGEMM, WMMA GEMM
  * Unit and performance tests

## [0.0.1](https://github.com/NVIDIA/cutlass/commit/d08ba8ac46e2fa3f745e070c390182edb56b2e91) (2017-12-04)

  * Initial release


## Copyright

Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.

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

