/******************************************************************************
 * Copyright (c) 2011-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
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

#pragma once

/**
 * \file
 * \brief C++ interface to CUDA device memory management functions.
 */

#include <memory>

#include "cutlass/util/debug.h"
#include "cutlass/util/platform.h"
#include "tools/util/exceptions.h"

namespace cutlass {
namespace device_memory {

/******************************************************************************
 * Allocation lifetime
 ******************************************************************************/

/// Allocate a buffer of \p count elements of type \p T on the current CUDA device
template <typename T>
T* allocate(size_t count = 1) {
  T* ptr = 0;
  size_t bytes = sizeof(T) * count;

  cudaError_t cuda_error = CUDA_PERROR(cudaMalloc((void**)&ptr, bytes));
  if (cuda_error != cudaSuccess) {
    throw cuda_exception("Failed to allocate memory", cuda_error);
  }

  return ptr;
}

/// Free the buffer pointed to by \p ptr
template <typename T>
void free(T* ptr) {
  if (ptr) {
    cudaError_t cuda_error = CUDA_PERROR(cudaFree(ptr));
    if (cuda_error != cudaSuccess) {
      throw cuda_exception("Failed to free device memory", cuda_error);
    }
  }
}

/******************************************************************************
 * Data movement
 ******************************************************************************/

template <typename T>
void copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind) {
  size_t bytes = count * sizeof(T);

  cudaError_t cuda_error = CUDA_PERROR(cudaMemcpy(dst, src, bytes, kind));
  if (cuda_error != cudaSuccess) {
    throw cuda_exception("cudaMemcpy() failed", cuda_error);
  }
}

template <typename T>
void copy_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToHost);
}

template <typename T>
void copy_device_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToDevice);
}

/// Copies elements from device memory to host-side range
template <typename OutputIterator, typename T>
void insert_to_host(OutputIterator begin, OutputIterator end, T const* device_begin) {
  size_t elements = end - begin;
  copy_to_host(&*begin, device_begin, elements);
}

/// Copies elements to device memory from host-side range
template <typename T, typename InputIterator>
void insert_to_device(T* device_begin, InputIterator begin, InputIterator end) {
  size_t elements = end - begin;
  copy_to_device(device_begin, &*begin, elements);
}

/******************************************************************************
 * "Smart" device memory allocation
 ******************************************************************************/

/// Device allocation abstraction that tracks size and capacity
template <typename T>
struct allocation {
  /// Delete functor for CUDA device memory
  struct deleter {
    void operator()(T* ptr) {
      cudaError_t cuda_error = CUDA_PERROR(cudaFree(ptr));
      if (cuda_error != cudaSuccess) {
        // noexcept
        //                throw cuda_exception("cudaFree() failed", cuda_error);
        return;
      }
    }
  };

  //
  // Data members
  //

  /// Number of elements of T allocated on the current CUDA device
  size_t capacity;

  /// Smart pointer
  platform::unique_ptr<T, deleter> smart_ptr;

  //
  // Methods
  //

  /// Constructor: allocates no memory
  allocation() : capacity(0) {}

  /// Constructor: allocates \p capacity elements on the current CUDA device
  allocation(size_t _capacity) : smart_ptr(allocate<T>(_capacity)), capacity(_capacity) {}

  /// Copy constructor
  allocation(allocation const &p): smart_ptr(allocate<T>(p.capacity)), capacity(p.capacity) {
    copy_device_to_device(smart_ptr.get(), p.get(), capacity);
  }

  /// Destructor
  ~allocation() { reset(); }

  /// Returns a pointer to the managed object
  T* get() const { return smart_ptr.get(); }

  /// Releases the ownership of the managed object (without deleting) and resets capacity to zero
  T* release() {
    capacity = 0;
    return smart_ptr.release();
  }

  /// Deletes the managed object and resets capacity to zero
  void reset() {
    capacity = 0;
    smart_ptr.reset();
  }

  /// Deletes managed object, if owned, and replaces its reference with a given pointer and capacity
  void reset(T* _ptr, size_t _capacity) {
    smart_ptr.reset(_ptr);
    capacity = _capacity;
  }

  /// Returns a pointer to the object owned by *this
  T* operator->() const { return smart_ptr.get(); }

  /// Returns the deleter object which would be used for destruction of the managed object.
  deleter& get_deleter() { return smart_ptr.get_deleter(); }

  /// Returns the deleter object which would be used for destruction of the managed object (const)
  const deleter& get_deleter() const { return smart_ptr.get_deleter(); }

  /// Copies a device-side memory allocation
  allocation & operator=(allocation const &p) {
    if (capacity != p.capacity) {
      smart_ptr.reset(allocate<T>(p.capacity));
      capacity = p.capacity;
    }
    copy_device_to_device(smart_ptr.get(), p.get(), capacity);
    return *this;
  }
};

}  // namespace device_memory
}  // namespace cutlass
