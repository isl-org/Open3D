/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

/*! \file
  \brief HostTensor contributes management for both host and device memory.

  HostTensor allocates host and device memory upon construction. Basic element-wise operations on
  host memory synchronize device memory automatically. Explicit copy operations provide abstractions
  for CUDA memcpy operations.

  Call device_{data, ref, view} for accessing device memory allocations.

  See cutlass/tensor_ref.h, cutlass/tensor_view.h, and tools/util/host_tensor_view.h for more details.
*/

#include "cutlass/cutlass.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/tensor_ref.h"
#include "tools/util/device_memory.h"
#include "tools/util/host_tensor_view.h"
#include "tools/util/type_traits.h"
#include <vector>

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Host tensor
template <
  /// Scalar data type (may be mapped to compatible types for use on host and device)
  typename T,
  /// Rank of logical tensor
  int Rank_ = 4,
  /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
  typename MapFunc_ = IdentityTensorMapFunc<Rank_>,
  /// Rank of internal n-D array
  int StorageRank_ = MapFunc_::kStorageRank,
  /// Index type used for coordinates
  typename Index_ = int,
  /// Index type used for offsets and pointer differences
  typename LongIndex_ = long long
>
class HostTensor : public HostTensorView<
  typename TypeTraits<T>::host_type,
  Rank_,
  MapFunc_,
  StorageRank_,
  Index_,
  LongIndex_> {
 public:
  /// Type used for host-side allocations
  typedef typename TypeTraits<T>::host_type HostType;

  /// Type used for device-side allocations
  typedef typename TypeTraits<T>::device_type DeviceType;

  /// Base class
  typedef HostTensorView<
    typename TypeTraits<T>::host_type,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> Base;

  /// Type used to compute the offset of an element to the base of a tensor
  typedef LongIndex_ LongIndex;

  /// Tensor reference to device memory
  typedef typename cutlass::TensorRef<
    DeviceType,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> DeviceTensorRef;

  /// Tensor reference to constant device memory
  typedef typename DeviceTensorRef::ConstTensorRef ConstDeviceTensorRef;

  /// TensorView to device memory
  typedef TensorView<
    DeviceType,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> DeviceTensorView;

  /// Tensor reference to constant device memory
  typedef typename DeviceTensorView::ConstTensorView ConstDeviceTensorView;

  /// Tensor reference to host memory
  typedef typename Base::TensorRef_t TensorRef_t;

  /// Tensor view to host memory
  typedef HostTensorView<
    typename TypeTraits<T>::host_type,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> HostTensorView_t;

  /// Tensor view to host memory
  typedef typename HostTensorView_t::ConstTensorView ConstHostTensorView;

  /// Coordinate in logical tensor space
  typedef typename TensorRef_t::TensorCoord TensorCoord;

  /// Coordinate in storage n-D array
  typedef typename TensorRef_t::StorageCoord StorageCoord;

  /// Stride vector in storage coordinate space
  /// Least significant stride is = 1 and not stored
  typedef typename TensorRef_t::StrideVector StrideVector;

  /// Rank of internal storage.
  static int const kStorageRank = Base::kStorageRank;

 private:

  /// Host-side memory allocation
  std::vector<HostType> host_;

  /// Device-side memory
  cutlass::device_memory::allocation<DeviceType> device_;

 public:
  //
  // Device and Host Methods
  //

  /// Default constructor
  HostTensor() {}

  /// Constructor for resizing the least significant rank
  HostTensor(Index_ size_1D, bool device_backed = true) {
    this->resize(size_1D, device_backed);
  }

  /// Helper to construct from pointer, stride, and size
  HostTensor(
    StorageCoord const &_stride,
    TensorCoord const& _size,
    bool _device_backed = true
  ) {

    this->reset(_stride, _size);
  }

  /// Clears the HostTensor allocation to size/capacity = 0
  void reset() {
    host_.clear();
    device_.reset();
    Base::reset();
  }

  /// Helper to resize the least significant rank
  void resize(
    Index_ size_1D,
    bool _device_backed = true) {

    TensorCoord _size;
    _size[Base::kRank - 1] = size_1D;
    for (int i = 0; i < Base::kRank - 1; ++i) {
      _size[i] = 1;
    }
    StorageCoord _stride;
    _stride[Base::kStorageRank - 1] = 1;
    for (int i = 0; i < Base::kStorageRank - 1; ++i) {
      _stride[i] = size_1D;
    }
    this->reset(_stride, _size, _device_backed);
  }

  /// Updates the reference and size of a Tensor_view object
  void reset(
    StorageCoord const& stride,
    TensorCoord const& size,
    bool _device_backed = true) {

    // Construct a temporary TensorView so we can calculate the new capacity
    size_t _capacity = Base(nullptr, stride, size).capacity();

    // Allocate memory
    DeviceType* _device_memory = nullptr;
    if (_device_backed) {
      _device_memory = cutlass::device_memory::allocate<DeviceType>(_capacity);
    }

    host_.clear();
    host_.resize(_capacity);
    device_.reset(_device_memory, _capacity);

    Base::reset(TensorRef_t(host_.data(), stride), size);
  }

  /// Accesses the tensor reference pointing to data
  TensorRef_t host_ref() { return Base::ref(); }

  /// Accesses the tensor reference pointing to data
  TensorRef_t host_ref() const { return Base::ref(); }

  /// Accesses the tensor reference pointing to data
  DeviceTensorRef device_ref() const {
    return DeviceTensorRef(device_data(), this->stride());
  }

  /// Accesses the tensor reference pointing to data
  HostTensorView_t host_view() {
    return HostTensorView_t(host_data(), this->stride(), this->size());
  }

  /// Accesses the tensor reference pointing to data
  ConstHostTensorView host_view() const {
    return HostTensorView_t(host_data(), this->stride(), this->size());
  }

  /// Accesses the tensor reference pointing to data
  DeviceTensorView device_view() const {
    return DeviceTensorView(device_data(), this->stride(), this->size());
  }

  /// Gets pointer to host data
  HostType * host_data() { return host_.data(); }

  /// Gets pointer to device data
  DeviceType* device_data() { return device_.get(); }

  /// Gets pointer to host data
  HostType const * host_data() const { return host_.data(); }

  /// Gets pointer to device data
  DeviceType * device_data() const { return device_.get(); }

  /// Returns true if device memory is allocated
  bool device_backed() const {
    return device_.get();
  }

  /// Copies data from device to host
  void sync_host() {
    if (device_.get()) {
      device_memory::copy_to_host(
          host_.data(), reinterpret_cast<HostType const*>(device_.get()), host_.size());
    }
  }

  /// Copies data from host to device
  void sync_device() {
    if (device_.get()) {
      device_memory::copy_to_device(
        device_.get(),
        reinterpret_cast<DeviceType const*>(host_.data()),
        host_.size());
    }
  }

  /// Copy data from a caller-supplied device pointer into host memory
  void copy_to_host(DeviceType const* ptr_device) {
    device_memory::copy_to_host(
      host_.data(), reinterpret_cast<HostType const*>(ptr_device), host_.size());
  }

  /// Copies device-to-device
  void copy_to_device(DeviceType* ptr_device) {
    device_memory::copy_to_device(
      ptr_device, reinterpret_cast<DeviceType const*>(host_.data()), host_.size());
  }

  /// Accumulate in place
  template <typename SrcTensorView>
  HostTensor& operator+=(SrcTensorView const& tensor) {
    Base::operator+=(tensor);
    sync_device();
    return *this;
  }

  /// Subtract in place
  template <typename SrcTensorView>
  HostTensor& operator-=(SrcTensorView const& tensor) {
    Base::operator-=(tensor);
    sync_device();
    return *this;
  }

  /// Multiply in place
  template <typename SrcTensorView>
  HostTensor& operator*=(SrcTensorView const& tensor) {
    Base::operator*=(tensor);
    sync_device();
    return *this;
  }

  /// Divide in place
  template <typename SrcTensorView>
  HostTensor& operator/=(SrcTensorView const& tensor) {
    Base::operator/=(tensor);
    sync_device();
    return *this;
  }

  /// Fills with random data
  template <typename Gen>
  void fill_random(Gen generator) {
    Base::fill_random(generator);
    sync_device();
  }

  /// Procedurally assigns elements
  template <typename Gen>
  void generate(Gen generator) {
    Base::generate(generator);
    sync_device();
  }

  /// Procedurally visits elements
  template <typename Gen>
  void visit(Gen& generator) const {
    Base::visit(generator);
  }

  /// initializes with identity
  void fill_identity() {
    Base::fill_identity();
    sync_device();
  }

  /// computes elements as a linear combination of their coordinates
  void fill_linear(TensorCoord v, HostType offset = HostType(0)) {
    Base::fill_linear(v, offset);
    sync_device();
  }

  /// computes elements as a linear combination of their coordinates
  void fill_sequential(HostType v = HostType(1), HostType offset = HostType(0)) {
    Base::fill_sequential(v, offset);
    sync_device();
  }

  /// fills with a value
  void fill(HostType val = HostType(0)) {
    Base::fill(val);
    sync_device();
  }

  /// copies from external data source and performs type conversion
  template <
    typename SrcType,
    typename SrcMapFunc_,
    int SrcStorageRank_,
    typename SrcIndex_,
    typename SrcLongIndex_
  >
  void fill(
    TensorView<SrcType, Base::kRank, SrcMapFunc_, SrcStorageRank_, SrcIndex_, SrcLongIndex_> const& tensor) {
    Base::fill(tensor);
    sync_device();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
