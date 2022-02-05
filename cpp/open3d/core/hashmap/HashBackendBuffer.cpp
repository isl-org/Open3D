// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/core/hashmap/HashBackendBuffer.h"

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

HashBackendBuffer::HashBackendBuffer(int64_t capacity,
                                     int64_t key_dsize,
                                     std::vector<int64_t> value_dsizes,
                                     const Device &device) {
    // First compute common bytesize divisor for fast copying values.
    const std::vector<int64_t> kDivisors = {16, 12, 8, 4, 2, 1};

    for (const auto &divisor : kDivisors) {
        bool valid = true;
        blocks_per_element_.clear();
        for (size_t i = 0; i < value_dsizes.size(); ++i) {
            int64_t bytesize = value_dsizes[i];
            valid = valid && (bytesize % divisor == 0);
            blocks_per_element_.push_back(bytesize / divisor);
        }
        if (valid) {
            common_block_size_ = divisor;
            break;
        }
    }

    heap_ = Tensor({capacity}, core::UInt32, device);

    key_buffer_ = Tensor({capacity},
                         Dtype(Dtype::DtypeCode::Object, key_dsize, "_hash_k"),
                         device);

    value_buffers_.clear();
    for (size_t i = 0; i < value_dsizes.size(); ++i) {
        int64_t dsize_value = value_dsizes[i];
        Tensor value_buffer_i({capacity},
                              Dtype(Dtype::DtypeCode::Object, dsize_value,
                                    "_hash_v_" + std::to_string(i)),
                              device);
        value_buffers_.push_back(value_buffer_i);
    }

    // Heap top is device specific
    if (device.GetType() == Device::DeviceType::CUDA) {
        heap_top_.cuda = Tensor({1}, Dtype::Int32, device);
    }

    ResetHeap();
}

void HashBackendBuffer::ResetHeap() {
    Device device = GetDevice();

    Tensor heap = GetIndexHeap();
    if (device.GetType() == Device::DeviceType::CPU) {
        CPUResetHeap(heap);
        heap_top_.cpu = 0;
    } else if (device.GetType() == Device::DeviceType::CUDA) {
        CUDA_CALL(CUDAResetHeap, heap);
        heap_top_.cuda.Fill<int>(0);
    }
}

Device HashBackendBuffer::GetDevice() const { return heap_.GetDevice(); }

int64_t HashBackendBuffer::GetCapacity() const { return heap_.GetLength(); }

int64_t HashBackendBuffer::GetKeyDsize() const {
    return key_buffer_.GetDtype().ByteSize();
}

std::vector<int64_t> HashBackendBuffer::GetValueDsizes() const {
    std::vector<int64_t> value_dsizes;
    for (auto &value_buffer : value_buffers_) {
        value_dsizes.push_back(value_buffer.GetDtype().ByteSize());
    }
    return value_dsizes;
}

int64_t HashBackendBuffer::GetCommonBlockSize() const {
    return common_block_size_;
}

std::vector<int64_t> HashBackendBuffer::GetValueBlocksPerElement() const {
    return blocks_per_element_;
}

Tensor HashBackendBuffer::GetIndexHeap() const { return heap_; }

HashBackendBuffer::HeapTop &HashBackendBuffer::GetHeapTop() {
    return heap_top_;
}

int HashBackendBuffer::GetHeapTopIndex() const {
    if (heap_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        return heap_top_.cuda[0].Item<int>();
    }
    return heap_top_.cpu.load();
}

Tensor HashBackendBuffer::GetKeyBuffer() const { return key_buffer_; }

std::vector<Tensor> HashBackendBuffer::GetValueBuffers() const {
    return value_buffers_;
}

Tensor HashBackendBuffer::GetValueBuffer(size_t i) const {
    if (i >= value_buffers_.size()) {
        utility::LogError("Value buffer index out-of-bound ({} >= {}).", i,
                          value_buffers_.size());
    }
    return value_buffers_[i];
}

}  // namespace core
}  // namespace open3d
