// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Core/DLPack/DLPackConverter.h"

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/DLPack/dlpack.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {
namespace dlpack {

struct Open3DDLManagedTensor {
    Tensor o3d_tensor_;
    DLManagedTensor dl_managed_tensor_;
    static void Deleter(DLManagedTensor* arg) {
        delete static_cast<Open3DDLManagedTensor*>(arg->manager_ctx);
    }
};

DLManagedTensor* ToDLPack(const Tensor& t) {
    DLDeviceType dl_device_type;
    switch (t.GetDevice().GetType()) {
        case Device::DeviceType::CPU:
            dl_device_type = DLDeviceType::kDLCPU;
            break;
        case Device::DeviceType::CUDA:
            dl_device_type = DLDeviceType::kDLGPU;
            break;
        default:
            utility::LogError("ToDLPack: unsupported device type {}",
                              t.GetDevice().ToString());
    }

    DLContext dl_context;
    dl_context.device_type = dl_device_type;
    dl_context.device_id = t.GetDevice().GetID();

    DLDataType dl_data_type;
    switch (t.GetDtype()) {
        case Dtype::Float32:
            dl_data_type.code = DLDataTypeCode::kDLFloat;
            break;
        case Dtype::Float64:
            dl_data_type.code = DLDataTypeCode::kDLFloat;
            break;
        case Dtype::Int32:
            dl_data_type.code = DLDataTypeCode::kDLInt;
            break;
        case Dtype::Int64:
            dl_data_type.code = DLDataTypeCode::kDLInt;
            break;
        case Dtype::UInt8:
            dl_data_type.code = DLDataTypeCode::kDLUInt;
            break;
        default:
            utility::LogError("Unsupported data type");
    }
    dl_data_type.bits =
            static_cast<uint8_t>(DtypeUtil::ByteSize(t.GetDtype()) * 8);
    dl_data_type.lanes = 1;

    Open3DDLManagedTensor* o3d_dl_tensor(new Open3DDLManagedTensor);
    o3d_dl_tensor->o3d_tensor_ = t;

    DLTensor dl_tensor;
    // Not Blob's data pointer.
    dl_tensor.data = const_cast<void*>(t.GetDataPtr());
    dl_tensor.ctx = dl_context;
    dl_tensor.ndim = static_cast<int>(t.GetShape().size());
    dl_tensor.dtype = dl_data_type;
    // The shape pointer is alive for the lifetime of Open3DDLManagedTensor.
    dl_tensor.shape = const_cast<int64_t*>(
            o3d_dl_tensor->o3d_tensor_.GetShapeRef().data());
    // The strides pointer is alive for the lifetime of Open3DDLManagedTensor.
    dl_tensor.strides = const_cast<int64_t*>(
            o3d_dl_tensor->o3d_tensor_.GetStridesRef().data());
    dl_tensor.byte_offset = 0;

    o3d_dl_tensor->dl_managed_tensor_.manager_ctx = o3d_dl_tensor;
    o3d_dl_tensor->dl_managed_tensor_.deleter = &Open3DDLManagedTensor::Deleter;
    o3d_dl_tensor->dl_managed_tensor_.dl_tensor = dl_tensor;

    return &(o3d_dl_tensor->dl_managed_tensor_);
}

Tensor FromDLPack(const DLManagedTensor* src) {
    Device device;
    switch (src->dl_tensor.ctx.device_type) {
        case DLDeviceType::kDLCPU:
            device = Device("CPU", src->dl_tensor.ctx.device_id);
            break;
        case DLDeviceType::kDLGPU:
            device = Device("CUDA", src->dl_tensor.ctx.device_id);
            break;
        default:
            utility::LogError("Unsupported device_type {}",
                              src->dl_tensor.ctx.device_type);
    }

    Dtype dtype;
    if (src->dl_tensor.dtype.lanes != 1) {
        utility::LogError("Only supports lanes == 1, but lanes == {}",
                          src->dl_tensor.dtype.lanes);
    }
    switch (src->dl_tensor.dtype.code) {
        case DLDataTypeCode::kDLUInt:
            switch (src->dl_tensor.dtype.bits) {
                case 8:
                    dtype = Dtype::UInt8;
                    break;
                default:
                    utility::LogError("Unsupported kDLUInt bits {}",
                                      src->dl_tensor.dtype.bits);
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (src->dl_tensor.dtype.bits) {
                case 32:
                    dtype = Dtype::Int32;
                    break;
                case 64:
                    dtype = Dtype::Int64;
                    break;
                default:
                    utility::LogError("Unsupported kDLInt bits {}",
                                      src->dl_tensor.dtype.bits);
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (src->dl_tensor.dtype.bits) {
                case 32:
                    dtype = Dtype::Float32;
                    break;
                case 64:
                    dtype = Dtype::Float64;
                    break;
                default:
                    utility::LogError("Unsupported kDLFloat bits {}",
                                      src->dl_tensor.dtype.bits);
            }
            break;
        default:
            utility::LogError("Unsupported dtype code {}",
                              src->dl_tensor.dtype.code);
    }

    // Open3D Blob's expects an std::function<void(void*)> deleter.
    auto deleter = [src](void* dummy) -> void {
        if (src->deleter != nullptr) {
            src->deleter(const_cast<DLManagedTensor*>(src));
        }
    };

    SizeVector shape(src->dl_tensor.shape,
                     src->dl_tensor.shape + src->dl_tensor.ndim);

    SizeVector strides;
    if (src->dl_tensor.strides == nullptr) {
        strides = Tensor::DefaultStrides(shape);
    } else {
        strides = SizeVector(src->dl_tensor.strides,
                             src->dl_tensor.strides + src->dl_tensor.ndim);
    }

    auto blob = std::make_shared<Blob>(device, src->dl_tensor.data, deleter);

    // src->dl_tensor.byte_offset is ignored in PyTorch and MXNet, but
    // according to dlpack.h, we added the offset here.
    return Tensor(shape, strides,
                  reinterpret_cast<char*>(blob->GetDataPtr()) +
                          src->dl_tensor.byte_offset,
                  dtype, blob);
}

}  // namespace dlpack
}  // namespace open3d
