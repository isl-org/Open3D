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

#pragma once

#include <initializer_list>
// #include <iostream>
#include <memory>
#include <vector>
#include "Mat.h"

#ifdef OPEN3D_USE_CUDA
#include "Open3D/Utility/CUDA.cuh"
#endif

namespace open3d {

// data set shape
typedef struct _Shape {
    size_t rows;
    size_t cols;
} Shape;

// data set element type
enum class DataType { FP_64 = 0, FP_32, INT_32 };

class ComplexType {};

template <typename V, typename T>
struct Blob {
    typedef struct _Type : ComplexType {
        _Type() {}
        _Type(const int &num_elements,
              const cuda::DeviceID::Type &device_id = cuda::DeviceID::CPU)
            : num_elements(num_elements), device_id(device_id) {
            Initialize();
        }
        // copy constructor
        _Type(const _Type &t)
            : num_elements(t.num_elements), device_id(t.device_id) {
            Initialize();

            // copy host data
            if (OnCPU())
                memcpy(h_data.data(), t.h_data.data(), num_of_bytes());

            // copy device data
            if (OnGPU())
                cuda::CopyDev2DevMemory(t.d_data, d_data, num_of_Ts());
        }
        // move constructor
        _Type(_Type &&t)
            : num_elements(t.num_elements),
              device_id(t.device_id),
              h_data(std::move(t.h_data)),
              d_data(t.d_data) {
            t.d_data = NULL;
            t.num_elements = 0;
            t.device_id = cuda::DeviceID::CPU;
        }
        // init constructor
        _Type(const std::vector<V> &v, const cuda::DeviceID::Type &device_id)
            : num_elements(v.size()), device_id(device_id) {
            Initialize();

            // init host data
            if (OnCPU())
                memcpy(h_data.data(), v.data(), num_of_bytes());

            // init device data
            if (OnGPU())
                cuda::CopyHst2DevMemory((const double *const)v.data(), d_data,
                                        num_of_Ts());
        }
        ~_Type() { Reset(); }

        // allocate memory
        void Initialize(const int &num_elements,
                        const cuda::DeviceID::Type &device_id) {
            this->num_elements = num_elements;
            this->device_id = device_id;

            if ((0 == h_data.size()) && (OnCPU()))
                h_data = std::vector<V>(num_elements);

            if ((NULL == d_data) && (OnGPU()))
                cuda::AllocateDeviceMemory(&d_data, num_of_Ts(), device_id);
        }
        void Initialize() { Initialize(num_elements, device_id); }
        // deallocate memory
        void Reset() {
            h_data.clear();
            cuda::ReleaseDeviceMemory(&d_data);
            num_elements = 0;
            device_id = cuda::DeviceID::CPU;
        }

        // total number of elements in this structure
        size_t num_elements{};
        // device id
        cuda::DeviceID::Type device_id = cuda::DeviceID::CPU;
        // host data container
        std::vector<V> h_data{};
        // device data pointer
        T *d_data{};

        // return true if the device_id includes the CPU.
        inline bool OnCPU() const {
            return cuda::DeviceID::CPU & device_id;
        }
        // return true if the device_id matches with one of the GPUs.
        inline bool OnGPU() const {
            return cuda::DeviceID::CPU != device_id;
        }

        // subscript operator: readwrite, host side only
        inline V &operator[](const uint &i) {
            if (OnCPU()) return h_data[i];
        }
        // subscript operator: readonly, host side only
        inline const V &operator[](const uint &i) const {
            if (OnCPU()) return h_data[i];
        }

        // compare contents for equality
        inline bool operator== (const _Type& r) {
            if (num_elements != r.num_elements)
                return false;

            // host-host
            if (OnCPU() && r.OnCPU())
                return h_data == r.h_data;

            // host-device
            if (OnCPU() && r.OnGPU())
                return false;

            // device-host
            if (OnGPU() && r.OnCPU())
                return false;

            // device-device
            if (OnGPU() && r.OnGPU())
                return false;

            return false;
        }
        // compare contents for inequality
        inline bool operator!= (const _Type& r) {
            return !(*this == r);
        }

        // copy from another Blob
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(const _Type &t) {
            Reset();

            num_elements = t.num_elements;
            device_id = t.device_id;

            Initialize();

            // copy host data
            if (OnCPU())
                memcpy(h_data.data(), t.h_data.data(), num_of_bytes());

            // copy device data
            if (OnGPU())
                cuda::CopyDev2DevMemory(t.d_data, d_data, num_of_Ts());

            return *this;
        }
        // move from another Blob
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(_Type &&t) {
            Reset();

            num_elements = t.num_elements;
            device_id = t.device_id;

            // move host data
            if (OnCPU()) {
                h_data = std::move(t.h_data);
            }

            // move device data
            if (OnGPU()) {
                d_data = t.d_data;
                t.d_data = NULL;
            }

            t.num_elements = 0;
            t.device_id = cuda::DeviceID::CPU;

            return *this;
        }
        // initialize with host data
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(const std::vector<V> &v) {
            cuda::DeviceID::Type bkp_device_id = device_id;

            Reset();

            num_elements = v.size();
            device_id = bkp_device_id;

            Initialize();

            // initialize host memory
            if (OnCPU())
                memcpy(h_data.data(), v.data(), num_of_bytes());

            // initialize device memory
            if (OnGPU())
                cuda::CopyHst2DevMemory((const T *const)v.data(), d_data,
                                        num_of_Ts());

            return *this;
        }
        // initialize from an initializer list
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(std::initializer_list<V> il) {
            cuda::DeviceID::Type bkp_device_id = device_id;

            Reset();

            std::vector<V> v(il);
            num_elements = v.size();
            device_id = bkp_device_id;

            Initialize();

            // initialize host memory
            if (OnCPU())
                memcpy(h_data.data(), v.data(), num_of_bytes());

            // initialize device memory
            if (OnGPU())
                cuda::CopyHst2DevMemory((const T *const)v.data(), d_data,
                                        num_of_Ts());

            return *this;
        }

        // redirect to std:vector<V>::data()
        inline V *data() noexcept {
            // host only
            if (OnCPU()) return h_data.data();

            return NULL;
        }
        // redirect to std:vector<V>::data()
        inline const V *data() const noexcept {
            // host only
            if (OnCPU()) return h_data.data();

            return NULL;
        }
        // redirect to std:vector<V>::begin()
        inline typename std::vector<V>::iterator begin() noexcept {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) output = h_data.begin();

            return output;
        }
        // redirect to std:vector<V>::begin()
        inline typename std::vector<V>::const_iterator begin() const noexcept {
            typename std::vector<V>::const_iterator output;

            // host only
            if (OnCPU()) output = h_data.begin();

            return output;
        }
        inline void clear() noexcept {
            // clear host memory
            // redirect to std:vector<V>::clear()
            if (OnCPU()) h_data.clear();

            // clear device memory
            if (OnGPU())
                cuda::ReleaseDeviceMemory(&d_data);

            num_elements = 0;
        }
        // redirect to std:vector<V>::end()
        inline typename std::vector<V>::iterator end() noexcept {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) output = h_data.end();

            return output;
        }
        // redirect to std:vector<V>::end()
        inline typename std::vector<V>::const_iterator end() const noexcept {
            typename std::vector<V>::const_iterator output;

            // host only
            if (OnCPU()) output = h_data.end();

            return output;
        }
        // redirect to std:vector<V>::empty()
        inline bool empty() const noexcept { return num_elements <= 0; }

        // TODO: insert works only on the host side for the moment.
        // Q: how to deal with device side?
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                const V &val) {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) {
                // redirect to std:vector<V>::insert(...)
                output = h_data.insert(position, val);

                num_elements = h_data.size();
            }

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                size_t n,
                const V &val) {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) {
                output = h_data.insert(position, n, val);

                num_elements = h_data.size();
            }

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        template <typename InputIterator>
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                InputIterator first,
                InputIterator last) {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) {
                output = h_data.insert(position, first, last);

                num_elements = h_data.size();
            }

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position, V &&val) {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) {
                output = h_data.insert(position, val);

                num_elements = h_data.size();
            }

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                std::initializer_list<V> il) {
            typename std::vector<V>::iterator output;

            // host only
            if (OnCPU()) {
                output = h_data.insert(position, il);

                num_elements = h_data.size();
            }

            return output;
        }
        inline void push_back(const V &val) {
            // host only
            if (OnCPU()) {
                // redirect to std:vector<V>::push_back(...)
                h_data.push_back(val);

                num_elements = h_data.size();
            }

            // device side
            // delete/reallocate device memory
            // Note: the overhead can be reduced at the cost of more complexity
            if (OnGPU()) {
                std::vector<V> data(num_elements);
                cuda::CopyDev2HstMemory(d_data, (T *const)data.data(),
                                        num_of_Ts());

                cuda::ReleaseDeviceMemory(&d_data);

                data.push_back(val);

                size_t new_size = data.size() * sizeof(V) / sizeof(T);
                cuda::AllocateDeviceMemory(&d_data, new_size, device_id);

                cuda::CopyHst2DevMemory((const T *const)data.data(), d_data,
                                        new_size);

                num_elements = data.size();
            }
        }
        inline void push_back(V &&val) { push_back(val); }
        // resize the memory allocated for storage.
        // this will actually resize both the host data and device data.
        // we could, in principle, avoid the resize depending on the usecase.
        // in the current mode memory is released/allocated on the spot.
        inline void resize(size_t n) {
            if (num_elements == n) return;

            // resize host data
            // redirect std:vector<V>::resize(...)
            if (OnCPU()) h_data.resize(n);

            // resize device data
            // delete/reallocate device memory
            // Note: the overhead can be reduced at the cost of more complexity
            if (OnGPU()) {
                std::vector<V> data(num_elements);
                cuda::CopyDev2HstMemory(d_data, (T *const)data.data(),
                                        num_of_Ts());

                cuda::ReleaseDeviceMemory(&d_data);

                data.resize(n);

                size_t new_size = n * sizeof(V) / sizeof(T);
                cuda::AllocateDeviceMemory(&d_data, new_size, device_id);

                cuda::CopyHst2DevMemory((const T *const)data.data(), d_data,
                                        new_size);
            }

            num_elements = n;
        }
        // redirect to std:vector<V>::resize(...)
        inline void resize(size_t n, const V &val) {
            if (num_elements == n) return;

            // resize host data
            // redirect std:vector<V>::resize(...)
            if (OnCPU()) h_data.resize(n, val);

            // resize device data
            // delete/reallocate device memory
            // Note: the overhead can be reduced at the cost of more complexity
            if (OnGPU()) {
                std::vector<V> data(num_elements);
                cuda::CopyDev2HstMemory(d_data, (T *const)data.data(),
                                        num_of_Ts());

                cuda::ReleaseDeviceMemory(&d_data);

                data.resize(n, val);

                size_t new_size = n * sizeof(V) / sizeof(T);
                cuda::AllocateDeviceMemory(&d_data, new_size, device_id);

                cuda::CopyHst2DevMemory((const T *const)data.data(), d_data,
                                        new_size);
            }

            num_elements = n;
        }
        // number of elements
        inline size_t size() const { return num_elements; }

    private:
        // number of T elements
        inline size_t num_of_Ts() const {
            return num_elements * sizeof(V) / sizeof(T);
        }
        // number of bytes
        inline size_t num_of_bytes() const { return num_elements * sizeof(V); }
    } Type;
};

typedef Blob<Eigen::Vector2i, int>::Type Blob2i;
typedef Blob<Eigen::Vector3i, int>::Type Blob3i;
typedef Blob<Eigen::Vector4i, int>::Type Blob4i;
typedef Blob<Eigen::Vector2d, double>::Type Blob2d;
typedef Blob<Eigen::Vector3d, double>::Type Blob3d;

typedef Blob3d Points;
typedef Blob3d Normals;
typedef Blob3d Colors;
typedef Blob3d Vertices;
typedef Blob3d Vertex_normals;
typedef Blob3d Vertex_colors;
typedef Blob3d Triangle_normals;
typedef Blob2i Lines;
typedef Blob3i Voxels;
typedef Blob3i Triangles;
typedef Blob2i CorrespondenceSet;

class Tensor {
public:
    // Tensor(const Shape& shape, const DataType& type);
    static std::shared_ptr<ComplexType> create(
            const Shape &shape,
            const DataType &type,
            const cuda::DeviceID::Type &device_id = cuda::DeviceID::CPU) {
        if (shape.cols == 3 && type == DataType::FP_64)
            return std::make_shared<Blob3d>(shape.rows, device_id);

        if (shape.cols == 3 && type == DataType::INT_32)
            return std::make_shared<Blob3i>(shape.rows, device_id);

        if (shape.cols == 2 && type == DataType::INT_32)
            return std::make_shared<Blob2i>(shape.rows, device_id);

        return NULL;
    };
};
}  // namespace open3d
