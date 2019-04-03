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
#include <vector>
#include "Mat.h"

#ifdef OPEN3D_USE_CUDA
#include "Open3D/Utility/CUDA.cuh"
#endif

namespace open3d {
// 1D tensor, row major
template <typename V, typename T>
struct Blob {
    typedef struct _Type {
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
            if (cuda::DeviceID::CPU & device_id)
                memcpy(h_data.data(), t.h_data.data(), num_of_bytes());

            // copy device data
            if (cuda::DeviceID::CPU != device_id)
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
            if (cuda::DeviceID::CPU & device_id)
                memcpy(h_data.data(), v.data(), num_of_bytes());

            // init device data
            if (cuda::DeviceID::CPU != device_id)
                cuda::CopyHst2DevMemory((const double *const)v.data(), d_data,
                                        num_of_Ts());
        }
        ~_Type() { Reset(); }

        // allocate memory
        void Initialize(const int &num_elements,
                        const cuda::DeviceID::Type &device_id) {
            if ((0 == h_data.size()) && (cuda::DeviceID::CPU & device_id))
                h_data = std::vector<V>(num_elements);

            if ((NULL == d_data) && (cuda::DeviceID::CPU != device_id))
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

        inline int GPU_ID() { return cuda::DeviceID::GPU_ID(device_id); }

        // subscript operator: readwrite, host side only
        inline V &operator[](const uint &i) { return h_data[i]; }
        // subscript operator: readonly, host side only
        inline const V &operator[](const uint &i) const { return h_data[i]; }

        // copy from another Blob
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(const _Type &t) {
            Reset();

            num_elements = t.num_elements;
            device_id = t.device_id;

            Initialize();

            // copy host data
            if (cuda::DeviceID::CPU & device_id)
                memcpy(h_data.data(), t.h_data.data(), num_of_bytes());

            // copy device data
            if (cuda::DeviceID::CPU != device_id)
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
            if (cuda::DeviceID::CPU & device_id) {
                h_data = std::move(t.h_data);
            }

            // move device data
            if (cuda::DeviceID::CPU != device_id) {
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
            if (cuda::DeviceID::CPU & device_id)
                memcpy(h_data.data(), v.data(), num_of_bytes());

            // initialize device memory
            if (cuda::DeviceID::CPU != device_id)
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
            if (cuda::DeviceID::CPU & device_id)
                memcpy(h_data.data(), v.data(), num_of_bytes());

            // initialize device memory
            if (cuda::DeviceID::CPU != device_id)
                cuda::CopyHst2DevMemory((const T *const)v.data(), d_data,
                                        num_of_Ts());

            return *this;
        }

        // redirect to std:vector<V>::data()
        inline V *data() noexcept {
            // host only
            if (cuda::DeviceID::CPU & device_id) return h_data.data();

            return NULL;
        }
        // redirect to std:vector<V>::data()
        inline const V *data() const noexcept {
            // host only
            if (cuda::DeviceID::CPU & device_id) return h_data.data();

            return NULL;
        }
        // redirect to std:vector<V>::begin()
        inline typename std::vector<V>::iterator begin() noexcept {
            typename std::vector<V>::iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id) output = h_data.begin();

            return output;
        }
        // redirect to std:vector<V>::begin()
        inline typename std::vector<V>::const_iterator begin() const noexcept {
            typename std::vector<V>::const_iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id) output = h_data.begin();

            return output;
        }
        inline void clear() noexcept {
            // clear host memory
            // redirect to std:vector<V>::clear()
            if (cuda::DeviceID::CPU & device_id) h_data.clear();

            // clear device memory
            if (cuda::DeviceID::CPU != device_id)
                cuda::ReleaseDeviceMemory(&d_data);

            num_elements = 0;
        }
        // redirect to std:vector<V>::end()
        inline typename std::vector<V>::iterator end() noexcept {
            typename std::vector<V>::iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id) output = h_data.end();

            return output;
        }
        // redirect to std:vector<V>::end()
        inline typename std::vector<V>::const_iterator end() const noexcept {
            typename std::vector<V>::const_iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id) output = h_data.end();

            return output;
        }
        // redirect to std:vector<V>::empty()
        inline bool empty() const noexcept { return num_elements <= 0; }

        // TODO: insert works only on the host side for the moment.
        // Q: how to deal with device side?
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                const V &val) {
            typename std::vector<V>::iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id)
                output = h_data.insert(position, val);

            num_elements = h_data.size();

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                size_t n,
                const V &val) {
            typename std::vector<V>::iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id)
                output = h_data.insert(position, n, val);

            num_elements = h_data.size();

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
            if (cuda::DeviceID::CPU & device_id)
                output = h_data.insert(position, first, last);

            num_elements = h_data.size();

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position, V &&val) {
            typename std::vector<V>::iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id)
                output = h_data.insert(position, val);

            num_elements = h_data.size();

            return output;
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                std::initializer_list<V> il) {
            typename std::vector<V>::iterator output;

            // host only
            if (cuda::DeviceID::CPU & device_id)
                output = h_data.insert(position, il);

            num_elements = h_data.size();

            return output;
        }
        // redirect to std:vector<V>::push_back(...)
        inline void push_back(const V &val) {
            // host only
            if (cuda::DeviceID::CPU & device_id) h_data.push_back(val);

            num_elements = h_data.size();
        }
        // redirect to std:vector<V>::push_back(...)
        inline void push_back(V &&val) {
            // host only
            if (cuda::DeviceID::CPU & device_id) h_data.push_back(val);

            num_elements = h_data.size();
        }
        //
        inline void resize(size_t n) {
            if (num_elements == n) return;

            num_elements = n;

            // resize host data
            // redirect std:vector<V>::resize(...)
            if (cuda::DeviceID::CPU & device_id) h_data.resize(n);

            // resize device data
            // delete memory and reallocate
            // doesn't preserve memory
            // initialize with zeros
            if (cuda::DeviceID::CPU != device_id) {
                cuda::ReleaseDeviceMemory(&d_data);
                cuda::AllocateDeviceMemory(&d_data, num_of_Ts(), device_id);
            }
        }
        // redirect to std:vector<V>::resize(...)
        inline void resize(size_t n, const V &val) {
            if (num_elements == n) return;

            num_elements = n;

            // resize host data
            // redirect std:vector<V>::resize(...)
            if (cuda::DeviceID::CPU & device_id) h_data.resize(n, val);

            // resize device data
            // delete memory and reallocate
            // doesn't preserve existing data
            // doesn't init with val but with zeros
            if (cuda::DeviceID::CPU != device_id) {
                cuda::ReleaseDeviceMemory(&d_data);
                cuda::AllocateDeviceMemory(&d_data, num_of_Ts(), device_id);
            }
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
}  // namespace open3d
