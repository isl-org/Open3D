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
              const DeviceID::Type &device_id = DeviceID::CPU)
            : num_elements(num_elements), device_id(device_id) {
            Initialize();
        }
        // copy constructor
        // note: how to deal with device data?
        _Type(const _Type &t)
            : h_data(t.h_data),
              num_elements(t.num_elements),
              device_id(t.device_id) {
            Initialize();
        }
        ~_Type() { Reset(); }

        // allocate memory
        void Initialize(const int &num_elements,
                        const DeviceID::Type &device_id) {
            if ((0 == h_data.size()) && (DeviceID::CPU & device_id))
                h_data = std::vector<V>(num_elements);

            if (NULL != d_data)
                AllocateDeviceMemory(&d_data, num_elements * sizeof(V),
                                     device_id);
        }
        void Initialize() { Initialize(num_elements, device_id); }
        // deallocate memory
        void Reset() {
            h_data.clear();
            ReleaseDeviceMemory(&d_data);
            num_elements = 0;
            device_id = DeviceID::CPU;
        }

        // total number of elements in this structure
        size_t num_elements{};
        // device id
        DeviceID::Type device_id = DeviceID::CPU;
        // host data container
        std::vector<V> h_data{};
        // device data pointer
        T *d_data{};

        inline int GPU_ID() { return DeviceID::GPU_ID(device_id); }

        // subscript operator: readwrite, host side only
        inline V &operator[](const uint &i) { return h_data[i]; }
        // subscript operator: readonly, host side only
        inline const V &operator[](const uint &i) const { return h_data[i]; }

        // initialize with host data
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(const std::vector<V> &v) {
            DeviceID::Type bkp_device_id = device_id;

            Reset();

            num_elements = v.size();
            device_id = bkp_device_id;

            Initialize();

            // initialize host memory
            if (DeviceID::CPU == device_id)
                memcpy(h_data.data(), v.data(), v.size() * sizeof(T));

            // initialize device memory
            if (DeviceID::CPU != device_id)
                CopyHst2DevMemory((const T *const)v.data(), d_data,
                                  num_elements * sizeof(V));

            return *this;
        }
        // initialize from another Blob
        // reset pointers, reinitialize and copy the data to hst/dev pointers
        inline _Type &operator=(const _Type &t) {
            Reset();

            num_elements = t.num_elements;
            device_id = t.device_id;

            Initialize();

            // copy host data
            if (DeviceID::CPU == device_id)
                memcpy(h_data.data(), t.h_data.data(),
                       t.h_data.size() * sizeof(T));

            // copy device data
            if (DeviceID::CPU != device_id)
                CopyDev2DevMemory(d_data, t.d_data, num_elements);

            return *this;
        }
        // redirect to std:vector<V>::operator=(...)
        // note: how to deal with device data?
        inline _Type &operator=(_Type &&t) {
            Reset();

            num_elements = t.num_elements;
            device_id = t.devide_id;

            Initialize();

            // copy host data
            if (DeviceID::CPU == device_id)
                memcpy(h_data.data(), t.h_data.data(),
                       t.h_data.size() * sizeof(T));

            // copy device data
            if (DeviceID::CPU != device_id)
                CopyDev2DevMemory(d_data, t.d_data, num_elements);

            return *this;
        }
        // redirect to std:vector<V>::operator=(...)
        inline _Type &operator=(std::initializer_list<V> il) {
            h_data = il;

            return *this;
        }

        // redirect to std:vector<V>::data()
        inline V *data() noexcept { return h_data.data(); }
        // redirect to std:vector<V>::data()
        inline const V *data() const noexcept { return h_data.data(); }
        // redirect to std:vector<V>::begin()
        inline typename std::vector<V>::iterator begin() noexcept {
            return h_data.begin();
        }
        // redirect to std:vector<V>::begin()
        inline typename std::vector<V>::const_iterator begin() const noexcept {
            return h_data.begin();
        }
        // redirect to std:vector<V>::clear()
        inline void clear() noexcept { h_data.clear(); }
        // redirect to std:vector<V>::end()
        inline typename std::vector<V>::iterator end() noexcept {
            return h_data.end();
        }
        // redirect to std:vector<V>::end()
        inline typename std::vector<V>::const_iterator end() const noexcept {
            return h_data.end();
        }
        // redirect to std:vector<V>::empty()
        inline bool empty() const noexcept { return h_data.empty(); }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                const V &val) {
            h_data.insert(position, val);
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                size_t n,
                const V &val) {
            h_data.insert(position, n, val);
        }
        // redirect to std:vector<V>::insert(...)
        template <typename InputIterator>
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                InputIterator first,
                InputIterator last) {
            h_data.insert(position, first, last);
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position, V &&val) {
            h_data.insert(position, val);
        }
        // redirect to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                std::initializer_list<V> il) {
            h_data.insert(position, il);
        }
        // redirect to std:vector<V>::push_back(...)
        inline void push_back(const V &val) { h_data.push_back(val); }
        // redirect to std:vector<V>::push_back(...)
        inline void push_back(V &&val) { h_data.push_back(val); }
        // redirect to std:vector<V>::resize(...)
        inline void resize(size_t n) { h_data.resize(n); }
        // redirect to std:vector<V>::resize(...)
        inline void resize(size_t n, const V &val) { h_data.resize(n, val); }
        // redirect to std:vector<V>::size()
        inline size_t size() const { return h_data.size(); }
        inline size_t d_size() const { return num_elements; }
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
