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

namespace open3d {
// 1D tensor, row major
template <typename V, typename T>
struct Blob {
    typedef struct _Type {
        // host data container
        std::vector<V> h_data{};
        // device data pointer
        T *d_data{};
        // device id
        // set to -1 to execute on the CPU
        int cuda_device_id = 0;

        // subscript operator: readwrite, host side only
        inline V &operator[](const uint &i) { return h_data[i]; }
        // subscript operator: readonly, host side only
        inline const V &operator[](const uint &i) const { return h_data[i]; }

        // forward the call to std:vector<V>::data()
        inline V *data() noexcept { return h_data.data(); }
        // forward the call to std:vector<V>::data()
        inline const V *data() const noexcept { return h_data.data(); }
        // forward the call to std:vector<V>::begin()
        inline typename std::vector<V>::iterator begin() noexcept {
            return h_data.begin();
        }
        // forward the call to std:vector<V>::begin()
        inline typename std::vector<V>::const_iterator begin() const noexcept {
            return h_data.begin();
        }
        // forward the call to std:vector<V>::clear()
        inline void clear() noexcept { h_data.clear(); }
        // forward the call to std:vector<V>::end()
        inline typename std::vector<V>::iterator end() noexcept {
            return h_data.end();
        }
        // forward the call to std:vector<V>::end()
        inline typename std::vector<V>::const_iterator end() const noexcept {
            return h_data.end();
        }
        // forward the call to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                const V &val) {
            h_data.insert(position, val);
        }
        // forward the call to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                size_t n,
                const V &val) {
            h_data.insert(position, n, val);
        }
        // forward the call to std:vector<V>::insert(...)
        template <typename InputIterator>
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position,
                InputIterator first,
                InputIterator last) {
            h_data.insert(position, first, last);
        }
        // forward the call to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert(
                typename std::vector<V>::const_iterator position, V &&val) {
            h_data.insert(position, val);
        }
        // forward the call to std:vector<V>::insert(...)
        inline typename std::vector<V>::iterator insert (typename
        std::vector<V>::const_iterator position, std::initializer_list<V> il) {
            h_data.insert(position, il);
        }
        // forward the call to std:vector<V>::push_back(...)
        inline void push_back(const V &val) { h_data.push_back(val); }
        // forward the call to std:vector<V>::push_back(...)
        inline void push_back(V &&val) { h_data.push_back(val); }
        // forward the call to std:vector<V>::resize(...)
        inline void resize(size_t n) { h_data.resize(n); }
        // forward the call to std:vector<V>::resize(...)
        inline void resize(size_t n, const V &val) { h_data.resize(n, val); }
        // forward the call to std:vector<V>::size()
        inline size_t size() const { return h_data.size(); }
    } Type;
};

typedef Blob<Eigen::Vector2i, int>::Type Blob2i;
typedef Blob<Eigen::Vector3i, int>::Type Blob3i;
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
}  // namespace open3d
