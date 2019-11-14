/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

// -- [TODO] Make as many of the functions (related to the solver) const as possible.
// -- [TODO] Move the point interpolation constraint scaling by 1<<maxDepth
// -- [TODO] Add support for staggered-grid test functions
// -- [TODO] Store signatures with constraints/systems/restriction-prolongations
// -- [TODO] Make a virtual evaluation that only needs to know the degree
// -- [TODO] Modify (public) functions so that template parameters don't need to be passed when they
// are called
// -- [TODO] Confirm that whenever _isValidFEM*Node is called, the flags have already been set.
// -- [TODO] Make weight evaluation more efficient in _getSamplesPerNode by reducing the number of
// calls to getNeighbors

// -- [TODO] For point evaluation:
//        1. Have the evaluator store stencils for all depths [DONE]
//        2. When testing centers/corners, don't use generic evaluation

// -- [TODO] Support nested parallelism with thread pools

#ifndef FEM_TREE_INCLUDED
#define FEM_TREE_INCLUDED

#include <atomic>
#include <functional>
#include <limits>
#include <string>
#include "BSplineData.h"
#include "BlockedVector.h"
#include "Geometry.h"
#include "MyMiscellany.h"
#include "PointStream.h"
#include "RegularTree.h"
#include "SparseMatrix.h"

#ifdef BIG_DATA
// The integer type used for indexing the nodes in the octree
typedef long long node_index_type;
// The integer type used for indexing the entries of the matrix
typedef int matrix_index_type;
#else   // !BIG_DATA
typedef int node_index_type;
typedef int matrix_index_type;
#endif  // BIG_DATA
#ifdef USE_DEEP_TREE_NODES
// The integer type used for storing the depth and offset within an octree node
typedef unsigned int depth_and_offset_type;
#else   // !USE_DEEP_TREE_NODES
typedef unsigned short depth_and_offset_type;
#endif  // USE_DEEP_TREE_NODES

template <unsigned int Dim, class Real>
class FEMTree;

enum {
    SHOW_GLOBAL_RESIDUAL_NONE,
    SHOW_GLOBAL_RESIDUAL_LAST,
    SHOW_GLOBAL_RESIDUAL_ALL,
    SHOW_GLOBAL_RESIDUAL_COUNT
};
const char* ShowGlobalResidualNames[] = {"show none", "show last", "show all"};

class FEMTreeNodeData {
public:
    enum {
        SPACE_FLAG = 1,
        FEM_FLAG_1 = 2,
        FEM_FLAG_2 = 4,
        REFINABLE_FLAG = 8,
        GHOST_FLAG = 1 << 7
    };
    node_index_type nodeIndex;
    mutable char flags;
    void setGhostFlag(bool f) const {
        if (f)
            flags |= GHOST_FLAG;
        else
            flags &= ~GHOST_FLAG;
    }
    bool getGhostFlag(void) const { return (flags & GHOST_FLAG) != 0; }
    FEMTreeNodeData(void);
    ~FEMTreeNodeData(void);
};

template <unsigned int Dim>
class SortedTreeNodes {
    typedef RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type> TreeNode;

protected:
    Pointer(Pointer(node_index_type)) _sliceStart;
    int _levels;

public:
    Pointer(TreeNode*) treeNodes;
    node_index_type begin(int depth) const { return _sliceStart[depth][0]; }
    node_index_type end(int depth) const { return _sliceStart[depth][(size_t)1 << depth]; }
    node_index_type begin(int depth, int slice) const {
        return _sliceStart[depth][slice < 0 ? 0 : (slice > (1 << depth) ? (1 << depth) : slice)];
    }
    node_index_type end(int depth, int slice) const { return begin(depth, slice + 1); }
    size_t size(void) const { return _sliceStart[_levels - 1][(size_t)1 << (_levels - 1)]; }
    size_t size(int depth) const {
        if (depth < 0 || depth >= _levels) ERROR_OUT("bad depth: 0 <= ", depth, " < ", _levels);
        return _sliceStart[depth][(size_t)1 << depth] - _sliceStart[depth][0];
    }
    size_t size(int depth, int slice) const { return end(depth, slice) - begin(depth, slice); }
    int levels(void) const { return _levels; }

    SortedTreeNodes(void);
    ~SortedTreeNodes(void);
    void set(TreeNode& root, std::vector<node_index_type>* map);
    size_t set(TreeNode& root);
};

template <typename T>
struct DotFunctor {};
template <>
struct DotFunctor<float> {
    double operator()(float v1, float v2) { return v1 * v2; }
    unsigned int dimension(void) const { return 1; }
};
template <>
struct DotFunctor<double> {
    double operator()(double v1, double v2) { return v1 * v2; }
    unsigned int dimension(void) const { return 1; }
};
template <class Real, unsigned int Dim>
struct DotFunctor<Point<Real, Dim>> {
    double operator()(Point<Real, Dim> v1, Point<Real, Dim> v2) {
        return Point<Real, Dim>::Dot(v1, v2);
    }
    unsigned int dimension(void) const { return Dim; }
};

template <typename Pack>
struct SupportKey {};
template <unsigned int... Degrees>
struct SupportKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template NeighborKey<UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...>,
                                   UIntPack<BSplineSupportSizes<Degrees>::SupportEnd...>> {
    typedef UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...> LeftRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportEnd)...> RightRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportSize)...> Sizes;
};
template <typename Pack>
struct ConstSupportKey {};
template <unsigned int... Degrees>
struct ConstSupportKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template ConstNeighborKey<UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...>,
                                        UIntPack<BSplineSupportSizes<Degrees>::SupportEnd...>> {
    typedef UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...> LeftRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportEnd)...> RightRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportSize)...> Sizes;
};
template <typename Pack>
struct OverlapKey {};
template <unsigned int... Degrees>
struct OverlapKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template NeighborKey<
                      UIntPack<(-BSplineOverlapSizes<Degrees, Degrees>::OverlapStart)...>,
                      UIntPack<BSplineOverlapSizes<Degrees, Degrees>::OverlapEnd...>> {
    typedef UIntPack<(-BSplineOverlapSizes<Degrees, Degrees>::OverlapStart)...> LeftRadii;
    typedef UIntPack<(BSplineOverlapSizes<Degrees, Degrees>::OverlapEnd)...> RightRadii;
    typedef UIntPack<(BSplineOverlapSizes<Degrees, Degrees>::OverlapSize)...> Sizes;
};
template <typename Pack>
struct ConstOverlapKey {};
template <unsigned int... Degrees>
struct ConstOverlapKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template ConstNeighborKey<
                      UIntPack<(-BSplineOverlapSizes<Degrees, Degrees>::OverlapStart)...>,
                      UIntPack<BSplineOverlapSizes<Degrees, Degrees>::OverlapEnd...>> {
    typedef UIntPack<(-BSplineOverlapSizes<Degrees, Degrees>::OverlapStart)...> LeftRadii;
    typedef UIntPack<(BSplineOverlapSizes<Degrees, Degrees>::OverlapEnd)...> RightRadii;
    typedef UIntPack<(BSplineOverlapSizes<Degrees, Degrees>::OverlapSize)...> Sizes;
};

template <typename Pack>
struct PointSupportKey {};
template <unsigned int... Degrees>
struct PointSupportKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template NeighborKey<UIntPack<BSplineSupportSizes<Degrees>::SupportEnd...>,
                                   UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...>> {
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportEnd)...> LeftRadii;
    typedef UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...> RightRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportEnd -
                      BSplineSupportSizes<Degrees>::SupportStart + 1)...>
            Sizes;
};
template <typename Pack>
struct ConstPointSupportKey {};
template <unsigned int... Degrees>
struct ConstPointSupportKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template ConstNeighborKey<
                      UIntPack<BSplineSupportSizes<Degrees>::SupportEnd...>,
                      UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...>> {
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportEnd)...> LeftRadii;
    typedef UIntPack<(-BSplineSupportSizes<Degrees>::SupportStart)...> RightRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::SupportEnd -
                      BSplineSupportSizes<Degrees>::SupportStart + 1)...>
            Sizes;
};

template <typename Pack>
struct CornerSupportKey {};
template <unsigned int... Degrees>
struct CornerSupportKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template NeighborKey<UIntPack<BSplineSupportSizes<Degrees>::BCornerEnd...>,
                                   UIntPack<(-BSplineSupportSizes<Degrees>::BCornerStart + 1)...>> {
    typedef UIntPack<(BSplineSupportSizes<Degrees>::BCornerEnd)...> LeftRadii;
    typedef UIntPack<(-BSplineSupportSizes<Degrees>::BCornerStart + 1)...> RightRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::BCornerSize + 1)...> Sizes;
};
template <typename Pack>
struct ConstCornerSupportKey {};
template <unsigned int... Degrees>
struct ConstCornerSupportKey<UIntPack<Degrees...>>
    : public RegularTreeNode<sizeof...(Degrees), FEMTreeNodeData, depth_and_offset_type>::
              template ConstNeighborKey<
                      UIntPack<BSplineSupportSizes<Degrees>::BCornerEnd...>,
                      UIntPack<(-BSplineSupportSizes<Degrees>::BCornerStart + 1)...>> {
    typedef UIntPack<(BSplineSupportSizes<Degrees>::BCornerEnd)...> LeftRadii;
    typedef UIntPack<(-BSplineSupportSizes<Degrees>::BCornerStart + 1)...> RightRadii;
    typedef UIntPack<(BSplineSupportSizes<Degrees>::BCornerSize + 1)...> Sizes;
};

template <class Data, typename Pack>
struct _SparseOrDenseNodeData {};
template <class Data, unsigned int... FEMSigs>
struct _SparseOrDenseNodeData<Data, UIntPack<FEMSigs...>> {
    static const unsigned int Dim = sizeof...(FEMSigs);
    typedef UIntPack<FEMSigs...> FEMSignatures;
    typedef Data data_type;

    // Methods for accessing as an array
    virtual size_t size(void) const = 0;
    virtual const Data& operator[](size_t idx) const = 0;
    virtual Data& operator[](size_t idx) = 0;

    // Method for accessing (and inserting if necessary) using a node
    virtual Data& operator[](
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) = 0;
    // Methods for accessing using a node
    virtual Data* operator()(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) = 0;
    virtual const Data* operator()(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) const = 0;

    // Method for getting the actual index associated with a node
    virtual node_index_type index(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) const = 0;
};

template <class Data, typename Pack>
struct SparseNodeData {};
template <class Data, unsigned int... FEMSigs>
struct SparseNodeData<Data, UIntPack<FEMSigs...>>
    : public _SparseOrDenseNodeData<Data, UIntPack<FEMSigs...>> {
    static const unsigned int Dim = sizeof...(FEMSigs);

    size_t size(void) const { return _data.size(); }
    const Data& operator[](size_t idx) const { return _data[idx]; }
    Data& operator[](size_t idx) { return _data[idx]; }

    void reserve(size_t sz) {
        if (sz > _indices.size()) _indices.resize(sz, -1);
    }
    Data* operator()(const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) {
        return (node->nodeData.nodeIndex < 0 ||
                node->nodeData.nodeIndex >= (node_index_type)_indices.size() ||
                _indices[node->nodeData.nodeIndex] == -1)
                       ? NULL
                       : &_data[_indices[node->nodeData.nodeIndex]];
    }
    const Data* operator()(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) const {
        return (node->nodeData.nodeIndex < 0 ||
                node->nodeData.nodeIndex >= (node_index_type)_indices.size() ||
                _indices[node->nodeData.nodeIndex] == -1)
                       ? NULL
                       : &_data[_indices[node->nodeData.nodeIndex]];
    }
    Data& operator[](const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) {
        static std::mutex _insertionMutex;

        // If the node hasn't been indexed yet
        if (node->nodeData.nodeIndex >= (node_index_type)_indices.size()) {
            std::lock_guard<std::mutex> lock(_insertionMutex);
            if (node->nodeData.nodeIndex >= (node_index_type)_indices.size())
                _indices.resize(node->nodeData.nodeIndex + 1, -1);
        }

        // If the node hasn't been allocated yet
        volatile node_index_type& _index = _indices[node->nodeData.nodeIndex];
        if (_index == -1) {
            std::lock_guard<std::mutex> lock(_insertionMutex);
            if (_index == -1) _index = (node_index_type)_data.push();
        }
        return _data[_index];
    }
    node_index_type index(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) const {
        if (!node || node->nodeData.nodeIndex < 0 ||
            node->nodeData.nodeIndex >= (node_index_type)_indices.size())
            return -1;
        else
            return _indices[node->nodeData.nodeIndex];
    }

protected:
    template <unsigned int _Dim, class _Real>
    friend class FEMTree;
    // Map should be the size of the old number of entries and map[i] should give the new index of
    // the old i-th node
    void _remapIndices(const node_index_type* newNodeIndices, size_t newNodeCount) {
        BlockedVector<node_index_type> newIndices;
        newIndices.resize(newNodeCount);
        for (node_index_type i = 0; i < (node_index_type)newNodeCount; i++) newIndices[i] = -1;
        for (node_index_type i = 0; i < (node_index_type)_indices.size(); i++)
            if (newNodeIndices[i] != -1 && newNodeIndices[i] < (node_index_type)newNodeCount)
                newIndices[newNodeIndices[i]] = _indices[i];
        _indices = newIndices;
    }
    BlockedVector<node_index_type> _indices;
    BlockedVector<Data> _data;
};

template <class Data, typename Pack>
struct DenseNodeData {};
template <class Data, unsigned int... FEMSigs>
struct DenseNodeData<Data, UIntPack<FEMSigs...>>
    : public _SparseOrDenseNodeData<Data, UIntPack<FEMSigs...>> {
    static const unsigned int Dim = sizeof...(FEMSigs);
    DenseNodeData(void) {
        _data = NullPointer(Data);
        _sz = 0;
    }
    DenseNodeData(size_t sz) {
        _sz = sz;
        if (sz)
            _data = NewPointer<Data>(sz);
        else
            _data = NullPointer(Data);
    }
    DenseNodeData(const DenseNodeData& d) : DenseNodeData() {
        _resize(d._sz);
        if (_sz) memcpy(_data, d._data, sizeof(Data) * _sz);
    }
    DenseNodeData(DenseNodeData&& d) {
        _data = d._data, _sz = d._sz;
        d._data = NullPointer(Data), d._sz = 0;
    }
    DenseNodeData& operator=(const DenseNodeData& d) {
        _resize(d._sz);
        if (_sz) memcpy(_data, d._data, sizeof(Data) * _sz);
        return *this;
    }
    DenseNodeData& operator=(DenseNodeData&& d) {
        size_t __sz = _sz;
        Pointer(Data) __data = _data;
        _data = d._data, _sz = d._sz;
        d._data = __data, d._sz = __sz;
        return *this;
    }
    ~DenseNodeData(void) {
        DeletePointer(_data);
        _sz = 0;
    }
    static void WriteSignatures(FILE* fp) {
        unsigned int dim = sizeof...(FEMSigs);
        fwrite(&dim, sizeof(unsigned int), 1, fp);
        unsigned int femSigs[] = {FEMSigs...};
        fwrite(femSigs, sizeof(unsigned int), dim, fp);
    }
    void write(FILE* fp) const {
        fwrite(&_sz, sizeof(size_t), 1, fp);
        fwrite(_data, sizeof(Data), _sz, fp);
    }
    void read(FILE* fp) {
        if (fread(&_sz, sizeof(size_t), 1, fp) != 1) ERROR_OUT("Failed to read size");
        _data = NewPointer<Data>(_sz);
        if (fread(_data, sizeof(Data), _sz, fp) != _sz) ERROR_OUT("failed to read data");
    }

    Data& operator[](size_t idx) { return _data[idx]; }
    const Data& operator[](size_t idx) const { return _data[idx]; }
    size_t size(void) const { return _sz; }
    Data& operator[](const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) {
        return _data[node->nodeData.nodeIndex];
    }
    Data* operator()(const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) {
        return (node == NULL || node->nodeData.nodeIndex >= (node_index_type)_sz)
                       ? NULL
                       : &_data[node->nodeData.nodeIndex];
    }
    const Data* operator()(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) const {
        return (node == NULL || node->nodeData.nodeIndex >= (node_index_type)_sz)
                       ? NULL
                       : &_data[node->nodeData.nodeIndex];
    }
    node_index_type index(
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) const {
        return (!node || node->nodeData.nodeIndex < 0 ||
                node->nodeData.nodeIndex >= (node_index_type)_sz)
                       ? -1
                       : node->nodeData.nodeIndex;
    }
    Pointer(Data) operator()(void) { return _data; }
    ConstPointer(Data) operator()(void) const { return (ConstPointer(Data))_data; }

protected:
    template <unsigned int _Dim, class _Real>
    friend class FEMTree;
    // Map should be the size of the old number of entries and map[i] should give the new index of
    // the old i-th node
    void _remapIndices(const node_index_type* newNodeIndices, size_t newNodeCount) {
        Pointer(Data) newData = NewPointer<Data>(newNodeCount);
        memset(newData, 0, sizeof(Data) * newNodeCount);
        for (size_t i = 0; i < _sz; i++)
            if (newNodeIndices[i] >= 0 && newNodeIndices[i] < newNodeCount)
                newData[newNodeIndices[i]] = _data[i];
        DeletePointer(_data);
        _data = newData;
        _sz = newNodeCount;
    }
    size_t _sz;
    void _resize(size_t sz) {
        DeletePointer(_data);
        if (sz)
            _data = NewPointer<Data>(sz);
        else
            _data = NullPointer(Data);
        _sz = sz;
    }
    Pointer(Data) _data;
};
enum FEMTreeRealType { FEM_TREE_REAL_FLOAT, FEM_TREE_REAL_DOUBLE, FEM_TREE_REAL_COUNT };
const char* FEMTreeRealNames[] = {"float", "double"};

void ReadFEMTreeParameter(FILE* fp, FEMTreeRealType& realType, unsigned int& dimension) {
    if (fread(&realType, sizeof(FEMTreeRealType), 1, fp) != 1)
        ERROR_OUT("Failed to read real type");
    if (fread(&dimension, sizeof(unsigned int), 1, fp) != 1) ERROR_OUT("Failed to read dimension");
}

unsigned int* ReadDenseNodeDataSignatures(FILE* fp, unsigned int& dim) {
    if (fread(&dim, sizeof(unsigned int), 1, fp) != 1) ERROR_OUT("Failed to read dimension");
    unsigned int* femSigs = new unsigned int[dim];
    if (fread(femSigs, sizeof(unsigned int), dim, fp) != dim)
        ERROR_OUT("Failed to read signatures");
    return femSigs;
}

// The Derivative method needs static members:
//		Dim: the dimensionality of the space in which derivatives are evaluated
//		Size: the total number of derivatives
// and static methods:
//		Index: takes the number of partials along each dimension and returns the index
//		Factor: takes an index and sets the number of partials along each dimension

template <typename T>
struct TensorDerivatives {};
template <class Real, typename T>
struct TensorDerivativeValues {};

// Specify the derivatives for each dimension separately
template <unsigned int D, unsigned int... Ds>
struct TensorDerivatives<UIntPack<D, Ds...>> {
    typedef TensorDerivatives<UIntPack<Ds...>> _TensorDerivatives;
    static const unsigned int LastDerivative = UIntPack<D, Ds...>::template Get<sizeof...(Ds)>();
    static const unsigned int Dim = _TensorDerivatives::Dim + 1;
    static const unsigned int Size = _TensorDerivatives::Size * (D + 1);
    static void Factor(unsigned int idx, unsigned int derivatives[Dim]) {
        derivatives[0] = idx / _TensorDerivatives::Size;
        _TensorDerivatives::Factor(idx % _TensorDerivatives::Size, derivatives + 1);
    }
    static unsigned int Index(const unsigned int derivatives[Dim]) {
        return _TensorDerivatives::Index(derivatives + 1) +
               _TensorDerivatives::Size * derivatives[0];
    }
};
template <unsigned int D>
struct TensorDerivatives<UIntPack<D>> {
    static const unsigned int LastDerivative = D;
    static const unsigned int Dim = 1;
    static const unsigned int Size = D + 1;
    static void Factor(unsigned int idx, unsigned int derivatives[1]) { derivatives[0] = idx; }
    static unsigned int Index(const unsigned int derivatives[1]) { return derivatives[0]; }
};
template <class Real, unsigned int... Ds>
struct TensorDerivativeValues<Real, UIntPack<Ds...>>
    : public Point<Real, TensorDerivatives<UIntPack<Ds...>>::Size> {};

// Specify the sum of the derivatives
template <unsigned int Dim, unsigned int D>
struct CumulativeDerivatives {
    typedef CumulativeDerivatives<Dim, D - 1> _CumulativeDerivatives;
    static const unsigned int LastDerivative = D;
    static const unsigned int Size = _CumulativeDerivatives::Size * Dim + 1;
    static void Factor(unsigned int idx, unsigned int d[Dim]) {
        if (idx < _CumulativeDerivatives::Size)
            return _CumulativeDerivatives::Factor(idx, d);
        else
            _Factor(idx - _CumulativeDerivatives::Size, d);
    }
    static unsigned int Index(const unsigned int derivatives[Dim]) {
        unsigned int dCount = 0;
        for (unsigned int d = 0; d < Dim; d++) dCount += derivatives[d];
        if (dCount >= D)
            ERROR_OUT("More derivatives than allowed");
        else if (dCount < D)
            return _CumulativeDerivatives::Index(derivatives);
        else
            return _CumulativeDerivatives::Size + _Index(derivatives);
    }

protected:
    static const unsigned int _Size = _CumulativeDerivatives::_Size * Dim;
    static void _Factor(unsigned int idx, unsigned int d[Dim]) {
        _CumulativeDerivatives::_Factor(idx % _CumulativeDerivatives::_Size, d);
        d[idx / _CumulativeDerivatives::_Size]++;
    }
    static unsigned int _Index(const unsigned int d[Dim]) {
        unsigned int _d[Dim];
        memcpy(_d, d, sizeof(_d));
        for (unsigned int i = 0; i < Dim; i++)
            if (_d[i]) {
                _d[i]--;
                return _CumulativeDerivatives::Index(_d) * Dim + i;
            }
        ERROR_OUT("No derivatives specified");
        return -1;
    }
    friend CumulativeDerivatives<Dim, D + 1>;
};
template <unsigned int Dim>
struct CumulativeDerivatives<Dim, 0> {
    static const unsigned int LastDerivative = 0;
    static const unsigned int Size = 1;
    static void Factor(unsigned int idx, unsigned int d[Dim]) {
        memset(d, 0, sizeof(unsigned int) * Dim);
    }
    static unsigned int Index(const unsigned int derivatives[Dim]) { return 0; }

protected:
    static const unsigned int _Size = 1;
    static void _Factor(unsigned int idx, unsigned int d[Dim]) {
        memset(d, 0, sizeof(unsigned int) * Dim);
    }
    friend CumulativeDerivatives<Dim, 1>;
};
template <typename Real, unsigned int Dim, unsigned int D>
using CumulativeDerivativeValues = Point<Real, CumulativeDerivatives<Dim, D>::Size>;

template <unsigned int Dim, class Real, unsigned int D>
CumulativeDerivativeValues<Real, Dim, D> Evaluate(const double dValues[Dim][D + 1]) {
    CumulativeDerivativeValues<Real, Dim, D> v;
    unsigned int _d[Dim];
    for (unsigned int d = 0; d < CumulativeDerivatives<Dim, D>::Size; d++) {
        CumulativeDerivatives<Dim, D>::Factor(d, _d);
        double value = dValues[0][_d[0]];
        for (unsigned int dd = 1; dd < Dim; dd++) value *= dValues[dd][_d[dd]];
        v[d] = (Real)value;
    }
    return v;
}

template <unsigned int Dim, class Real, typename T, unsigned int D>
struct DualPointInfo {
    Point<Real, Dim> position;
    Real weight;
    CumulativeDerivativeValues<T, Dim, D> dualValues;
    DualPointInfo operator+(const DualPointInfo& p) const {
        return DualPointInfo(position + p.position, dualValues + p.dualValues, weight + p.weight);
    }
    DualPointInfo& operator+=(const DualPointInfo& p) {
        position += p.position;
        weight += p.weight, dualValues += p.dualValues;
        return *this;
    }
    DualPointInfo operator*(Real s) const {
        return DualPointInfo(position * s, weight * s, dualValues * s);
    }
    DualPointInfo& operator*=(Real s) {
        position *= s, weight *= s, dualValues *= s;
        return *this;
    }
    DualPointInfo operator/(Real s) const {
        return DualPointInfo(position / s, weight / s, dualValues / s);
    }
    DualPointInfo& operator/=(Real s) {
        position /= s, weight /= s, dualValues /= s;
        return *this;
    }
    DualPointInfo(void) : weight(0) {}
    DualPointInfo(Point<Real, Dim> p, CumulativeDerivativeValues<T, Dim, D> c, Real w) {
        position = p, dualValues = c, weight = w;
    }
};
template <unsigned int Dim, class Real, typename Data, typename T, unsigned int D>
struct DualPointAndDataInfo {
    DualPointInfo<Dim, Real, T, D> pointInfo;
    Data data;
    DualPointAndDataInfo operator+(const DualPointAndDataInfo& p) const {
        return DualPointAndDataInfo(pointInfo + p.pointInfo, data + p.data);
    }
    DualPointAndDataInfo operator*(Real s) const {
        return DualPointAndDataInfo(pointInfo * s, data * s);
    }
    DualPointAndDataInfo operator/(Real s) const {
        return DualPointAndDataInfo(pointInfo / s, data / s);
    }
    DualPointAndDataInfo& operator+=(const DualPointAndDataInfo& p) {
        pointInfo += p.pointInfo;
        data += p.data;
        return *this;
    }
    DualPointAndDataInfo& operator*=(Real s) {
        pointInfo *= s, data *= s;
        return *this;
    }
    DualPointAndDataInfo& operator/=(Real s) {
        pointInfo /= s, data /= s;
        return *this;
    }
    DualPointAndDataInfo(void) {}
    DualPointAndDataInfo(DualPointInfo<Dim, Real, T, D> p, Data d) { pointInfo = p, data = d; }
};
template <unsigned int Dim, class Real, typename T, unsigned int D>
struct DualPointInfoBrood {
    DualPointInfo<Dim, Real, T, D>& operator[](size_t idx) { return _dpInfo[idx]; }
    const DualPointInfo<Dim, Real, T, D>& operator[](size_t idx) const { return _dpInfo[idx]; }
    void finalize(void) {
        _size = 0;
        for (unsigned int i = 0; i < (1 << Dim); i++)
            if (_dpInfo[i].weight > 0) _dpInfo[_size++] = _dpInfo[i];
    }
    unsigned int size(void) const { return _size; }

    DualPointInfoBrood operator+(const DualPointInfoBrood& p) const {
        DualPointInfoBrood d;
        for (unsigned int i = 0; i < (1 << Dim); i++) d._dpInfo[i] = _dpInfo[i] + p._dpInfo[i];
        return d;
    }
    DualPointInfoBrood operator*(Real s) const {
        DualPointInfoBrood d;
        for (unsigned int i = 0; i < (1 << Dim); i++) d._dpInfo[i] = _dpInfo[i] * s;
        return d;
    }
    DualPointInfoBrood operator/(Real s) const {
        DualPointInfoBrood d;
        for (unsigned int i = 0; i < (1 << Dim); i++) d._dpInfo[i] = _dpInfo[i] / s;
        return d;
    }
    DualPointInfoBrood& operator+=(const DualPointInfoBrood& p) {
        for (unsigned int i = 0; i < (1 << Dim); i++) _dpInfo[i] += p._dpInfo[i];
        return *this;
    }
    DualPointInfoBrood& operator*=(Real s) {
        for (unsigned int i = 0; i < (1 << Dim); i++) _dpInfo[i] *= s;
        return *this;
    }
    DualPointInfoBrood& operator/=(Real s) {
        for (unsigned int i = 0; i < (1 << Dim); i++) _dpInfo[i] /= s;
        return *this;
    }

protected:
    DualPointInfo<Dim, Real, T, D> _dpInfo[1 << Dim];
    unsigned int _size;
};
template <unsigned int Dim, class Real, typename Data, typename T, unsigned int D>
struct DualPointAndDataInfoBrood {
    DualPointAndDataInfo<Dim, Real, Data, T, D>& operator[](size_t idx) { return _dpInfo[idx]; }
    const DualPointAndDataInfo<Dim, Real, Data, T, D>& operator[](size_t idx) const {
        return _dpInfo[idx];
    }
    void finalize(void) {
        _size = 0;
        for (unsigned int i = 0; i < (1 << Dim); i++)
            if (_dpInfo[i].pointInfo.weight > 0) _dpInfo[_size++] = _dpInfo[i];
    }
    unsigned int size(void) const { return _size; }

    DualPointAndDataInfoBrood operator+(const DualPointAndDataInfoBrood& p) const {
        DualPointAndDataInfoBrood d;
        for (unsigned int i = 0; i < (1 << Dim); i++) d._dpInfo[i] = _dpInfo[i] + p._dpInfo[i];
        return d;
    }
    DualPointAndDataInfoBrood operator*(Real s) const {
        DualPointAndDataInfoBrood d;
        for (unsigned int i = 0; i < (1 << Dim); i++) d._dpInfo[i] = _dpInfo[i] * s;
        return d;
    }
    DualPointAndDataInfoBrood operator/(Real s) const {
        DualPointAndDataInfoBrood d;
        for (unsigned int i = 0; i < (1 << Dim); i++) d._dpInfo[i] = _dpInfo[i] / s;
        return d;
    }
    DualPointAndDataInfoBrood& operator+=(const DualPointAndDataInfoBrood& p) {
        for (unsigned int i = 0; i < (1 << Dim); i++) _dpInfo[i] += p._dpInfo[i];
        return *this;
    }
    DualPointAndDataInfoBrood& operator*=(Real s) {
        for (unsigned int i = 0; i < (1 << Dim); i++) _dpInfo[i] *= s;
        return *this;
    }
    DualPointAndDataInfoBrood& operator/=(Real s) {
        for (unsigned int i = 0; i < (1 << Dim); i++) _dpInfo[i] /= s;
        return *this;
    }

protected:
    DualPointAndDataInfo<Dim, Real, Data, T, D> _dpInfo[1 << Dim];
    unsigned int _size;
};

////////////////////////////
// The virtual integrator //
////////////////////////////
struct BaseFEMIntegrator {
    template <typename TDegreePack>
    struct System {};
    template <typename TDegreePack>
    struct RestrictionProlongation {};
    template <typename TDegreePack, typename CDegreePack, unsigned int CDim>
    struct Constraint {};
    template <typename TDegreePack>
    struct SystemConstraint {};
    template <typename TDegreePack>
    struct PointEvaluator {};

protected:
    template <unsigned int Degree, unsigned int... Degrees>
    static typename std::enable_if<sizeof...(Degrees) == 0, bool>::type _IsInteriorlySupported(
            UIntPack<Degree, Degrees...>, unsigned int depth, const int off[]) {
        int begin, end;
        BSplineSupportSizes<Degree>::InteriorSupportedSpan(depth, begin, end);
        return off[0] >= begin && off[0] < end;
    }
    template <unsigned int Degree, unsigned int... Degrees>
    static typename std::enable_if<sizeof...(Degrees) != 0, bool>::type _IsInteriorlySupported(
            UIntPack<Degree, Degrees...>, unsigned int depth, const int off[]) {
        int begin, end;
        BSplineSupportSizes<Degree>::InteriorSupportedSpan(depth, begin, end);
        return (off[0] >= begin && off[0] < end) &&
               _IsInteriorlySupported(UIntPack<Degrees...>(), depth, off + 1);
    }
    template <unsigned int Degree, unsigned int... Degrees>
    static typename std::enable_if<sizeof...(Degrees) == 0, bool>::type _IsInteriorlySupported(
            UIntPack<Degree, Degrees...>,
            unsigned int depth,
            const int off[],
            const double begin[],
            const double end[]) {
        int res = 1 << depth;
        double b = (0. + off[0] + BSplineSupportSizes<Degree>::SupportStart) / res;
        double e = (1. + off[0] + BSplineSupportSizes<Degree>::SupportEnd) / res;
        return b >= begin[0] && e <= end[0];
    }
    template <unsigned int Degree, unsigned int... Degrees>
    static typename std::enable_if<sizeof...(Degrees) != 0, bool>::type _IsInteriorlySupported(
            UIntPack<Degree, Degrees...>,
            unsigned int depth,
            const int off[],
            const double begin[],
            const double end[]) {
        int res = 1 << depth;
        double b = (0. + off[0] + BSplineSupportSizes<Degree>::SupportStart) / res;
        double e = (1. + off[0] + BSplineSupportSizes<Degree>::SupportEnd) / res;
        return b >= begin[0] && e <= end[0] &&
               _IsInteriorlySupported(UIntPack<Degrees...>(), depth, off + 1, begin + 1, end + 1);
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) == 0>::type _InteriorOverlappedSpan(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            int depth,
            int begin[],
            int end[]) {
        BSplineIntegrationData<FEMDegreeAndBType<Degree1, BOUNDARY_NEUMANN>::Signature,
                               FEMDegreeAndBType<Degree2, BOUNDARY_NEUMANN>::Signature>::
                InteriorOverlappedSpan(depth, begin[0], end[0]);
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) != 0>::type _InteriorOverlappedSpan(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            int depth,
            int begin[],
            int end[]) {
        BSplineIntegrationData<FEMDegreeAndBType<Degree1, BOUNDARY_NEUMANN>::Signature,
                               FEMDegreeAndBType<Degree2, BOUNDARY_NEUMANN>::Signature>::
                InteriorOverlappedSpan(depth, begin[0], end[0]);
        _InteriorOverlappedSpan(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), depth, begin + 1,
                                end + 1);
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) == 0, bool>::type _IsInteriorlyOverlapped(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            unsigned int depth,
            const int off[]) {
        int begin, end;
        BSplineIntegrationData<FEMDegreeAndBType<Degree1, BOUNDARY_NEUMANN>::Signature,
                               FEMDegreeAndBType<Degree2, BOUNDARY_NEUMANN>::Signature>::
                InteriorOverlappedSpan(depth, begin, end);
        return off[0] >= begin && off[0] < end;
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) != 0, bool>::type _IsInteriorlyOverlapped(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            unsigned int depth,
            const int off[]) {
        int begin, end;
        BSplineIntegrationData<FEMDegreeAndBType<Degree1, BOUNDARY_NEUMANN>::Signature,
                               FEMDegreeAndBType<Degree2, BOUNDARY_NEUMANN>::Signature>::
                InteriorOverlappedSpan(depth, begin, end);
        return (off[0] >= begin && off[0] < end) &&
               _IsInteriorlyOverlapped(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), depth,
                                       off + 1);
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) == 0>::type _ParentOverlapBounds(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            unsigned int depth,
            const int off[],
            int start[],
            int end[]) {
        const int OverlapStart = BSplineOverlapSizes<Degree1, Degree2>::OverlapStart;
        start[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapStart[off[0] & 1] -
                   OverlapStart;
        end[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapEnd[off[0] & 1] -
                 OverlapStart + 1;
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) != 0>::type _ParentOverlapBounds(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            unsigned int depth,
            const int off[],
            int start[],
            int end[]) {
        const int OverlapStart = BSplineOverlapSizes<Degree1, Degree2>::OverlapStart;
        start[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapStart[off[0] & 1] -
                   OverlapStart;
        end[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapEnd[off[0] & 1] -
                 OverlapStart + 1;
        _ParentOverlapBounds(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), depth, off + 1,
                             start + 1, end + 1);
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) == 0>::type _ParentOverlapBounds(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            int corner,
            int start[],
            int end[]) {
        const int OverlapStart = BSplineOverlapSizes<Degree1, Degree2>::OverlapStart;
        start[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapStart[corner & 1] -
                   OverlapStart;
        end[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapEnd[corner & 1] -
                 OverlapStart + 1;
    }
    template <unsigned int Degree1,
              unsigned int... Degrees1,
              unsigned int Degree2,
              unsigned int... Degrees2>
    static typename std::enable_if<sizeof...(Degrees1) != 0>::type _ParentOverlapBounds(
            UIntPack<Degree1, Degrees1...>,
            UIntPack<Degree2, Degrees2...>,
            int corner,
            int start[],
            int end[]) {
        const int OverlapStart = BSplineOverlapSizes<Degree1, Degree2>::OverlapStart;
        start[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapStart[corner & 1] -
                   OverlapStart;
        end[0] = BSplineOverlapSizes<Degree1, Degree2>::ParentOverlapEnd[corner & 1] -
                 OverlapStart + 1;
        _ParentOverlapBounds(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), corner >> 1,
                             start + 1, end + 1);
    }

public:
    template <unsigned int... Degrees>
    static bool IsInteriorlySupported(UIntPack<Degrees...>, int depth, const int offset[]) {
        return depth >= 0 && _IsInteriorlySupported(UIntPack<Degrees...>(), depth, offset);
    }
    template <unsigned int... Degrees>
    static bool IsInteriorlySupported(UIntPack<Degrees...>,
                                      int depth,
                                      const int offset[],
                                      const double begin[],
                                      const double end[]) {
        return depth >= 0 &&
               _IsInteriorlySupported(UIntPack<Degrees...>(), depth, offset, begin, end);
    }

    template <unsigned int... Degrees1, unsigned int... Degrees2>
    static void InteriorOverlappedSpan(
            UIntPack<Degrees1...>, UIntPack<Degrees2...>, int depth, int begin[], int end[]) {
        static_assert(sizeof...(Degrees1) == sizeof...(Degrees2), "[ERROR] Dimensions don't match");
        _InteriorOverlappedSpan(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), depth, begin,
                                end);
    }
    template <unsigned int... Degrees1, unsigned int... Degrees2>
    static bool IsInteriorlyOverlapped(UIntPack<Degrees1...>,
                                       UIntPack<Degrees2...>,
                                       int depth,
                                       const int offset[]) {
        static_assert(sizeof...(Degrees1) == sizeof...(Degrees2), "[ERROR] Dimensions don't match");
        return depth >= 0 && _IsInteriorlyOverlapped(UIntPack<Degrees1...>(),
                                                     UIntPack<Degrees2...>(), depth, offset);
    }

    template <unsigned int... Degrees1, unsigned int... Degrees2>
    static void ParentOverlapBounds(UIntPack<Degrees1...>,
                                    UIntPack<Degrees2...>,
                                    int depth,
                                    const int offset[],
                                    int start[],
                                    int end[]) {
        static_assert(sizeof...(Degrees1) == sizeof...(Degrees2), "[ERROR] Dimensions don't match");
        if (depth > 0)
            _ParentOverlapBounds(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), depth, offset,
                                 start, end);
    }
    template <unsigned int... Degrees1, unsigned int... Degrees2>
    static void ParentOverlapBounds(
            UIntPack<Degrees1...>, UIntPack<Degrees2...>, int corner, int start[], int end[]) {
        static_assert(sizeof...(Degrees1) == sizeof...(Degrees2), "[ERROR] Dimensions don't match");
        _ParentOverlapBounds(UIntPack<Degrees1...>(), UIntPack<Degrees2...>(), corner, start, end);
    }

    template <unsigned int Dim>
    struct PointEvaluatorState {
        virtual double value(const int offset[], const unsigned int d[]) const = 0;
        virtual double subValue(const int offset[], const unsigned int d[]) const = 0;
        template <class Real, typename DerivativeType>
        Point<Real, DerivativeType::Size> dValues(const int offset[]) const {
            Point<Real, DerivativeType::Size> v;
            unsigned int _d[Dim];
            for (int d = 0; d < DerivativeType::Size; d++) {
                DerivativeType::Factor(d, _d);
                v[d] = (Real)value(offset, _d);
            }
            return v;
        }
        template <class Real, typename DerivativeType>
        Point<Real, DerivativeType::LastDerivative + 1> partialDotDValues(
                Point<Real, DerivativeType::Size> v, const int offset[]) const {
            Point<Real, DerivativeType::LastDerivative + 1> dot;
            unsigned int _d[Dim];
            for (int d = 0; d < DerivativeType::Size; d++) {
                DerivativeType::Factor(d, _d);
                dot[_d[Dim - 1]] += (Real)(subValue(offset, _d) * v[d]);
            }
            return dot;
        }
    };

    template <unsigned int... TDegrees>
    struct PointEvaluator<UIntPack<TDegrees...>> {
        static const unsigned int Dim = sizeof...(TDegrees);
    };

    template <unsigned int... TDegrees>
    struct RestrictionProlongation<UIntPack<TDegrees...>> {
        virtual void init(void) {}
        virtual double upSampleCoefficient(const int pOff[], const int cOff[]) const = 0;

        typedef DynamicWindow<double,
                              UIntPack<(-BSplineSupportSizes<TDegrees>::DownSample0Start +
                                        BSplineSupportSizes<TDegrees>::DownSample1End + 1)...>>
                DownSampleStencil;
        struct UpSampleStencil
            : public DynamicWindow<double,
                                   UIntPack<BSplineSupportSizes<TDegrees>::UpSampleSize...>> {};
        struct DownSampleStencils
            : public DynamicWindow<DownSampleStencil, IsotropicUIntPack<sizeof...(TDegrees), 2>> {};

        void init(int highDepth) {
            _highDepth = highDepth;
            init();
        }
        void setStencil(UpSampleStencil& stencil) const;
        void setStencils(DownSampleStencils& stencils) const;
        int highDepth(void) const { return _highDepth; }

    protected:
        int _highDepth;
    };

    template <unsigned int... TDegrees>
    struct System<UIntPack<TDegrees...>> {
        virtual void init(void) {}
        virtual double ccIntegrate(const int off1[], const int off2[]) const = 0;
        virtual double pcIntegrate(const int off1[], const int off2[]) const = 0;
        virtual bool vanishesOnConstants(void) const { return false; }
        virtual RestrictionProlongation<UIntPack<TDegrees...>>& restrictionProlongation(void) = 0;

        struct CCStencil
            : public DynamicWindow<
                      double,
                      UIntPack<BSplineOverlapSizes<TDegrees, TDegrees>::OverlapSize...>> {};
#ifdef SHOW_WARNINGS
#pragma message("[WARNING] Why are the parent/child stencils so big?")
#endif  // SHOW_WARNINGS
        struct PCStencils
            : public DynamicWindow<CCStencil, IsotropicUIntPack<sizeof...(TDegrees), 2>> {};

        void init(int highDepth) {
            _highDepth = highDepth;
            init();
        }
        template <bool IterateFirst>
        void setStencil(CCStencil& stencil) const;
        template <bool IterateFirst>
        void setStencils(PCStencils& stencils) const;
        int highDepth(void) const { return _highDepth; }

    protected:
        int _highDepth;
    };

    template <unsigned int... TDegrees, unsigned int... CDegrees, unsigned int CDim>
    struct Constraint<UIntPack<TDegrees...>, UIntPack<CDegrees...>, CDim> {
        static_assert(sizeof...(TDegrees) == sizeof...(CDegrees),
                      "[ERROR] BaseFEMIntegrator::Constraint: Test and constraint dimensions don't "
                      "match");

        virtual void init(void) { ; }
        virtual Point<double, CDim> ccIntegrate(const int off1[], const int off2[]) const = 0;
        virtual Point<double, CDim> pcIntegrate(const int off1[], const int off2[]) const = 0;
        virtual Point<double, CDim> cpIntegrate(const int off1[], const int off2[]) const = 0;
        virtual RestrictionProlongation<UIntPack<TDegrees...>>& tRestrictionProlongation(void) = 0;
        virtual RestrictionProlongation<UIntPack<CDegrees...>>& cRestrictionProlongation(void) = 0;

        struct CCStencil
            : public DynamicWindow<
                      Point<double, CDim>,
                      UIntPack<BSplineOverlapSizes<TDegrees, CDegrees>::OverlapSize...>> {};
#ifdef SHOW_WARNINGS
#pragma message("[WARNING] Why are the parent/child stencils so big?")
#endif  // SHOW_WARNINGS
        struct PCStencils
            : public DynamicWindow<CCStencil, IsotropicUIntPack<sizeof...(TDegrees), 2>> {};
        struct CPStencils
            : public DynamicWindow<CCStencil, IsotropicUIntPack<sizeof...(TDegrees), 2>> {};

        void init(int highDepth) {
            _highDepth = highDepth;
            init();
        }
        template <bool IterateFirst>
        void setStencil(CCStencil& stencil) const;
        template <bool IterateFirst>
        void setStencils(PCStencils& stencils) const;
        template <bool IterateFirst>
        void setStencils(CPStencils& stencils) const;
        int highDepth(void) const { return _highDepth; }

    protected:
        int _highDepth;
    };

    template <unsigned int... TDegrees>
    struct SystemConstraint<UIntPack<TDegrees...>>
        : public Constraint<UIntPack<TDegrees...>, UIntPack<TDegrees...>, 1> {
        typedef Constraint<UIntPack<TDegrees...>, UIntPack<TDegrees...>, 1> Base;
        SystemConstraint(System<UIntPack<TDegrees...>>& sys) : _sys(sys) { ; }
        void init(void) {
            _sys.init(Base::highDepth());
            _sys.init();
        }
        Point<double, 1> ccIntegrate(const int off1[], const int off2[]) const {
            return Point<double, 1>(_sys.ccIntegrate(off1, off2));
        }
        Point<double, 1> pcIntegrate(const int off1[], const int off2[]) const {
            return Point<double, 1>(_sys.pcIntegrate(off1, off2));
        }
        Point<double, 1> cpIntegrate(const int off1[], const int off2[]) const {
            return Point<double, 1>(_sys.pcIntegrate(off2, off1));
        }
        RestrictionProlongation<UIntPack<TDegrees...>>& tRestrictionProlongation(void) {
            return _sys.restrictionProlongation();
        }
        RestrictionProlongation<UIntPack<TDegrees...>>& cRestrictionProlongation(void) {
            return _sys.restrictionProlongation();
        }

    protected:
        System<UIntPack<TDegrees...>>& _sys;
    };
};

/////////////////////////////////////////////////
// An implementation of the virtual integrator //
/////////////////////////////////////////////////
struct FEMIntegrator {
protected:
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) == 0, bool>::type _IsValidFEMNode(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, const int offset[]) {
        return !BSplineEvaluationData<FEMSig>::OutOfBounds(depth, offset[0]);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) != 0, bool>::type _IsValidFEMNode(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, const int offset[]) {
        return !BSplineEvaluationData<FEMSig>::OutOfBounds(depth, offset[0]) &&
               _IsValidFEMNode(UIntPack<FEMSigs...>(), depth, offset + 1);
    }
    template <unsigned int FEMSig, unsigned... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) == 0, bool>::type _IsOutOfBounds(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, const int offset[]) {
        return BSplineEvaluationData<FEMSig>::OutOfBounds(depth, offset[0]);
    }
    template <unsigned int FEMSig, unsigned... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) != 0, bool>::type _IsOutOfBounds(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, const int offset[]) {
        return BSplineEvaluationData<FEMSig>::OutOfBounds(depth, offset[0]) ||
               _IsOutOfBounds(UIntPack<FEMSigs...>(), depth, offset + 1);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) == 0>::type _BSplineBegin(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, int begin[]) {
        begin[0] = BSplineEvaluationData<FEMSig>::Begin(depth);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) != 0>::type _BSplineBegin(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, int begin[]) {
        begin[0] = BSplineEvaluationData<FEMSig>::Begin(depth);
        _BSplineBegin(UIntPack<FEMSigs...>(), depth, begin + 1);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) == 0>::type _BSplineEnd(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, int end[]) {
        end[0] = BSplineEvaluationData<FEMSig>::End(depth);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) != 0>::type _BSplineEnd(
            UIntPack<FEMSig, FEMSigs...>, unsigned int depth, int end[]) {
        end[0] = BSplineEvaluationData<FEMSig>::End(depth);
        _BSplineEnd(UIntPack<FEMSigs...>(), depth, end + 1);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) == 0, double>::type _Integral(
            UIntPack<FEMSig, FEMSigs...>,
            unsigned int depth,
            const int offset[],
            const double begin[],
            const double end[]) {
        return BSplineEvaluationData<FEMSig>::Integral(depth, offset[0], begin[0], end[0], 0);
    }
    template <unsigned int FEMSig, unsigned int... FEMSigs>
    static typename std::enable_if<sizeof...(FEMSigs) != 0, double>::type _Integral(
            UIntPack<FEMSig, FEMSigs...>,
            unsigned int depth,
            const int offset[],
            const double begin[],
            const double end[]) {
        return BSplineEvaluationData<FEMSig>::Integral(depth, offset[0], begin[0], end[0], 0) *
               _Integral(UIntPack<FEMSigs...>(), depth, offset + 1, begin + 1, end + 1);
    }

public:
    template <unsigned int... FEMSigs>
    static double Integral(UIntPack<FEMSigs...>,
                           int depth,
                           const int offset[],
                           const double begin[],
                           const double end[]) {
        if (depth < 0)
            return 0;
        else
            return _Integral(UIntPack<FEMSigs...>(), depth, offset, begin, end);
    }
    template <unsigned int... FEMSigs>
    static bool IsValidFEMNode(UIntPack<FEMSigs...>, int depth, const int offset[]) {
        return _IsValidFEMNode(UIntPack<FEMSigs...>(), depth, offset);
    }
    template <unsigned int... FEMSigs>
    static bool IsOutOfBounds(UIntPack<FEMSigs...>, int depth, const int offset[]) {
        return depth < 0 || _IsOutOfBounds(UIntPack<FEMSigs...>(), depth, offset);
    }
    template <unsigned int... FEMSigs>
    static void BSplineBegin(UIntPack<FEMSigs...>, int depth, int begin[]) {
        if (depth >= 0) _BSplineBegin(UIntPack<FEMSigs...>(), depth, begin);
    }
    template <unsigned int... FEMSigs>
    static void BSplineEnd(UIntPack<FEMSigs...>, int depth, int end[]) {
        if (depth >= 0) _BSplineEnd(UIntPack<FEMSigs...>(), depth, end);
    }

    template <typename TSignatures, typename TDerivatives>
    struct System {};
    template <typename TSignatures,
              typename TDerivatives,
              typename CSignatures,
              typename CDerivatives,
              unsigned int CDim>
    struct Constraint {};
    template <typename TSignatures,
              typename TDerivatives,
              typename CSignatures,
              typename CDerivatives>
    struct ScalarConstraint {};
    template <typename TSignatures>
    struct RestrictionProlongation {};
    template <typename TSignatures, typename TDerivatives>
    struct PointEvaluator {};
    template <typename TSignatures, typename TDerivatives>
    struct PointEvaluatorState {};

    template <unsigned int... TSignatures, unsigned int... TDs>
    struct PointEvaluatorState<UIntPack<TSignatures...>, UIntPack<TDs...>>
        : public BaseFEMIntegrator::template PointEvaluatorState<sizeof...(TSignatures)> {
        static_assert(sizeof...(TSignatures) == sizeof...(TDs),
                      "[ERROR] Degree and derivative dimensions don't match");
        static_assert(UIntPack<FEMSignature<TSignatures>::Degree...>::template Compare<
                              UIntPack<TDs...>>::GreaterThanOrEqual,
                      "[ERROR] PointEvaluatorState: More derivatives than degrees");

        static const unsigned int Dim = sizeof...(TSignatures);

        double value(const int offset[], const unsigned int derivatives[]) const {
            return _value<Dim>(offset, derivatives);
        }
        double subValue(const int offset[], const unsigned int derivatives[]) const {
            return _value<Dim - 1>(offset, derivatives);
        }
        // Bypassing the "auto" keyword
        template <unsigned int _Dim>
        const double (*(values)(void)const)[UIntPack<TDs...>::template Get<_Dim>() + 1] {
            return std::template get<_Dim>(_oneDValues).values;
        }

    protected:
        int _pointOffset[Dim];

        template <unsigned int Degree, unsigned int D>
        struct _OneDValues {
            double values[BSplineSupportSizes<Degree>::SupportSize][D + 1];
            double value(int dOff, unsigned int d) const {
                if (dOff >= -BSplineSupportSizes<Degree>::SupportEnd &&
                    dOff <= -BSplineSupportSizes<Degree>::SupportStart && d <= D)
                    return values[dOff + BSplineSupportSizes<Degree>::SupportEnd][d];
                else
                    return 0;
            }
        };
        std::tuple<_OneDValues<FEMSignature<TSignatures>::Degree, TDs>...> _oneDValues;
        template <unsigned int MaxDim = Dim, unsigned int I = 0>
        typename std::enable_if<I == MaxDim, double>::type _value(const int off[],
                                                                  const unsigned int d[]) const {
            return 1.;
        }
        template <unsigned int MaxDim = Dim, unsigned int I = 0>
        typename std::enable_if<I != MaxDim, double>::type _value(const int off[],
                                                                  const unsigned int d[]) const {
            return std::get<I>(_oneDValues).value(off[I] - _pointOffset[I], d[I]) *
                   _value<MaxDim, I + 1>(off, d);
        }
        template <typename T1, typename T2>
        friend struct PointEvaluator;
    };

    template <unsigned int... TSignatures, unsigned int... TDs>
    struct PointEvaluator<UIntPack<TSignatures...>, UIntPack<TDs...>>
        : public BaseFEMIntegrator::template PointEvaluator<
                  UIntPack<FEMSignature<TSignatures>::Degree...>> {
        static_assert(sizeof...(TSignatures) == sizeof...(TDs),
                      "[ERROR] PointEvaluator: Degree and derivative dimensions don't match");
        static_assert(UIntPack<FEMSignature<TSignatures>::Degree...>::template Compare<
                              UIntPack<TDs...>>::GreaterThanOrEqual,
                      "[ERROR] PointEvaluator: More derivatives than degrees");

        static const unsigned int Dim = sizeof...(TSignatures);

        typedef typename BaseFEMIntegrator::template PointEvaluator<
                UIntPack<FEMSignature<TSignatures>::Degree...>>
                Base;

        PointEvaluator(unsigned int maxDepth) : _maxDepth(maxDepth) { _init(); }
        template <unsigned int... EDs>
        void initEvaluationState(
                Point<double, Dim> p,
                unsigned int depth,
                PointEvaluatorState<UIntPack<TSignatures...>, UIntPack<EDs...>>& state) const {
            unsigned int res = 1 << depth;
            for (int d = 0; d < Dim; d++) state._pointOffset[d] = (int)(p[d] * res);
            initEvaluationState(p, depth, state._pointOffset, state);
        }
        template <unsigned int... EDs>
        void initEvaluationState(
                Point<double, Dim> p,
                unsigned int depth,
                const int* offset,
                PointEvaluatorState<UIntPack<TSignatures...>, UIntPack<EDs...>>& state) const {
            static_assert(UIntPack<TDs...>::template Compare<UIntPack<EDs...>>::GreaterThanOrEqual,
                          "[ERROR] PointEvaluator::init: More evaluation derivatives than stored "
                          "derivatives");
            for (int d = 0; d < Dim; d++) state._pointOffset[d] = (int)offset[d];
            _initEvaluationState(UIntPack<TSignatures...>(), UIntPack<EDs...>(), &p[0], depth,
                                 state);
        }

    protected:
        unsigned int _maxDepth;
        std::tuple<BSplineData<TSignatures, TDs>...> _bSplineData;
        template <unsigned int I = 0>
        typename std::enable_if<I == Dim>::type _init(void) {}
        template <unsigned int I = 0>
                typename std::enable_if < I<Dim>::type _init(void) {
            std::get<I>(_bSplineData).reset(_maxDepth);
            _init<I + 1>();
        }

        template <unsigned int I, unsigned int TSig, unsigned int D, typename State>
        void _setEvaluationState(const double* p, unsigned int depth, State& state) const {
            static const int LeftSupportRadius =
                    -BSplineSupportSizes<FEMSignature<TSig>::Degree>::SupportStart;
            static const int LeftPointSupportRadius =
                    BSplineSupportSizes<FEMSignature<TSig>::Degree>::SupportEnd;
            static const int RightSupportRadius =
                    BSplineSupportSizes<FEMSignature<TSig>::Degree>::SupportEnd;
            static const int RightPointSupportRadius =
                    -BSplineSupportSizes<FEMSignature<TSig>::Degree>::SupportStart;
            for (int s = -LeftPointSupportRadius; s <= RightPointSupportRadius; s++) {
                int pIdx = state._pointOffset[I];
                int fIdx = state._pointOffset[I] + s;
                double _p = p[I];
                const Polynomial<FEMSignature<TSig>::Degree>* components =
                        std::get<I>(_bSplineData)[depth].polynomialsAndOffset(_p, pIdx, fIdx);
                for (int d = 0; d <= D; d++)
                    std::get<I>(state._oneDValues).values[s + LeftPointSupportRadius][d] =
                            components[d](_p);
            }
        }
        template <typename State,
                  unsigned int TSig,
                  unsigned int... TSigs,
                  unsigned int D,
                  unsigned int... Ds>
        typename std::enable_if<sizeof...(TSigs) == 0>::type _initEvaluationState(
                UIntPack<TSig, TSigs...>,
                UIntPack<D, Ds...>,
                const double* p,
                unsigned int depth,
                State& state) const {
            _setEvaluationState<Dim - 1, TSig, D>(p, depth, state);
        }
        template <typename State,
                  unsigned int TSig,
                  unsigned int... TSigs,
                  unsigned int D,
                  unsigned int... Ds>
        typename std::enable_if<sizeof...(TSigs) != 0>::type _initEvaluationState(
                UIntPack<TSig, TSigs...>,
                UIntPack<D, Ds...>,
                const double* p,
                unsigned int depth,
                State& state) const {
            _setEvaluationState<Dim - 1 - sizeof...(TSigs), TSig, D>(p, depth, state);
            _initEvaluationState(UIntPack<TSigs...>(), UIntPack<Ds...>(), p, depth, state);
        }
    };

    template <unsigned int... TSignatures>
    struct RestrictionProlongation<UIntPack<TSignatures...>>
        : public BaseFEMIntegrator::template RestrictionProlongation<
                  UIntPack<FEMSignature<TSignatures>::Degree...>> {
        static const unsigned int Dim = sizeof...(TSignatures);
        typedef typename BaseFEMIntegrator::template RestrictionProlongation<
                UIntPack<FEMSignature<TSignatures>::Degree...>>
                Base;

        double upSampleCoefficient(const int pOff[], const int cOff[]) const {
            return _coefficient(pOff, cOff);
        }
        void init(unsigned int depth) { Base::init(depth); }
        void init(void) { _init(Base::highDepth()); }

    protected:
        std::tuple<typename BSplineEvaluationData<TSignatures>::UpSampleEvaluator...> _upSamplers;

        template <unsigned int D = 0>
        typename std::enable_if<D == Dim>::type _init(int highDepth) {}
        template <unsigned int D = 0>
                typename std::enable_if < D<Dim>::type _init(int highDepth) {
            std::get<D>(_upSamplers).set(highDepth - 1);
            _init<D + 1>(highDepth);
        }
        template <unsigned int D = 0>
        typename std::enable_if<D == Dim, double>::type _coefficient(const int pOff[],
                                                                     const int cOff[]) const {
            return 1.;
        }
        template <unsigned int D = 0>
                typename std::enable_if <
                D<Dim, double>::type _coefficient(const int pOff[], const int cOff[]) const {
            return _coefficient<D + 1>(pOff, cOff) *
                   std::get<D>(_upSamplers).value(pOff[D], cOff[D]);
        }
    };

    template <unsigned int... TSignatures,
              unsigned int... TDerivatives,
              unsigned int... CSignatures,
              unsigned int... CDerivatives,
              unsigned int CDim>
    struct Constraint<UIntPack<TSignatures...>,
                      UIntPack<TDerivatives...>,
                      UIntPack<CSignatures...>,
                      UIntPack<CDerivatives...>,
                      CDim>
        : public BaseFEMIntegrator::template Constraint<
                  UIntPack<FEMSignature<TSignatures>::Degree...>,
                  UIntPack<FEMSignature<CSignatures>::Degree...>,
                  CDim> {
        static_assert(
                sizeof...(TSignatures) == sizeof...(CSignatures),
                "[ERROR] Test signatures and contraint signatures must have the same dimension");
        static_assert(sizeof...(TSignatures) == sizeof...(TDerivatives),
                      "[ERROR] Test signatures and derivatives must have the same dimension");
        static_assert(sizeof...(CSignatures) == sizeof...(CDerivatives),
                      "[ERROR] Constraint signatures and derivatives must have the same dimension");
        static_assert(UIntPack<FEMSignature<TSignatures>::Degree...>::template Compare<
                              UIntPack<TDerivatives...>>::GreaterThanOrEqual,
                      "[ERROR] Test functions cannot have more derivatives than the degree");
        static_assert(UIntPack<FEMSignature<CSignatures>::Degree...>::template Compare<
                              UIntPack<CDerivatives...>>::GreaterThanOrEqual,
                      "[ERROR] Test functions cannot have more derivatives than the degree");

        static const unsigned int Dim = sizeof...(TSignatures);
        typedef typename BaseFEMIntegrator::template Constraint<
                UIntPack<FEMSignature<TSignatures>::Degree...>,
                UIntPack<FEMSignature<CSignatures>::Degree...>,
                CDim>
                Base;

        static const unsigned int TDerivativeSize =
                TensorDerivatives<UIntPack<TDerivatives...>>::Size;
        static const unsigned int CDerivativeSize =
                TensorDerivatives<UIntPack<CDerivatives...>>::Size;
        static inline void TFactorDerivatives(unsigned int idx, unsigned int d[Dim]) {
            TensorDerivatives<UIntPack<TDerivatives...>>::Factor(idx, d);
        }
        static inline void CFactorDerivatives(unsigned int idx, unsigned int d[Dim]) {
            TensorDerivatives<UIntPack<CDerivatives...>>::Factor(idx, d);
        }
        static inline unsigned int TDerivativeIndex(const unsigned int d[Dim]) {
            return TensorDerivatives<UIntPack<TDerivatives...>>::Index(d);
        }
        static inline unsigned int CDerivativeIndex(const unsigned int d[Dim]) {
            return TensorDerivatives<UIntPack<CDerivatives...>>::Index(d);
        }
        Matrix<double, TDerivativeSize, CDerivativeSize> weights[CDim];

        Point<double, CDim> ccIntegrate(const int off1[], const int off2[]) const {
            return _integrate(INTEGRATE_CHILD_CHILD, off1, off2);
        }
        Point<double, CDim> pcIntegrate(const int off1[], const int off2[]) const {
            return _integrate(INTEGRATE_PARENT_CHILD, off1, off2);
        }
        Point<double, CDim> cpIntegrate(const int off1[], const int off2[]) const {
            return _integrate(INTEGRATE_CHILD_PARENT, off1, off2);
        }

        void init(unsigned int depth) { Base::init(depth); }
        void init(void) {
            _init(Base::highDepth());
            _weightedIndices.resize(0);
            for (unsigned int d1 = 0; d1 < TDerivativeSize; d1++)
                for (unsigned int d2 = 0; d2 < CDerivativeSize; d2++) {
                    _WeightedIndices w(d1, d2);
                    for (unsigned int c = 0; c < CDim; c++)
                        if (weights[c](d1, d2) > 0)
                            w.indices.push_back(
                                    std::pair<unsigned int, double>(c, weights[c](d1, d2)));
                    if (w.indices.size()) _weightedIndices.push_back(w);
                }
        }
        typename BaseFEMIntegrator::template RestrictionProlongation<
                UIntPack<FEMSignature<TSignatures>::Degree...>>&
        tRestrictionProlongation(void) {
            return _tRestrictionProlongation;
        }
        typename BaseFEMIntegrator::template RestrictionProlongation<
                UIntPack<FEMSignature<CSignatures>::Degree...>>&
        cRestrictionProlongation(void) {
            return _cRestrictionProlongation;
        }

    protected:
        RestrictionProlongation<UIntPack<TSignatures...>> _tRestrictionProlongation;
        RestrictionProlongation<UIntPack<CSignatures...>> _cRestrictionProlongation;
        struct _WeightedIndices {
            _WeightedIndices(unsigned int _d1 = 0, unsigned int _d2 = 0) : d1(_d1), d2(_d2) { ; }
            unsigned int d1, d2;
            std::vector<std::pair<unsigned int, double>> indices;
        };
        std::vector<_WeightedIndices> _weightedIndices;
        enum IntegrationType {
            INTEGRATE_CHILD_CHILD,
            INTEGRATE_PARENT_CHILD,
            INTEGRATE_CHILD_PARENT
        };

        template <unsigned int _TSig,
                  unsigned int _TDerivatives,
                  unsigned int _CSig,
                  unsigned int _CDerivatives>
        struct _Integrators {
            typename BSplineIntegrationData<_TSig, _CSig>::FunctionIntegrator::
                    template Integrator<_TDerivatives, _CDerivatives>
                            ccIntegrator;
            typename BSplineIntegrationData<_TSig, _CSig>::FunctionIntegrator::
                    template ChildIntegrator<_TDerivatives, _CDerivatives>
                            pcIntegrator;
            typename BSplineIntegrationData<_CSig, _TSig>::FunctionIntegrator::
                    template ChildIntegrator<_CDerivatives, _TDerivatives>
                            cpIntegrator;
        };
        std::tuple<_Integrators<TSignatures, TDerivatives, CSignatures, CDerivatives>...>
                _integrators;

        template <unsigned int D = 0>
        typename std::enable_if<D == Dim>::type _init(int depth) {
            ;
        }
        template <unsigned int D = 0>
                typename std::enable_if < D<Dim>::type _init(int depth) {
            std::get<D>(_integrators).ccIntegrator.set(depth);
            if (depth)
                std::get<D>(_integrators).pcIntegrator.set(depth - 1),
                        std::get<D>(_integrators).cpIntegrator.set(depth - 1);
            _init<D + 1>(depth);
        }
        template <unsigned int D = 0>
        typename std::enable_if<D == Dim, double>::type _integral(IntegrationType iType,
                                                                  const int off1[],
                                                                  const int off2[],
                                                                  const unsigned int d1[],
                                                                  const unsigned int d2[]) const {
            return 1.;
        }
        template <unsigned int D = 0>
                typename std::enable_if <
                D<Dim, double>::type _integral(IntegrationType iType,
                                               const int off1[],
                                               const int off2[],
                                               const unsigned int d1[],
                                               const unsigned int d2[]) const {
            double remainingIntegral = _integral<D + 1>(iType, off1, off2, d1, d2);
            switch (iType) {
                case INTEGRATE_CHILD_CHILD:
                    return std::get<D>(_integrators)
                                   .ccIntegrator.dot(off1[D], off2[D], d1[D], d2[D]) *
                           remainingIntegral;
                case INTEGRATE_PARENT_CHILD:
                    return std::get<D>(_integrators)
                                   .pcIntegrator.dot(off1[D], off2[D], d1[D], d2[D]) *
                           remainingIntegral;
                case INTEGRATE_CHILD_PARENT:
                    return std::get<D>(_integrators)
                                   .cpIntegrator.dot(off2[D], off1[D], d2[D], d1[D]) *
                           remainingIntegral;
                default:
                    ERROR_OUT("Undefined integration type");
            }
            return 0;
        }
        Point<double, CDim> _integrate(IntegrationType iType,
                                       const int off1[],
                                       const int off[]) const;
    };

    template <unsigned int... TSignatures,
              unsigned int... TDerivatives,
              unsigned int... CSignatures,
              unsigned int... CDerivatives>
    struct ScalarConstraint<UIntPack<TSignatures...>,
                            UIntPack<TDerivatives...>,
                            UIntPack<CSignatures...>,
                            UIntPack<CDerivatives...>>
        : public Constraint<UIntPack<TSignatures...>,
                            UIntPack<TDerivatives...>,
                            UIntPack<CSignatures...>,
                            UIntPack<CDerivatives...>,
                            1> {
        static const unsigned int Dim = sizeof...(TSignatures);
        typedef typename BaseFEMIntegrator::template Constraint<
                UIntPack<FEMSignature<TSignatures>::Degree...>,
                UIntPack<FEMSignature<CSignatures>::Degree...>,
                1>
                Base;

        typedef Constraint<UIntPack<TSignatures...>,
                           UIntPack<TDerivatives...>,
                           UIntPack<CSignatures...>,
                           UIntPack<CDerivatives...>,
                           1>
                FullConstraint;
        using FullConstraint::weights;
        // [NOTE] We define the constructor using a recursive function call to take into account
        // multiplicity (e.g. so that d^2/dxdy and d^2/dydx each contribute)
        ScalarConstraint(const std::initializer_list<double>& w) {
            std::function<void(unsigned int[], const double[], unsigned int)> SetDerivativeWeights =
                    [&](unsigned int derivatives[Dim], const double w[], unsigned int d) {
                        unsigned int idx1 = FullConstraint::TDerivativeIndex(derivatives),
                                     idx2 = FullConstraint::CDerivativeIndex(derivatives);
                        weights[0][idx1][idx2] += w[0];
                        if (d > 0)
                            for (int dd = 0; dd < Dim; dd++) {
                                derivatives[dd]++;
                                SetDerivativeWeights(derivatives, w + 1, d - 1);
                                derivatives[dd]--;
                            }
                    };
            static const unsigned int DMax = std::min<unsigned int>(
                    UIntPack<TDerivatives...>::Min(), UIntPack<CDerivatives...>::Min());

            unsigned int derivatives[Dim];
            double _w[DMax + 1];
            memset(_w, 0, sizeof(_w));
            {
                unsigned int dd = 0;
                for (typename std::initializer_list<double>::const_iterator iter = w.begin();
                     iter != w.end() && dd <= DMax; dd++, iter++)
                    _w[dd] = *iter;
            }
            for (int d = 0; d < Dim; d++) derivatives[d] = 0;
            if (w.size())
                SetDerivativeWeights(derivatives, _w,
                                     std::min<unsigned int>(DMax + 1, (unsigned int)w.size()) - 1);
        }
    };

    template <unsigned int... TSignatures, unsigned int... TDerivatives>
    struct System<UIntPack<TSignatures...>, UIntPack<TDerivatives...>>
        : public BaseFEMIntegrator::template System<
                  UIntPack<FEMSignature<TSignatures>::Degree...>> {
        static_assert(sizeof...(TSignatures) == sizeof...(TDerivatives),
                      "[ERROR] Test signatures and derivatives must have the same dimension");

        static const unsigned int Dim = sizeof...(TSignatures);
        typedef typename BaseFEMIntegrator::template System<
                UIntPack<FEMSignature<TSignatures>::Degree...>>
                Base;

        System(const std::initializer_list<double>& w) : _sc(w) { ; }
        void init(unsigned int depth) { Base::init(depth); }
        void init(void) {
            ((BaseFEMIntegrator::template Constraint<UIntPack<FEMSignature<TSignatures>::Degree...>,
                                                     UIntPack<FEMSignature<TSignatures>::Degree...>,
                                                     1>&)_sc)
                    .init(BaseFEMIntegrator::template System<
                            UIntPack<FEMSignature<TSignatures>::Degree...>>::_highDepth);
        }
        double ccIntegrate(const int off1[], const int off2[]) const {
            return _sc.ccIntegrate(off1, off2)[0];
        }
        double pcIntegrate(const int off1[], const int off2[]) const {
            return _sc.pcIntegrate(off1, off2)[0];
        }
        bool vanishesOnConstants(void) const { return _sc.weights[0][0][0] == 0; }

        typename BaseFEMIntegrator::template RestrictionProlongation<
                UIntPack<FEMSignature<TSignatures>::Degree...>>&
        restrictionProlongation(void) {
            return _sc.tRestrictionProlongation();
        }

    protected:
        ScalarConstraint<UIntPack<TSignatures...>,
                         UIntPack<TDerivatives...>,
                         UIntPack<TSignatures...>,
                         UIntPack<TDerivatives...>>
                _sc;
    };
};

//////////////////////////////////////////

template <unsigned int Dim>
inline void SetGhostFlag(RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node,
                         bool flag) {
    if (node && node->parent) node->parent->nodeData.setGhostFlag(flag);
}
template <unsigned int Dim>
inline bool GetGhostFlag(const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) {
    return node == NULL || node->parent == NULL || node->parent->nodeData.getGhostFlag();
}
template <unsigned int Dim>
inline bool IsActiveNode(const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node) {
    return !GetGhostFlag<Dim>(node);
}

template <unsigned int Dim, class Real, class Vertex>
struct IsoSurfaceExtractor;

template <unsigned int Dim, class Data>
struct NodeSample {
    RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node;
    Data data;
};
template <unsigned int Dim, class Real>
struct NodeAndPointSample {
    RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* node;
    ProjectiveData<Point<Real, Dim>, Real> sample;
};
template <unsigned int Dim, class Real>
using NodeSimplices = NodeSample<Dim, std::vector<Simplex<Real, Dim, Dim - 1>>>;

template <typename T>
struct WindowLoopData {};

template <unsigned int... Sizes>
struct WindowLoopData<UIntPack<Sizes...>> {
    static const int Dim = sizeof...(Sizes);
    unsigned int size[1 << Dim];
    unsigned int indices[1 << Dim][WindowSize<UIntPack<Sizes...>>::Size];
    template <typename BoundsFunction>
    WindowLoopData(const BoundsFunction& boundsFunction) {
        int start[Dim], end[Dim];
        for (int c = 0; c < (1 << Dim); c++) {
            size[c] = 0;
            boundsFunction(c, start, end);
            unsigned int idx[Dim];
            WindowLoop<Dim>::Run(start, end, [&](int d, int i) { idx[d] = i; },
                                 [&](void) {
                                     indices[c][size[c]++] =
                                             GetWindowIndex(UIntPack<Sizes...>(), idx);
                                 });
        }
    }
};

template <class Real, unsigned int Dim>
void AddAtomic(Point<Real, Dim>& a, const Point<Real, Dim>& b) {
    for (int d = 0; d < Dim; d++) AddAtomic(a[d], b[d]);
}

template <class Data>
bool IsZero(const Data& data) {
    return false;
}
template <class Real, unsigned int Dim>
bool IsZero(const Point<Real, Dim>& d) {
    bool zero = true;
    for (int i = 0; i < Dim; i++) zero &= (d[i] == 0);
    return zero;
}
bool IsZero(const float& f) { return f == 0.f; }
bool IsZero(const double& f) { return f == 0.; }

template <unsigned int Dim, class Real>
class FEMTree {
public:
    typedef RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type> FEMTreeNode;
    std::vector<Allocator<FEMTreeNode>*> nodeAllocators;

protected:
    template <unsigned int _Dim, class _Real, class Vertex>
    friend struct IsoSurfaceExtractor;
    std::atomic<node_index_type> _nodeCount;
    struct _NodeInitializer {
        FEMTree& femTree;
        _NodeInitializer(FEMTree& f) : femTree(f) { ; }
        void operator()(FEMTreeNode& node) { node.nodeData.nodeIndex = femTree._nodeCount++; }
    };
    _NodeInitializer _nodeInitializer;

public:
    typedef int LocalDepth;
    typedef int LocalOffset[Dim];

    node_index_type nodeCount(void) const { return _nodeCount; }

    typedef NodeAndPointSample<Dim, Real> PointSample;

    typedef typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 1>,
                                                       IsotropicUIntPack<Dim, 1>>
            OneRingNeighborKey;
    typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>,
                                                            IsotropicUIntPack<Dim, 1>>
            ConstOneRingNeighborKey;
    typedef typename FEMTreeNode::template Neighbors<IsotropicUIntPack<Dim, 3>> OneRingNeighbors;
    typedef typename FEMTreeNode::template ConstNeighbors<IsotropicUIntPack<Dim, 3>>
            ConstOneRingNeighbors;

    template <typename FEMDegreePack>
    using BaseSystem = typename BaseFEMIntegrator::template System<FEMDegreePack>;
    template <typename FEMSigPack, typename DerivativePack>
    using PointEvaluator =
            typename FEMIntegrator::template PointEvaluator<FEMSigPack, DerivativePack>;
    template <typename FEMSigPack, typename DerivativePack>
    using PointEvaluatorState =
            typename FEMIntegrator::template PointEvaluatorState<FEMSigPack, DerivativePack>;
    template <typename FEMDegreePack>
    using CCStencil = typename BaseSystem<FEMDegreePack>::CCStencil;
    template <typename FEMDegreePack>
    using PCStencils = typename BaseSystem<FEMDegreePack>::PCStencils;

    template <unsigned int... FEMSigs>
    bool isValidFEMNode(UIntPack<FEMSigs...>, const FEMTreeNode* node) const;
    bool isValidSpaceNode(const FEMTreeNode* node) const;
    const FEMTreeNode* leaf(Point<Real, Dim> p) const;

protected:
    template <bool ThreadSafe>
    FEMTreeNode* _leaf(Allocator<FEMTreeNode>* nodeAllocator,
                       Point<Real, Dim> p,
                       LocalDepth maxDepth = -1);

public:
    // [NOTE] In the case that T != double, we require both operators() for computing the system
    // dual
    template <typename T, unsigned int PointD>
    struct InterpolationInfo {
        virtual void range(const FEMTreeNode* node, size_t& begin, size_t& end) const = 0;
        virtual Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx) const = 0;
        virtual Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const = 0;
        virtual Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const = 0;
        virtual const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const = 0;
        virtual bool constrainsDCTerm(void) const = 0;
        virtual ~InterpolationInfo(void) {}

        DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIndex) {
            return const_cast<DualPointInfo<Dim, Real, T, PointD>&>(
                    ((const InterpolationInfo*)this)->operator[](pointIndex));
        }
    };
    template <unsigned int PointD>
    struct InterpolationInfo<double, PointD> {
        virtual void range(const FEMTreeNode* node, size_t& begin, size_t& end) const = 0;
        virtual Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx) const = 0;
        virtual Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const = 0;
        virtual const DualPointInfo<Dim, Real, double, PointD>& operator[](
                size_t pointIdx) const = 0;
        virtual bool constrainsDCTerm(void) const = 0;
        virtual ~InterpolationInfo(void) {}

        DualPointInfo<Dim, Real, double, PointD>& operator[](size_t pointIndex) {
            return const_cast<DualPointInfo<Dim, Real, double, PointD>&>(
                    ((const InterpolationInfo*)this)->operator[](pointIndex));
        }
    };

    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ApproximatePointInterpolationInfo : public InterpolationInfo<T, PointD> {
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = idx, end = idx + 1;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx];
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].position);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].position, dValues);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].position, dValues);
        }

        ApproximatePointInterpolationInfo(ConstraintDual constraintDual,
                                          SystemDual systemDual,
                                          bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        SparseNodeData<DualPointInfo<Dim, Real, T, PointD>, ZeroUIntPack<Dim>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ApproximatePointInterpolationInfo<double, PointD, ConstraintDual, SystemDual>
        : public InterpolationInfo<double, PointD> {
        typedef double T;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = idx, end = idx + 1;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx];
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].position);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].position, dValues);
        }

        ApproximatePointInterpolationInfo(ConstraintDual constraintDual,
                                          SystemDual systemDual,
                                          bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        SparseNodeData<DualPointInfo<Dim, Real, T, PointD>, ZeroUIntPack<Dim>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    struct ApproximatePointAndDataInterpolationInfo : public InterpolationInfo<T, PointD> {
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = idx, end = idx + 1;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx].pointInfo;
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].pointInfo.position, _iData[pointIdx].data);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].pointInfo.position, _iData[(int)pointIdx].data,
                               dValues);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[(int)pointIdx].pointInfo.position, _iData[(int)pointIdx].data,
                               dValues);
        }

        ApproximatePointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                                 SystemDual systemDual,
                                                 bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        SparseNodeData<DualPointAndDataInfo<Dim, Real, Data, T, PointD>, ZeroUIntPack<Dim>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <typename Data, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ApproximatePointAndDataInterpolationInfo<double,
                                                    Data,
                                                    PointD,
                                                    ConstraintDual,
                                                    SystemDual>
        : public InterpolationInfo<double, PointD> {
        typedef double T;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = idx, end = idx + 1;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx].pointInfo;
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].pointInfo.position, _iData[pointIdx].data);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].pointInfo.position, _iData[pointIdx].data, dValues);
        }

        ApproximatePointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                                 SystemDual systemDual,
                                                 bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        SparseNodeData<DualPointAndDataInfo<Dim, Real, Data, T, PointD>, ZeroUIntPack<Dim>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };

    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ApproximateChildPointInterpolationInfo : public InterpolationInfo<T, PointD> {
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = (idx << Dim), end = (idx << Dim) | _iData[idx].size();
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return __iData(pointIdx);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(__iData(pointIdx).position);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(__iData(pointIdx).position, dValues);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(__iData(pointIdx).position, dValues);
        }

        ApproximateChildPointInterpolationInfo(ConstraintDual constraintDual,
                                               SystemDual systemDual,
                                               bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        static const unsigned int _Mask = (1 << Dim) - 1;
        SparseNodeData<DualPointInfoBrood<Dim, Real, T, PointD>, ZeroUIntPack<Dim>> _iData;
        DualPointInfo<Dim, Real, T, PointD>& __iData(size_t pointIdx) {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        const DualPointInfo<Dim, Real, T, PointD>& __iData(size_t pointIdx) const {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ApproximateChildPointInterpolationInfo<double, PointD, ConstraintDual, SystemDual>
        : public InterpolationInfo<double, PointD> {
        typedef double T;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx < 0)
                begin = end = 0;
            else
                begin = (idx << Dim), end = (idx << Dim) | _iData[idx].size();
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return __iData(pointIdx);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(__iData(pointIdx).position);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(__iData(pointIdx).position, dValues);
        }

        ApproximateChildPointInterpolationInfo(ConstraintDual constraintDual,
                                               SystemDual systemDual,
                                               bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        static const unsigned int _Mask = (1 << Dim) - 1;
        SparseNodeData<DualPointInfoBrood<Dim, Real, T, PointD>, ZeroUIntPack<Dim>> _iData;
        DualPointInfo<Dim, Real, T, PointD>& __iData(size_t pointIdx) {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        const DualPointInfo<Dim, Real, T, PointD>& __iData(size_t pointIdx) const {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    struct ApproximateChildPointAndDataInterpolationInfo : public InterpolationInfo<T, PointD> {
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = (idx << Dim), end = (idx << Dim) | _iData[idx].size();
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return __iData(pointIdx).pointInfo;
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(__iData(pointIdx).pointInfo.position, __iData(pointIdx).data);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(__iData(pointIdx).pointInfo.position, __iData(pointIdx).data,
                               dValues);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(__iData(pointIdx).pointInfo.position, __iData(pointIdx).data,
                               dValues);
        }

        ApproximateChildPointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                                      SystemDual systemDual,
                                                      bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        static const unsigned int _Mask = (1 << Dim) - 1;
        SparseNodeData<DualPointAndDataInfoBrood<Dim, Real, Data, T, PointD>, ZeroUIntPack<Dim>>
                _iData;
        DualPointAndDataInfo<Dim, Real, Data, T, PointD>& __iData(size_t pointIdx) {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        const DualPointAndDataInfo<Dim, Real, Data, T, PointD>& __iData(size_t pointIdx) const {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <typename Data, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ApproximateChildPointAndDataInterpolationInfo<double,
                                                         Data,
                                                         PointD,
                                                         ConstraintDual,
                                                         SystemDual>
        : public InterpolationInfo<double, PointD> {
        typedef double T;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            node_index_type idx = _iData.index(node);
            if (idx == -1)
                begin = end = 0;
            else
                begin = (idx << Dim), end = (idx << Dim) | _iData[idx].size();
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return __iData(pointIdx).pointInfo;
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(__iData(pointIdx).pointInfo.position, __iData(pointIdx).data);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(__iData(pointIdx).pointInfo.position, __iData(pointIdx).data,
                               dValues);
        }

        ApproximateChildPointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                                      SystemDual systemDual,
                                                      bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        static const unsigned int _Mask = (1 << Dim) - 1;
        SparseNodeData<DualPointAndDataInfoBrood<Dim, Real, Data, T, PointD>, ZeroUIntPack<Dim>>
                _iData;
        DualPointAndDataInfo<Dim, Real, Data, T, PointD>& __iData(size_t pointIdx) {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        const DualPointAndDataInfo<Dim, Real, Data, T, PointD>& __iData(size_t pointIdx) const {
            return _iData[pointIdx >> Dim][pointIdx & _Mask];
        }
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ExactPointInterpolationInfo : public InterpolationInfo<T, PointD> {
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            begin = _sampleSpan[node->nodeData.nodeIndex].first,
            end = _sampleSpan[node->nodeData.nodeIndex].second;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx];
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].position);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].position, dValues);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].position, dValues);
        }

        ExactPointInterpolationInfo(ConstraintDual constraintDual,
                                    SystemDual systemDual,
                                    bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        void _init(const class FEMTree<Dim, Real>& tree,
                   const std::vector<PointSample>& samples,
                   bool noRescale);

        std::vector<std::pair<node_index_type, node_index_type>> _sampleSpan;
        std::vector<DualPointInfo<Dim, Real, T, PointD>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ExactPointInterpolationInfo<double, PointD, ConstraintDual, SystemDual>
        : public InterpolationInfo<double, PointD> {
        typedef double T;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            begin = _sampleSpan[node->nodeData.nodeIndex].first,
            end = _sampleSpan[node->nodeData.nodeIndex].second;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx];
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].position);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].position, dValues);
        }

        ExactPointInterpolationInfo(ConstraintDual constraintDual,
                                    SystemDual systemDual,
                                    bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        void _init(const class FEMTree<Dim, Real>& tree,
                   const std::vector<PointSample>& samples,
                   bool noRescale);

        std::vector<std::pair<node_index_type, node_index_type>> _sampleSpan;
        std::vector<DualPointInfo<Dim, Real, T, PointD>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    struct _ExactPointAndDataInterpolationInfo : public InterpolationInfo<T, PointD> {
        _ExactPointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                            SystemDual systemDual,
                                            bool constrainsDCTerm)
            : _constraintDual(constraintDual),
              _systemDual(systemDual),
              _constrainsDCTerm(constrainsDCTerm) {}

    protected:
        void _init(const class FEMTree<Dim, Real>& tree,
                   const std::vector<PointSample>& samples,
                   ConstPointer(Data) sampleData,
                   bool noRescale);

        std::vector<std::pair<node_index_type, node_index_type>> _sampleSpan;
        std::vector<DualPointAndDataInfo<Dim, Real, Data, T, PointD>> _iData;
        bool _constrainsDCTerm;
        ConstraintDual _constraintDual;
        SystemDual _systemDual;

        friend class FEMTree<Dim, Real>;
    };

    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    struct ExactPointAndDataInterpolationInfo
        : public _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual> {
        using _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>::
                _sampleSpan;
        using _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>::
                _constrainsDCTerm;
        using _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>::
                _iData;
        using _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>::
                _constraintDual;
        using _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>::
                _systemDual;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            begin = _sampleSpan[node->nodeData.nodeIndex].first,
            end = _sampleSpan[node->nodeData.nodeIndex].second;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, T, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx].pointInfo;
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].pointInfo.position, _iData[pointIdx].data);
        }
        Point<T, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<T, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].pointInfo.position, _iData[pointIdx].data, dValues);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].pointInfo.position, _iData[(int)pointIdx].data,
                               dValues);
        }

        ExactPointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                           SystemDual systemDual,
                                           bool constrainsDCTerm)
            : _ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>(
                      constraintDual, systemDual, constrainsDCTerm) {}
    };
    template <typename Data, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    struct ExactPointAndDataInterpolationInfo<double, Data, PointD, ConstraintDual, SystemDual>
        : public _ExactPointAndDataInterpolationInfo<double,
                                                     Data,
                                                     PointD,
                                                     ConstraintDual,
                                                     SystemDual> {
        using _ExactPointAndDataInterpolationInfo<double,
                                                  Data,
                                                  PointD,
                                                  ConstraintDual,
                                                  SystemDual>::_sampleSpan;
        using _ExactPointAndDataInterpolationInfo<double,
                                                  Data,
                                                  PointD,
                                                  ConstraintDual,
                                                  SystemDual>::_constrainsDCTerm;
        using _ExactPointAndDataInterpolationInfo<double,
                                                  Data,
                                                  PointD,
                                                  ConstraintDual,
                                                  SystemDual>::_iData;
        using _ExactPointAndDataInterpolationInfo<double,
                                                  Data,
                                                  PointD,
                                                  ConstraintDual,
                                                  SystemDual>::_constraintDual;
        using _ExactPointAndDataInterpolationInfo<double,
                                                  Data,
                                                  PointD,
                                                  ConstraintDual,
                                                  SystemDual>::_systemDual;
        void range(const FEMTreeNode* node, size_t& begin, size_t& end) const {
            begin = _sampleSpan[node->nodeData.nodeIndex].first,
            end = _sampleSpan[node->nodeData.nodeIndex].second;
        }
        bool constrainsDCTerm(void) const { return _constrainsDCTerm; }
        const DualPointInfo<Dim, Real, double, PointD>& operator[](size_t pointIdx) const {
            return _iData[pointIdx].pointInfo;
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(size_t pointIdx) const {
            return _constraintDual(_iData[pointIdx].pointInfo.position, _iData[pointIdx].data);
        }
        Point<double, CumulativeDerivatives<Dim, PointD>::Size> operator()(
                size_t pointIdx,
                const Point<double, CumulativeDerivatives<Dim, PointD>::Size>& dValues) const {
            return _systemDual(_iData[pointIdx].pointInfo.position, _iData[(int)pointIdx].data,
                               dValues);
        }
        ExactPointAndDataInterpolationInfo(ConstraintDual constraintDual,
                                           SystemDual systemDual,
                                           bool constrainsDCTerm)
            : _ExactPointAndDataInterpolationInfo<double, Data, PointD, ConstraintDual, SystemDual>(
                      constraintDual, systemDual, constrainsDCTerm) {}
    };

    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    static ApproximatePointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>*
    InitializeApproximatePointInterpolationInfo(const class FEMTree<Dim, Real>& tree,
                                                const std::vector<PointSample>& samples,
                                                ConstraintDual constraintDual,
                                                SystemDual systemDual,
                                                bool constrainsDCTerm,
                                                int adaptiveExponent) {
        ApproximatePointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>* a =
                new ApproximatePointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>(
                        constraintDual, systemDual, constrainsDCTerm);
        a->_iData = tree._densifyInterpolationInfoAndSetDualConstraints<T, PointD>(
                samples, constraintDual, adaptiveExponent);
        return a;
    }
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    static ApproximatePointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>*
    InitializeApproximatePointAndDataInterpolationInfo(const class FEMTree<Dim, Real>& tree,
                                                       const std::vector<PointSample>& samples,
                                                       ConstPointer(Data) sampleData,
                                                       ConstraintDual constraintDual,
                                                       SystemDual systemDual,
                                                       bool constrainsDCTerm,
                                                       int adaptiveExponent) {
        ApproximatePointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>* a =
                new ApproximatePointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual,
                                                             SystemDual>(constraintDual, systemDual,
                                                                         constrainsDCTerm);
        a->_iData = tree._densifyInterpolationInfoAndSetDualConstraints<T, Data, PointD>(
                samples, sampleData, constraintDual, adaptiveExponent);
        return a;
    }
    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    static ApproximateChildPointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>*
    InitializeApproximateChildPointInterpolationInfo(const class FEMTree<Dim, Real>& tree,
                                                     const std::vector<PointSample>& samples,
                                                     ConstraintDual constraintDual,
                                                     SystemDual systemDual,
                                                     bool constrainsDCTerm,
                                                     bool noRescale) {
        ApproximateChildPointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>* a =
                new ApproximateChildPointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>(
                        constraintDual, systemDual, constrainsDCTerm);
        a->_iData = tree._densifyChildInterpolationInfoAndSetDualConstraints<T, PointD>(
                samples, constraintDual, noRescale);
        return a;
    }
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    static ApproximateChildPointAndDataInterpolationInfo<T,
                                                         Data,
                                                         PointD,
                                                         ConstraintDual,
                                                         SystemDual>*
    InitializeApproximateChildPointAndDataInterpolationInfo(const class FEMTree<Dim, Real>& tree,
                                                            const std::vector<PointSample>& samples,
                                                            ConstPointer(Data) sampleData,
                                                            ConstraintDual constraintDual,
                                                            SystemDual systemDual,
                                                            bool constrainsDCTerm,
                                                            bool noRescale) {
        ApproximateChildPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>*
                a = new ApproximateChildPointAndDataInterpolationInfo<T, Data, PointD,
                                                                      ConstraintDual, SystemDual>(
                        constraintDual, systemDual, constrainsDCTerm);
        a->_iData = tree._densifyChildInterpolationInfoAndSetDualConstraints<T, Data, PointD>(
                samples, sampleData, constraintDual, noRescale);
        return a;
    }

    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    static ExactPointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>*
    InitializeExactPointInterpolationInfo(const class FEMTree<Dim, Real>& tree,
                                          const std::vector<PointSample>& samples,
                                          ConstraintDual constraintDual,
                                          SystemDual systemDual,
                                          bool constrainsDCTerm,
                                          bool noRescale) {
        ExactPointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>* e =
                new ExactPointInterpolationInfo<T, PointD, ConstraintDual, SystemDual>(
                        constraintDual, systemDual, constrainsDCTerm);
        e->_init(tree, samples, noRescale);
        return e;
    }
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    static ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>*
    InitializeExactPointAndDataInterpolationInfo(const class FEMTree<Dim, Real>& tree,
                                                 const std::vector<PointSample>& samples,
                                                 ConstPointer(Data) sampleData,
                                                 ConstraintDual constraintDual,
                                                 SystemDual systemDual,
                                                 bool constrainsDCTerm,
                                                 bool noRescale) {
        ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>* e =
                new ExactPointAndDataInterpolationInfo<T, Data, PointD, ConstraintDual, SystemDual>(
                        constraintDual, systemDual, constrainsDCTerm);
        e->_init(tree, samples, sampleData, noRescale);
        return e;
    }

    template <typename T, unsigned int PointD, typename ConstraintDual, typename SystemDual>
    friend struct ExactPointInterpolationInfo;
    template <typename T,
              typename Data,
              unsigned int PointD,
              typename ConstraintDual,
              typename SystemDual>
    friend struct ExactPointAndDataInterpolationInfo;

    template <typename T, unsigned int PointD, unsigned int... PointDs>
    static bool ConstrainsDCTerm(const InterpolationInfo<T, PointD>* iInfo,
                                 const InterpolationInfo<T, PointDs>*... iInfos) {
        return ConstrainsDCTerm(iInfo) || ConstrainsDCTerm(iInfos...);
    }
    template <typename T, unsigned int PointD>
    static bool ConstrainsDCTerm(const InterpolationInfo<T, PointD>* iInfo) {
        return iInfo && iInfo->constrainsDCTerm();
    }
    static bool ConstrainsDCTerm(void) { return false; }

#ifdef SHOW_WARNINGS
#pragma message("[WARNING] This should not be isotropic")
#endif  // SHOW_WARNINGS
    template <unsigned int DensityDegree>
    struct DensityEstimator
        : public SparseNodeData<
                  Real,
                  IsotropicUIntPack<Dim, FEMDegreeAndBType<DensityDegree>::Signature>> {
        DensityEstimator(int kernelDepth, int coDimension)
            : _kernelDepth(kernelDepth), _coDimension(coDimension) {
            ;
        }
        int coDimension(void) const { return _coDimension; }
        int kernelDepth(void) const { return _kernelDepth; }

    protected:
        int _kernelDepth, _coDimension;
    };

protected:
    bool _isValidSpaceNode(const FEMTreeNode* node) const {
        return !GetGhostFlag<Dim>(node) && (node->nodeData.flags & FEMTreeNodeData::SPACE_FLAG);
    }
    bool _isValidFEM1Node(const FEMTreeNode* node) const {
        return !GetGhostFlag<Dim>(node) && (node->nodeData.flags & FEMTreeNodeData::FEM_FLAG_1);
    }
    bool _isValidFEM2Node(const FEMTreeNode* node) const {
        return !GetGhostFlag<Dim>(node) && (node->nodeData.flags & FEMTreeNodeData::FEM_FLAG_2);
    }
    bool _isRefinableNode(const FEMTreeNode* node) const {
        return !GetGhostFlag<Dim>(node) && (node->nodeData.flags & FEMTreeNodeData::REFINABLE_FLAG);
    }

    FEMTreeNode* _tree;
    FEMTreeNode* _spaceRoot;
    SortedTreeNodes<Dim> _sNodes;
    LocalDepth _maxDepth;
    int _depthOffset;
    mutable unsigned int _femSigs1[Dim];
    mutable unsigned int _femSigs2[Dim];
    mutable unsigned int _refinableSigs[Dim];

    static bool _InBounds(Point<Real, Dim> p);
    int _localToGlobal(LocalDepth d) const { return d + _depthOffset; }
    LocalDepth _localDepth(const FEMTreeNode* node) const { return node->depth() - _depthOffset; }
    int _localInset(LocalDepth d) const {
        return _depthOffset <= 1 ? 0 : 1 << (d + _depthOffset - 1);
    }
    void _localDepthAndOffset(const FEMTreeNode* node, LocalDepth& d, LocalOffset& off) const {
        node->depthAndOffset(d, off);
        d -= _depthOffset;
        int inset = _localInset(d);
        for (int d = 0; d < Dim; d++) off[d] -= inset;
    }
    template <unsigned int FEMSig>
    static int _BSplineBegin(LocalDepth depth) {
        return BSplineEvaluationData<FEMSig>::Begin(depth);
    }
    template <unsigned int FEMSig>
    static int _BSplineEnd(LocalDepth depth) {
        return BSplineEvaluationData<FEMSig>::End(depth);
    }
    template <unsigned int... FEMSigs>
    bool _outOfBounds(UIntPack<FEMSigs...>, const FEMTreeNode* node) const {
        if (!node) return true;
        LocalDepth d;
        LocalOffset off;
        _localDepthAndOffset(node, d, off);
        return FEMIntegrator::IsOutOfBounds(UIntPack<FEMSigs...>(), d, off);
    }
    node_index_type _sNodesBegin(LocalDepth d) const { return _sNodes.begin(_localToGlobal(d)); }
    node_index_type _sNodesBegin(LocalDepth d, int slice) const {
        return _sNodes.begin(_localToGlobal(d), slice + _localInset(d));
    }
    node_index_type _sNodesBeginSlice(LocalDepth d) const { return _localInset(d); }
    node_index_type _sNodesEnd(LocalDepth d) const { return _sNodes.end(_localToGlobal(d)); }
    node_index_type _sNodesEnd(LocalDepth d, int slice) const {
        return _sNodes.end(_localToGlobal(d), slice + _localInset(d));
    }
    node_index_type _sNodesEndSlice(LocalDepth d) const {
        return (1 << _localToGlobal(d)) - _localInset(d) - 1;
    }
    size_t _sNodesSize(LocalDepth d) const { return _sNodes.size(_localToGlobal(d)); }
    size_t _sNodesSize(LocalDepth d, int slice) const {
        return _sNodes.size(_localToGlobal(d), slice + _localInset(d));
    }

    template <unsigned int FEMDegree>
    static bool _IsInteriorlySupported(LocalDepth depth, const LocalOffset off) {
        if (depth >= 0) {
            int begin, end;
            BSplineSupportSizes<FEMDegree>::InteriorSupportedSpan(depth, begin, end);
            bool interior = true;
            for (int dd = 0; dd < Dim; dd++) interior &= off[dd] >= begin && off[dd] < end;
            return interior;
        } else
            return false;
    }
    template <unsigned int FEMDegree>
    bool _isInteriorlySupported(const FEMTreeNode* node) const {
        if (!node) return false;
        LocalDepth d;
        LocalOffset off;
        _localDepthAndOffset(node, d, off);
        return _IsInteriorlySupported<FEMDegree>(d, off);
    }
    template <unsigned int... FEMDegrees>
    static bool _IsInteriorlySupported(UIntPack<FEMDegrees...>,
                                       LocalDepth depth,
                                       const LocalOffset off) {
        return BaseFEMIntegrator::IsInteriorlySupported(UIntPack<FEMDegrees...>(), depth, off);
    }
    template <unsigned int... FEMDegrees>
    bool _isInteriorlySupported(UIntPack<FEMDegrees...>, const FEMTreeNode* node) const {
        if (!node) return false;
        LocalDepth d;
        LocalOffset off;
        _localDepthAndOffset(node, d, off);
        return _IsInteriorlySupported<FEMDegrees...>(UIntPack<FEMDegrees...>(), d, off);
    }
    template <unsigned int FEMDegree1, unsigned int FEMDegree2>
    static bool _IsInteriorlyOverlapped(LocalDepth depth, const LocalOffset off) {
        if (depth >= 0) {
            int begin, end;
            BSplineIntegrationData<FEMDegreeAndBType<FEMDegree1, BOUNDARY_NEUMANN>::Signature,
                                   FEMDegreeAndBType<FEMDegree2, BOUNDARY_NEUMANN>::Signature>::
                    InteriorOverlappedSpan(depth, begin, end);
            bool interior = true;
            for (int dd = 0; dd < Dim; dd++) interior &= off[dd] >= begin && off[dd] < end;
            return interior;
        } else
            return false;
    }
    template <unsigned int FEMDegree1, unsigned int FEMDegree2>
    bool _isInteriorlyOverlapped(const FEMTreeNode* node) const {
        if (!node) return false;
        LocalDepth d;
        LocalOffset off;
        _localDepthAndOffset(node, d, off);
        return _IsInteriorlyOverlapped<FEMDegree1, FEMDegree2>(d, off);
    }
    template <unsigned int... FEMDegrees1, unsigned int... FEMDegrees2>
    static bool _IsInteriorlyOverlapped(UIntPack<FEMDegrees1...>,
                                        UIntPack<FEMDegrees2...>,
                                        LocalDepth depth,
                                        const LocalOffset off) {
        return BaseFEMIntegrator::IsInteriorlyOverlapped(UIntPack<FEMDegrees1...>(),
                                                         UIntPack<FEMDegrees2...>(), depth, off);
    }
    template <unsigned int... FEMDegrees1, unsigned int... FEMDegrees2>
    bool _isInteriorlyOverlapped(UIntPack<FEMDegrees1...>,
                                 UIntPack<FEMDegrees2...>,
                                 const FEMTreeNode* node) const {
        if (!node) return false;
        LocalDepth d;
        LocalOffset off;
        _localDepthAndOffset(node, d, off);
        return _IsInteriorlyOverlapped(UIntPack<FEMDegrees1...>(), UIntPack<FEMDegrees2...>(), d,
                                       off);
    }
    void _startAndWidth(const FEMTreeNode* node, Point<Real, Dim>& start, Real& width) const {
        LocalDepth d;
        LocalOffset off;
        _localDepthAndOffset(node, d, off);
        if (d >= 0)
            width = Real(1.0 / (1 << d));
        else
            width = Real(1.0 * (1 << (-d)));
        for (int dd = 0; dd < Dim; dd++) start[dd] = Real(off[dd]) * width;
    }
    void _centerAndWidth(const FEMTreeNode* node, Point<Real, Dim>& center, Real& width) const {
        int d, off[Dim];
        _localDepthAndOffset(node, d, off);
        width = Real(1.0 / (1 << d));
        for (int dd = 0; dd < Dim; dd++) center[dd] = Real(off[dd] + 0.5) * width;
    }
    int _childIndex(const FEMTreeNode* node, Point<Real, Dim> p) const {
        Point<Real, Dim> c;
        Real w;
        _centerAndWidth(node, c, w);
        int cIdx = 0;
        for (int d = 0; d < Dim; d++)
            if (p[d] >= c[d]) cIdx |= (1 << d);
        return cIdx;
    }

    template <bool ThreadSafe, unsigned int... Degrees>
    void _setFullDepth(UIntPack<Degrees...>,
                       Allocator<FEMTreeNode>* nodeAllocator,
                       FEMTreeNode* node,
                       LocalDepth depth);
    template <bool ThreadSafe, unsigned int... Degrees>
    void _setFullDepth(UIntPack<Degrees...>,
                       Allocator<FEMTreeNode>* nodeAllocator,
                       LocalDepth depth);
    template <unsigned int... Degrees>
    LocalDepth _getFullDepth(UIntPack<Degrees...>, const FEMTreeNode* node) const;

public:
    template <unsigned int... Degrees>
    LocalDepth getFullDepth(UIntPack<Degrees...>) const;

    LocalDepth depth(const FEMTreeNode* node) const { return _localDepth(node); }
    void depthAndOffset(const FEMTreeNode* node, LocalDepth& depth, LocalOffset& offset) const {
        _localDepthAndOffset(node, depth, offset);
    }

    size_t nodesSize(void) const { return _sNodes.size(); }
    node_index_type nodesBegin(LocalDepth d) const { return _sNodes.begin(_localToGlobal(d)); }
    node_index_type nodesEnd(LocalDepth d) const { return _sNodes.end(_localToGlobal(d)); }
    size_t nodesSize(LocalDepth d) const { return _sNodes.size(_localToGlobal(d)); }
    node_index_type nodesBegin(LocalDepth d, int slice) const {
        return _sNodes.begin(_localToGlobal(d), slice + _localInset(d));
    }
    node_index_type nodesEnd(LocalDepth d, int slice) const {
        return _sNodes.end(_localToGlobal(d), slice + _localInset(d));
    }
    size_t nodesSize(LocalDepth d, int slice) const {
        return _sNodes.size(_localToGlobal(d), slice + _localInset(d));
    }
    const FEMTreeNode* node(node_index_type idx) const { return _sNodes.treeNodes[idx]; }
    void centerAndWidth(node_index_type idx, Point<Real, Dim>& center, Real& width) const {
        _centerAndWidth(_sNodes.treeNodes[idx], center, width);
    }
    void startAndWidth(node_index_type idx, Point<Real, Dim>& center, Real& width) const {
        _startAndWidth(_sNodes.treeNodes[idx], center, width);
    }

protected:
    /////////////////////////////////////
    // System construction code        //
    // MultiGridFEMTreeData.System.inl //
    /////////////////////////////////////
public:
    template <unsigned int... FEMSigs>
    void setMultiColorIndices(UIntPack<FEMSigs...>,
                              int depth,
                              std::vector<std::vector<size_t>>& indices) const;

protected:
    template <unsigned int... FEMSigs>
    void _setMultiColorIndices(UIntPack<FEMSigs...>,
                               node_index_type start,
                               node_index_type end,
                               std::vector<std::vector<size_t>>& indices) const;

    struct _SolverStats {
        double constraintUpdateTime, systemTime, solveTime;
        double bNorm2, inRNorm2, outRNorm2;
    };
    template <unsigned int... FEMSigs, typename T, unsigned int PointD, unsigned int... PointDs>
    typename std::enable_if<(sizeof...(PointDs) != 0)>::type _addPointValues(
            UIntPack<FEMSigs...>,
            StaticWindow<
                    Real,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pointValues,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointD>* iInfo,
            const InterpolationInfo<T, PointDs>*... iInfos) const {
        _addPointValues(UIntPack<FEMSigs...>(), pointValues, neighbors, bsData, iInfo),
                _addPointValues(UIntPack<FEMSigs...>(), pointValues, neighbors, bsData, iInfos...);
    }
    template <unsigned int... FEMSigs>
    void _addPointValues(
            UIntPack<FEMSigs...>,
            StaticWindow<
                    Real,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pointValues,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData) const {}
    template <unsigned int... FEMSigs, typename T, unsigned int PointD>
    void _addPointValues(
            UIntPack<FEMSigs...>,
            StaticWindow<
                    Real,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pointValues,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointD>* interpolationInfo) const;

    template <unsigned int... FEMSigs, typename T, unsigned int PointD, unsigned int... PointDs>
    typename std::enable_if<(sizeof...(PointDs) > 1)>::type _addProlongedPointValues(
            UIntPack<FEMSigs...>,
            WindowSlice<
                    Real,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>
                    pointValues,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointD>* iInfo,
            const InterpolationInfo<T, PointDs>*... iInfos) const {
        _addProlongedPointValues(UIntPack<FEMSigs...>(), pointValues, neighbors, pNeighbors, bsData,
                                 iInfo),
                _addProlongedPointValues(UIntPack<FEMSigs...>(), pointValues, neighbors, pNeighbors,
                                         bsData, iInfos...);
    }
    template <unsigned int... FEMSigs>
    void _addProlongedPointValues(
            UIntPack<FEMSigs...>,
            WindowSlice<
                    Real,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>
                    pointValues,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData) const {}
    template <unsigned int... FEMSigs, typename T, unsigned int PointD>
    void _addProlongedPointValues(
            UIntPack<FEMSigs...>,
            WindowSlice<
                    Real,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>
                    pointValues,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointD>* iInfo) const;

    template <unsigned int... FEMSigs, typename T, unsigned int PointD, unsigned int... PointDs>
    typename std::enable_if<(sizeof...(PointDs) != 0)>::type _setPointValuesFromProlongedSolution(
            LocalDepth highDepth,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) prolongedSolution,
            InterpolationInfo<T, PointD>* iInfo,
            InterpolationInfo<T, PointDs>*... iInfos) const {
        _setPointValuesFromProlongedSolution(highDepth, bsData, prolongedSolution, iInfo),
                _setPointValuesFromProlongedSolution(highDepth, bsData, prolongedSolution,
                                                     iInfos...);
    }
    template <unsigned int... FEMSigs, typename T>
    void _setPointValuesFromProlongedSolution(
            LocalDepth highDepth,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) prolongedSolution) const {}
    template <unsigned int... FEMSigs, typename T, unsigned int PointD>
    void _setPointValuesFromProlongedSolution(
            LocalDepth highDepth,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) prolongedSolution,
            InterpolationInfo<T, PointD>* interpolationInfo) const;

    template <unsigned int... FEMSigs, typename T, unsigned int PointD, unsigned int... PointDs>
    typename std::enable_if<(sizeof...(PointDs) != 0), T>::type
    _getInterpolationConstraintFromProlongedSolution(
            const typename FEMTreeNode::template ConstNeighbors<UIntPack<
                    BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>& neighbors,
            const FEMTreeNode* node,
            ConstPointer(T) prolongedSolution,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointD>* iInfo,
            const InterpolationInfo<T, PointDs>*... iInfos) const {
        return _getInterpolationConstraintFromProlongedSolution(neighbors, node, prolongedSolution,
                                                                bsData, iInfo) +
               _getInterpolationConstraintFromProlongedSolution(neighbors, node, prolongedSolution,
                                                                bsData, iInfos...);
    }
    template <unsigned int... FEMSigs, typename T>
    T _getInterpolationConstraintFromProlongedSolution(
            const typename FEMTreeNode::template ConstNeighbors<UIntPack<
                    BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>& neighbors,
            const FEMTreeNode* node,
            ConstPointer(T) prolongedSolution,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData) const {
        return T();
    }
    template <unsigned int... FEMSigs, typename T, unsigned int PointD>
    T _getInterpolationConstraintFromProlongedSolution(
            const typename FEMTreeNode::template ConstNeighbors<UIntPack<
                    BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>& neighbors,
            const FEMTreeNode* node,
            ConstPointer(T) prolongedSolution,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointD>* iInfo) const;

    template <unsigned int... FEMSigs, typename T, unsigned int PointD, unsigned int... PointDs>
    typename std::enable_if<(sizeof...(PointDs) != 0)>::type
    _updateRestrictedInterpolationConstraints(
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth highDepth,
            ConstPointer(T) solution,
            Pointer(T) cumulativeConstraints,
            const InterpolationInfo<T, PointD>* iInfo,
            const InterpolationInfo<T, PointDs>*... iInfos) const {
        _updateRestrictedInterpolationConstraints(bsData, highDepth, solution,
                                                  cumulativeConstraints, iInfo),
                _updateRestrictedInterpolationConstraints(bsData, highDepth, solution,
                                                          cumulativeConstraints, iInfos...);
    }
    template <unsigned int... FEMSigs, typename T>
    void _updateRestrictedInterpolationConstraints(
            PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth highDepth,
            ConstPointer(T) solution,
            Pointer(T) cumulativeConstraints) const {
        ;
    }
    template <unsigned int... FEMSigs, typename T, unsigned int PointD>
    void _updateRestrictedInterpolationConstraints(
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth highDepth,
            ConstPointer(T) solution,
            Pointer(T) cumulativeConstraints,
            const InterpolationInfo<T, PointD>* interpolationInfo) const;

    template <unsigned int FEMDegree1, unsigned int FEMDegree2>
    static void _SetParentOverlapBounds(const FEMTreeNode* node, int start[Dim], int end[Dim]);
    template <unsigned int FEMDegree1, unsigned int FEMDegree2>
    static void _SetParentOverlapBounds(int cIdx, int start[Dim], int end[Dim]);
    template <unsigned int... FEMDegrees1, unsigned int... FEMDegrees2>
    static void _SetParentOverlapBounds(UIntPack<FEMDegrees1...>,
                                        UIntPack<FEMDegrees2...>,
                                        const FEMTreeNode* node,
                                        int start[Dim],
                                        int end[Dim]) {
        if (node) {
            int d, off[Dim];
            node->depthAndOffset(d, off);
            BaseFEMIntegrator::template ParentOverlapBounds(
                    UIntPack<FEMDegrees1...>(), UIntPack<FEMDegrees2...>(), d, off, start, end);
        }
    }
    template <unsigned int... FEMDegrees1, unsigned int... FEMDegrees2>
    static void _SetParentOverlapBounds(UIntPack<FEMDegrees1...>,
                                        UIntPack<FEMDegrees2...>,
                                        int cIdx,
                                        int start[Dim],
                                        int end[Dim]) {
        BaseFEMIntegrator::template ParentOverlapBounds(
                UIntPack<FEMDegrees1...>(), UIntPack<FEMDegrees2...>(), cIdx, start, end);
    }

    template <unsigned int... FEMSigs>
    int _getProlongedMatrixRowSize(
            const FEMTreeNode* node,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors) const;
    // #if defined( __GNUC__ ) && __GNUC__ < 5
    // 	#warning "you've got me gcc version<5"
    // 		template< unsigned int ... FEMSigs >
    // 	int _getMatrixRowSize( UIntPack< FEMSigs ... > , const typename FEMTreeNode::template
    // ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize
    // ... > >& neighbors ) const; #else // !__GNUC__ || __GNUC__ >=5
    template <unsigned int... FEMSigs>
    int _getMatrixRowSize(
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors) const;
    // #endif // __GNUC__ || __GNUC__ < 4
    template <typename T, unsigned int... PointDs, unsigned int... FEMSigs>
    T _setMatrixRowAndGetConstraintFromProlongation(
            UIntPack<FEMSigs...>,
            const BaseSystem<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            size_t idx,
            SparseMatrix<Real,
                         matrix_index_type,
                         WindowSize<UIntPack<BSplineOverlapSizes<
                                 FEMSignature<FEMSigs>::Degree>::OverlapSize...>>::Size>& M,
            node_index_type offset,
            const PCStencils<UIntPack<FEMSignature<FEMSigs>::Degree...>>& pcStencils,
            const CCStencil<UIntPack<FEMSignature<FEMSigs>::Degree...>>& ccStencil,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) prolongedSolution,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <typename T, unsigned int... PointDs, unsigned int... FEMSigs>
    T _setMatrixRowAndGetConstraintFromProlongation(
            UIntPack<FEMSigs...>,
            const BaseSystem<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            Pointer(MatrixEntry<Real, matrix_index_type>) row,
            node_index_type offset,
            const PCStencils<UIntPack<FEMSignature<FEMSigs>::Degree...>>& pcStencils,
            const CCStencil<UIntPack<FEMSignature<FEMSigs>::Degree...>>& ccStencil,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) prolongedSolution,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <typename T, unsigned int... PointDs, unsigned int... FEMSigs>
    int _setProlongedMatrixRow(
            const typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            Pointer(MatrixEntry<Real, matrix_index_type>) row,
            node_index_type offset,
            const DynamicWindow<
                    double,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    stencil,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;

    // Updates the constraints @(depth) based on the solution coefficients @(depth-1)
    template <unsigned int... FEMSigs, typename T, unsigned int... PointDs>
    T _getConstraintFromProlongedSolution(
            UIntPack<FEMSigs...>,
            const BaseSystem<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    neighbors,
            const typename FEMTreeNode::template ConstNeighbors<
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    pNeighbors,
            const FEMTreeNode* node,
            ConstPointer(T) prolongedSolution,
            const DynamicWindow<
                    double,
                    UIntPack<BSplineOverlapSizes<FEMSignature<FEMSigs>::Degree>::OverlapSize...>>&
                    stencil,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;

    template <unsigned int... FEMSigs,
              typename T,
              typename TDotT,
              typename SORWeights,
              unsigned int... PointDs>
    int _solveFullSystemGS(
            UIntPack<FEMSigs...>,
            const typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth depth,
            Pointer(T) solution,
            ConstPointer(T) prolongedSolution,
            ConstPointer(T) constraints,
            TDotT Dot,
            int iters,
            bool coarseToFine,
            SORWeights sorWeights,
            _SolverStats& stats,
            bool computeNorms,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <unsigned int... FEMSigs,
              typename T,
              typename TDotT,
              typename SORWeights,
              unsigned int... PointDs>
    int _solveSlicedSystemGS(
            UIntPack<FEMSigs...>,
            const typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth depth,
            Pointer(T) solution,
            ConstPointer(T) prolongedSolution,
            ConstPointer(T) constraints,
            TDotT Dot,
            int iters,
            bool coarseToFine,
            unsigned int sliceBlockSize,
            SORWeights sorWeights,
            _SolverStats& stats,
            bool computeNorms,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <unsigned int... FEMSigs,
              typename T,
              typename TDotT,
              typename SORWeights,
              unsigned int... PointDs>
    int _solveSystemGS(
            UIntPack<FEMSigs...>,
            bool sliced,
            const typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth depth,
            Pointer(T) solution,
            ConstPointer(T) prolongedSolution,
            ConstPointer(T) constraints,
            TDotT Dot,
            int iters,
            bool coarseToFine,
            unsigned int sliceBlockSize,
            SORWeights sorWeights,
            _SolverStats& stats,
            bool computeNorms,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const {
        if (sliced)
            return _solveSlicedSystemGS(UIntPack<FEMSigs...>(), F, bsData, depth, solution,
                                        prolongedSolution, constraints, Dot, iters, coarseToFine,
                                        sliceBlockSize, sorWeights, stats, computeNorms,
                                        interpolationInfo...);
        else
            return _solveFullSystemGS(UIntPack<FEMSigs...>(), F, bsData, depth, solution,
                                      prolongedSolution, constraints, Dot, iters, coarseToFine,
                                      sorWeights, stats, computeNorms, interpolationInfo...);
    }
    template <unsigned int... FEMSigs, typename T, typename TDotT, unsigned int... PointDs>
    int _solveSystemCG(
            UIntPack<FEMSigs...>,
            const typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth depth,
            Pointer(T) solution,
            ConstPointer(T) prolongedSolution,
            ConstPointer(T) constraints,
            TDotT Dot,
            int iters,
            bool coarseToFine,
            _SolverStats& stats,
            bool computeNorms,
            double cgAccuracy,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <unsigned int... FEMSigs, typename T, typename TDotT, unsigned int... PointDs>
    void _solveRegularMG(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth depth,
            Pointer(T) solution,
            ConstPointer(T) constraints,
            TDotT Dot,
            int vCycles,
            int iters,
            _SolverStats& stats,
            bool computeNorms,
            double cgAccuracy,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;

    // Updates the cumulative integral constraints @(depth-1) based on the change in solution
    // coefficients @(depth)
    template <unsigned int... FEMSigs, typename T>
    void _updateRestrictedIntegralConstraints(
            UIntPack<FEMSigs...>,
            const typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            LocalDepth highDepth,
            ConstPointer(T) solution,
            Pointer(T) cumulativeConstraints) const;

    template <unsigned int PointD, typename T, unsigned int... FEMSigs>
    CumulativeDerivativeValues<T, Dim, PointD> _coarserFunctionValues(
            UIntPack<FEMSigs...>,
            Point<Real, Dim> p,
            const ConstPointSupportKey<UIntPack<FEMSignature<FEMSigs>::Degree...>>& neighborKey,
            const FEMTreeNode* node,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) coefficients) const;
    template <unsigned int PointD, typename T, unsigned int... FEMSigs>
    CumulativeDerivativeValues<T, Dim, PointD> _finerFunctionValues(
            UIntPack<FEMSigs...>,
            Point<Real, Dim> p,
            const ConstPointSupportKey<UIntPack<FEMSignature<FEMSigs>::Degree...>>& neighborKey,
            const FEMTreeNode* node,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            ConstPointer(T) coefficients) const;

    template <unsigned int... FEMSigs, typename T, unsigned int... PointDs>
    int _getSliceMatrixAndProlongationConstraints(
            UIntPack<FEMSigs...>,
            const BaseSystem<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            SparseMatrix<Real,
                         matrix_index_type,
                         WindowSize<UIntPack<BSplineOverlapSizes<
                                 FEMSignature<FEMSigs>::Degree>::OverlapSize...>>::Size>& matrix,
            Pointer(Real) diagonalR,
            const PointEvaluator<UIntPack<FEMSigs...>, UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    bsData,
            LocalDepth depth,
            node_index_type nBegin,
            node_index_type nEnd,
            ConstPointer(T) prolongedSolution,
            Pointer(T) constraints,
            const CCStencil<UIntPack<FEMSignature<FEMSigs>::Degree...>>& ccStencil,
            const PCStencils<UIntPack<FEMSignature<FEMSigs>::Degree...>>& pcStencils,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;

    // Down samples constraints @(depth) to constraints @(depth-1)
    template <class C, unsigned... Degrees, unsigned int... FEMSigs>
    void _downSample(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::template RestrictionProlongation<UIntPack<Degrees...>>& RP,
            LocalDepth highDepth,
            Pointer(C) constraints) const;
    // Up samples coefficients @(depth-1) to coefficients @(depth)
    template <class C, unsigned... Degrees, unsigned int... FEMSigs>
    void _upSample(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::template RestrictionProlongation<UIntPack<Degrees...>>& RP,
            LocalDepth highDepth,
            Pointer(C) coefficients) const;

    template <bool XMajor, class C, unsigned int... FEMSigs>
    static void _RegularGridUpSample(UIntPack<FEMSigs...>,
                                     LocalDepth highDepth,
                                     ConstPointer(C) lowCoefficients,
                                     Pointer(C) highCoefficients);
    template <bool XMajor, class C, unsigned int... FEMSigs>
    static void _RegularGridUpSample(UIntPack<FEMSigs...>,
                                     const int lowBegin[],
                                     const int lowEnd[],
                                     const int highBegin[],
                                     const int highEnd[],
                                     LocalDepth highDepth,
                                     ConstPointer(C) lowCoefficients,
                                     Pointer(C) highCoefficients);

public:
    template <class C, unsigned int... FEMSigs>
    DenseNodeData<C, UIntPack<FEMSigs...>> coarseCoefficients(
            const DenseNodeData<C, UIntPack<FEMSigs...>>& coefficients) const;
    template <class C, unsigned int... FEMSigs>
    DenseNodeData<C, UIntPack<FEMSigs...>> coarseCoefficients(
            const SparseNodeData<C, UIntPack<FEMSigs...>>& coefficients) const;

    // For each (valid) fem node, compute the ratio of the sum of active prolongation weights to the
    // sum of total prolongation weights If the prolongToChildren flag is set, then these weights
    // are pushed to the children by computing the ratio of the prolongation of the above weights to
    // the prolongation of unity weights

    template <unsigned int... FEMSigs>
    DenseNodeData<Real, UIntPack<FEMSigs...>> supportWeights(UIntPack<FEMSigs...>) const;
    template <unsigned int... FEMSigs>
    DenseNodeData<Real, UIntPack<FEMSigs...>> prolongationWeights(UIntPack<FEMSigs...>,
                                                                  bool prolongToChildren) const;

protected:
    //////////////////////////////////////////////
    // Code for splatting point-sample data     //
    // MultiGridFEMTreeData.WeightedSamples.inl //
    //////////////////////////////////////////////
    template <bool ThreadSafe, unsigned int WeightDegree>
    void _addWeightContribution(Allocator<FEMTreeNode>* nodeAllocator,
                                DensityEstimator<WeightDegree>& densityWeights,
                                FEMTreeNode* node,
                                Point<Real, Dim> position,
                                PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
                                Real weight = Real(1.0));
    template <unsigned int WeightDegree, class PointSupportKey>
    Real _getSamplesPerNode(const DensityEstimator<WeightDegree>& densityWeights,
                            const FEMTreeNode* node,
                            Point<Real, Dim> position,
                            PointSupportKey& weightKey) const;
    template <unsigned int WeightDegree, class WeightKey>
    void _getSampleDepthAndWeight(const DensityEstimator<WeightDegree>& densityWeights,
                                  const FEMTreeNode* node,
                                  Point<Real, Dim> position,
                                  WeightKey& weightKey,
                                  Real& depth,
                                  Real& weight) const;
    template <unsigned int WeightDegree, class WeightKey>
    void _getSampleDepthAndWeight(const DensityEstimator<WeightDegree>& densityWeights,
                                  Point<Real, Dim> position,
                                  WeightKey& weightKey,
                                  Real& depth,
                                  Real& weight) const;

    template <bool CreateNodes, bool ThreadSafe, class V, unsigned int... DataSigs>
    void _splatPointData(Allocator<FEMTreeNode>* nodeAllocator,
                         FEMTreeNode* node,
                         Point<Real, Dim> point,
                         V v,
                         SparseNodeData<V, UIntPack<DataSigs...>>& data,
                         PointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey);
    template <bool CreateNodes,
              bool ThreadSafe,
              unsigned int WeightDegree,
              class V,
              unsigned int... DataSigs>
    Real _splatPointData(Allocator<FEMTreeNode>* nodeAllocator,
                         const DensityEstimator<WeightDegree>& densityWeights,
                         Point<Real, Dim> point,
                         V v,
                         SparseNodeData<V, UIntPack<DataSigs...>>& data,
                         PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
                         PointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey,
                         LocalDepth minDepth,
                         LocalDepth maxDepth,
                         int dim,
                         Real depthBias);
    template <bool CreateNodes,
              bool ThreadSafe,
              unsigned int WeightDegree,
              class V,
              unsigned int... DataSigs>
    Real _multiSplatPointData(Allocator<FEMTreeNode>* nodeAllocator,
                              const DensityEstimator<WeightDegree>* densityWeights,
                              FEMTreeNode* node,
                              Point<Real, Dim> point,
                              V v,
                              SparseNodeData<V, UIntPack<DataSigs...>>& data,
                              PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
                              PointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey,
                              int dim);
    template <unsigned int WeightDegree, class V, unsigned int... DataSigs>
    Real _nearestMultiSplatPointData(
            const DensityEstimator<WeightDegree>* densityWeights,
            FEMTreeNode* node,
            Point<Real, Dim> point,
            V v,
            SparseNodeData<V, UIntPack<DataSigs...>>& data,
            PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
            int dim = Dim);
    template <class V, class Coefficients, unsigned int D, unsigned int... DataSigs>
    V _evaluate(
            const Coefficients& coefficients,
            Point<Real, Dim> p,
            const PointEvaluator<UIntPack<DataSigs...>, IsotropicUIntPack<Dim, D>>& pointEvaluator,
            const ConstPointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey) const;

public:
    template <bool XMajor, class V, unsigned int... DataSigs>
    Pointer(V) regularGridEvaluate(const DenseNodeData<V, UIntPack<DataSigs...>>& coefficients,
                                   int& res,
                                   LocalDepth depth = -1,
                                   bool primal = false) const;
    template <bool XMajor, class V, unsigned int... DataSigs>
    Pointer(V) regularGridUpSample(const DenseNodeData<V, UIntPack<DataSigs...>>& coefficients,
                                   LocalDepth depth = -1) const;
    template <bool XMajor, class V, unsigned int... DataSigs>
    Pointer(V) regularGridUpSample(const DenseNodeData<V, UIntPack<DataSigs...>>& coefficients,
                                   const int begin[Dim],
                                   const int end[Dim],
                                   LocalDepth depth = -1) const;
    template <class V, unsigned int... DataSigs>
    V average(const DenseNodeData<V, UIntPack<DataSigs...>>& coefficients) const;
    template <class V, unsigned int... DataSigs>
    V average(const DenseNodeData<V, UIntPack<DataSigs...>>& coefficients,
              const Real begin[Dim],
              const Real end[Dim]) const;
    template <typename T>
    struct HasNormalDataFunctor {};
    template <unsigned int... NormalSigs>
    struct HasNormalDataFunctor<UIntPack<NormalSigs...>> {
        const SparseNodeData<Point<Real, Dim>, UIntPack<NormalSigs...>>& normalInfo;
        HasNormalDataFunctor(const SparseNodeData<Point<Real, Dim>, UIntPack<NormalSigs...>>& ni)
            : normalInfo(ni) {
            ;
        }
        bool operator()(const FEMTreeNode* node) const {
            const Point<Real, Dim>* n = normalInfo(node);
            if (n) {
                const Point<Real, Dim>& normal = *n;
                for (int d = 0; d < Dim; d++)
                    if (normal[d] != 0) return true;
            }
            if (node->children)
                for (int c = 0; c < (1 << Dim); c++)
                    if ((*this)(node->children + c)) return true;
            return false;
        }
    };
    struct TrivialHasDataFunctor {
        bool operator()(const FEMTreeNode* node) const { return true; }
    };

protected:
    // [NOTE] The input/output for this method is pre-scaled by weight
    template <typename T>
    bool _setInterpolationInfoFromChildren(
            FEMTreeNode* node,
            SparseNodeData<T, IsotropicUIntPack<Dim, FEMTrivialSignature>>& iInfo) const;
    template <typename T, unsigned int PointD, typename ConstraintDual>
    SparseNodeData<DualPointInfo<Dim, Real, T, PointD>, IsotropicUIntPack<Dim, FEMTrivialSignature>>
    _densifyInterpolationInfoAndSetDualConstraints(const std::vector<PointSample>& samples,
                                                   ConstraintDual constraintDual,
                                                   int adaptiveExponent) const;
    template <typename T, typename Data, unsigned int PointD, typename ConstraintDual>
    SparseNodeData<DualPointAndDataInfo<Dim, Real, Data, T, PointD>,
                   IsotropicUIntPack<Dim, FEMTrivialSignature>>
    _densifyInterpolationInfoAndSetDualConstraints(const std::vector<PointSample>& samples,
                                                   ConstPointer(Data) sampleData,
                                                   ConstraintDual constraintDual,
                                                   int adaptiveExponent) const;
    template <typename T, unsigned int PointD, typename ConstraintDual>
    SparseNodeData<DualPointInfoBrood<Dim, Real, T, PointD>,
                   IsotropicUIntPack<Dim, FEMTrivialSignature>>
    _densifyChildInterpolationInfoAndSetDualConstraints(const std::vector<PointSample>& samples,
                                                        ConstraintDual constraintDual,
                                                        bool noRescale) const;
    template <typename T, typename Data, unsigned int PointD, typename ConstraintDual>
    SparseNodeData<DualPointAndDataInfoBrood<Dim, Real, Data, T, PointD>,
                   IsotropicUIntPack<Dim, FEMTrivialSignature>>
    _densifyChildInterpolationInfoAndSetDualConstraints(const std::vector<PointSample>& samples,
                                                        ConstPointer(Data) sampleData,
                                                        ConstraintDual constraintDual,
                                                        bool noRescale) const;

    void _setSpaceValidityFlags(void) const;
    template <unsigned int... FEMSigs>
    void _setRefinabilityFlags(UIntPack<FEMSigs...>) const;
    template <unsigned int... FEMSigs1>
    void _setFEM1ValidityFlags(UIntPack<FEMSigs1...>) const;
    template <unsigned int... FEMSigs2>
    void _setFEM2ValidityFlags(UIntPack<FEMSigs2...>) const;
    template <class HasDataFunctor>
    void _clipTree(const HasDataFunctor& f, LocalDepth fullDepth);

public:
    template <unsigned int PointD, unsigned int... FEMSigs>
    SparseNodeData<CumulativeDerivativeValues<Real, Dim, PointD>,
                   IsotropicUIntPack<Dim, FEMTrivialSignature>>
    leafValues(const DenseNodeData<Real, UIntPack<FEMSigs...>>& coefficients,
               int maxDepth = -1) const;

protected:
    /////////////////////////////////////
    // Evaluation Methods              //
    // MultiGridFEMTreeData.Evaluation //
    /////////////////////////////////////
    static const unsigned int CHILDREN = 1 << Dim;
    template <typename Pack, unsigned int PointD>
    struct _Evaluator {};
    template <unsigned int... FEMSigs, unsigned int PointD>
    struct _Evaluator<UIntPack<FEMSigs...>, PointD> {
        static_assert(Dim == sizeof...(FEMSigs),
                      "[ERROR] Number of signatures doesn't match dimension");

        typedef DynamicWindow<
                CumulativeDerivativeValues<double, Dim, PointD>,
                UIntPack<BSplineSupportSizes<FEMSignature<FEMSigs>::Degree>::SupportSize...>>
                CenterStencil;
        typedef DynamicWindow<
                CumulativeDerivativeValues<double, Dim, PointD>,
                UIntPack<BSplineSupportSizes<FEMSignature<FEMSigs>::Degree>::SupportSize...>>
                CornerStencil;
        typedef DynamicWindow<
                CumulativeDerivativeValues<double, Dim, PointD>,
                UIntPack<(BSplineSupportSizes<FEMSignature<FEMSigs>::Degree>::BCornerSize + 1)...>>
                BCornerStencil;

        typedef std::tuple<typename BSplineEvaluationData<FEMSigs>::template Evaluator<PointD>...>
                Evaluators;
        typedef std::tuple<
                typename BSplineEvaluationData<FEMSigs>::template ChildEvaluator<PointD>...>
                ChildEvaluators;
        struct StencilData {
            CenterStencil ccCenterStencil, pcCenterStencils[CHILDREN];
            CornerStencil ccCornerStencil[CHILDREN], pcCornerStencils[CHILDREN][CHILDREN];
            BCornerStencil ccBCornerStencil[CHILDREN], pcBCornerStencils[CHILDREN][CHILDREN];
        };
        Pointer(StencilData) stencilData;
        Pointer(Evaluators) evaluators;
        Pointer(ChildEvaluators) childEvaluators;

        void set(LocalDepth depth);
        _Evaluator(void) {
            _pointEvaluator = NULL;
            stencilData = NullPointer(StencilData), evaluators = NullPointer(Evaluators),
            childEvaluators = NullPointer(ChildEvaluators);
        }
        ~_Evaluator(void) {
            if (_pointEvaluator) delete _pointEvaluator, _pointEvaluator = NULL;
            if (stencilData) DeletePointer(stencilData);
            if (evaluators) DeletePointer(evaluators);
            if (childEvaluators) DeletePointer(childEvaluators);
        }

    protected:
        enum _CenterOffset { CENTER = -1, BACK = 0, FRONT = 1 };
        template <unsigned int _PointD = PointD>
        CumulativeDerivativeValues<double, Dim, _PointD> _values(unsigned int d,
                                                                 const int fIdx[Dim],
                                                                 const int idx[Dim],
                                                                 const _CenterOffset off[Dim],
                                                                 bool parentChild) const;
        template <unsigned int _PointD = PointD>
        CumulativeDerivativeValues<double, Dim, _PointD> _centerValues(unsigned int d,
                                                                       const int fIdx[Dim],
                                                                       const int idx[Dim],
                                                                       bool parentChild) const;
        template <unsigned int _PointD = PointD>
        CumulativeDerivativeValues<double, Dim, _PointD> _cornerValues(unsigned int d,
                                                                       const int fIdx[Dim],
                                                                       const int idx[Dim],
                                                                       int corner,
                                                                       bool parentChild) const;
        template <unsigned int _PointD = PointD, unsigned int I = 0>
        typename std::enable_if<I == Dim>::type _setDValues(unsigned int d,
                                                            const int fIdx[],
                                                            const int cIdx[],
                                                            const _CenterOffset off[],
                                                            bool pc,
                                                            double dValues[][_PointD + 1]) const {}
        template <unsigned int _PointD = PointD, unsigned int I = 0>
                typename std::enable_if <
                I<Dim>::type _setDValues(unsigned int d,
                                         const int fIdx[],
                                         const int cIdx[],
                                         const _CenterOffset off[],
                                         bool pc,
                                         double dValues[][_PointD + 1]) const {
            if (pc)
                for (int dd = 0; dd <= _PointD; dd++)
                    dValues[I][dd] = off[I] == CENTER
                                             ? std::get<I>(childEvaluators[d])
                                                       .centerValue(fIdx[I], cIdx[I], dd)
                                             : std::get<I>(childEvaluators[d])
                                                       .cornerValue(fIdx[I], cIdx[I] + off[I], dd);
            else
                for (int dd = 0; dd <= _PointD; dd++)
                    dValues[I][dd] =
                            off[I] == CENTER
                                    ? std::get<I>(evaluators[d]).centerValue(fIdx[I], cIdx[I], dd)
                                    : std::get<I>(evaluators[d])
                                              .cornerValue(fIdx[I], cIdx[I] + off[I], dd);
            _setDValues<_PointD, I + 1>(d, fIdx, cIdx, off, pc, dValues);
        }

        template <unsigned int I = 0>
        typename std::enable_if<I == Dim>::type _setEvaluators(unsigned int maxDepth) {}
        template <unsigned int I = 0>
                typename std::enable_if < I<Dim>::type _setEvaluators(unsigned int maxDepth) {
            static const unsigned int FEMSig = UIntPack<FEMSigs...>::template Get<I>();
            for (unsigned int d = 0; d <= maxDepth; d++)
                BSplineEvaluationData<FEMSig>::SetEvaluator(std::template get<I>(evaluators[d]), d);
            for (unsigned int d = 1; d <= maxDepth; d++)
                BSplineEvaluationData<FEMSig>::SetChildEvaluator(
                        std::template get<I>(childEvaluators[d]), d - 1);
            _setEvaluators<I + 1>(maxDepth);
        }
        typename FEMIntegrator::template PointEvaluator<UIntPack<FEMSigs...>,
                                                        IsotropicUIntPack<Dim, PointD>>*
                _pointEvaluator;
        friend FEMTree;
    };
    template <class V, unsigned int _PointD, unsigned int... FEMSigs, unsigned int PointD>
    CumulativeDerivativeValues<V, Dim, _PointD> _getCenterValues(
            const ConstPointSupportKey<UIntPack<FEMSignature<FEMSigs>::Degree...>>& neighborKey,
            const FEMTreeNode* node,
            ConstPointer(V) solution,
            ConstPointer(V) coarseSolution,
            const _Evaluator<UIntPack<FEMSigs...>, PointD>& evaluator,
            int maxDepth,
            bool isInterior) const;
    template <class V, unsigned int _PointD, unsigned int... FEMSigs, unsigned int PointD>
    CumulativeDerivativeValues<V, Dim, _PointD> _getCornerValues(
            const ConstPointSupportKey<UIntPack<FEMSignature<FEMSigs>::Degree...>>& neighborKey,
            const FEMTreeNode* node,
            int corner,
            ConstPointer(V) solution,
            ConstPointer(V) coarseSolution,
            const _Evaluator<UIntPack<FEMSigs...>, PointD>& evaluator,
            int maxDepth,
            bool isInterior) const;
    template <class V, unsigned int _PointD, unsigned int... FEMSigs, unsigned int PointD>
    CumulativeDerivativeValues<V, Dim, _PointD> _getValues(
            const ConstPointSupportKey<UIntPack<FEMSignature<FEMSigs>::Degree...>>& neighborKey,
            const FEMTreeNode* node,
            Point<Real, Dim> p,
            ConstPointer(V) solution,
            ConstPointer(V) coarseSolution,
            const _Evaluator<UIntPack<FEMSigs...>, PointD>& evaluator,
            int maxDepth) const;
    template <class V, unsigned int _PointD, unsigned int... FEMSigs, unsigned int PointD>
    CumulativeDerivativeValues<V, Dim, _PointD> _getCornerValues(
            const ConstCornerSupportKey<UIntPack<FEMSignature<FEMSigs>::Degree...>>& neighborKey,
            const FEMTreeNode* node,
            int corner,
            ConstPointer(V) solution,
            ConstPointer(V) coarseSolution,
            const _Evaluator<UIntPack<FEMSigs...>, PointD>& evaluator,
            int maxDepth,
            bool isInterior) const;
    template <unsigned int... SupportSizes>
    struct CornerLoopData {
        typedef UIntPack<SupportSizes...> _SupportSizes;
        //		static const unsigned int supportSizes[] = { SupportSizes ... };
        static const unsigned int supportSizes[];
        unsigned int ccSize[1 << Dim], pcSize[1 << Dim][1 << Dim];
        unsigned int ccIndices[1 << Dim][WindowSize<_SupportSizes>::Size];
        unsigned int pcIndices[1 << Dim][1 << Dim][WindowSize<_SupportSizes>::Size];
        CornerLoopData(void) {
            int start[Dim], end[Dim], _start[Dim], _end[Dim];
            for (int c = 0; c < (1 << Dim); c++) {
                ccSize[c] = 0;
                for (int dd = 0; dd < Dim; dd++) {
                    start[dd] = 0, end[dd] = supportSizes[dd];
                    if ((c >> dd) & 1)
                        start[dd]++;
                    else
                        end[dd]--;
                }
                unsigned int idx[Dim];
                WindowLoop<Dim>::Run(start, end, [&](int d, int i) { idx[d] = i; },
                                     [&](void) {
                                         ccIndices[c][ccSize[c]++] =
                                                 GetWindowIndex(_SupportSizes(), idx);
                                     });

                for (int _c = 0; _c < (1 << Dim); _c++) {
                    pcSize[c][_c] = 0;
                    for (int dd = 0; dd < Dim; dd++) {
                        if (((_c >> dd) & 1) != ((c >> dd) & 1))
                            _start[dd] = 0, _end[dd] = supportSizes[dd];
                        else
                            _start[dd] = start[dd], _end[dd] = end[dd];
                    }

                    unsigned int idx[Dim];
                    WindowLoop<Dim>::Run(_start, _end, [&](int d, int i) { idx[d] = i; },
                                         [&](void) {
                                             pcIndices[c][_c][pcSize[c][_c]++] =
                                                     GetWindowIndex(_SupportSizes(), idx);
                                         });
                }
            }
        }
    };

public:
    template <typename Pack, unsigned int PointD, typename T>
    struct _MultiThreadedEvaluator {};
    template <unsigned int... FEMSigs, unsigned int PointD, typename T>
    struct _MultiThreadedEvaluator<UIntPack<FEMSigs...>, PointD, T> {
        typedef UIntPack<FEMSigs...> FEMSignatures;
        typedef UIntPack<FEMSignature<FEMSigs>::Degree...> FEMDegrees;
        const FEMTree* _tree;
        int _threads;
        std::vector<ConstPointSupportKey<FEMDegrees>> _pointNeighborKeys;
        std::vector<ConstCornerSupportKey<FEMDegrees>> _cornerNeighborKeys;
        _Evaluator<FEMSignatures, PointD> _evaluator;
        const DenseNodeData<T, FEMSignatures>& _coefficients;
        DenseNodeData<T, FEMSignatures> _coarseCoefficients;

    public:
        _MultiThreadedEvaluator(const FEMTree* tree,
                                const DenseNodeData<T, FEMSignatures>& coefficients,
                                int threads = std::thread::hardware_concurrency());
        template <unsigned int _PointD = PointD>
        CumulativeDerivativeValues<T, Dim, _PointD> values(Point<Real, Dim> p,
                                                           int thread = 0,
                                                           const FEMTreeNode* node = NULL);
        template <unsigned int _PointD = PointD>
        CumulativeDerivativeValues<T, Dim, _PointD> centerValues(const FEMTreeNode* node,
                                                                 int thread = 0);
        template <unsigned int _PointD = PointD>
        CumulativeDerivativeValues<T, Dim, _PointD> cornerValues(const FEMTreeNode* node,
                                                                 int corner,
                                                                 int thread = 0);
    };
    template <typename Pack, unsigned int PointD, typename T = Real>
    using MultiThreadedEvaluator = _MultiThreadedEvaluator<Pack, PointD, T>;
    template <unsigned int DensityDegree>
    struct MultiThreadedWeightEvaluator {
        const FEMTree* _tree;
        int _threads;
        std::vector<ConstPointSupportKey<IsotropicUIntPack<Dim, DensityDegree>>> _neighborKeys;
        const DensityEstimator<DensityDegree>& _density;

    public:
        MultiThreadedWeightEvaluator(const FEMTree* tree,
                                     const DensityEstimator<DensityDegree>& density,
                                     int threads = std::thread::hardware_concurrency());
        Real weight(Point<Real, Dim> p, int thread = 0);
    };

    static double _MaxMemoryUsage, _LocalMemoryUsage;
    void _reorderDenseOrSparseNodeData(const node_index_type*, size_t) { ; }
    template <class Data, unsigned int... FEMSigs, class... DenseOrSparseNodeData>
    void _reorderDenseOrSparseNodeData(const node_index_type* map,
                                       size_t sz,
                                       SparseNodeData<Data, UIntPack<FEMSigs...>>* sData,
                                       DenseOrSparseNodeData*... data) {
        if (sData) sData->_remapIndices(map, sz);
        _reorderDenseOrSparseNodeData(map, sz, data...);
    }
    template <class Data, unsigned int... FEMSigs, class... DenseOrSparseNodeData>
    void _reorderDenseOrSparseNodeData(const node_index_type* map,
                                       size_t sz,
                                       DenseNodeData<Data, UIntPack<FEMSigs...>>* dData,
                                       DenseOrSparseNodeData*... data) {
        if (dData) dData->_remapIndices(map, sz);
        _reorderDenseOrSparseNodeData(map, sz, data...);
    }

public:
    static double MaxMemoryUsage(void) { return _MaxMemoryUsage; }
    static double LocalMemoryUsage(void) { return _LocalMemoryUsage; }
    static void ResetLocalMemoryUsage(void) { _LocalMemoryUsage = 0; }
    static double MemoryUsage(void);
    FEMTree(size_t blockSize);
    FEMTree(FILE* fp, XForm<Real, Dim + 1>& xForm, size_t blockSize);
    ~FEMTree(void) {
        if (_tree)
            for (int c = 0; c < (1 << Dim); c++) _tree[c].cleanChildren(!nodeAllocators.size());
        for (size_t i = 0; i < nodeAllocators.size(); i++) delete nodeAllocators[i];
    }
    void write(FILE* fp, XForm<Real, Dim + 1> xForm) const;
    static void WriteParameter(FILE* fp) {
        FEMTreeRealType realType;
        if (typeid(Real) == typeid(float))
            realType = FEM_TREE_REAL_FLOAT;
        else if (typeid(Real) == typeid(double))
            realType = FEM_TREE_REAL_DOUBLE;
        else
            ERROR_OUT("Unrecognized real type");
        fwrite(&realType, sizeof(FEMTreeRealType), 1, fp);
        int dim = Dim;
        fwrite(&dim, sizeof(int), 1, fp);
    }

    template <unsigned int LeftRadius, unsigned int RightRadius, class... DenseOrSparseNodeData>
    void thicken(FEMTreeNode** nodes, size_t nodeCount, DenseOrSparseNodeData*... data);
    template <unsigned int LeftRadius,
              unsigned int RightRadius,
              class IsThickenNode,
              class... DenseOrSparseNodeData>
    void thicken(IsThickenNode F, DenseOrSparseNodeData*... data);
    template <unsigned int Radius, class... DenseOrSparseNodeData>
    void thicken(FEMTreeNode** nodes, size_t nodeCount, DenseOrSparseNodeData*... data) {
        thicken<Radius, Radius>(nodes, nodeCount, data...);
    }
    template <unsigned int Radius, class IsThickenNode, class... DenseOrSparseNodeData>
    void thicken(IsThickenNode F, DenseOrSparseNodeData*... data) {
        thicken<Radius, Radius>(F, data...);
    }
    template <unsigned int DensityDegree>
    typename FEMTree::template DensityEstimator<DensityDegree>* setDensityEstimator(
            const std::vector<PointSample>& samples,
            LocalDepth splatDepth,
            Real samplesPerNode,
            int coDimension);
    template <unsigned int... DataSigs, unsigned int DensityDegree, class InData, class OutData>
    SparseNodeData<OutData, UIntPack<DataSigs...>> setDataField(
            UIntPack<DataSigs...>,
            const std::vector<PointSample>& samples,
            const std::vector<InData>& data,
            const DensityEstimator<DensityDegree>* density,
            Real& pointWeightSum,
            std::function<bool(InData, OutData&, Real&)> ConversionAndBiasFunction);
    template <unsigned int... DataSigs, unsigned int DensityDegree, class InData, class OutData>
#if defined(_WIN32) || defined(_WIN64)
    SparseNodeData<OutData, UIntPack<DataSigs...>> setDataField(
            UIntPack<DataSigs...>,
            const std::vector<PointSample>& samples,
            const std::vector<InData>& data,
            const DensityEstimator<DensityDegree>* density,
            Real& pointWeightSum,
            std::function<bool(InData, OutData&)> ConversionFunction,
            std::function<Real(InData)> BiasFunction = [](InData) { return 0.f; });
#else   // !_WIN32 && !_WIN64
    SparseNodeData<OutData, UIntPack<DataSigs...>> setDataField(
            UIntPack<DataSigs...>,
            const std::vector<PointSample>& samples,
            const std::vector<InData>& data,
            const DensityEstimator<DensityDegree>* density,
            Real& pointWeightSum,
            std::function<bool(InData, OutData&)> ConversionFunction,
            std::function<Real(InData)> BiasFunction = [](InData) { return (Real)0; });
#endif  // _WIN32 || _WIN64

    template <unsigned int DataSig, bool CreateNodes, unsigned int DensityDegree, class Data>
    SparseNodeData<Data, IsotropicUIntPack<Dim, DataSig>> setSingleDepthDataField(
            const std::vector<PointSample>& samples,
            const std::vector<Data>& sampleData,
            const DensityEstimator<DensityDegree>* density);
    template <unsigned int DataSig, bool CreateNodes, unsigned int DensityDegree, class Data>
    SparseNodeData<ProjectiveData<Data, Real>, IsotropicUIntPack<Dim, DataSig>>
    setMultiDepthDataField(const std::vector<PointSample>& samples,
                           std::vector<Data>& sampleData,
                           const DensityEstimator<DensityDegree>* density,
                           bool nearest = false);
    template <unsigned int MaxDegree, class HasDataFunctor, class... DenseOrSparseNodeData>
    void finalizeForMultigrid(LocalDepth fullDepth,
                              const HasDataFunctor F,
                              DenseOrSparseNodeData*... data);

    template <unsigned int... FEMSigs>
    DenseNodeData<Real, UIntPack<FEMSigs...>> initDenseNodeData(UIntPack<FEMSigs...>) const;
    template <class Data, unsigned int... FEMSigs>
    DenseNodeData<Data, UIntPack<FEMSigs...>> initDenseNodeData(UIntPack<FEMSigs...>) const;

    // Add multiple-dimensions -> one-dimension constraints
    template <typename T,
              unsigned int... FEMDegrees,
              unsigned int... FEMSigs,
              unsigned int... CDegrees,
              unsigned int... CSigs,
              unsigned int CDim>
    void addFEMConstraints(
            typename BaseFEMIntegrator::
                    template Constraint<UIntPack<FEMDegrees...>, UIntPack<CDegrees...>, CDim>& F,
            const _SparseOrDenseNodeData<Point<T, CDim>, UIntPack<CSigs...>>& coefficients,
            DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
            LocalDepth maxDepth) const {
        typedef SparseNodeData<Point<T, CDim>, UIntPack<CSigs...>> SparseType;
        typedef DenseNodeData<Point<T, CDim>, UIntPack<CSigs...>> DenseType;
        static_assert(sizeof...(FEMDegrees) == Dim && sizeof...(FEMSigs) == Dim &&
                              sizeof...(CDegrees) == Dim && sizeof...(CSigs) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs>::Degree...>>::Equal,
                      "[ERROR] FEM signature and degrees don't match");
        static_assert(UIntPack<CDegrees...>::template Compare<
                              UIntPack<FEMSignature<CSigs>::Degree...>>::Equal,
                      "[ERROR] Constraint signature and degrees don't match");
        if (typeid(coefficients) == typeid(SparseType))
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<CSigs...>(), F,
                                         static_cast<const SparseType&>(coefficients),
                                         constraints(), maxDepth);
        else if (typeid(coefficients) == typeid(DenseType))
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<CSigs...>(), F,
                                         static_cast<const DenseType&>(coefficients), constraints(),
                                         maxDepth);
        else
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<CSigs...>(), F,
                                         coefficients, constraints(), maxDepth);
    }
    // Add one-dimensions -> one-dimension constraints (with distinct signatures)
    template <typename T,
              unsigned int... FEMDegrees,
              unsigned int... FEMSigs,
              unsigned int... CDegrees,
              unsigned int... CSigs>
    void addFEMConstraints(typename BaseFEMIntegrator::template Constraint<UIntPack<FEMDegrees...>,
                                                                           UIntPack<CDegrees...>,
                                                                           1>& F,
                           const _SparseOrDenseNodeData<T, UIntPack<CSigs...>>& coefficients,
                           DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
                           LocalDepth maxDepth) const {
        typedef SparseNodeData<T, UIntPack<CSigs...>> SparseType;
        typedef DenseNodeData<T, UIntPack<CSigs...>> DenseType;
        static_assert(sizeof...(FEMDegrees) == Dim && sizeof...(FEMSigs) == Dim &&
                              sizeof...(CDegrees) == Dim && sizeof...(CSigs) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs>::Degree...>>::Equal,
                      "[ERROR] FEM signature and degrees don't match");
        static_assert(UIntPack<CDegrees...>::template Compare<
                              UIntPack<FEMSignature<CSigs>::Degree...>>::Equal,
                      "[ERROR] Constaint signature and degrees don't match");
        if (typeid(coefficients) == typeid(SparseType))
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<CSigs...>(), F,
                                         static_cast<const SparseType&>(coefficients),
                                         constraints(), maxDepth);
        else if (typeid(coefficients) == typeid(DenseType))
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<CSigs...>(), F,
                                         static_cast<const DenseType&>(coefficients), constraints(),
                                         maxDepth);
        else
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<CSigs...>(), F,
                                         coefficients, constraints(), maxDepth);
    }
    // Add one-dimensions -> one-dimension constraints (with the same signatures)
    template <typename T, unsigned int... FEMDegrees, unsigned int... FEMSigs>
    void addFEMConstraints(typename BaseFEMIntegrator::template System<UIntPack<FEMDegrees...>>& F,
                           const _SparseOrDenseNodeData<T, UIntPack<FEMSigs...>>& coefficients,
                           DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
                           LocalDepth maxDepth) const {
        typedef SparseNodeData<T, UIntPack<FEMSigs...>> SparseType;
        typedef DenseNodeData<T, UIntPack<FEMSigs...>> DenseType;
        static_assert(sizeof...(FEMDegrees) == Dim && sizeof...(FEMSigs) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs>::Degree...>>::Equal,
                      "[ERROR] FEM signatures and degrees don't match");
        typename BaseFEMIntegrator::template SystemConstraint<UIntPack<FEMDegrees...>> _F(F);
        if (typeid(coefficients) == typeid(SparseType))
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                                         static_cast<const SparseType&>(coefficients),
                                         constraints(), maxDepth);
        else if (typeid(coefficients) == typeid(DenseType))
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                                         static_cast<const DenseType&>(coefficients), constraints(),
                                         maxDepth);
        else
            return _addFEMConstraints<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                                         coefficients, constraints(), maxDepth);
    }

    template <typename T, unsigned int... FEMSigs, unsigned int PointD, unsigned int... PointDs>
    typename std::enable_if<(sizeof...(PointDs) != 0)>::type addInterpolationConstraints(
            DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
            LocalDepth maxDepth,
            const InterpolationInfo<T, PointD>& iInfo,
            const InterpolationInfo<T, PointDs>&... iInfos) const {
        addInterpolationConstraints<T, FEMSigs...>(constraints, maxDepth, iInfo);
        addInterpolationConstraints<T, FEMSigs...>(constraints, maxDepth, iInfos...);
    }
    template <typename T, unsigned int... FEMSigs, unsigned int PointD>
    void addInterpolationConstraints(DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
                                     LocalDepth maxDepth,
                                     const InterpolationInfo<T, PointD>& interpolationInfo) const;

    // Real
    template <unsigned int... FEMDegrees1,
              unsigned int... FEMSigs1,
              unsigned int... FEMDegrees2,
              unsigned int... FEMSigs2>
    double dot(typename BaseFEMIntegrator::
                       Constraint<UIntPack<FEMDegrees1...>, UIntPack<FEMDegrees2...>, 1>& F,
               const _SparseOrDenseNodeData<Real, UIntPack<FEMSigs1...>>& coefficients1,
               const _SparseOrDenseNodeData<Real, UIntPack<FEMSigs2...>>& coefficients2) const {
        typedef SparseNodeData<Real, UIntPack<FEMSigs1...>> SparseType1;
        typedef DenseNodeData<Real, UIntPack<FEMSigs1...>> DenseType1;
        typedef SparseNodeData<Real, UIntPack<FEMSigs2...>> SparseType2;
        typedef DenseNodeData<Real, UIntPack<FEMSigs2...>> DenseType2;
        static_assert(sizeof...(FEMDegrees1) == Dim && sizeof...(FEMSigs1) == Dim &&
                              sizeof...(FEMDegrees2) == Dim && sizeof...(FEMSigs2) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees1...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs1>::Degree...>>::Equal,
                      "[ERROR] FEM signature and degrees don't match");
        static_assert(UIntPack<FEMDegrees2...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs2>::Degree...>>::Equal,
                      "[ERROR] FEM signature and degrees don't match");
        if (typeid(coefficients1) == typeid(SparseType1) &&
            typeid(coefficients2) == typeid(SparseType2))
            return _dot<Real>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                              static_cast<const SparseType1&>(coefficients1),
                              static_cast<const SparseType2&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients1) == typeid(SparseType1) &&
                 typeid(coefficients2) == typeid(DenseType2))
            return _dot<Real>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                              static_cast<const SparseType1&>(coefficients1),
                              static_cast<const DenseType2&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients1) == typeid(DenseType1) &&
                 typeid(coefficients2) == typeid(DenseType2))
            return _dot<Real>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                              static_cast<const DenseType1&>(coefficients1),
                              static_cast<const DenseType2&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients1) == typeid(DenseType1) &&
                 typeid(coefficients2) == typeid(SparseType2))
            return _dot<Real>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                              static_cast<const DenseType1&>(coefficients1),
                              static_cast<const SparseType2&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else
            return _dot<Real>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F, coefficients1,
                              coefficients2, [](Real v, Real w) { return v * w; });
    }
    template <unsigned int... FEMDegrees, unsigned int... FEMSigs>
    double dot(typename BaseFEMIntegrator::System<UIntPack<FEMDegrees...>>& F,
               const _SparseOrDenseNodeData<Real, UIntPack<FEMSigs...>>& coefficients1,
               const _SparseOrDenseNodeData<Real, UIntPack<FEMSigs...>>& coefficients2) const {
        typedef SparseNodeData<Real, UIntPack<FEMSigs...>> SparseType;
        typedef DenseNodeData<Real, UIntPack<FEMSigs...>> DenseType;
        static_assert(sizeof...(FEMDegrees) == Dim && sizeof...(FEMSigs) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs>::Degree...>>::Equal,
                      "[ERROR] FEM signatures and degrees don't match");
        typename BaseFEMIntegrator::template SystemConstraint<UIntPack<FEMDegrees...>> _F(F);
        if (typeid(coefficients1) == typeid(SparseType) &&
            typeid(coefficients2) == typeid(SparseType))
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                              static_cast<const SparseType&>(coefficients1),
                              static_cast<const SparseType&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients1) == typeid(SparseType) &&
                 typeid(coefficients2) == typeid(DenseType))
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                              static_cast<const SparseType&>(coefficients1),
                              static_cast<const DenseType&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients1) == typeid(DenseType) &&
                 typeid(coefficients2) == typeid(DenseType))
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                              static_cast<const DenseType&>(coefficients1),
                              static_cast<const DenseType&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients1) == typeid(DenseType) &&
                 typeid(coefficients2) == typeid(SparseType))
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                              static_cast<const DenseType&>(coefficients1),
                              static_cast<const SparseType&>(coefficients2),
                              [](Real v, Real w) { return v * w; });
        else
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F, coefficients1,
                              coefficients2, [](Real v, Real w) { return v * w; });
    }
    template <unsigned int... FEMDegrees, unsigned int... FEMSigs>
    double squareNorm(
            typename BaseFEMIntegrator::template System<UIntPack<FEMDegrees...>>& F,
            const _SparseOrDenseNodeData<Real, UIntPack<FEMSigs...>>& coefficients) const {
        typedef SparseNodeData<Real, UIntPack<FEMSigs...>> SparseType;
        typedef DenseNodeData<Real, UIntPack<FEMSigs...>> DenseType;
        typename BaseFEMIntegrator::template SystemConstraint<UIntPack<FEMDegrees...>> _F(F);
        if (typeid(coefficients) == typeid(SparseType))
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                              static_cast<const SparseType&>(coefficients),
                              static_cast<const SparseType&>(coefficients),
                              [](Real v, Real w) { return v * w; });
        else if (typeid(coefficients) == typeid(DenseType))
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                              static_cast<const DenseType&>(coefficients),
                              static_cast<const DenseType&>(coefficients),
                              [](Real v, Real w) { return v * w; });
        else
            return _dot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F, coefficients,
                              coefficients, [](Real v, Real w) { return v * w; });
    }

    template <unsigned int... FEMSigs1, unsigned int... FEMSigs2, unsigned int... PointDs>
    double interpolationDot(const DenseNodeData<Real, UIntPack<FEMSigs1...>>& coefficients1,
                            const DenseNodeData<Real, UIntPack<FEMSigs2...>>& coefficients2,
                            const InterpolationInfo<Real, PointDs>*... iInfos) const {
        static_assert(sizeof...(FEMSigs1) == Dim && sizeof...(FEMSigs2) == Dim,
                      "[ERROR] Dimensions don't match");
        return _interpolationDot(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), coefficients1,
                                 coefficients2, [](Real v, Real w) { return v * w; }, iInfos...);
    }
    template <unsigned int... FEMSigs, unsigned int... PointDs>
    double interpolationSquareNorm(const DenseNodeData<Real, UIntPack<FEMSigs...>>& coefficients,
                                   const InterpolationInfo<Real, PointDs>*... iInfos) const {
        static_assert(sizeof...(FEMSigs) == Dim, "[ERROR] Dimensions don't match");
        return _interpolationDot<Real>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), coefficients,
                                       coefficients, [](Real v, Real w) { return v * w; },
                                       iInfos...);
    }
    // Generic
    template <typename T,
              typename TDotT,
              unsigned int... FEMDegrees1,
              unsigned int... FEMSigs1,
              unsigned int... FEMDegrees2,
              unsigned int... FEMSigs2>
    double dot(TDotT Dot,
               typename BaseFEMIntegrator::
                       Constraint<UIntPack<FEMDegrees1...>, UIntPack<FEMDegrees2...>, 1>& F,
               const _SparseOrDenseNodeData<T, UIntPack<FEMSigs1...>>& coefficients1,
               const _SparseOrDenseNodeData<T, UIntPack<FEMSigs2...>>& coefficients2) const {
        typedef SparseNodeData<T, UIntPack<FEMSigs1...>> SparseType1;
        typedef DenseNodeData<T, UIntPack<FEMSigs1...>> DenseType1;
        typedef SparseNodeData<T, UIntPack<FEMSigs2...>> SparseType2;
        typedef DenseNodeData<T, UIntPack<FEMSigs2...>> DenseType2;
        static_assert(sizeof...(FEMDegrees1) == Dim && sizeof...(FEMSigs1) == Dim &&
                              sizeof...(FEMDegrees2) == Dim && sizeof...(FEMSigs2) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees1...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs1>::Degree...>>::Equal,
                      "[ERROR] FEM signature and degrees don't match");
        static_assert(UIntPack<FEMDegrees2...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs2>::Degree...>>::Equal,
                      "[ERROR] FEM signature and degrees don't match");
        if (typeid(coefficients1) == typeid(SparseType1) &&
            typeid(coefficients2) == typeid(SparseType2))
            return _dot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                           static_cast<const SparseType1&>(coefficients1),
                           static_cast<const SparseType2&>(coefficients2), Dot);
        else if (typeid(coefficients1) == typeid(SparseType1) &&
                 typeid(coefficients2) == typeid(DenseType2))
            return _dot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                           static_cast<const SparseType1&>(coefficients1),
                           static_cast<const DenseType2&>(coefficients2), Dot);
        else if (typeid(coefficients1) == typeid(DenseType1) &&
                 typeid(coefficients2) == typeid(DenseType2))
            return _dot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                           static_cast<const DenseType1&>(coefficients1),
                           static_cast<const DenseType2&>(coefficients2), Dot);
        else if (typeid(coefficients1) == typeid(DenseType1) &&
                 typeid(coefficients2) == typeid(SparseType2))
            return _dot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F,
                           static_cast<const DenseType1&>(coefficients1),
                           static_cast<const SparseType2&>(coefficients2), Dot);
        else
            return _dot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), F, coefficients1,
                           coefficients2, Dot);
    }
    template <typename T, typename TDotT, unsigned int... FEMDegrees, unsigned int... FEMSigs>
    double dot(TDotT Dot,
               typename BaseFEMIntegrator::System<UIntPack<FEMDegrees...>>& F,
               const _SparseOrDenseNodeData<T, UIntPack<FEMSigs...>>& coefficients1,
               const _SparseOrDenseNodeData<T, UIntPack<FEMSigs...>>& coefficients2) const {
        typedef SparseNodeData<T, UIntPack<FEMSigs...>> SparseType;
        typedef DenseNodeData<T, UIntPack<FEMSigs...>> DenseType;
        static_assert(sizeof...(FEMDegrees) == Dim && sizeof...(FEMSigs) == Dim,
                      "[ERROR] Dimensions don't match");
        static_assert(UIntPack<FEMDegrees...>::template Compare<
                              UIntPack<FEMSignature<FEMSigs>::Degree...>>::Equal,
                      "[ERROR] FEM signatures and degrees don't match");
        typename BaseFEMIntegrator::template SystemConstraint<UIntPack<FEMDegrees...>> _F(F);
        if (typeid(coefficients1) == typeid(SparseType) &&
            typeid(coefficients2) == typeid(SparseType))
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                           static_cast<const SparseType&>(coefficients1),
                           static_cast<const SparseType&>(coefficients2), Dot);
        else if (typeid(coefficients1) == typeid(SparseType) &&
                 typeid(coefficients2) == typeid(DenseType))
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                           static_cast<const SparseType&>(coefficients1),
                           static_cast<const DenseType&>(coefficients2), Dot);
        else if (typeid(coefficients1) == typeid(DenseType) &&
                 typeid(coefficients2) == typeid(DenseType))
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                           static_cast<const DenseType&>(coefficients1),
                           static_cast<const DenseType&>(coefficients2), Dot);
        else if (typeid(coefficients1) == typeid(DenseType) &&
                 typeid(coefficients2) == typeid(SparseType))
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                           static_cast<const DenseType&>(coefficients1),
                           static_cast<const SparseType&>(coefficients2), Dot);
        else
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F, coefficients1,
                           coefficients2, Dot);
    }
    template <typename T, typename TDotT, unsigned int... FEMDegrees, unsigned int... FEMSigs>
    double squareNorm(TDotT Dot,
                      typename BaseFEMIntegrator::template System<UIntPack<FEMDegrees...>>& F,
                      const _SparseOrDenseNodeData<T, UIntPack<FEMSigs...>>& coefficients) const {
        typedef SparseNodeData<T, UIntPack<FEMSigs...>> SparseType;
        typedef DenseNodeData<T, UIntPack<FEMSigs...>> DenseType;
        typename BaseFEMIntegrator::template SystemConstraint<UIntPack<FEMDegrees...>> _F(F);
        if (typeid(coefficients) == typeid(SparseType))
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                           static_cast<const SparseType&>(coefficients),
                           static_cast<const SparseType&>(coefficients), Dot);
        else if (typeid(coefficients) == typeid(DenseType))
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F,
                           static_cast<const DenseType&>(coefficients),
                           static_cast<const DenseType&>(coefficients), Dot);
        else
            return _dot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), _F, coefficients,
                           coefficients, Dot);
    }

    template <typename T,
              typename TDotT,
              unsigned int... FEMSigs1,
              unsigned int... FEMSigs2,
              unsigned int... PointDs>
    double interpolationDot(TDotT Dot,
                            const DenseNodeData<T, UIntPack<FEMSigs1...>>& coefficients1,
                            const DenseNodeData<T, UIntPack<FEMSigs2...>>& coefficients2,
                            const InterpolationInfo<T, PointDs>*... iInfos) const {
        static_assert(sizeof...(FEMSigs1) == Dim && sizeof...(FEMSigs2) == Dim,
                      "[ERROR] Dimensions don't match");
        return _interpolationDot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), coefficients1,
                                    coefficients2, Dot, iInfos...);
    }
    template <typename T, typename TDotT, unsigned int... FEMSigs, unsigned int... PointDs>
    double interpolationSquareNorm(TDotT Dot,
                                   const DenseNodeData<T, UIntPack<FEMSigs...>>& coefficients,
                                   const InterpolationInfo<T, PointDs>*... iInfos) const {
        static_assert(sizeof...(FEMSigs) == Dim, "[ERROR] Dimensions don't match");
        return _interpolationDot<T>(UIntPack<FEMSigs...>(), UIntPack<FEMSigs...>(), coefficients,
                                    coefficients, Dot, iInfos...);
    }

    template <typename T, unsigned int... PointDs, unsigned int... FEMSigs>
    SparseMatrix<Real, matrix_index_type> systemMatrix(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            LocalDepth depth,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <typename T, unsigned int... PointDs, unsigned int... FEMSigs>
    SparseMatrix<Real, matrix_index_type> prolongedSystemMatrix(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            LocalDepth highDepth,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;
    template <unsigned int... FEMSigs>
    SparseMatrix<Real, matrix_index_type> downSampleMatrix(UIntPack<FEMSigs...>,
                                                           LocalDepth highDepth) const;
    template <typename T, unsigned int... PointDs, unsigned int... FEMSigs>
    SparseMatrix<Real, matrix_index_type> fullSystemMatrix(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::System<UIntPack<FEMSignature<FEMSigs>::Degree...>>& F,
            LocalDepth depth,
            bool nonRefinableOnly,
            const InterpolationInfo<T, PointDs>*... interpolationInfo) const;

    struct SolverInfo {
    protected:
        struct _IterFunction {
            _IterFunction(int i) : _i0(i), _type(0) {}
            _IterFunction(std::function<int(int)> iFunction) : _i1(iFunction), _type(1) {}
            _IterFunction(std::function<int(bool, int)> iFunction) : _i2(iFunction), _type(2) {}
            _IterFunction(std::function<int(int, bool, int)> iFunction)
                : _i3(iFunction), _type(3) {}
            _IterFunction& operator=(int i) {
                *this = _IterFunction(i);
                return *this;
            }
            _IterFunction& operator=(std::function<int(int)> iFunction) {
                *this = _IterFunction(iFunction);
                return *this;
            }
            _IterFunction& operator=(std::function<int(bool, int)> iFunction) {
                *this = _IterFunction(iFunction);
                return *this;
            }
            _IterFunction& operator=(std::function<int(int, bool, int)> iFunction) {
                *this = _IterFunction(iFunction);
                return *this;
            }

            int operator()(int vCycle, bool restriction, int depth) const {
                switch (_type) {
                    case 0:
                        return _i0;
                    case 1:
                        return _i1(depth);
                    case 2:
                        return _i2(restriction, depth);
                    case 3:
                        return _i3(vCycle, restriction, depth);
                    default:
                        return 0;
                }
            }

        protected:
            int _i0;
            std::function<int(int)> _i1;
            std::function<int(bool, int)> _i2;
            std::function<int(int i3, bool, int)> _i3;
            int _type;
        };

    public:
        // How to solve
        bool wCycle;
        LocalDepth cgDepth;
        bool cascadic;
        unsigned int sliceBlockSize;
        bool useSupportWeights, useProlongationSupportWeights;
        std::function<Real(Real, Real)> sorRestrictionFunction;
        std::function<Real(Real, Real)> sorProlongationFunction;
        _IterFunction iters;
        int vCycles;
        double cgAccuracy;
        int baseDepth, baseVCycles;
        // What to output
        bool verbose, showResidual;
        int showGlobalResidual;

        SolverInfo(void)
            : cgDepth(0),
              wCycle(false),
              cascadic(true),
              iters(1),
              vCycles(1),
              cgAccuracy(0.),
              verbose(false),
              showResidual(false),
              showGlobalResidual(SHOW_GLOBAL_RESIDUAL_NONE),
              sliceBlockSize(1),
              sorRestrictionFunction([](Real, Real) { return (Real)1; }),
              sorProlongationFunction([](Real, Real) { return (Real)1; }),
              useSupportWeights(false),
              useProlongationSupportWeights(false),
              baseDepth(0),
              baseVCycles(1) {}
    };
    // Solve the linear system
    template <unsigned int... FEMSigs, typename T, typename TDotT, unsigned int... PointDs>
    void solveSystem(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::template System<UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    F,
            const DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
            DenseNodeData<T, UIntPack<FEMSigs...>>& solution,
            TDotT Dot,
            LocalDepth maxSolveDepth,
            const SolverInfo& solverInfo,
            InterpolationInfo<T, PointDs>*... iData) const;
    template <unsigned int... FEMSigs, typename T, typename TDotT, unsigned int... PointDs>
    DenseNodeData<T, UIntPack<FEMSigs...>> solveSystem(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::template System<UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    F,
            const DenseNodeData<T, UIntPack<FEMSigs...>>& constraints,
            TDotT Dot,
            LocalDepth maxSolveDepth,
            const SolverInfo& solverInfo,
            InterpolationInfo<T, PointDs>*... iData) const;
    template <unsigned int... FEMSigs, unsigned int... PointDs>
    void solveSystem(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::template System<UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    F,
            const DenseNodeData<Real, UIntPack<FEMSigs...>>& constraints,
            DenseNodeData<Real, UIntPack<FEMSigs...>>& solution,
            LocalDepth maxSolveDepth,
            const SolverInfo& solverInfo,
            InterpolationInfo<Real, PointDs>*... iData) const {
        return solveSystem<FEMSigs..., Real>(UIntPack<FEMSigs...>(), F, constraints, solution,
                                             [](Real v, Real w) { return v * w; }, maxSolveDepth,
                                             solverInfo, iData...);
    }
    template <unsigned int... FEMSigs, unsigned int... PointDs>
    DenseNodeData<Real, UIntPack<FEMSigs...>> solveSystem(
            UIntPack<FEMSigs...>,
            typename BaseFEMIntegrator::template System<UIntPack<FEMSignature<FEMSigs>::Degree...>>&
                    F,
            const DenseNodeData<Real, UIntPack<FEMSigs...>>& constraints,
            LocalDepth maxSolveDepth,
            const SolverInfo& solverInfo,
            InterpolationInfo<Real, PointDs>*... iData) const {
        return solveSystem(UIntPack<FEMSigs...>(), F, constraints,
                           [](Real v, Real w) { return v * w; }, maxSolveDepth, solverInfo,
                           iData...);
    }

    FEMTreeNode& spaceRoot(void) { return *_spaceRoot; }
    const FEMTreeNode& tree(void) const { return *_tree; }
    _NodeInitializer& initializer(void) { return _nodeInitializer; }
    size_t leaves(void) const { return _tree->leaves(); }
    size_t nodes(void) const {
        int count = 0;
        for (const FEMTreeNode* n = _tree->nextNode(); n; n = _tree->nextNode(n))
            if (IsActiveNode<Dim>(n)) count++;
        return count;
    }
    size_t ghostNodes(void) const {
        int count = 0;
        for (const FEMTreeNode* n = _tree->nextNode(); n; n = _tree->nextNode(n))
            if (!IsActiveNode<Dim>(n)) count++;
        return count;
    }
    inline size_t validSpaceNodes(void) const {
        int count = 0;
        for (const FEMTreeNode* n = _tree->nextNode(); n; n = _tree->nextNode(n))
            if (isValidSpaceNode(n)) count++;
        return count;
    }
    inline size_t validSpaceNodes(LocalDepth d) const {
        int count = 0;
        for (const FEMTreeNode* n = _tree->nextNode(); n; n = _tree->nextNode(n))
            if (_localDepth(n) == d && isValidSpaceNode(n)) count++;
        return count;
    }
    template <unsigned int... FEMSigs>
    size_t validFEMNodes(UIntPack<FEMSigs...>) const {
        int count = 0;
        for (const FEMTreeNode* n = _tree->nextNode(); n; n = _tree->nextNode(n))
            if (isValidFEMNode(UIntPack<FEMSigs...>(), n)) count++;
        return count;
    }
    template <unsigned int... FEMSigs>
    size_t validFEMNodes(UIntPack<FEMSigs...>, LocalDepth d) const {
        int count = 0;
        for (const FEMTreeNode* n = _tree->nextNode(); n; n = _tree->nextNode(n))
            if (_localDepth(n) == d && isValidFEMNode(UIntPack<FEMSigs...>(), n)) count++;
        return count;
    }
    LocalDepth depth(void) const { return _spaceRoot->maxDepth(); }
    void resetNodeIndices(void) {
        _nodeCount = 0;
        for (FEMTreeNode* node = _tree->nextNode(); node; node = _tree->nextNode(node))
            _nodeInitializer(*node), node->nodeData.flags = 0;
    }

    std::vector<node_index_type> merge(FEMTree* tree);

protected:
    template <class Real1, unsigned int _Dim>
    static bool _IsZero(Point<Real1, _Dim> p);
    template <class Real1>
    static bool _IsZero(Real1 p);
    template <class SReal, class Data, unsigned int _Dim>
    static Data _StencilDot(Point<SReal, _Dim> p1, Point<Data, _Dim> p2);
    template <class SReal, class Data>
    static Data _StencilDot(Point<SReal, 1> p1, Point<Data, 1> p2);
    template <class SReal, class Data>
    static Data _StencilDot(SReal p1, Point<Data, 1> p2);
    template <class SReal, class Data>
    static Data _StencilDot(Point<SReal, 1> p1, Data p2);
    template <class SReal, class Data>
    static Data _StencilDot(SReal p1, Data p2);

    // We need the signatures to test if nodes are valid
    template <typename T,
              unsigned int... FEMSigs,
              unsigned int... CSigs,
              unsigned int... FEMDegrees,
              unsigned int... CDegrees,
              unsigned int CDim,
              class Coefficients>
    void _addFEMConstraints(UIntPack<FEMSigs...>,
                            UIntPack<CSigs...>,
                            typename BaseFEMIntegrator::Constraint<UIntPack<FEMDegrees...>,
                                                                   UIntPack<CDegrees...>,
                                                                   CDim>& F,
                            const Coefficients& coefficients,
                            Pointer(T) constraints,
                            LocalDepth maxDepth) const;
    template <typename T,
              typename TDotT,
              unsigned int... FEMSigs1,
              unsigned int... FEMSigs2,
              unsigned int... Degrees1,
              unsigned int... Degrees2,
              class Coefficients1,
              class Coefficients2>
    double
    _dot(UIntPack<FEMSigs1...>,
         UIntPack<FEMSigs2...>,
         typename BaseFEMIntegrator::Constraint<UIntPack<Degrees1...>, UIntPack<Degrees2...>, 1>& F,
         const Coefficients1& coefficients1,
         const Coefficients2& coefficients2,
         TDotT Dot) const;
    template <typename T,
              typename TDotT,
              unsigned int... FEMSigs1,
              unsigned int... FEMSigs2,
              class Coefficients1,
              class Coefficients2,
              unsigned int PointD>
    double _interpolationDot(UIntPack<FEMSigs1...>,
                             UIntPack<FEMSigs2...>,
                             const Coefficients1& coefficients1,
                             const Coefficients2& coefficients2,
                             TDotT Dot,
                             const InterpolationInfo<T, PointD>* iInfo) const;
    template <typename T,
              typename TDotT,
              unsigned int... FEMSigs1,
              unsigned int... FEMSigs2,
              class Coefficients1,
              class Coefficients2,
              unsigned int PointD,
              unsigned int... PointDs>
    double _interpolationDot(UIntPack<FEMSigs1...>,
                             UIntPack<FEMSigs2...>,
                             const Coefficients1& coefficients1,
                             const Coefficients2& coefficients2,
                             TDotT Dot,
                             const InterpolationInfo<T, PointD>* iInfo,
                             const InterpolationInfo<T, PointDs>*... iInfos) const {
        return _interpolationDot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), coefficients1,
                                    coefficients2, Dot, iInfo) +
               _interpolationDot<T>(UIntPack<FEMSigs1...>(), UIntPack<FEMSigs2...>(), coefficients1,
                                    coefficients2, Dot, iInfos...);
    }
    template <typename T,
              typename TDotT,
              unsigned int... FEMSigs1,
              unsigned int... FEMSigs2,
              class Coefficients1,
              class Coefficients2>
    double _interpolationDot(UIntPack<FEMSigs1...>,
                             UIntPack<FEMSigs2...>,
                             const Coefficients1& coefficients1,
                             const Coefficients2& coefficients2,
                             TDotT Dot) const {
        return 0;
    }
};
template <unsigned int Dim, class Real>
double FEMTree<Dim, Real>::_MaxMemoryUsage = 0;
template <unsigned int Dim, class Real>
double FEMTree<Dim, Real>::_LocalMemoryUsage = 0;

template <unsigned int Dim, class Real, class Vertex>
struct IsoSurfaceExtractor {
    struct IsoStats {
        std::string toString(void) const {
            return std::string("Iso-surface extraction not supported for dimension %d", Dim);
        }
    };
    template <typename Data,
              typename SetVertexFunction,
              unsigned int... FEMSigs,
              unsigned int WeightDegree,
              unsigned int DataSig>
    static IsoStats Extract(
            UIntPack<FEMSigs...>,
            UIntPack<WeightDegree>,
            UIntPack<DataSig>,               // Dummy variables for grouping the parameter
            const FEMTree<Dim, Real>& tree,  // The tree over which the system is discretized
            const typename FEMTree<Dim, Real>::template DensityEstimator<WeightDegree>*
                    densityWeights,  // Density weights
            const SparseNodeData<ProjectiveData<Data, Real>, IsotropicUIntPack<Dim, DataSig>>*
                    data,  // Auxiliary spatial data
            const DenseNodeData<Real, UIntPack<FEMSigs...>>&
                    coefficients,  // The coefficients of the function
            Real isoValue,         // The value at which to extract the level-set
            CoredMeshData<Vertex, node_index_type>& mesh,  // The mesh in which to store the output
            const SetVertexFunction&
                    SetVertex,    // A function for setting the depth and data of a vertex
            bool nonLinearFit,    // Should a linear interpolant be used
            bool addBarycenter,   // Should we triangulate polygons by adding a mid-point
            bool polygonMesh,     // Should we output triangles or polygons
            bool flipOrientation  // Should we flip the orientation
    ) {
        // The unspecialized implementation is not supported
        WARN("Iso-surface extraction not supported for dimension ", Dim);
        return IsoStats();
    }
};

template <unsigned int Dim, class Real>
struct FEMTreeInitializer {
    typedef RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type> FEMTreeNode;
    typedef NodeAndPointSample<Dim, Real> PointSample;

    template <class Data>
    struct DerivativeStream {
        virtual void resolution(unsigned int res[]) const = 0;
        virtual bool nextDerivative(unsigned int idx[], unsigned int& dir, Data& dValue) = 0;
    };

    // Initialize the tree using a refinement avatar
    static size_t Initialize(FEMTreeNode& root,
                             int maxDepth,
                             std::function<bool(int, int[])> Refine,
                             Allocator<FEMTreeNode>* nodeAllocator,
                             std::function<void(FEMTreeNode&)> NodeInitializer);

    // Initialize the tree using a point stream
    static size_t Initialize(FEMTreeNode& root,
                             InputPointStream<Real, Dim>& pointStream,
                             int maxDepth,
                             std::vector<PointSample>& samplePoints,
                             Allocator<FEMTreeNode>* nodeAllocator,
                             std::function<void(FEMTreeNode&)> NodeInitializer);
    template <class Data>
    static size_t Initialize(FEMTreeNode& root,
                             InputPointStreamWithData<Real, Dim, Data>& pointStream,
                             int maxDepth,
                             std::vector<PointSample>& samplePoints,
                             std::vector<Data>& sampleData,
                             bool mergeNodeSamples,
                             Allocator<FEMTreeNode>* nodeAllocator,
                             std::function<void(FEMTreeNode&)> NodeInitializer,
                             std::function<Real(const Point<Real, Dim>&, Data&)> ProcessData =
                                     [](const Point<Real, Dim>&, Data&) { return (Real)1.; });

    // Initialize the tree using simplices
    static void Initialize(FEMTreeNode& root,
                           const std::vector<Point<Real, Dim>>& vertices,
                           const std::vector<SimplexIndex<Dim - 1, node_index_type>>& simplices,
                           int maxDepth,
                           std::vector<PointSample>& samples,
                           bool mergeNodeSamples,
                           std::vector<Allocator<FEMTreeNode>*>& nodeAllocators,
                           std::function<void(FEMTreeNode&)> NodeInitializer);
    static void Initialize(FEMTreeNode& root,
                           const std::vector<Point<Real, Dim>>& vertices,
                           const std::vector<SimplexIndex<Dim - 1, node_index_type>>& simplices,
                           int maxDepth,
                           std::vector<NodeSimplices<Dim, Real>>& nodeSimplices,
                           Allocator<FEMTreeNode>* nodeAllocator,
                           std::function<void(FEMTreeNode&)> NodeInitializer);

    template <class Data, class _Data, bool Dual = true>
    static size_t Initialize(FEMTreeNode& root,
                             ConstPointer(Data) values,
                             ConstPointer(int) labels,
                             int resolution[Dim],
                             std::vector<NodeSample<Dim, _Data>> derivatives[Dim],
                             Allocator<FEMTreeNode>* nodeAllocator,
                             std::function<void(FEMTreeNode&)> NodeInitializer,
                             std::function<_Data(const Data&)> DataConverter = [](const Data& d) {
                                 return (_Data)d;
                             });
    template <bool Dual, class Data>
    static unsigned int Initialize(FEMTreeNode& root,
                                   DerivativeStream<Data>& dStream,
                                   std::vector<NodeSample<Dim, Data>> derivatives[Dim],
                                   Allocator<FEMTreeNode>* nodeAllocator,
                                   std::function<void(FEMTreeNode&)> NodeInitializer);

protected:
    template <bool ThreadSafe>
    static size_t _AddSimplex(FEMTreeNode& root,
                              Simplex<Real, Dim, Dim - 1>& s,
                              int maxDepth,
                              std::vector<PointSample>& samples,
                              std::vector<node_index_type>* nodeToIndexMap,
                              Allocator<FEMTreeNode>* nodeAllocator,
                              std::function<void(FEMTreeNode&)> NodeInitializer);
    template <bool ThreadSafe>
    static size_t _AddSimplex(FEMTreeNode& root,
                              Simplex<Real, Dim, Dim - 1>& s,
                              int maxDepth,
                              std::vector<NodeSimplices<Dim, Real>>& simplices,
                              std::vector<node_index_type>& nodeToIndexMap,
                              Allocator<FEMTreeNode>* nodeAllocator,
                              std::function<void(FEMTreeNode&)> NodeInitializer);
    template <bool ThreadSafe>
    static size_t _AddSimplex(FEMTreeNode* node,
                              Simplex<Real, Dim, Dim - 1>& s,
                              int maxDepth,
                              std::vector<PointSample>& samples,
                              std::vector<node_index_type>* nodeToIndexMap,
                              Allocator<FEMTreeNode>* nodeAllocator,
                              std::function<void(FEMTreeNode&)> NodeInitializer);
    template <bool ThreadSafe>
    static size_t _AddSimplex(FEMTreeNode* node,
                              Simplex<Real, Dim, Dim - 1>& s,
                              int maxDepth,
                              std::vector<NodeSimplices<Dim, Real>>& simplices,
                              std::vector<node_index_type>& nodeToIndexMap,
                              Allocator<FEMTreeNode>* nodeAllocator,
                              std::function<void(FEMTreeNode&)> NodeInitializer);
};
template <unsigned int Dim, class Real>
template <unsigned int... SupportSizes>
const unsigned int FEMTree<Dim, Real>::CornerLoopData<SupportSizes...>::supportSizes[] = {
        SupportSizes...};

#include "FEMTree.Evaluation.inl"
#include "FEMTree.Initialize.inl"
#include "FEMTree.IsoSurface.specialized.inl"
#include "FEMTree.SortedTreeNodes.inl"
#include "FEMTree.System.inl"
#include "FEMTree.WeightedSamples.inl"
#include "FEMTree.inl"
#endif  // FEM_TREE_INCLUDED
