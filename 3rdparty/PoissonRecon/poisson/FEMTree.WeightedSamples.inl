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

template <class Real, unsigned int DataDegree, unsigned int... DataDegrees>
typename std::enable_if<sizeof...(DataDegrees) == 0>::type __SetBSplineComponentValues(
        const Real* position, const Real* start, Real width, double* values, unsigned int stride) {
    Polynomial<DataDegree>::BSplineComponentValues((position[0] - start[0]) / width, values);
}
template <class Real, unsigned int DataDegree, unsigned int... DataDegrees>
typename std::enable_if<sizeof...(DataDegrees) != 0>::type __SetBSplineComponentValues(
        const Real* position, const Real* start, Real width, double* values, unsigned int stride) {
    Polynomial<DataDegree>::BSplineComponentValues((position[0] - start[0]) / width, values);
    __SetBSplineComponentValues<Real, DataDegrees...>(position + 1, start + 1, width,
                                                      values + stride, stride);
}

// evaluate the result of splatting along a plane and then evaluating at a point on the plane.
template <unsigned int Degree>
double GetScaleValue(void) {
    double centerValues[Degree + 1];
    Polynomial<Degree>::BSplineComponentValues(0.5, centerValues);
    double scaleValue = 0;
    for (int i = 0; i <= Degree; i++) scaleValue += centerValues[i] * centerValues[i];
    return 1. / scaleValue;
}
template <unsigned int Dim, class Real>
template <bool ThreadSafe, unsigned int WeightDegree>
void FEMTree<Dim, Real>::_addWeightContribution(
        Allocator<FEMTreeNode>* nodeAllocator,
        DensityEstimator<WeightDegree>& densityWeights,
        FEMTreeNode* node,
        Point<Real, Dim> position,
        PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
        Real weight) {
    static const double ScaleValue = GetScaleValue<WeightDegree>();
    double values[Dim][BSplineSupportSizes<WeightDegree>::SupportSize];
    typename FEMTreeNode::template Neighbors<
            IsotropicUIntPack<Dim, BSplineSupportSizes<WeightDegree>::SupportSize>>& neighbors =
            weightKey.template getNeighbors<true, ThreadSafe>(node, nodeAllocator,
                                                              _nodeInitializer);

    densityWeights.reserve(nodeCount());

    Point<Real, Dim> start;
    Real w;
    _startAndWidth(node, start, w);

    for (int dim = 0; dim < Dim; dim++)
        Polynomial<WeightDegree>::BSplineComponentValues((position[dim] - start[dim]) / w,
                                                         values[dim]);

    weight *= (Real)ScaleValue;
    double scratch[Dim + 1];
    scratch[0] = weight;
    WindowLoop<Dim>::Run(IsotropicUIntPack<Dim, 0>(),
                         IsotropicUIntPack<Dim, BSplineSupportSizes<WeightDegree>::SupportSize>(),
                         [&](int d, int i) { scratch[d + 1] = scratch[d] * values[d][i]; },
                         [&](FEMTreeNode* node) {
                             if (node) {
                                 AddAtomic(densityWeights[node], (Real)scratch[Dim]);
                             }
                         },
                         neighbors.neighbors());
}

template <unsigned int Dim, class Real>
template <unsigned int WeightDegree, class PointSupportKey>
Real FEMTree<Dim, Real>::_getSamplesPerNode(const DensityEstimator<WeightDegree>& densityWeights,
                                            const FEMTreeNode* node,
                                            Point<Real, Dim> position,
                                            PointSupportKey& weightKey) const {
    Real weight = 0;
    typedef typename PointSupportKey::NeighborType Neighbors;
    double values[Dim][BSplineSupportSizes<WeightDegree>::SupportSize];
    Neighbors neighbors = weightKey.getNeighbors(node);
    Point<Real, Dim> start;
    Real w;
    _startAndWidth(node, start, w);

    for (int dim = 0; dim < Dim; dim++)
        Polynomial<WeightDegree>::BSplineComponentValues((position[dim] - start[dim]) / w,
                                                         values[dim]);
    double scratch[Dim + 1];
    scratch[0] = 1;
    WindowLoop<Dim>::Run(IsotropicUIntPack<Dim, 0>(),
                         IsotropicUIntPack<Dim, BSplineSupportSizes<WeightDegree>::SupportSize>(),
                         [&](int d, int i) { scratch[d + 1] = scratch[d] * values[d][i]; },
                         [&](typename Neighbors::Window::data_type node) {
                             if (node) {
                                 const Real* w = densityWeights(node);
                                 if (w) weight += (Real)(scratch[Dim] * (*w));
                             }
                         },
                         neighbors.neighbors());
    return weight;
}
template <unsigned int Dim, class Real>
template <unsigned int WeightDegree, class PointSupportKey>
void FEMTree<Dim, Real>::_getSampleDepthAndWeight(
        const DensityEstimator<WeightDegree>& densityWeights,
        const FEMTreeNode* node,
        Point<Real, Dim> position,
        PointSupportKey& weightKey,
        Real& depth,
        Real& weight) const {
    const FEMTreeNode* temp = node;
    while (_localDepth(temp) > densityWeights.kernelDepth()) temp = temp->parent;
    weight = _getSamplesPerNode(densityWeights, temp, position, weightKey);
    if (weight >= (Real)1.)
        depth = Real(_localDepth(temp) +
                     log(weight) / log(double(1 << (Dim - densityWeights.coDimension()))));
    else {
        Real oldWeight, newWeight;
        oldWeight = newWeight = weight;
        while (newWeight < (Real)1. && _localDepth(temp)) {
            temp = temp->parent;
            oldWeight = newWeight;
            newWeight = _getSamplesPerNode(densityWeights, temp, position, weightKey);
        }
        depth = Real(_localDepth(temp) + log(newWeight) / log(newWeight / oldWeight));
    }
    weight = Real(pow(double(1 << (Dim - densityWeights.coDimension())), -double(depth)));
}
template <unsigned int Dim, class Real>
template <unsigned int WeightDegree, class PointSupportKey>
void FEMTree<Dim, Real>::_getSampleDepthAndWeight(
        const DensityEstimator<WeightDegree>& densityWeights,
        Point<Real, Dim> position,
        PointSupportKey& weightKey,
        Real& depth,
        Real& weight) const {
    FEMTreeNode* temp;
    Point<Real, Dim> myCenter;
    for (int d = 0; d < Dim; d++) myCenter[d] = (Real)0.5;
    Real myWidth = Real(1.);

    // Get the finest node with depth less than or equal to the splat depth that contains the point
    temp = _spaceRoot;
    while (_localDepth(temp) < densityWeights.kernelDepth()) {
        if (!IsActiveNode<Dim>(temp->children)) break;  // ERROR_OUT( "" );
        int cIndex = FEMTreeNode::ChildIndex(myCenter, position);
        temp = temp->children + cIndex;
        myWidth /= 2;
        for (int d = 0; d < Dim; d++)
            if ((cIndex >> d) & 1)
                myCenter[d] += myWidth / 2;
            else
                myCenter[d] -= myWidth / 2;
    }
    return _getSampleDepthAndWeight(densityWeights, temp, position, weightKey, depth, weight);
}

template <unsigned int Dim, class Real>
template <bool CreateNodes, bool ThreadSafe, class V, unsigned int... DataSigs>
void FEMTree<Dim, Real>::_splatPointData(
        Allocator<FEMTreeNode>* nodeAllocator,
        FEMTreeNode* node,
        Point<Real, Dim> position,
        V v,
        SparseNodeData<V, UIntPack<DataSigs...>>& dataInfo,
        PointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey) {
    typedef UIntPack<BSplineSupportSizes<FEMSignature<DataSigs>::Degree>::SupportSize...>
            SupportSizes;
    double values[Dim][SupportSizes::Max()];
    typename FEMTreeNode::template Neighbors<
            UIntPack<BSplineSupportSizes<FEMSignature<DataSigs>::Degree>::SupportSize...>>&
            neighbors = dataKey.template getNeighbors<CreateNodes, ThreadSafe>(node, nodeAllocator,
                                                                               _nodeInitializer);
    Point<Real, Dim> start;
    Real w;
    _startAndWidth(node, start, w);
    __SetBSplineComponentValues<Real, FEMSignature<DataSigs>::Degree...>(
            &position[0], &start[0], w, &values[0][0], SupportSizes::Max());
    double scratch[Dim + 1];
    scratch[0] = 1;
    WindowLoop<Dim>::Run(
            ZeroUIntPack<Dim>(),
            UIntPack<BSplineSupportSizes<FEMSignature<DataSigs>::Degree>::SupportSize...>(),
            [&](int d, int i) { scratch[d + 1] = scratch[d] * values[d][i]; },
            [&](FEMTreeNode* node) {
                if (IsActiveNode<Dim>(node)) {
                    AddAtomic(dataInfo[node], v * (Real)scratch[Dim]);
                }
            },
            neighbors.neighbors());
}
template <unsigned int Dim, class Real>
template <bool CreateNodes,
          bool ThreadSafe,
          unsigned int WeightDegree,
          class V,
          unsigned int... DataSigs>
Real FEMTree<Dim, Real>::_splatPointData(
        Allocator<FEMTreeNode>* nodeAllocator,
        const DensityEstimator<WeightDegree>& densityWeights,
        Point<Real, Dim> position,
        V v,
        SparseNodeData<V, UIntPack<DataSigs...>>& dataInfo,
        PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
        PointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey,
        LocalDepth minDepth,
        LocalDepth maxDepth,
        int dim,
        Real depthBias) {
    double dx;
    V _v;
    FEMTreeNode* temp;
    double width;
    Point<Real, Dim> myCenter;
    for (int d = 0; d < Dim; d++) myCenter[d] = (Real)0.5;
    Real myWidth = (Real)1.;
    temp = _spaceRoot;
    while (_localDepth(temp) < densityWeights.kernelDepth()) {
        if (!IsActiveNode<Dim>(temp->children)) break;
        int cIndex = FEMTreeNode::ChildIndex(myCenter, position);
        temp = temp->children + cIndex;
        myWidth /= 2;
        for (int d = 0; d < Dim; d++)
            if ((cIndex >> d) & 1)
                myCenter[d] += myWidth / 2;
            else
                myCenter[d] -= myWidth / 2;
    }
    Real weight, depth;
    _getSampleDepthAndWeight(densityWeights, temp, position, weightKey, depth, weight);
    depth += depthBias;

    if (depth < minDepth) depth = Real(minDepth);
    if (depth > maxDepth) depth = Real(maxDepth);
    int topDepth = int(ceil(depth));

    dx = 1.0 - (topDepth - depth);
    if (topDepth <= minDepth)
        topDepth = minDepth, dx = 1;
    else if (topDepth > maxDepth)
        topDepth = maxDepth, dx = 1;

    while (_localDepth(temp) > topDepth) temp = temp->parent;
    while (_localDepth(temp) < topDepth) {
        if (!temp->children)
            temp->template initChildren<ThreadSafe>(nodeAllocator, _nodeInitializer);
        int cIndex = FEMTreeNode::ChildIndex(myCenter, position);
        temp = &temp->children[cIndex];
        myWidth /= 2;
        for (int d = 0; d < Dim; d++)
            if ((cIndex >> d) & 1)
                myCenter[d] += myWidth / 2;
            else
                myCenter[d] -= myWidth / 2;
    }

    width = 1.0 / (1 << _localDepth(temp));
    _v = v * weight / Real(pow(width, dim)) * Real(dx);
    // #if defined( __GNUC__ ) && __GNUC__ < 5
    // #warning "you've got me gcc version<5"
    // 	_splatPointData< CreateNodes , ThreadSafe , V >( nodeAllocator , temp , position , _v ,
    // dataInfo , dataKey ); #else // !__GNUC__ || __GNUC__ >=5
    _splatPointData<CreateNodes, ThreadSafe, V, DataSigs...>(nodeAllocator, temp, position, _v,
                                                             dataInfo, dataKey);
    // #endif // __GNUC__ || __GNUC__ < 4
    if (fabs(1.0 - dx) > 1e-6) {
        dx = Real(1.0 - dx);
        temp = temp->parent;
        width = 1.0 / (1 << _localDepth(temp));

        _v = v * weight / Real(pow(width, dim)) * Real(dx);
        // #if defined( __GNUC__ ) && __GNUC__ < 5
        // #warning "you've got me gcc version<5"
        // 		_splatPointData< CreateNodes , ThreadSafe , V >( nodeAllocator , temp ,
        // position , _v , dataInfo , dataKey ); #else // !__GNUC__ || __GNUC__ >=5
        _splatPointData<CreateNodes, ThreadSafe, V, DataSigs...>(nodeAllocator, temp, position, _v,
                                                                 dataInfo, dataKey);
        // #endif // __GNUC__ || __GNUC__ < 4
    }
    return weight;
}
template <unsigned int Dim, class Real>
template <bool CreateNodes,
          bool ThreadSafe,
          unsigned int WeightDegree,
          class V,
          unsigned int... DataSigs>
Real FEMTree<Dim, Real>::_multiSplatPointData(
        Allocator<FEMTreeNode>* nodeAllocator,
        const DensityEstimator<WeightDegree>* densityWeights,
        FEMTreeNode* node,
        Point<Real, Dim> position,
        V v,
        SparseNodeData<V, UIntPack<DataSigs...>>& dataInfo,
        PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
        PointSupportKey<UIntPack<FEMSignature<DataSigs>::Degree...>>& dataKey,
        int dim) {
    typedef UIntPack<BSplineSupportSizes<FEMSignature<DataSigs>::Degree>::SupportSize...>
            SupportSizes;
    Real _depth, weight;
    if (densityWeights)
        _getSampleDepthAndWeight(*densityWeights, position, weightKey, _depth, weight);
    else
        weight = (Real)1.;
    V _v = v * weight;

    double values[Dim][SupportSizes::Max()];
    dataKey.template getNeighbors<CreateNodes, ThreadSafe>(node, nodeAllocator, _nodeInitializer);

    for (FEMTreeNode* _node = node; _localDepth(_node) >= 0; _node = _node->parent) {
        V __v = _v * (Real)pow(1 << _localDepth(_node), dim);
        Point<Real, Dim> start;
        Real w;
        _startAndWidth(_node, start, w);
        __SetBSplineComponentValues<Real, FEMSignature<DataSigs>::Degree...>(
                &position[0], &start[0], w, &values[0][0], SupportSizes::Max());
        typename FEMTreeNode::template Neighbors<
                UIntPack<BSplineSupportSizes<FEMSignature<DataSigs>::Degree>::SupportSize...>>&
                neighbors = dataKey.neighbors[_localToGlobal(_localDepth(_node))];
        double scratch[Dim + 1];
        scratch[0] = 1.;
        WindowLoop<Dim>::Run(
                ZeroUIntPack<Dim>(),
                UIntPack<BSplineSupportSizes<FEMSignature<DataSigs>::Degree>::SupportSize...>(),
                [&](int d, int i) { scratch[d + 1] = scratch[d] * values[d][i]; },
                [&](FEMTreeNode* node) {
                    if (IsActiveNode<Dim>(node)) dataInfo[node] += __v * (Real)scratch[Dim];
                },
                neighbors.neighbors());
    }
    return weight;
}

template <unsigned int Dim, class Real>
template <unsigned int WeightDegree, class V, unsigned int... DataSigs>
Real FEMTree<Dim, Real>::_nearestMultiSplatPointData(
        const DensityEstimator<WeightDegree>* densityWeights,
        FEMTreeNode* node,
        Point<Real, Dim> position,
        V v,
        SparseNodeData<V, UIntPack<DataSigs...>>& dataInfo,
        PointSupportKey<IsotropicUIntPack<Dim, WeightDegree>>& weightKey,
        int dim) {
    Real _depth, weight;
    if (densityWeights)
        _getSampleDepthAndWeight(*densityWeights, position, weightKey, _depth, weight);
    else
        weight = (Real)1.;
    V _v = v * weight;

    for (FEMTreeNode* _node = node; _localDepth(_node) >= 0; _node = _node->parent)
        if (IsActiveNode<Dim>(_node))
            dataInfo[_node] += _v * (Real)pow(1 << _localDepth(_node), dim);
    return weight;
}
//////////////////////////////////
// MultiThreadedWeightEvaluator //
//////////////////////////////////
template <unsigned int Dim, class Real>
template <unsigned int DensityDegree>
FEMTree<Dim, Real>::MultiThreadedWeightEvaluator<DensityDegree>::MultiThreadedWeightEvaluator(
        const FEMTree<Dim, Real>* tree, const DensityEstimator<DensityDegree>& density, int threads)
    : _density(density), _tree(tree) {
    _threads = std::max<int>(1, threads);
    _neighborKeys.resize(_threads);
    for (int t = 0; t < _neighborKeys.size(); t++)
        _neighborKeys[t].set(tree->_localToGlobal(density.kernelDepth()));
}
template <unsigned int Dim, class Real>
template <unsigned int DensityDegree>
Real FEMTree<Dim, Real>::MultiThreadedWeightEvaluator<DensityDegree>::weight(Point<Real, Dim> p,
                                                                             int thread) {
    ConstPointSupportKey<IsotropicUIntPack<Dim, DensityDegree>>& nKey = _neighborKeys[thread];
    Real depth, weight;
    _tree->_getSampleDepthAndWeight(_density, p, nKey, depth, weight);
    return weight;
}
