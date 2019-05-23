/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_KDTREE_SINGLE_INDEX_H_
#define FLANN_KDTREE_SINGLE_INDEX_H_

#include <algorithm>
#include <map>
#include <cassert>
#include <cstring>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"

namespace flann
{

struct KDTreeSingleIndexParams : public IndexParams
{
    KDTreeSingleIndexParams(int leaf_max_size = 10, bool reorder = true)
    {
        (*this)["algorithm"] = FLANN_INDEX_KDTREE_SINGLE;
        (*this)["leaf_max_size"] = leaf_max_size;
        (*this)["reorder"] = reorder;
    }
};


/**
 * Single kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template <typename Distance>
class KDTreeSingleIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    typedef NNIndex<Distance> BaseClass;

    typedef bool needs_kdtree_distance;

    /**
     * KDTree constructor
     *
     * Params:
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeSingleIndex(const IndexParams& params = KDTreeSingleIndexParams(), Distance d = Distance() ) :
        BaseClass(params, d), root_node_(NULL)
    {
        leaf_max_size_ = get_param(params,"leaf_max_size",10);
        reorder_ = get_param(params, "reorder", true);
    }

    /**
     * KDTree constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeSingleIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KDTreeSingleIndexParams(),
                      Distance d = Distance() ) : BaseClass(params, d), root_node_(NULL)
    {
        leaf_max_size_ = get_param(params,"leaf_max_size",10);
        reorder_ = get_param(params, "reorder", true);

        setDataset(inputData);
    }


    KDTreeSingleIndex(const KDTreeSingleIndex& other) : BaseClass(other),
            leaf_max_size_(other.leaf_max_size_),
            reorder_(other.reorder_),
            vind_(other.vind_),
            root_bbox_(other.root_bbox_)
    {
        if (reorder_) {
            data_ = flann::Matrix<ElementType>(new ElementType[size_*veclen_], size_, veclen_);
            std::copy(other.data_[0], other.data_[0]+size_*veclen_, data_[0]);
        }
        copyTree(root_node_, other.root_node_);
    }

    KDTreeSingleIndex& operator=(KDTreeSingleIndex other)
    {
        this->swap(other);
        return *this;
    }
    
    /**
     * Standard destructor
     */
    virtual ~KDTreeSingleIndex()
    {
        freeIndex();
    }
    
    BaseClass* clone() const
    {
        return new KDTreeSingleIndex(*this);
    }

    using BaseClass::buildIndex;

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        assert(points.cols==veclen_);
        extendDataset(points);
        buildIndex();
    }

    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_KDTREE_SINGLE;
    }


    template<typename Archive>
    void serialize(Archive& ar)
    {
        ar.setObject(this);

        if (reorder_) index_params_["save_dataset"] = false;

        ar & *static_cast<NNIndex<Distance>*>(this);

        ar & reorder_;
        ar & leaf_max_size_;
        ar & root_bbox_;
        ar & vind_;

        if (reorder_) {
            ar & data_;
        }

        if (Archive::is_loading::value) {
            root_node_ = new(pool_) Node();
        }

        ar & *root_node_;

        if (Archive::is_loading::value) {
            index_params_["algorithm"] = getType();
            index_params_["leaf_max_size"] = leaf_max_size_;
            index_params_["reorder"] = reorder_;
        }
    }


    void saveIndex(FILE* stream)
    {
        serialization::SaveArchive sa(stream);
        sa & *this;
    }


    void loadIndex(FILE* stream)
    {
        freeIndex();
        serialization::LoadArchive la(stream);
        la & *this;
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const
    {
        return pool_.usedMemory+pool_.wastedMemory+size_*sizeof(int);  // pool memory and vind array memory
    }

    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
        float epsError = 1+searchParams.eps;

        std::vector<DistanceType> dists(veclen_,0);
        DistanceType distsq = computeInitialDistances(vec, dists);
        if (removed_) {
            searchLevel<true>(result, vec, root_node_, distsq, dists, epsError);
        }
        else {
            searchLevel<false>(result, vec, root_node_, distsq, dists, epsError);
        }
    }

protected:

    /**
     * Builds the index
     */
    void buildIndexImpl()
    {
        // Create a permutable array of indices to the input vectors.
        vind_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            vind_[i] = i;
        }

        computeBoundingBox(root_bbox_);
        root_node_ = divideTree(0, size_, root_bbox_ );   // construct the tree

        if (reorder_) {
            data_ = flann::Matrix<ElementType>(new ElementType[size_*veclen_], size_, veclen_);
            for (size_t i=0; i<size_; ++i) {
                std::copy(points_[vind_[i]], points_[vind_[i]]+veclen_, data_[i]);
            }
        }
    }

private:


    /*--------------------- Internal Data Structures --------------------------*/
    struct Node
    {
    	/**
    	 * Indices of points in leaf node
    	 */
    	int left, right;
    	/**
    	 * Dimension used for subdivision.
    	 */
    	int divfeat;
    	/**
    	 * The values used for subdivision.
    	 */
    	DistanceType divlow, divhigh;
        /**
         * The child nodes.
         */
        Node* child1, * child2;
        
        ~Node()
        {
            if (child1) child1->~Node();
            if (child2) child2->~Node();
        }

    private:
        template<typename Archive>
        void serialize(Archive& ar)
        {
            typedef KDTreeSingleIndex<Distance> Index;
            Index* obj = static_cast<Index*>(ar.getObject());

            ar & left;
            ar & right;
            ar & divfeat;
            ar & divlow;
            ar & divhigh;

            bool leaf_node = false;
            if (Archive::is_saving::value) {
                leaf_node = ((child1==NULL) && (child2==NULL));
            }
            ar & leaf_node;

            if (!leaf_node) {
                if (Archive::is_loading::value) {
                    child1 = new(obj->pool_) Node();
                    child2 = new(obj->pool_) Node();
                }
                ar & *child1;
                ar & *child2;
            }
        }
        friend struct serialization::access;
    };
    typedef Node* NodePtr;


    struct Interval
    {
        DistanceType low, high;
        
    private:
        template <typename Archive>
        void serialize(Archive& ar)
        {
            ar & low;
            ar & high;
        }
        friend struct serialization::access;
    };

    typedef std::vector<Interval> BoundingBox;

    typedef BranchStruct<NodePtr, DistanceType> BranchSt;
    typedef BranchSt* Branch;


    
    void freeIndex()
    {
        if (data_.ptr()) {
            delete[] data_.ptr();
            data_ = flann::Matrix<ElementType>();
        }
        if (root_node_) root_node_->~Node();
        pool_.free();
    }
    
    void copyTree(NodePtr& dst, const NodePtr& src)
    {
        dst = new(pool_) Node();
        *dst = *src;
        if (src->child1!=NULL && src->child2!=NULL) {
            copyTree(dst->child1, src->child1);
            copyTree(dst->child2, src->child2);
        }
    }



    void computeBoundingBox(BoundingBox& bbox)
    {
        bbox.resize(veclen_);
        for (size_t i=0; i<veclen_; ++i) {
            bbox[i].low = (DistanceType)points_[0][i];
            bbox[i].high = (DistanceType)points_[0][i];
        }
        for (size_t k=1; k<size_; ++k) {
            for (size_t i=0; i<veclen_; ++i) {
                if (points_[k][i]<bbox[i].low) bbox[i].low = (DistanceType)points_[k][i];
                if (points_[k][i]>bbox[i].high) bbox[i].high = (DistanceType)points_[k][i];
            }
        }
    }


    /**
     * Create a tree node that subdivides the list of vecs from vind[first]
     * to vind[last].  The routine is called recursively on each sublist.
     * Place a pointer to this new tree node in the location pTree.
     *
     * Params: pTree = the new node to create
     *                  first = index of the first vector
     *                  last = index of the last vector
     */
    NodePtr divideTree(int left, int right, BoundingBox& bbox)
    {
        NodePtr node = new (pool_) Node(); // allocate memory

        /* If too few exemplars remain, then make this a leaf node. */
        if ( (right-left) <= leaf_max_size_) {
            node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
            node->left = left;
            node->right = right;

            // compute bounding-box of leaf points
            for (size_t i=0; i<veclen_; ++i) {
                bbox[i].low = (DistanceType)points_[vind_[left]][i];
                bbox[i].high = (DistanceType)points_[vind_[left]][i];
            }
            for (int k=left+1; k<right; ++k) {
                for (size_t i=0; i<veclen_; ++i) {
                    if (bbox[i].low>points_[vind_[k]][i]) bbox[i].low=(DistanceType)points_[vind_[k]][i];
                    if (bbox[i].high<points_[vind_[k]][i]) bbox[i].high=(DistanceType)points_[vind_[k]][i];
                }
            }
        }
        else {
            int idx;
            int cutfeat;
            DistanceType cutval;
            middleSplit(&vind_[0]+left, right-left, idx, cutfeat, cutval, bbox);

            node->divfeat = cutfeat;

            BoundingBox left_bbox(bbox);
            left_bbox[cutfeat].high = cutval;
            node->child1 = divideTree(left, left+idx, left_bbox);

            BoundingBox right_bbox(bbox);
            right_bbox[cutfeat].low = cutval;
            node->child2 = divideTree(left+idx, right, right_bbox);

            node->divlow = left_bbox[cutfeat].high;
            node->divhigh = right_bbox[cutfeat].low;

            for (size_t i=0; i<veclen_; ++i) {
            	bbox[i].low = std::min(left_bbox[i].low, right_bbox[i].low);
            	bbox[i].high = std::max(left_bbox[i].high, right_bbox[i].high);
            }
        }

        return node;
    }

    void computeMinMax(int* ind, int count, int dim, ElementType& min_elem, ElementType& max_elem)
    {
        min_elem = points_[ind[0]][dim];
        max_elem = points_[ind[0]][dim];
        for (int i=1; i<count; ++i) {
            ElementType val = points_[ind[i]][dim];
            if (val<min_elem) min_elem = val;
            if (val>max_elem) max_elem = val;
        }
    }

    void middleSplit(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval, const BoundingBox& bbox)
    {
        // find the largest span from the approximate bounding box
        ElementType max_span = bbox[0].high-bbox[0].low;
        cutfeat = 0;
        cutval = (bbox[0].high+bbox[0].low)/2;
        for (size_t i=1; i<veclen_; ++i) {
            ElementType span = bbox[i].high-bbox[i].low;
            if (span>max_span) {
                max_span = span;
                cutfeat = i;
                cutval = (bbox[i].high+bbox[i].low)/2;
            }
        }

        // compute exact span on the found dimension
        ElementType min_elem, max_elem;
        computeMinMax(ind, count, cutfeat, min_elem, max_elem);
        cutval = (min_elem+max_elem)/2;
        max_span = max_elem - min_elem;

        // check if a dimension of a largest span exists
        size_t k = cutfeat;
        for (size_t i=0; i<veclen_; ++i) {
            if (i==k) continue;
            ElementType span = bbox[i].high-bbox[i].low;
            if (span>max_span) {
                computeMinMax(ind, count, i, min_elem, max_elem);
                span = max_elem - min_elem;
                if (span>max_span) {
                    max_span = span;
                    cutfeat = i;
                    cutval = (min_elem+max_elem)/2;
                }
            }
        }
        int lim1, lim2;
        planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1>count/2) index = lim1;
        else if (lim2<count/2) index = lim2;
        else index = count/2;
        
        assert(index > 0 && index < count);
    }


    void middleSplit_(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval, const BoundingBox& bbox)
    {
        const float eps_val=0.00001f;
        DistanceType max_span = bbox[0].high-bbox[0].low;
        for (size_t i=1; i<veclen_; ++i) {
            DistanceType span = bbox[i].high-bbox[i].low;
            if (span>max_span) {
                max_span = span;
            }
        }
        DistanceType max_spread = -1;
        cutfeat = 0;
        for (size_t i=0; i<veclen_; ++i) {
            DistanceType span = bbox[i].high-bbox[i].low;
            if (span>(DistanceType)((1-eps_val)*max_span)) {
                ElementType min_elem, max_elem;
                computeMinMax(ind, count, cutfeat, min_elem, max_elem);
                DistanceType spread = (DistanceType)(max_elem-min_elem);
                if (spread>max_spread) {
                    cutfeat = i;
                    max_spread = spread;
                }
            }
        }
        // split in the middle
        DistanceType split_val = (bbox[cutfeat].low+bbox[cutfeat].high)/2;
        ElementType min_elem, max_elem;
        computeMinMax(ind, count, cutfeat, min_elem, max_elem);

        if (split_val<min_elem) cutval = (DistanceType)min_elem;
        else if (split_val>max_elem) cutval = (DistanceType)max_elem;
        else cutval = split_val;

        int lim1, lim2;
        planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1>count/2) index = lim1;
        else if (lim2<count/2) index = lim2;
        else index = count/2;
        
        assert(index > 0 && index < count);
    }


    /**
     *  Subdivide the list of points by a plane perpendicular on axe corresponding
     *  to the 'cutfeat' dimension at 'cutval' position.
     *
     *  On return:
     *  dataset[ind[0..lim1-1]][cutfeat]<cutval
     *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
     *  dataset[ind[lim2..count]][cutfeat]>cutval
     */
    void planeSplit(int* ind, int count, int cutfeat, DistanceType cutval, int& lim1, int& lim2)
    {
        int left = 0;
        int right = count-1;
        for (;; ) {
            while (left<=right && points_[ind[left]][cutfeat]<cutval) ++left;
            while (left<=right && points_[ind[right]][cutfeat]>=cutval) --right;
            if (left>right) break;
            std::swap(ind[left], ind[right]); ++left; --right;
        }

        lim1 = left;
        right = count-1;
        for (;; ) {
            while (left<=right && points_[ind[left]][cutfeat]<=cutval) ++left;
            while (left<=right && points_[ind[right]][cutfeat]>cutval) --right;
            if (left>right) break;
            std::swap(ind[left], ind[right]); ++left; --right;
        }
        lim2 = left;
    }

    DistanceType computeInitialDistances(const ElementType* vec, std::vector<DistanceType>& dists) const
    {
        DistanceType distsq = 0.0;

        for (size_t i = 0; i < veclen_; ++i) {
            if (vec[i] < root_bbox_[i].low) {
                dists[i] = distance_.accum_dist(vec[i], root_bbox_[i].low, i);
                distsq += dists[i];
            }
            if (vec[i] > root_bbox_[i].high) {
                dists[i] = distance_.accum_dist(vec[i], root_bbox_[i].high, i);
                distsq += dists[i];
            }
        }

        return distsq;
    }

    /**
     * Performs an exact search in the tree starting from a node.
     */
    template <bool with_removed>
    void searchLevel(ResultSet<DistanceType>& result_set, const ElementType* vec, const NodePtr node, DistanceType mindistsq,
                     std::vector<DistanceType>& dists, const float epsError) const
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            DistanceType worst_dist = result_set.worstDist();
            for (int i=node->left; i<node->right; ++i) {
                if (with_removed) {
                    if (removed_points_.test(vind_[i])) continue;
                }
                ElementType* point = reorder_ ? data_[i] : points_[vind_[i]];
                DistanceType dist = distance_(vec, point, veclen_, worst_dist);
                if (dist<worst_dist) {
                    result_set.addPoint(dist,vind_[i]);
                }
            }
            return;
        }

        /* Which child branch should be taken first? */
        int idx = node->divfeat;
        ElementType val = vec[idx];
        DistanceType diff1 = val - node->divlow;
        DistanceType diff2 = val - node->divhigh;

        NodePtr bestChild;
        NodePtr otherChild;
        DistanceType cut_dist;
        if ((diff1+diff2)<0) {
            bestChild = node->child1;
            otherChild = node->child2;
            cut_dist = distance_.accum_dist(val, node->divhigh, idx);
        }
        else {
            bestChild = node->child2;
            otherChild = node->child1;
            cut_dist = distance_.accum_dist( val, node->divlow, idx);
        }

        /* Call recursively to search next level down. */
        searchLevel<with_removed>(result_set, vec, bestChild, mindistsq, dists, epsError);

        DistanceType dst = dists[idx];
        mindistsq = mindistsq + cut_dist - dst;
        dists[idx] = cut_dist;
        if (mindistsq*epsError<=result_set.worstDist()) {
            searchLevel<with_removed>(result_set, vec, otherChild, mindistsq, dists, epsError);
        }
        dists[idx] = dst;
    }

    
    void swap(KDTreeSingleIndex& other)
    {
        BaseClass::swap(other);
        std::swap(leaf_max_size_, other.leaf_max_size_);
        std::swap(reorder_, other.reorder_);
        std::swap(vind_, other.vind_);
        std::swap(data_, other.data_);
        std::swap(root_node_, other.root_node_);
        std::swap(root_bbox_, other.root_bbox_);
        std::swap(pool_, other.pool_);
    }
    
private:



    int leaf_max_size_;
    
    
    bool reorder_;

    /**
     *  Array of indices to vectors in the dataset.
     */
    std::vector<int> vind_;

    Matrix<ElementType> data_;

    /**
     * Array of k-d trees used to find neighbours.
     */
    NodePtr root_node_;

    /**
     * Root bounding box
     */
    BoundingBox root_bbox_;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool_;

    USING_BASECLASS_SYMBOLS

};   // class KDTreeSingleIndex

}

#endif //FLANN_KDTREE_SINGLE_INDEX_H_
