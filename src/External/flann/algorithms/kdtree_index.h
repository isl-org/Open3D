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

#ifndef FLANN_KDTREE_INDEX_H_
#define FLANN_KDTREE_INDEX_H_

#include <algorithm>
#include <map>
#include <cassert>
#include <cstring>
#include <stdarg.h>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/util/dynamic_bitset.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"


namespace flann
{

struct KDTreeIndexParams : public IndexParams
{
    KDTreeIndexParams(int trees = 4)
    {
        (*this)["algorithm"] = FLANN_INDEX_KDTREE;
        (*this)["trees"] = trees;
    }
};


/**
 * Randomized kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template <typename Distance>
class KDTreeIndex : public NNIndex<Distance>
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
     *          inputData = dataset with the input features
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeIndex(const IndexParams& params = KDTreeIndexParams(), Distance d = Distance() ) :
    	BaseClass(params, d), mean_(NULL), var_(NULL)
    {
        trees_ = get_param(index_params_,"trees",4);
    }


    /**
     * KDTree constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeIndex(const Matrix<ElementType>& dataset, const IndexParams& params = KDTreeIndexParams(),
                Distance d = Distance() ) : BaseClass(params,d ), mean_(NULL), var_(NULL)
    {
        trees_ = get_param(index_params_,"trees",4);

        setDataset(dataset);
    }

    KDTreeIndex(const KDTreeIndex& other) : BaseClass(other),
    		trees_(other.trees_)
    {
        tree_roots_.resize(other.tree_roots_.size());
        for (size_t i=0;i<tree_roots_.size();++i) {
        	copyTree(tree_roots_[i], other.tree_roots_[i]);
        }
    }

    KDTreeIndex& operator=(KDTreeIndex other)
    {
    	this->swap(other);
    	return *this;
    }

    /**
     * Standard destructor
     */
    virtual ~KDTreeIndex()
    {
    	freeIndex();
    }

    BaseClass* clone() const
    {
    	return new KDTreeIndex(*this);
    }

    using BaseClass::buildIndex;
    
    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        assert(points.cols==veclen_);

        size_t old_size = size_;
        extendDataset(points);
        
        if (rebuild_threshold>1 && size_at_build_*rebuild_threshold<size_) {
            buildIndex();
        }
        else {
            for (size_t i=old_size;i<size_;++i) {
                for (int j = 0; j < trees_; j++) {
                    addPointToTree(tree_roots_[j], i);
                }
            }
        }        
    }

    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_KDTREE;
    }


    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	ar & trees_;

    	if (Archive::is_loading::value) {
    		tree_roots_.resize(trees_);
    	}
    	for (size_t i=0;i<tree_roots_.size();++i) {
    		if (Archive::is_loading::value) {
    			tree_roots_[i] = new(pool_) Node();
    		}
    		ar & *tree_roots_[i];
    	}

    	if (Archive::is_loading::value) {
            index_params_["algorithm"] = getType();
            index_params_["trees"] = trees_;
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
        return int(pool_.usedMemory+pool_.wastedMemory+size_*sizeof(int));  // pool memory and vind array memory
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
        int maxChecks = searchParams.checks;
        float epsError = 1+searchParams.eps;

        if (maxChecks==FLANN_CHECKS_UNLIMITED) {
        	if (removed_) {
        		getExactNeighbors<true>(result, vec, epsError);
        	}
        	else {
        		getExactNeighbors<false>(result, vec, epsError);
        	}
        }
        else {
        	if (removed_) {
        		getNeighbors<true>(result, vec, maxChecks, epsError);
        	}
        	else {
        		getNeighbors<false>(result, vec, maxChecks, epsError);
        	}
        }
    }

protected:

    /**
     * Builds the index
     */
    void buildIndexImpl()
    {
        // Create a permutable array of indices to the input vectors.
    	std::vector<int> ind(size_);
        for (size_t i = 0; i < size_; ++i) {
            ind[i] = int(i);
        }

        mean_ = new DistanceType[veclen_];
        var_ = new DistanceType[veclen_];

        tree_roots_.resize(trees_);
        /* Construct the randomized trees. */
        for (int i = 0; i < trees_; i++) {
            /* Randomize the order of vectors to allow for unbiased sampling. */
            std::random_shuffle(ind.begin(), ind.end());
            tree_roots_[i] = divideTree(&ind[0], int(size_) );
        }
        delete[] mean_;
        delete[] var_;
    }

    void freeIndex()
    {
    	for (size_t i=0;i<tree_roots_.size();++i) {
    		// using placement new, so call destructor explicitly
    		if (tree_roots_[i]!=NULL) tree_roots_[i]->~Node();
    	}
    	pool_.free();
    }


private:

    /*--------------------- Internal Data Structures --------------------------*/
    struct Node
    {
    	/**
         * Dimension used for subdivision.
         */
        int divfeat;
        /**
         * The values used for subdivision.
         */
        DistanceType divval;
        /**
         * Point data
         */
        ElementType* point;
        /**
         * The child nodes.
         */
        Node* child1, * child2;

        ~Node() {
        	if (child1!=NULL) child1->~Node();
        	if (child2!=NULL) child2->~Node();
        }

    private:
    	template<typename Archive>
    	void serialize(Archive& ar)
    	{
    		typedef KDTreeIndex<Distance> Index;
    		Index* obj = static_cast<Index*>(ar.getObject());

    		ar & divfeat;
    		ar & divval;

    		bool leaf_node = false;
    		if (Archive::is_saving::value) {
    			leaf_node = ((child1==NULL) && (child2==NULL));
    		}
    		ar & leaf_node;

    		if (leaf_node) {
    			if (Archive::is_loading::value) {
    				point = obj->points_[divfeat];
    			}
    		}

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
    typedef BranchStruct<NodePtr, DistanceType> BranchSt;
    typedef BranchSt* Branch;


    void copyTree(NodePtr& dst, const NodePtr& src)
    {
    	dst = new(pool_) Node();
    	dst->divfeat = src->divfeat;
    	dst->divval = src->divval;
    	if (src->child1==NULL && src->child2==NULL) {
    		dst->point = points_[dst->divfeat];
    		dst->child1 = NULL;
    		dst->child2 = NULL;
    	}
    	else {
    		copyTree(dst->child1, src->child1);
    		copyTree(dst->child2, src->child2);
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
    NodePtr divideTree(int* ind, int count)
    {
        NodePtr node = new(pool_) Node(); // allocate memory

        /* If too few exemplars remain, then make this a leaf node. */
        if (count == 1) {
            node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
            node->divfeat = *ind;    /* Store index of this vec. */
            node->point = points_[*ind];
        }
        else {
            int idx;
            int cutfeat;
            DistanceType cutval;
            meanSplit(ind, count, idx, cutfeat, cutval);

            node->divfeat = cutfeat;
            node->divval = cutval;
            node->child1 = divideTree(ind, idx);
            node->child2 = divideTree(ind+idx, count-idx);
        }

        return node;
    }


    /**
     * Choose which feature to use in order to subdivide this set of vectors.
     * Make a random choice among those with the highest variance, and use
     * its variance as the threshold value.
     */
    void meanSplit(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval)
    {
        memset(mean_,0,veclen_*sizeof(DistanceType));
        memset(var_,0,veclen_*sizeof(DistanceType));

        /* Compute mean values.  Only the first SAMPLE_MEAN values need to be
            sampled to get a good estimate.
         */
        int cnt = std::min((int)SAMPLE_MEAN+1, count);
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = points_[ind[j]];
            for (size_t k=0; k<veclen_; ++k) {
                mean_[k] += v[k];
            }
        }
        DistanceType div_factor = DistanceType(1)/cnt;
        for (size_t k=0; k<veclen_; ++k) {
            mean_[k] *= div_factor;
        }

        /* Compute variances (no need to divide by count). */
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = points_[ind[j]];
            for (size_t k=0; k<veclen_; ++k) {
                DistanceType dist = v[k] - mean_[k];
                var_[k] += dist * dist;
            }
        }
        /* Select one of the highest variance indices at random. */
        cutfeat = selectDivision(var_);
        cutval = mean_[cutfeat];

        int lim1, lim2;
        planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1>count/2) index = lim1;
        else if (lim2<count/2) index = lim2;
        else index = count/2;

        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
        if ((lim1==count)||(lim2==0)) index = count/2;
    }


    /**
     * Select the top RAND_DIM largest values from v and return the index of
     * one of these selected at random.
     */
    int selectDivision(DistanceType* v)
    {
        int num = 0;
        size_t topind[RAND_DIM];

        /* Create a list of the indices of the top RAND_DIM values. */
        for (size_t i = 0; i < veclen_; ++i) {
            if ((num < RAND_DIM)||(v[i] > v[topind[num-1]])) {
                /* Put this element at end of topind. */
                if (num < RAND_DIM) {
                    topind[num++] = i;            /* Add to list. */
                }
                else {
                    topind[num-1] = i;         /* Replace last element. */
                }
                /* Bubble end value down to right location by repeated swapping. */
                int j = num - 1;
                while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
                    std::swap(topind[j], topind[j-1]);
                    --j;
                }
            }
        }
        /* Select a random integer in range [0,num-1], and return that index. */
        int rnd = rand_int(num);
        return (int)topind[rnd];
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
        /* Move vector indices for left subtree to front of list. */
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

    /**
     * Performs an exact nearest neighbor search. The exact search performs a full
     * traversal of the tree.
     */
    template<bool with_removed>
    void getExactNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, float epsError) const
    {
        //		checkID -= 1;  /* Set a different unique ID for each search. */

        if (trees_ > 1) {
            fprintf(stderr,"It doesn't make any sense to use more than one tree for exact search");
        }
        if (trees_>0) {
            searchLevelExact<with_removed>(result, vec, tree_roots_[0], 0.0, epsError);
        }
    }

    /**
     * Performs the approximate nearest-neighbor search. The search is approximate
     * because the tree traversal is abandoned after a given number of descends in
     * the tree.
     */
    template<bool with_removed>
    void getNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, int maxCheck, float epsError) const
    {
        int i;
        BranchSt branch;

        int checkCount = 0;
        Heap<BranchSt>* heap = new Heap<BranchSt>((int)size_);
        DynamicBitset checked(size_);

        /* Search once through each tree down to root. */
        for (i = 0; i < trees_; ++i) {
            searchLevel<with_removed>(result, vec, tree_roots_[i], 0, checkCount, maxCheck, epsError, heap, checked);
        }

        /* Keep searching other branches from heap until finished. */
        while ( heap->popMin(branch) && (checkCount < maxCheck || !result.full() )) {
            searchLevel<with_removed>(result, vec, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
        }

        delete heap;

    }

    /**
     *  Search starting from a given node of the tree.  Based on any mismatches at
     *  higher levels, all exemplars below this level must have a distance of
     *  at least "mindistsq".
     */
    template<bool with_removed>
    void searchLevel(ResultSet<DistanceType>& result_set, const ElementType* vec, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
                     float epsError, Heap<BranchSt>* heap, DynamicBitset& checked) const
    {
        if (result_set.worstDist()<mindist) {
            //			printf("Ignoring branch, too far\n");
            return;
        }

        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            int index = node->divfeat;
            if (with_removed) {
            	if (removed_points_.test(index)) return;
            }
            /*  Do not check same node more than once when searching multiple trees. */
            if ( checked.test(index) || ((checkCount>=maxCheck)&& result_set.full()) ) return;
            checked.set(index);
            checkCount++;

            DistanceType dist = distance_(node->point, vec, veclen_);
            result_set.addPoint(dist,index);
            return;
        }

        /* Which child branch should be taken first? */
        ElementType val = vec[node->divfeat];
        DistanceType diff = val - node->divval;
        NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
        NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

        /* Create a branch record for the branch not taken.  Add distance
            of this feature boundary (we don't attempt to correct for any
            use of this feature in a parent node, which is unlikely to
            happen and would have only a small effect).  Don't bother
            adding more branches to heap after halfway point, as cost of
            adding exceeds their value.
         */

        DistanceType new_distsq = mindist + distance_.accum_dist(val, node->divval, node->divfeat);
        //		if (2 * checkCount < maxCheck  ||  !result.full()) {
        if ((new_distsq*epsError < result_set.worstDist())||  !result_set.full()) {
            heap->insert( BranchSt(otherChild, new_distsq) );
        }

        /* Call recursively to search next level down. */
        searchLevel<with_removed>(result_set, vec, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
    }

    /**
     * Performs an exact search in the tree starting from a node.
     */
    template<bool with_removed>
    void searchLevelExact(ResultSet<DistanceType>& result_set, const ElementType* vec, const NodePtr node, DistanceType mindist, const float epsError) const
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            int index = node->divfeat;
            if (with_removed) {
            	if (removed_points_.test(index)) return; // ignore removed points
            }
            DistanceType dist = distance_(node->point, vec, veclen_);
            result_set.addPoint(dist,index);

            return;
        }

        /* Which child branch should be taken first? */
        ElementType val = vec[node->divfeat];
        DistanceType diff = val - node->divval;
        NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
        NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

        /* Create a branch record for the branch not taken.  Add distance
            of this feature boundary (we don't attempt to correct for any
            use of this feature in a parent node, which is unlikely to
            happen and would have only a small effect).  Don't bother
            adding more branches to heap after halfway point, as cost of
            adding exceeds their value.
         */

        DistanceType new_distsq = mindist + distance_.accum_dist(val, node->divval, node->divfeat);

        /* Call recursively to search next level down. */
        searchLevelExact<with_removed>(result_set, vec, bestChild, mindist, epsError);

        if (mindist*epsError<=result_set.worstDist()) {
            searchLevelExact<with_removed>(result_set, vec, otherChild, new_distsq, epsError);
        }
    }
    
    void addPointToTree(NodePtr node, int ind)
    {
        ElementType* point = points_[ind];
        
        if ((node->child1==NULL) && (node->child2==NULL)) {
            ElementType* leaf_point = node->point;
            ElementType max_span = 0;
            size_t div_feat = 0;
            for (size_t i=0;i<veclen_;++i) {
                ElementType span = abs(point[i]-leaf_point[i]);
                if (span > max_span) {
                    max_span = span;
                    div_feat = i;
                }
            }
            NodePtr left = new(pool_) Node();
            left->child1 = left->child2 = NULL;
            NodePtr right = new(pool_) Node();
            right->child1 = right->child2 = NULL;

            if (point[div_feat]<leaf_point[div_feat]) {
                left->divfeat = ind;
                left->point = point;
                right->divfeat = node->divfeat;
                right->point = node->point;
            }
            else {
                left->divfeat = node->divfeat;
                left->point = node->point;
                right->divfeat = ind;
                right->point = point;
            }
            node->divfeat = div_feat;
            node->divval = (point[div_feat]+leaf_point[div_feat])/2;
            node->child1 = left;
            node->child2 = right;            
        }
        else {
            if (point[node->divfeat]<node->divval) {
                addPointToTree(node->child1,ind);
            }
            else {
                addPointToTree(node->child2,ind);                
            }
        }
    }
private:
    void swap(KDTreeIndex& other)
    {
    	BaseClass::swap(other);
    	std::swap(trees_, other.trees_);
    	std::swap(tree_roots_, other.tree_roots_);
    	std::swap(pool_, other.pool_);
    }

private:

    enum
    {
        /**
         * To improve efficiency, only SAMPLE_MEAN random values are used to
         * compute the mean and variance at each level when building a tree.
         * A value of 100 seems to perform as well as using all values.
         */
        SAMPLE_MEAN = 100,
        /**
         * Top random dimensions to consider
         *
         * When creating random trees, the dimension on which to subdivide is
         * selected at random from among the top RAND_DIM dimensions with the
         * highest variance.  A value of 5 works well.
         */
        RAND_DIM=5
    };


    /**
     * Number of randomized trees that are used
     */
    int trees_;

    DistanceType* mean_;
    DistanceType* var_;

    /**
     * Array of k-d trees used to find neighbours.
     */
    std::vector<NodePtr> tree_roots_;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool_;

    USING_BASECLASS_SYMBOLS
};   // class KDTreeIndex

}

#endif //FLANN_KDTREE_INDEX_H_
