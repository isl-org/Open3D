/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2011  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2011  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

#ifndef FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_
#define FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/dist.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"
#include "flann/util/serialization.h"

namespace flann
{

struct HierarchicalClusteringIndexParams : public IndexParams
{
    HierarchicalClusteringIndexParams(int branching = 32,
                                      flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
                                      int trees = 4, int leaf_max_size = 100)
    {
        (*this)["algorithm"] = FLANN_INDEX_HIERARCHICAL;
        // The branching factor used in the hierarchical clustering
        (*this)["branching"] = branching;
        // Algorithm used for picking the initial cluster centers
        (*this)["centers_init"] = centers_init;
        // number of parallel trees to build
        (*this)["trees"] = trees;
        // maximum leaf size
        (*this)["leaf_max_size"] = leaf_max_size;
    }
};



/**
 * Hierarchical index
 *
 * Contains a tree constructed through a hierarchical clustering
 * and other information for indexing a set of points for nearest-neighbour matching.
 */
template <typename Distance>
class HierarchicalClusteringIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    typedef NNIndex<Distance> BaseClass;

    /**
     * Constructor.
     *
     * @param index_params
     * @param d
     */
    HierarchicalClusteringIndex(const IndexParams& index_params = HierarchicalClusteringIndexParams(), Distance d = Distance())
        : BaseClass(index_params, d)
    {
        memoryCounter_ = 0;

        branching_ = get_param(index_params_,"branching",32);
        centers_init_ = get_param(index_params_,"centers_init", FLANN_CENTERS_RANDOM);
        trees_ = get_param(index_params_,"trees",4);
        leaf_max_size_ = get_param(index_params_,"leaf_max_size",100);

        initCenterChooser();
    }


    /**
     * Index constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the hierarchical k-means algorithm
     */
    HierarchicalClusteringIndex(const Matrix<ElementType>& inputData, const IndexParams& index_params = HierarchicalClusteringIndexParams(),
                                Distance d = Distance())
        : BaseClass(index_params, d)
    {
        memoryCounter_ = 0;

        branching_ = get_param(index_params_,"branching",32);
        centers_init_ = get_param(index_params_,"centers_init", FLANN_CENTERS_RANDOM);
        trees_ = get_param(index_params_,"trees",4);
        leaf_max_size_ = get_param(index_params_,"leaf_max_size",100);

        initCenterChooser();
        chooseCenters_->setDataset(inputData);
        
        setDataset(inputData);
    }


    HierarchicalClusteringIndex(const HierarchicalClusteringIndex& other) : BaseClass(other),
    		memoryCounter_(other.memoryCounter_),
    		branching_(other.branching_),
    		trees_(other.trees_),
    		centers_init_(other.centers_init_),
    		leaf_max_size_(other.leaf_max_size_)

    {
    	initCenterChooser();
        tree_roots_.resize(other.tree_roots_.size());
        for (size_t i=0;i<tree_roots_.size();++i) {
        	copyTree(tree_roots_[i], other.tree_roots_[i]);
        }
    }

    HierarchicalClusteringIndex& operator=(HierarchicalClusteringIndex other)
    {
    	this->swap(other);
    	return *this;
    }


    void initCenterChooser()
    {
        switch(centers_init_) {
        case FLANN_CENTERS_RANDOM:
        	chooseCenters_ = new RandomCenterChooser<Distance>(distance_);
        	break;
        case FLANN_CENTERS_GONZALES:
        	chooseCenters_ = new GonzalesCenterChooser<Distance>(distance_);
        	break;
        case FLANN_CENTERS_KMEANSPP:
            chooseCenters_ = new KMeansppCenterChooser<Distance>(distance_);
        	break;
        default:
            throw FLANNException("Unknown algorithm for choosing initial centers.");
        }
    }

    /**
     * Index destructor.
     *
     * Release the memory used by the index.
     */
    virtual ~HierarchicalClusteringIndex()
    {
    	delete chooseCenters_;
    	freeIndex();
    }

    BaseClass* clone() const
    {
    	return new HierarchicalClusteringIndex(*this);
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const
    {
        return pool_.usedMemory+pool_.wastedMemory+memoryCounter_;
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
            for (size_t i=0;i<points.rows;++i) {
                for (int j = 0; j < trees_; j++) {
                    addPointToTree(tree_roots_[j], old_size + i);
                }
            }            
        }
    }


    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_HIERARCHICAL;
    }


    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	ar & branching_;
    	ar & trees_;
    	ar & centers_init_;
    	ar & leaf_max_size_;

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
            index_params_["branching"] = branching_;
            index_params_["trees"] = trees_;
            index_params_["centers_init"] = centers_init_;
            index_params_["leaf_size"] = leaf_max_size_;
    	}
    }

    void saveIndex(FILE* stream)
    {
    	serialization::SaveArchive sa(stream);
    	sa & *this;
    }


    void loadIndex(FILE* stream)
    {
    	serialization::LoadArchive la(stream);
    	la & *this;
    }


    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     searchParams = parameters that influence the search algorithm (checks)
     */

    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
    	if (removed_) {
    		findNeighborsWithRemoved<true>(result, vec, searchParams);
    	}
    	else {
    		findNeighborsWithRemoved<false>(result, vec, searchParams);
    	}
    }

protected:

    /**
     * Builds the index
     */
    void buildIndexImpl()
    {
        if (branching_<2) {
            throw FLANNException("Branching factor must be at least 2");
        }
        tree_roots_.resize(trees_);
        std::vector<int> indices(size_);
        for (int i=0; i<trees_; ++i) {
            for (size_t j=0; j<size_; ++j) {
                indices[j] = j;
            }
            tree_roots_[i] = new(pool_) Node();
            computeClustering(tree_roots_[i], &indices[0], size_);
        }
    }

private:

    struct PointInfo
    {
    	/** Point index */
    	size_t index;
    	/** Point data */
    	ElementType* point;

    private:
    	template<typename Archive>
    	void serialize(Archive& ar)
    	{
    		typedef HierarchicalClusteringIndex<Distance> Index;
    		Index* obj = static_cast<Index*>(ar.getObject());

    		ar & index;
//    		ar & point;

			if (Archive::is_loading::value) {
				point = obj->points_[index];
			}
    	}
    	friend struct serialization::access;
    };

    /**
     * Struture representing a node in the hierarchical k-means tree.
     */
    struct Node
    {
        /**
         * The cluster center
         */
    	ElementType* pivot;
    	size_t pivot_index;
        /**
         * Child nodes (only for non-terminal nodes)
         */
        std::vector<Node*> childs;
        /**
         * Node points (only for terminal nodes)
         */
        std::vector<PointInfo> points;

        /**
         * destructor
         * calling Node destructor explicitly
         */
        ~Node()
        {
        	for(size_t i=0; i<childs.size(); i++){
        		childs[i]->~Node();
        	}
        };

    private:
    	template<typename Archive>
    	void serialize(Archive& ar)
    	{
    		typedef HierarchicalClusteringIndex<Distance> Index;
    		Index* obj = static_cast<Index*>(ar.getObject());
    		ar & pivot_index;
    		if (Archive::is_loading::value) {
    			pivot = obj->points_[pivot_index];
    		}
    		size_t childs_size;
    		if (Archive::is_saving::value) {
    			childs_size = childs.size();
    		}
    		ar & childs_size;

    		if (childs_size==0) {
    			ar & points;
    		}
    		else {
    			if (Archive::is_loading::value) {
    				childs.resize(childs_size);
    			}
    			for (size_t i=0;i<childs_size;++i) {
    				if (Archive::is_loading::value) {
    					childs[i] = new(obj->pool_) Node();
    				}
    				ar & *childs[i];
    			}
    		}

    	}
    	friend struct serialization::access;
    };
    typedef Node* NodePtr;



    /**
     * Alias definition for a nicer syntax.
     */
    typedef BranchStruct<NodePtr, DistanceType> BranchSt;


    /**
     * Clears Node tree
     * calling Node destructor explicitly
     */
    void freeIndex(){
    	for (size_t i=0; i<tree_roots_.size(); ++i) {
    		tree_roots_[i]->~Node();
    	}
    	pool_.free();
    }

    void copyTree(NodePtr& dst, const NodePtr& src)
    {
    	dst = new(pool_) Node();
    	dst->pivot_index = src->pivot_index;
    	dst->pivot = points_[dst->pivot_index];

    	if (src->childs.size()==0) {
    		dst->points = src->points;
    	}
    	else {
    		dst->childs.resize(src->childs.size());
    		for (size_t i=0;i<src->childs.size();++i) {
    			copyTree(dst->childs[i], src->childs[i]);
    		}
    	}
    }



    void computeLabels(int* indices, int indices_length,  int* centers, int centers_length, int* labels, DistanceType& cost)
    {
        cost = 0;
        for (int i=0; i<indices_length; ++i) {
            ElementType* point = points_[indices[i]];
            DistanceType dist = distance_(point, points_[centers[0]], veclen_);
            labels[i] = 0;
            for (int j=1; j<centers_length; ++j) {
                DistanceType new_dist = distance_(point, points_[centers[j]], veclen_);
                if (dist>new_dist) {
                    labels[i] = j;
                    dist = new_dist;
                }
            }
            cost += dist;
        }
    }

    /**
     * The method responsible with actually doing the recursive hierarchical
     * clustering
     *
     * Params:
     *     node = the node to cluster
     *     indices = indices of the points belonging to the current node
     *     branching = the branching factor to use in the clustering
     *
     */
    void computeClustering(NodePtr node, int* indices, int indices_length)
    {
        if (indices_length < leaf_max_size_) { // leaf node
            node->points.resize(indices_length);
            for (int i=0;i<indices_length;++i) {
            	node->points[i].index = indices[i];
            	node->points[i].point = points_[indices[i]];
            }
            node->childs.clear();
            return;
        }

        std::vector<int> centers(branching_);
        std::vector<int> labels(indices_length);

        int centers_length;
        (*chooseCenters_)(branching_, indices, indices_length, &centers[0], centers_length);

        if (centers_length<branching_) {
            node->points.resize(indices_length);
            for (int i=0;i<indices_length;++i) {
            	node->points[i].index = indices[i];
            	node->points[i].point = points_[indices[i]];
            }
            node->childs.clear();
            return;
        }


        //  assign points to clusters
        DistanceType cost;
        computeLabels(indices, indices_length, &centers[0], centers_length, &labels[0], cost);

        node->childs.resize(branching_);
        int start = 0;
        int end = start;
        for (int i=0; i<branching_; ++i) {
            for (int j=0; j<indices_length; ++j) {
                if (labels[j]==i) {
                    std::swap(indices[j],indices[end]);
                    std::swap(labels[j],labels[end]);
                    end++;
                }
            }

            node->childs[i] = new(pool_) Node();
            node->childs[i]->pivot_index = centers[i];
            node->childs[i]->pivot = points_[centers[i]];
            node->childs[i]->points.clear();
            computeClustering(node->childs[i],indices+start, end-start);
            start=end;
        }
    }


    template<bool with_removed>
    void findNeighborsWithRemoved(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
        int maxChecks = searchParams.checks;

        // Priority queue storing intermediate branches in the best-bin-first search
        Heap<BranchSt>* heap = new Heap<BranchSt>(size_);

        DynamicBitset checked(size_);
        int checks = 0;
        for (int i=0; i<trees_; ++i) {
            findNN<with_removed>(tree_roots_[i], result, vec, checks, maxChecks, heap, checked);
        }

        BranchSt branch;
        while (heap->popMin(branch) && (checks<maxChecks || !result.full())) {
            NodePtr node = branch.node;
            findNN<with_removed>(node, result, vec, checks, maxChecks, heap, checked);
        }

        delete heap;
    }


    /**
     * Performs one descent in the hierarchical k-means tree. The branches not
     * visited are stored in a priority queue.
     *
     * Params:
     *      node = node to explore
     *      result = container for the k-nearest neighbors found
     *      vec = query points
     *      checks = how many points in the dataset have been checked so far
     *      maxChecks = maximum dataset points to checks
     */

    template<bool with_removed>
    void findNN(NodePtr node, ResultSet<DistanceType>& result, const ElementType* vec, int& checks, int maxChecks,
                Heap<BranchSt>* heap,  DynamicBitset& checked) const
    {
        if (node->childs.empty()) {
            if (checks>=maxChecks) {
                if (result.full()) return;
            }

            for (size_t i=0; i<node->points.size(); ++i) {
            	PointInfo& pointInfo = node->points[i];
            	if (with_removed) {
            		if (removed_points_.test(pointInfo.index)) continue;
            	}
                if (checked.test(pointInfo.index)) continue;
                DistanceType dist = distance_(pointInfo.point, vec, veclen_);
                result.addPoint(dist, pointInfo.index);
                checked.set(pointInfo.index);
                ++checks;
            }
        }
        else {
            DistanceType* domain_distances = new DistanceType[branching_];
            int best_index = 0;
            domain_distances[best_index] = distance_(vec, node->childs[best_index]->pivot, veclen_);
            for (int i=1; i<branching_; ++i) {
                domain_distances[i] = distance_(vec, node->childs[i]->pivot, veclen_);
                if (domain_distances[i]<domain_distances[best_index]) {
                    best_index = i;
                }
            }
            for (int i=0; i<branching_; ++i) {
                if (i!=best_index) {
                    heap->insert(BranchSt(node->childs[i],domain_distances[i]));
                }
            }
            delete[] domain_distances;
            findNN<with_removed>(node->childs[best_index],result,vec, checks, maxChecks, heap, checked);
        }
    }
    
    void addPointToTree(NodePtr node, size_t index)
    {
        ElementType* point = points_[index];
        
        if (node->childs.empty()) { // leaf node
        	PointInfo pointInfo;
        	pointInfo.point = point;
        	pointInfo.index = index;
            node->points.push_back(pointInfo);

            if (node->points.size()>=size_t(branching_)) {
                std::vector<int> indices(node->points.size());

                for (size_t i=0;i<node->points.size();++i) {
                	indices[i] = node->points[i].index;
                }
                computeClustering(node, &indices[0], indices.size());
            }
        }
        else {            
            // find the closest child
            int closest = 0;
            ElementType* center = node->childs[closest]->pivot;
            DistanceType dist = distance_(center, point, veclen_);
            for (size_t i=1;i<size_t(branching_);++i) {
                center = node->childs[i]->pivot;
                DistanceType crt_dist = distance_(center, point, veclen_);
                if (crt_dist<dist) {
                    dist = crt_dist;
                    closest = i;
                }
            }
            addPointToTree(node->childs[closest], index);
        }                
    }

    void swap(HierarchicalClusteringIndex& other)
    {
    	BaseClass::swap(other);

    	std::swap(tree_roots_, other.tree_roots_);
    	std::swap(pool_, other.pool_);
    	std::swap(memoryCounter_, other.memoryCounter_);
    	std::swap(branching_, other.branching_);
    	std::swap(trees_, other.trees_);
    	std::swap(centers_init_, other.centers_init_);
    	std::swap(leaf_max_size_, other.leaf_max_size_);
    	std::swap(chooseCenters_, other.chooseCenters_);
    }

private:

    /**
     * The root nodes in the tree.
     */
    std::vector<Node*> tree_roots_;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool_;

    /**
     * Memory occupied by the index.
     */
    int memoryCounter_;

    /** index parameters */
    /**
     * Branching factor to use for clustering
     */
    int branching_;
    
    /**
     * How many parallel trees to build
     */
    int trees_;
    
    /**
     * Algorithm to use for choosing cluster centers
     */
    flann_centers_init_t centers_init_;
    
    /**
     * Max size of leaf nodes
     */
    int leaf_max_size_;
    
    /**
     * Algorithm used to choose initial centers
     */
    CenterChooser<Distance>* chooseCenters_;

    USING_BASECLASS_SYMBOLS
};

}

#endif /* FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_ */
