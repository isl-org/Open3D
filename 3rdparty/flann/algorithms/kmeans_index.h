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

#ifndef FLANN_KMEANS_INDEX_H_
#define FLANN_KMEANS_INDEX_H_

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/dist.h"
#include "flann/algorithms/center_chooser.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"
#include "flann/util/logger.h"



namespace flann
{

struct KMeansIndexParams : public IndexParams
{
    KMeansIndexParams(int branching = 32, int iterations = 11,
                      flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM, float cb_index = 0.2 )
    {
        (*this)["algorithm"] = FLANN_INDEX_KMEANS;
        // branching factor
        (*this)["branching"] = branching;
        // max iterations to perform in one kmeans clustering (kmeans tree)
        (*this)["iterations"] = iterations;
        // algorithm used for picking the initial cluster centers for kmeans tree
        (*this)["centers_init"] = centers_init;
        // cluster boundary index. Used when searching the kmeans tree
        (*this)["cb_index"] = cb_index;
    }
};


/**
 * Hierarchical kmeans index
 *
 * Contains a tree constructed through a hierarchical kmeans clustering
 * and other information for indexing a set of points for nearest-neighbour matching.
 */
template <typename Distance>
class KMeansIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    typedef NNIndex<Distance> BaseClass;

    typedef bool needs_vector_space_distance;



    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_KMEANS;
    }

    /**
     * Index constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the hierarchical k-means algorithm
     */
    KMeansIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KMeansIndexParams(),
                Distance d = Distance())
        : BaseClass(params,d), root_(NULL), memoryCounter_(0)
    {
        branching_ = get_param(params,"branching",32);
        iterations_ = get_param(params,"iterations",11);
        if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }
        centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);
        cb_index_  = get_param(params,"cb_index",0.4f);

        initCenterChooser();
        chooseCenters_->setDataset(inputData);

        setDataset(inputData);
    }


    /**
     * Index constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the hierarchical k-means algorithm
     */
    KMeansIndex(const IndexParams& params = KMeansIndexParams(), Distance d = Distance())
        : BaseClass(params, d), root_(NULL), memoryCounter_(0)
    {
        branching_ = get_param(params,"branching",32);
        iterations_ = get_param(params,"iterations",11);
        if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }
        centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);
        cb_index_  = get_param(params,"cb_index",0.4f);

        initCenterChooser();
    }


    KMeansIndex(const KMeansIndex& other) : BaseClass(other),
    		branching_(other.branching_),
    		iterations_(other.iterations_),
    		centers_init_(other.centers_init_),
    		cb_index_(other.cb_index_),
    		memoryCounter_(other.memoryCounter_)
    {
    	initCenterChooser();

    	copyTree(root_, other.root_);
    }

    KMeansIndex& operator=(KMeansIndex other)
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
    virtual ~KMeansIndex()
    {
    	delete chooseCenters_;
    	freeIndex();
    }

    BaseClass* clone() const
    {
    	return new KMeansIndex(*this);
    }


    void set_cb_index( float index)
    {
        cb_index_ = index;
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
                DistanceType dist = distance_(root_->pivot, points[i], veclen_);
                addPointToTree(root_, old_size + i, dist);
            }
        }
    }

    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	ar & branching_;
    	ar & iterations_;
    	ar & memoryCounter_;
    	ar & cb_index_;
    	ar & centers_init_;

    	if (Archive::is_loading::value) {
    		root_ = new(pool_) Node();
    	}
    	ar & *root_;

    	if (Archive::is_loading::value) {
            index_params_["algorithm"] = getType();
            index_params_["branching"] = branching_;
            index_params_["iterations"] = iterations_;
            index_params_["centers_init"] = centers_init_;
            index_params_["cb_index"] = cb_index_;
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
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     searchParams = parameters that influence the search algorithm (checks, cb_index)
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

    /**
     * Clustering function that takes a cut in the hierarchical k-means
     * tree and return the clusters centers of that clustering.
     * Params:
     *     numClusters = number of clusters to have in the clustering computed
     * Returns: number of cluster centers
     */
    int getClusterCenters(Matrix<DistanceType>& centers)
    {
        int numClusters = centers.rows;
        if (numClusters<1) {
            throw FLANNException("Number of clusters must be at least 1");
        }

        DistanceType variance;
        std::vector<NodePtr> clusters(numClusters);

        int clusterCount = getMinVarianceClusters(root_, clusters, numClusters, variance);

        Logger::info("Clusters requested: %d, returning %d\n",numClusters, clusterCount);

        for (int i=0; i<clusterCount; ++i) {
            DistanceType* center = clusters[i]->pivot;
            for (size_t j=0; j<veclen_; ++j) {
                centers[i][j] = center[j];
            }
        }

        return clusterCount;
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

        std::vector<int> indices(size_);
        for (size_t i=0; i<size_; ++i) {
        	indices[i] = int(i);
        }

        root_ = new(pool_) Node();
        computeNodeStatistics(root_, indices);
        computeClustering(root_, &indices[0], (int)size_, branching_);
    }

private:

    struct PointInfo
    {
    	size_t index;
    	ElementType* point;
    private:
    	template<typename Archive>
    	void serialize(Archive& ar)
    	{
    		typedef KMeansIndex<Distance> Index;
    		Index* obj = static_cast<Index*>(ar.getObject());

    		ar & index;
//    		ar & point;

			if (Archive::is_loading::value) point = obj->points_[index];
    	}
    	friend struct serialization::access;
    };

    /**
     * Struture representing a node in the hierarchical k-means tree.
     */
    struct Node
    {
        /**
         * The cluster center.
         */
        DistanceType* pivot;
        /**
         * The cluster radius.
         */
        DistanceType radius;
        /**
         * The cluster variance.
         */
        DistanceType variance;
        /**
         * The cluster size (number of points in the cluster)
         */
        int size;
        /**
         * Child nodes (only for non-terminal nodes)
         */
        std::vector<Node*> childs;
        /**
         * Node points (only for terminal nodes)
         */
        std::vector<PointInfo> points;
        /**
         * Level
         */
//        int level;

        ~Node()
        {
            delete[] pivot;
            if (!childs.empty()) {
                for (size_t i=0; i<childs.size(); ++i) {
                    childs[i]->~Node();
                }
            }
        }

    	template<typename Archive>
    	void serialize(Archive& ar)
    	{
    		typedef KMeansIndex<Distance> Index;
    		Index* obj = static_cast<Index*>(ar.getObject());

    		if (Archive::is_loading::value) {
    			pivot = new DistanceType[obj->veclen_];
    		}
    		ar & serialization::make_binary_object(pivot, obj->veclen_*sizeof(DistanceType));
    		ar & radius;
    		ar & variance;
    		ar & size;

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
     * Helper function
     */
    void freeIndex()
    {
    	if (root_) root_->~Node();
    	root_ = NULL;
    	pool_.free();
    }

    void copyTree(NodePtr& dst, const NodePtr& src)
    {
    	dst = new(pool_) Node();
    	dst->pivot = new DistanceType[veclen_];
    	std::copy(src->pivot, src->pivot+veclen_, dst->pivot);
    	dst->radius = src->radius;
    	dst->variance = src->variance;
    	dst->size = src->size;

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


    /**
     * Computes the statistics of a node (mean, radius, variance).
     *
     * Params:
     *     node = the node to use
     *     indices = the indices of the points belonging to the node
     */
    void computeNodeStatistics(NodePtr node, const std::vector<int>& indices)
    {
        size_t size = indices.size();

        DistanceType* mean = new DistanceType[veclen_];
        memoryCounter_ += int(veclen_*sizeof(DistanceType));
        memset(mean,0,veclen_*sizeof(DistanceType));

        for (size_t i=0; i<size; ++i) {
            ElementType* vec = points_[indices[i]];
            for (size_t j=0; j<veclen_; ++j) {
                mean[j] += vec[j];
            }
        }
        DistanceType div_factor = DistanceType(1)/size;
        for (size_t j=0; j<veclen_; ++j) {
            mean[j] *= div_factor;
        }

        DistanceType radius = 0;
        DistanceType variance = 0;
        for (size_t i=0; i<size; ++i) {
            DistanceType dist = distance_(mean, points_[indices[i]], veclen_);
            if (dist>radius) {
                radius = dist;
            }
            variance += dist;
        }
        variance /= size;

        node->variance = variance;
        node->radius = radius;
        node->pivot = mean;
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
     * TODO: for 1-sized clusters don't store a cluster center (it's the same as the single cluster point)
     */
    void computeClustering(NodePtr node, int* indices, int indices_length, int branching)
    {
        node->size = indices_length;

        if (indices_length < branching) {
            node->points.resize(indices_length);
            for (int i=0;i<indices_length;++i) {
            	node->points[i].index = indices[i];
            	node->points[i].point = points_[indices[i]];
            }
            node->childs.clear();
            return;
        }

        std::vector<int> centers_idx(branching);
        int centers_length;
        (*chooseCenters_)(branching, indices, indices_length, &centers_idx[0], centers_length);

        if (centers_length<branching) {
            node->points.resize(indices_length);
            for (int i=0;i<indices_length;++i) {
            	node->points[i].index = indices[i];
            	node->points[i].point = points_[indices[i]];
            }
            node->childs.clear();
            return;
        }


        Matrix<double> dcenters(new double[branching*veclen_],branching,veclen_);
        for (int i=0; i<centers_length; ++i) {
            ElementType* vec = points_[centers_idx[i]];
            for (size_t k=0; k<veclen_; ++k) {
                dcenters[i][k] = double(vec[k]);
            }
        }

        std::vector<DistanceType> radiuses(branching,0);
        std::vector<int> count(branching,0);

        //	assign points to clusters
        std::vector<int> belongs_to(indices_length);
        for (int i=0; i<indices_length; ++i) {

            DistanceType sq_dist = distance_(points_[indices[i]], dcenters[0], veclen_);
            belongs_to[i] = 0;
            for (int j=1; j<branching; ++j) {
                DistanceType new_sq_dist = distance_(points_[indices[i]], dcenters[j], veclen_);
                if (sq_dist>new_sq_dist) {
                    belongs_to[i] = j;
                    sq_dist = new_sq_dist;
                }
            }
            if (sq_dist>radiuses[belongs_to[i]]) {
                radiuses[belongs_to[i]] = sq_dist;
            }
            count[belongs_to[i]]++;
        }

        bool converged = false;
        int iteration = 0;
        while (!converged && iteration<iterations_) {
            converged = true;
            iteration++;

            // compute the new cluster centers
            for (int i=0; i<branching; ++i) {
                memset(dcenters[i],0,sizeof(double)*veclen_);
                radiuses[i] = 0;
            }
            for (int i=0; i<indices_length; ++i) {
                ElementType* vec = points_[indices[i]];
                double* center = dcenters[belongs_to[i]];
                for (size_t k=0; k<veclen_; ++k) {
                    center[k] += vec[k];
                }
            }
            for (int i=0; i<branching; ++i) {
                int cnt = count[i];
                double div_factor = 1.0/cnt;
                for (size_t k=0; k<veclen_; ++k) {
                    dcenters[i][k] *= div_factor;
                }
            }

            // reassign points to clusters
            for (int i=0; i<indices_length; ++i) {
                DistanceType sq_dist = distance_(points_[indices[i]], dcenters[0], veclen_);
                int new_centroid = 0;
                for (int j=1; j<branching; ++j) {
                    DistanceType new_sq_dist = distance_(points_[indices[i]], dcenters[j], veclen_);
                    if (sq_dist>new_sq_dist) {
                        new_centroid = j;
                        sq_dist = new_sq_dist;
                    }
                }
                if (sq_dist>radiuses[new_centroid]) {
                    radiuses[new_centroid] = sq_dist;
                }
                if (new_centroid != belongs_to[i]) {
                    count[belongs_to[i]]--;
                    count[new_centroid]++;
                    belongs_to[i] = new_centroid;

                    converged = false;
                }
            }

            for (int i=0; i<branching; ++i) {
                // if one cluster converges to an empty cluster,
                // move an element into that cluster
                if (count[i]==0) {
                    int j = (i+1)%branching;
                    while (count[j]<=1) {
                        j = (j+1)%branching;
                    }

                    for (int k=0; k<indices_length; ++k) {
                        if (belongs_to[k]==j) {
                            belongs_to[k] = i;
                            count[j]--;
                            count[i]++;
                            break;
                        }
                    }
                    converged = false;
                }
            }

        }

        std::vector<DistanceType*> centers(branching);

        for (int i=0; i<branching; ++i) {
            centers[i] = new DistanceType[veclen_];
            memoryCounter_ += veclen_*sizeof(DistanceType);
            for (size_t k=0; k<veclen_; ++k) {
                centers[i][k] = (DistanceType)dcenters[i][k];
            }
        }


        // compute kmeans clustering for each of the resulting clusters
        node->childs.resize(branching);
        int start = 0;
        int end = start;
        for (int c=0; c<branching; ++c) {
            int s = count[c];

            DistanceType variance = 0;
            for (int i=0; i<indices_length; ++i) {
                if (belongs_to[i]==c) {
                    variance += distance_(centers[c], points_[indices[i]], veclen_);
                    std::swap(indices[i],indices[end]);
                    std::swap(belongs_to[i],belongs_to[end]);
                    end++;
                }
            }
            variance /= s;

            node->childs[c] = new(pool_) Node();
            node->childs[c]->radius = radiuses[c];
            node->childs[c]->pivot = centers[c];
            node->childs[c]->variance = variance;
            computeClustering(node->childs[c],indices+start, end-start, branching);
            start=end;
        }

        delete[] dcenters.ptr();
    }


    template<bool with_removed>
    void findNeighborsWithRemoved(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {

        int maxChecks = searchParams.checks;

        if (maxChecks==FLANN_CHECKS_UNLIMITED) {
            findExactNN<with_removed>(root_, result, vec);
        }
        else {
            // Priority queue storing intermediate branches in the best-bin-first search
            Heap<BranchSt>* heap = new Heap<BranchSt>((int)size_);

            int checks = 0;
            findNN<with_removed>(root_, result, vec, checks, maxChecks, heap);

            BranchSt branch;
            while (heap->popMin(branch) && (checks<maxChecks || !result.full())) {
                NodePtr node = branch.node;
                findNN<with_removed>(node, result, vec, checks, maxChecks, heap);
            }

            delete heap;
        }

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
                Heap<BranchSt>* heap) const
    {
        // Ignore those clusters that are too far away
        {
            DistanceType bsq = distance_(vec, node->pivot, veclen_);
            DistanceType rsq = node->radius;
            DistanceType wsq = result.worstDist();

            DistanceType val = bsq-rsq-wsq;
            DistanceType val2 = val*val-4*rsq*wsq;

            //if (val>0) {
            if ((val>0)&&(val2>0)) {
                return;
            }
        }

        if (node->childs.empty()) {
            if (checks>=maxChecks) {
                if (result.full()) return;
            }
            for (int i=0; i<node->size; ++i) {
            	PointInfo& point_info = node->points[i];
                int index = point_info.index;
                if (with_removed) {
                	if (removed_points_.test(index)) continue;
                }
                DistanceType dist = distance_(point_info.point, vec, veclen_);
                result.addPoint(dist, index);
                ++checks;
            }
        }
        else {
            int closest_center = exploreNodeBranches(node, vec, heap);
            findNN<with_removed>(node->childs[closest_center],result,vec, checks, maxChecks, heap);
        }
    }

    /**
     * Helper function that computes the nearest childs of a node to a given query point.
     * Params:
     *     node = the node
     *     q = the query point
     *     distances = array with the distances to each child node.
     * Returns:
     */
    int exploreNodeBranches(NodePtr node, const ElementType* q, Heap<BranchSt>* heap) const
    {
        std::vector<DistanceType> domain_distances(branching_);
        int best_index = 0;
        domain_distances[best_index] = distance_(q, node->childs[best_index]->pivot, veclen_);
        for (int i=1; i<branching_; ++i) {
            domain_distances[i] = distance_(q, node->childs[i]->pivot, veclen_);
            if (domain_distances[i]<domain_distances[best_index]) {
                best_index = i;
            }
        }

        //		float* best_center = node->childs[best_index]->pivot;
        for (int i=0; i<branching_; ++i) {
            if (i != best_index) {
                domain_distances[i] -= cb_index_*node->childs[i]->variance;

                //				float dist_to_border = getDistanceToBorder(node.childs[i].pivot,best_center,q);
                //				if (domain_distances[i]<dist_to_border) {
                //					domain_distances[i] = dist_to_border;
                //				}
                heap->insert(BranchSt(node->childs[i],domain_distances[i]));
            }
        }

        return best_index;
    }


    /**
     * Function the performs exact nearest neighbor search by traversing the entire tree.
     */
    template<bool with_removed>
    void findExactNN(NodePtr node, ResultSet<DistanceType>& result, const ElementType* vec) const
    {
        // Ignore those clusters that are too far away
        {
            DistanceType bsq = distance_(vec, node->pivot, veclen_);
            DistanceType rsq = node->radius;
            DistanceType wsq = result.worstDist();

            DistanceType val = bsq-rsq-wsq;
            DistanceType val2 = val*val-4*rsq*wsq;

            //                  if (val>0) {
            if ((val>0)&&(val2>0)) {
                return;
            }
        }

        if (node->childs.empty()) {
            for (int i=0; i<node->size; ++i) {
            	PointInfo& point_info = node->points[i];
                int index = point_info.index;
                if (with_removed) {
                	if (removed_points_.test(index)) continue;
                }
                DistanceType dist = distance_(point_info.point, vec, veclen_);
                result.addPoint(dist, index);
            }
        }
        else {
            std::vector<int> sort_indices(branching_);
            getCenterOrdering(node, vec, sort_indices);

            for (int i=0; i<branching_; ++i) {
                findExactNN<with_removed>(node->childs[sort_indices[i]],result,vec);
            }

        }
    }


    /**
     * Helper function.
     *
     * I computes the order in which to traverse the child nodes of a particular node.
     */
    void getCenterOrdering(NodePtr node, const ElementType* q, std::vector<int>& sort_indices) const
    {
        std::vector<DistanceType> domain_distances(branching_);
        for (int i=0; i<branching_; ++i) {
            DistanceType dist = distance_(q, node->childs[i]->pivot, veclen_);

            int j=0;
            while (domain_distances[j]<dist && j<i) j++;
            for (int k=i; k>j; --k) {
                domain_distances[k] = domain_distances[k-1];
                sort_indices[k] = sort_indices[k-1];
            }
            domain_distances[j] = dist;
            sort_indices[j] = i;
        }
    }

    /**
     * Method that computes the squared distance from the query point q
     * from inside region with center c to the border between this
     * region and the region with center p
     */
    DistanceType getDistanceToBorder(DistanceType* p, DistanceType* c, DistanceType* q) const
    {
        DistanceType sum = 0;
        DistanceType sum2 = 0;

        for (int i=0; i<veclen_; ++i) {
            DistanceType t = c[i]-p[i];
            sum += t*(q[i]-(c[i]+p[i])/2);
            sum2 += t*t;
        }

        return sum*sum/sum2;
    }


    /**
     * Helper function the descends in the hierarchical k-means tree by spliting those clusters that minimize
     * the overall variance of the clustering.
     * Params:
     *     root = root node
     *     clusters = array with clusters centers (return value)
     *     varianceValue = variance of the clustering (return value)
     * Returns:
     */
    int getMinVarianceClusters(NodePtr root, std::vector<NodePtr>& clusters, int clusters_length, DistanceType& varianceValue) const
    {
        int clusterCount = 1;
        clusters[0] = root;

        DistanceType meanVariance = root->variance*root->size;

        while (clusterCount<clusters_length) {
            DistanceType minVariance = (std::numeric_limits<DistanceType>::max)();
            int splitIndex = -1;

            for (int i=0; i<clusterCount; ++i) {
                if (!clusters[i]->childs.empty()) {

                    DistanceType variance = meanVariance - clusters[i]->variance*clusters[i]->size;

                    for (int j=0; j<branching_; ++j) {
                        variance += clusters[i]->childs[j]->variance*clusters[i]->childs[j]->size;
                    }
                    if (variance<minVariance) {
                        minVariance = variance;
                        splitIndex = i;
                    }
                }
            }

            if (splitIndex==-1) break;
            if ( (branching_+clusterCount-1) > clusters_length) break;

            meanVariance = minVariance;

            // split node
            NodePtr toSplit = clusters[splitIndex];
            clusters[splitIndex] = toSplit->childs[0];
            for (int i=1; i<branching_; ++i) {
                clusters[clusterCount++] = toSplit->childs[i];
            }
        }

        varianceValue = meanVariance/root->size;
        return clusterCount;
    }

    void addPointToTree(NodePtr node, size_t index, DistanceType dist_to_pivot)
    {
        ElementType* point = points_[index];
        if (dist_to_pivot>node->radius) {
            node->radius = dist_to_pivot;
        }
        // if radius changed above, the variance will be an approximation
        node->variance = (node->size*node->variance+dist_to_pivot)/(node->size+1);
        node->size++;

        if (node->childs.empty()) { // leaf node
        	PointInfo point_info;
        	point_info.index = index;
        	point_info.point = point;
        	node->points.push_back(point_info);

            std::vector<int> indices(node->points.size());
            for (size_t i=0;i<node->points.size();++i) {
            	indices[i] = node->points[i].index;
            }
            computeNodeStatistics(node, indices);
            if (indices.size()>=size_t(branching_)) {
                computeClustering(node, &indices[0], indices.size(), branching_);
            }
        }
        else {
            // find the closest child
            int closest = 0;
            DistanceType dist = distance_(node->childs[closest]->pivot, point, veclen_);
            for (size_t i=1;i<size_t(branching_);++i) {
                DistanceType crt_dist = distance_(node->childs[i]->pivot, point, veclen_);
                if (crt_dist<dist) {
                    dist = crt_dist;
                    closest = i;
                }
            }
            addPointToTree(node->childs[closest], index, dist);
        }
    }


    void swap(KMeansIndex& other)
    {
    	std::swap(branching_, other.branching_);
    	std::swap(iterations_, other.iterations_);
    	std::swap(centers_init_, other.centers_init_);
    	std::swap(cb_index_, other.cb_index_);
    	std::swap(root_, other.root_);
    	std::swap(pool_, other.pool_);
    	std::swap(memoryCounter_, other.memoryCounter_);
    	std::swap(chooseCenters_, other.chooseCenters_);
    }


private:
    /** The branching factor used in the hierarchical k-means clustering */
    int branching_;

    /** Maximum number of iterations to use when performing k-means clustering */
    int iterations_;

    /** Algorithm for choosing the cluster centers */
    flann_centers_init_t centers_init_;

    /**
     * Cluster border index. This is used in the tree search phase when determining
     * the closest cluster to explore next. A zero value takes into account only
     * the cluster centres, a value greater then zero also take into account the size
     * of the cluster.
     */
    float cb_index_;

    /**
     * The root node in the tree.
     */
    NodePtr root_;

    /**
     * Pooled memory allocator.
     */
    PooledAllocator pool_;

    /**
     * Memory occupied by the index.
     */
    int memoryCounter_;

    /**
     * Algorithm used to choose initial centers
     */
    CenterChooser<Distance>* chooseCenters_;

    USING_BASECLASS_SYMBOLS
};

}

#endif //FLANN_KMEANS_INDEX_H_
