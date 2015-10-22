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

#ifndef FLANN_HPP_
#define FLANN_HPP_


#include <vector>
#include <string>
#include <cassert>
#include <cstdio>

#include "flann/general.h"
#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include "flann/util/saving.h"

#include "flann/algorithms/all_indices.h"

namespace flann
{

/**
 * Sets the log level used for all flann functions
 * @param level Verbosity level
 */
inline void log_verbosity(int level)
{
    if (level >= 0) {
        Logger::setLevel(level);
    }
}

/**
 * (Deprecated) Index parameters for creating a saved index.
 */
struct SavedIndexParams : public IndexParams
{
    SavedIndexParams(std::string filename)
    {
        (*this)["algorithm"] = FLANN_INDEX_SAVED;
        (*this)["filename"] = filename;
    }
};



template<typename Distance>
class Index
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    typedef NNIndex<Distance> IndexType;

    Index(const IndexParams& params, Distance distance = Distance() )
        : index_params_(params)
    {
        flann_algorithm_t index_type = get_param<flann_algorithm_t>(params,"algorithm");
        loaded_ = false;

        Matrix<ElementType> features;
        if (index_type == FLANN_INDEX_SAVED) {
            nnIndex_ = load_saved_index(features, get_param<std::string>(params,"filename"), distance);
            loaded_ = true;
        }
        else {
        	flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
            nnIndex_ = create_index_by_type<Distance>(index_type, features, params, distance);
        }
    }


    Index(const Matrix<ElementType>& features, const IndexParams& params, Distance distance = Distance() )
        : index_params_(params)
    {
        flann_algorithm_t index_type = get_param<flann_algorithm_t>(params,"algorithm");
        loaded_ = false;

        if (index_type == FLANN_INDEX_SAVED) {
            nnIndex_ = load_saved_index(features, get_param<std::string>(params,"filename"), distance);
            loaded_ = true;
        }
        else {
        	flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
            nnIndex_ = create_index_by_type<Distance>(index_type, features, params, distance);
        }
    }


    Index(const Index& other) : loaded_(other.loaded_), index_params_(other.index_params_)
    {
    	nnIndex_ = other.nnIndex_->clone();
    }

    Index& operator=(Index other)
    {
    	this->swap(other);
    	return *this;
    }

    virtual ~Index()
    {
        delete nnIndex_;
    }

    /**
     * Builds the index.
     */
    void buildIndex()
    {
        if (!loaded_) {
            nnIndex_->buildIndex();
        }
    }

    void buildIndex(const Matrix<ElementType>& points)
    {
    	nnIndex_->buildIndex(points);
    }

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        nnIndex_->addPoints(points, rebuild_threshold);
    }

    /**
     * Remove point from the index
     * @param index Index of point to be removed
     */
    void removePoint(size_t point_id)
    {
    	nnIndex_->removePoint(point_id);
    }

    /**
     * Returns pointer to a data point with the specified id.
     * @param point_id the id of point to retrieve
     * @return
     */
    ElementType* getPoint(size_t point_id)
    {
    	return nnIndex_->getPoint(point_id);
    }

    /**
     * Save index to file
     * @param filename
     */
    void save(std::string filename)
    {
        FILE* fout = fopen(filename.c_str(), "wb");
        if (fout == NULL) {
            throw FLANNException("Cannot open file");
        }
        nnIndex_->saveIndex(fout);
        fclose(fout);
    }

    /**
     * \returns number of features in this index.
     */
    size_t veclen() const
    {
        return nnIndex_->veclen();
    }

    /**
     * \returns The dimensionality of the features in this index.
     */
    size_t size() const
    {
        return nnIndex_->size();
    }

    /**
     * \returns The index type (kdtree, kmeans,...)
     */
    flann_algorithm_t getType() const
    {
        return nnIndex_->getType();
    }

    /**
     * \returns The amount of memory (in bytes) used by the index.
     */
    int usedMemory() const
    {
        return nnIndex_->usedMemory();
    }


    /**
     * \returns The index parameters
     */
    IndexParams getParameters() const
    {
        return nnIndex_->getParameters();
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 Matrix<size_t>& indices,
                                 Matrix<DistanceType>& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param knn
     * @param params
     * @return
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 Matrix<int>& indices,
                                 Matrix<DistanceType>& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 std::vector< std::vector<size_t> >& indices,
                                 std::vector<std::vector<DistanceType> >& dists,
                                 size_t knn,
                           const SearchParams& params)
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param knn
     * @param params
     * @return
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 std::vector< std::vector<int> >& indices,
                                 std::vector<std::vector<DistanceType> >& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * \brief Perform radius search
     * \param[in] queries The query points
     * \param[out] indices The indinces of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    Matrix<size_t>& indices,
                                    Matrix<DistanceType>& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param radius
     * @param params
     * @return
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    Matrix<int>& indices,
                                    Matrix<DistanceType>& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     * \brief Perform radius search
     * \param[in] queries The query points
     * \param[out] indices The indinces of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    std::vector< std::vector<size_t> >& indices,
                                    std::vector<std::vector<DistanceType> >& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param radius
     * @param params
     * @return
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    std::vector< std::vector<int> >& indices,
                                    std::vector<std::vector<DistanceType> >& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

private:
    IndexType* load_saved_index(const Matrix<ElementType>& dataset, const std::string& filename, Distance distance)
    {
        FILE* fin = fopen(filename.c_str(), "rb");
        if (fin == NULL) {
            return NULL;
        }
        IndexHeader header = load_header(fin);
        if (header.data_type != flann_datatype_value<ElementType>::value) {
            throw FLANNException("Datatype of saved index is different than of the one to be created.");
        }

        IndexParams params;
        params["algorithm"] = header.index_type;
        IndexType* nnIndex = create_index_by_type<Distance>(header.index_type, dataset, params, distance);
        rewind(fin);
        nnIndex->loadIndex(fin);
        fclose(fin);

        return nnIndex;
    }

    void swap( Index& other)
    {
    	std::swap(nnIndex_, other.nnIndex_);
    	std::swap(loaded_, other.loaded_);
    	std::swap(index_params_, other.index_params_);
    }

private:
    /** Pointer to actual index class */
    IndexType* nnIndex_;
    /** Indices if the index was loaded from a file */
    bool loaded_;
    /** Parameters passed to the index */
    IndexParams index_params_;
};





/**
 * Performs a hierarchical clustering of the points passed as argument and then takes a cut in the
 * the clustering tree to return a flat clustering.
 * @param[in] points Points to be clustered
 * @param centers The computed cluster centres. Matrix should be preallocated and centers.rows is the
 *  number of clusters requested.
 * @param params Clustering parameters (The same as for flann::KMeansIndex)
 * @param d Distance to be used for clustering (eg: flann::L2)
 * @return number of clusters computed (can be different than clusters.rows and is the highest number
 * of the form (branching-1)*K+1 smaller than clusters.rows).
 */
template <typename Distance>
int hierarchicalClustering(const Matrix<typename Distance::ElementType>& points, Matrix<typename Distance::ResultType>& centers,
                           const KMeansIndexParams& params, Distance d = Distance())
{
    KMeansIndex<Distance> kmeans(points, params, d);
    kmeans.buildIndex();

    int clusterNum = kmeans.getClusterCenters(centers);
    return clusterNum;
}

}
#endif /* FLANN_HPP_ */
