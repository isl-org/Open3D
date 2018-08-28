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

#ifndef FLANN_NNINDEX_H
#define FLANN_NNINDEX_H

#include <vector>

#include "flann/general.h"
#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include "flann/util/result_set.h"
#include "flann/util/dynamic_bitset.h"
#include "flann/util/saving.h"

namespace flann
{

#define KNN_HEAP_THRESHOLD 250


class IndexBase
{
public:
    virtual ~IndexBase() {};

    virtual size_t veclen() const = 0;

    virtual size_t size() const = 0;

    virtual flann_algorithm_t getType() const = 0;

    virtual int usedMemory() const = 0;

    virtual IndexParams getParameters() const = 0;

    virtual void loadIndex(FILE* stream) = 0;

    virtual void saveIndex(FILE* stream) = 0;
};

/**
 * Nearest-neighbour index base class
 */
template <typename Distance>
class NNIndex : public IndexBase
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

	NNIndex(Distance d) : distance_(d), last_id_(0), size_(0), size_at_build_(0), veclen_(0),
			removed_(false), removed_count_(0), data_ptr_(NULL)
	{
	}

	NNIndex(const IndexParams& params, Distance d) : distance_(d), last_id_(0), size_(0), size_at_build_(0), veclen_(0),
			index_params_(params), removed_(false), removed_count_(0), data_ptr_(NULL)
	{
	}

	NNIndex(const NNIndex& other) :
		distance_(other.distance_),
		last_id_(other.last_id_),
		size_(other.size_),
		size_at_build_(other.size_at_build_),
		veclen_(other.veclen_),
		index_params_(other.index_params_),
		removed_(other.removed_),
		removed_points_(other.removed_points_),
		removed_count_(other.removed_count_),
		ids_(other.ids_),
		points_(other.points_),
		data_ptr_(NULL)
	{
		if (other.data_ptr_) {
			data_ptr_ = new ElementType[size_*veclen_];
			std::copy(other.data_ptr_, other.data_ptr_+size_*veclen_, data_ptr_);
			for (size_t i=0;i<size_;++i) {
				points_[i] = data_ptr_ + i*veclen_;
			}
		}
	}

	virtual ~NNIndex()
	{
		if (data_ptr_) {
			delete[] data_ptr_;
		}
	}


	virtual NNIndex* clone() const = 0;

	/**
	 * Builds the index
	 */
	virtual void buildIndex()
	{
    	freeIndex();
    	cleanRemovedPoints();

    	// building index
		buildIndexImpl();

        size_at_build_ = size_;

	}

	/**
	 * Builds th index using using the specified dataset
	 * @param dataset the dataset to use
	 */
    virtual void buildIndex(const Matrix<ElementType>& dataset)
    {
        setDataset(dataset);
        this->buildIndex();
    }

	/**
	 * @brief Incrementally add points to the index.
	 * @param points Matrix with points to be added
	 * @param rebuild_threshold
	 */
    virtual void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        throw FLANNException("Functionality not supported by this index");
    }

    /**
     * Remove point from the index
     * @param index Index of point to be removed
     */
    virtual void removePoint(size_t id)
    {
    	if (!removed_) {
    		ids_.resize(size_);
    		for (size_t i=0;i<size_;++i) {
    			ids_[i] = i;
    		}
    		removed_points_.resize(size_);
    		removed_points_.reset();
    		last_id_ = size_;
        	removed_ = true;
    	}

    	size_t point_index = id_to_index(id);
    	if (point_index!=size_t(-1) && !removed_points_.test(point_index)) {
    		removed_points_.set(point_index);
    		removed_count_++;
    	}
    }


    /**
     * Get point with specific id
     * @param id
     * @return
     */
    virtual ElementType* getPoint(size_t id)
    {
    	size_t index = id_to_index(id);
    	if (index!=size_t(-1)) {
    		return points_[index];
    	}
    	else {
    		return NULL;
    	}
    }

    /**
     * @return number of features in this index.
     */
    inline size_t size() const
    {
    	return size_ - removed_count_;
    }

    /**
     * @return The dimensionality of the features in this index.
     */
    inline size_t veclen() const
    {
        return veclen_;
    }

    /**
     * Returns the parameters used by the index.
     *
     * @return The index parameters
     */
    IndexParams getParameters() const
    {
        return index_params_;
    }


    template<typename Archive>
    void serialize(Archive& ar)
    {
    	IndexHeader header;

    	if (Archive::is_saving::value) {
    		header.data_type = flann_datatype_value<ElementType>::value;
    		header.index_type = getType();
    		header.rows = size_;
    		header.cols = veclen_;
    	}
    	ar & header;

    	// sanity checks
    	if (Archive::is_loading::value) {
    	    if (strcmp(header.signature,FLANN_SIGNATURE_)!=0) {
    	        throw FLANNException("Invalid index file, wrong signature");
    	    }
            if (header.data_type != flann_datatype_value<ElementType>::value) {
                throw FLANNException("Datatype of saved index is different than of the one to be created.");
            }
            if (header.index_type != getType()) {
                throw FLANNException("Saved index type is different then the current index type.");
            }
            // TODO: check for distance type

    	}

    	ar & size_;
    	ar & veclen_;
    	ar & size_at_build_;

    	bool save_dataset;
    	if (Archive::is_saving::value) {
    		save_dataset = get_param(index_params_,"save_dataset", false);
    	}
    	ar & save_dataset;

    	if (save_dataset) {
    		if (Archive::is_loading::value) {
    			if (data_ptr_) {
    				delete[] data_ptr_;
    			}
    			data_ptr_ = new ElementType[size_*veclen_];
    			points_.resize(size_);
        		for (size_t i=0;i<size_;++i) {
        			points_[i] = data_ptr_ + i*veclen_;
        		}
    		}
    		for (size_t i=0;i<size_;++i) {
    			ar & serialization::make_binary_object (points_[i], veclen_*sizeof(ElementType));
    		}
    	} else {
    		if (points_.size()!=size_) {
    			throw FLANNException("Saved index does not contain the dataset and no dataset was provided.");
    		}
    	}

    	ar & last_id_;
    	ar & ids_;
    	ar & removed_;
    	if (removed_) {
    		ar & removed_points_;
    	}
    	ar & removed_count_;
    }


    /**
     * @brief Perform k-nearest neighbor search
     * @param[in] queries The query points for which to find the nearest neighbors
     * @param[out] indices The indices of the nearest neighbors found
     * @param[out] dists Distances to the nearest neighbors found
     * @param[in] knn Number of nearest neighbors to return
     * @param[in] params Search parameters
     */
    virtual int knnSearch(const Matrix<ElementType>& queries,
    		Matrix<size_t>& indices,
    		Matrix<DistanceType>& dists,
    		size_t knn,
    		const SearchParams& params) const
    {
    	assert(queries.cols == veclen());
    	assert(indices.rows >= queries.rows);
    	assert(dists.rows >= queries.rows);
    	assert(indices.cols >= knn);
    	assert(dists.cols >= knn);
    	bool use_heap;

    	if (params.use_heap==FLANN_Undefined) {
    		use_heap = (knn>KNN_HEAP_THRESHOLD)?true:false;
    	}
    	else {
    		use_heap = (params.use_heap==FLANN_True)?true:false;
    	}
    	int count = 0;

    	if (use_heap) {
#pragma omp parallel num_threads(params.cores)
    		{
    			KNNResultSet2<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
    			for (int i = 0; i < (int)queries.rows; i++) {
    				resultSet.clear();
    				findNeighbors(resultSet, queries[i], params);
    				size_t n = std::min(resultSet.size(), knn);
    				resultSet.copy(indices[i], dists[i], n, params.sorted);
    				indices_to_ids(indices[i], indices[i], n);
    				count += n;
    			}
    		}
    	}
    	else {
#pragma omp parallel num_threads(params.cores)
    		{
    			KNNSimpleResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
    			for (int i = 0; i < (int)queries.rows; i++) {
    				resultSet.clear();
    				findNeighbors(resultSet, queries[i], params);
    				size_t n = std::min(resultSet.size(), knn);
    				resultSet.copy(indices[i], dists[i], n, params.sorted);
    				indices_to_ids(indices[i], indices[i], n);
    				count += n;
    			}
    		}
    	}
    	return count;
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
    	flann::Matrix<size_t> indices_(new size_t[indices.rows*indices.cols], indices.rows, indices.cols);
    	int result = knnSearch(queries, indices_, dists, knn, params);

    	for (size_t i=0;i<indices.rows;++i) {
    		for (size_t j=0;j<indices.cols;++j) {
    			indices[i][j] = indices_[i][j];
    		}
    	}
        delete[] indices_.ptr();
    	return result;
    }


    /**
     * @brief Perform k-nearest neighbor search
     * @param[in] queries The query points for which to find the nearest neighbors
     * @param[out] indices The indices of the nearest neighbors found
     * @param[out] dists Distances to the nearest neighbors found
     * @param[in] knn Number of nearest neighbors to return
     * @param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
					std::vector< std::vector<size_t> >& indices,
					std::vector<std::vector<DistanceType> >& dists,
    				size_t knn,
    				const SearchParams& params) const
    {
        assert(queries.cols == veclen());
        bool use_heap;
        if (params.use_heap==FLANN_Undefined) {
        	use_heap = (knn>KNN_HEAP_THRESHOLD)?true:false;
        }
        else {
        	use_heap = (params.use_heap==FLANN_True)?true:false;
        }

        if (indices.size() < queries.rows ) indices.resize(queries.rows);
		if (dists.size() < queries.rows ) dists.resize(queries.rows);

		int count = 0;
		if (use_heap) {
#pragma omp parallel num_threads(params.cores)
			{
				KNNResultSet2<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					findNeighbors(resultSet, queries[i], params);
					size_t n = std::min(resultSet.size(), knn);
					indices[i].resize(n);
					dists[i].resize(n);
					if (n>0) {
						resultSet.copy(&indices[i][0], &dists[i][0], n, params.sorted);
						indices_to_ids(&indices[i][0], &indices[i][0], n);
					}
					count += n;
				}
			}
		}
		else {
#pragma omp parallel num_threads(params.cores)
			{
				KNNSimpleResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					findNeighbors(resultSet, queries[i], params);
					size_t n = std::min(resultSet.size(), knn);
					indices[i].resize(n);
					dists[i].resize(n);
					if (n>0) {
						resultSet.copy(&indices[i][0], &dists[i][0], n, params.sorted);
						indices_to_ids(&indices[i][0], &indices[i][0], n);
					}
					count += n;
				}
			}
		}

		return count;
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
    	std::vector<std::vector<size_t> > indices_;
    	int result = knnSearch(queries, indices_, dists, knn, params);

    	indices.resize(indices_.size());
    	for (size_t i=0;i<indices_.size();++i) {
            indices[i].assign(indices_[i].begin(), indices_[i].end());
    	}
    	return result;
    }

    /**
     * @brief Perform radius search
     * @param[in] query The query point
     * @param[out] indices The indinces of the neighbors found within the given radius
     * @param[out] dists The distances to the nearest neighbors found
     * @param[in] radius The radius used for search
     * @param[in] params Search parameters
     * @return Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
    		Matrix<size_t>& indices,
    		Matrix<DistanceType>& dists,
    		float radius,
    		const SearchParams& params) const
    {
    	assert(queries.cols == veclen());
    	int count = 0;
    	size_t num_neighbors = std::min(indices.cols, dists.cols);
    	int max_neighbors = params.max_neighbors;
    	if (max_neighbors<0) max_neighbors = num_neighbors;
    	else max_neighbors = std::min(max_neighbors,(int)num_neighbors);

    	if (max_neighbors==0) {
#pragma omp parallel num_threads(params.cores)
    		{
    			CountRadiusResultSet<DistanceType> resultSet(radius);
#pragma omp for schedule(static) reduction(+:count)
    			for (int i = 0; i < (int)queries.rows; i++) {
    				resultSet.clear();
    				findNeighbors(resultSet, queries[i], params);
    				count += resultSet.size();
    			}
    		}
    	}
    	else {
    		// explicitly indicated to use unbounded radius result set
    		// and we know there'll be enough room for resulting indices and dists
    		if (params.max_neighbors<0 && (num_neighbors>=size())) {
#pragma omp parallel num_threads(params.cores)
    			{
    				RadiusResultSet<DistanceType> resultSet(radius);
#pragma omp for schedule(static) reduction(+:count)
    				for (int i = 0; i < (int)queries.rows; i++) {
    					resultSet.clear();
    					findNeighbors(resultSet, queries[i], params);
    					size_t n = resultSet.size();
    					count += n;
    					if (n>num_neighbors) n = num_neighbors;
    					resultSet.copy(indices[i], dists[i], n, params.sorted);

    					// mark the next element in the output buffers as unused
    					if (n<indices.cols) indices[i][n] = size_t(-1);
    					if (n<dists.cols) dists[i][n] = std::numeric_limits<DistanceType>::infinity();
    					indices_to_ids(indices[i], indices[i], n);
    				}
    			}
    		}
    		else {
    			// number of neighbors limited to max_neighbors
#pragma omp parallel num_threads(params.cores)
    			{
    				KNNRadiusResultSet<DistanceType> resultSet(radius, max_neighbors);
#pragma omp for schedule(static) reduction(+:count)
    				for (int i = 0; i < (int)queries.rows; i++) {
    					resultSet.clear();
    					findNeighbors(resultSet, queries[i], params);
    					size_t n = resultSet.size();
    					count += n;
    					if ((int)n>max_neighbors) n = max_neighbors;
    					resultSet.copy(indices[i], dists[i], n, params.sorted);

    					// mark the next element in the output buffers as unused
    					if (n<indices.cols) indices[i][n] = size_t(-1);
    					if (n<dists.cols) dists[i][n] = std::numeric_limits<DistanceType>::infinity();
    					indices_to_ids(indices[i], indices[i], n);
    				}
    			}
    		}
    	}
        return count;
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
    	flann::Matrix<size_t> indices_(new size_t[indices.rows*indices.cols], indices.rows, indices.cols);
    	int result = radiusSearch(queries, indices_, dists, radius, params);

    	for (size_t i=0;i<indices.rows;++i) {
    		for (size_t j=0;j<indices.cols;++j) {
    			indices[i][j] = indices_[i][j];
    		}
    	}
        delete[] indices_.ptr();
    	return result;
    }

    /**
     * @brief Perform radius search
     * @param[in] query The query point
     * @param[out] indices The indinces of the neighbors found within the given radius
     * @param[out] dists The distances to the nearest neighbors found
     * @param[in] radius The radius used for search
     * @param[in] params Search parameters
     * @return Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
    		std::vector< std::vector<size_t> >& indices,
    		std::vector<std::vector<DistanceType> >& dists,
    		float radius,
    		const SearchParams& params) const
    {
        assert(queries.cols == veclen());
    	int count = 0;
    	// just count neighbors
    	if (params.max_neighbors==0) {
#pragma omp parallel num_threads(params.cores)
    		{
    			CountRadiusResultSet<DistanceType> resultSet(radius);
#pragma omp for schedule(static) reduction(+:count)
    			for (int i = 0; i < (int)queries.rows; i++) {
    				resultSet.clear();
    				findNeighbors(resultSet, queries[i], params);
    				count += resultSet.size();
    			}
    		}
    	}
    	else {
    		if (indices.size() < queries.rows ) indices.resize(queries.rows);
    		if (dists.size() < queries.rows ) dists.resize(queries.rows);

    		if (params.max_neighbors<0) {
    			// search for all neighbors
#pragma omp parallel num_threads(params.cores)
    			{
    				RadiusResultSet<DistanceType> resultSet(radius);
#pragma omp for schedule(static) reduction(+:count)
    				for (int i = 0; i < (int)queries.rows; i++) {
    					resultSet.clear();
    					findNeighbors(resultSet, queries[i], params);
    					size_t n = resultSet.size();
    					count += n;
    					indices[i].resize(n);
    					dists[i].resize(n);
    					if (n > 0) {
    						resultSet.copy(&indices[i][0], &dists[i][0], n, params.sorted);
        					indices_to_ids(&indices[i][0], &indices[i][0], n);
    					}
    				}
    			}
    		}
    		else {
    			// number of neighbors limited to max_neighbors
#pragma omp parallel num_threads(params.cores)
    			{
    				KNNRadiusResultSet<DistanceType> resultSet(radius, params.max_neighbors);
#pragma omp for schedule(static) reduction(+:count)
    				for (int i = 0; i < (int)queries.rows; i++) {
    					resultSet.clear();
    					findNeighbors(resultSet, queries[i], params);
    					size_t n = resultSet.size();
    					count += n;
    					if ((int)n>params.max_neighbors) n = params.max_neighbors;
    					indices[i].resize(n);
    					dists[i].resize(n);
    					if (n > 0) {
    						resultSet.copy(&indices[i][0], &dists[i][0], n, params.sorted);
        					indices_to_ids(&indices[i][0], &indices[i][0], n);
    					}
    				}
    			}
    		}
    	}
    	return count;
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
    	std::vector<std::vector<size_t> > indices_;
    	int result = radiusSearch(queries, indices_, dists, radius, params);

    	indices.resize(indices_.size());
    	for (size_t i=0;i<indices_.size();++i) {
            indices[i].assign(indices_[i].begin(), indices_[i].end());
    	}
    	return result;
    }


    virtual void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const = 0;

protected:

    virtual void freeIndex() = 0;

    virtual void buildIndexImpl() = 0;

    size_t id_to_index(size_t id)
    {
    	if (ids_.size()==0) {
    		return id;
    	}
    	size_t point_index = size_t(-1);
    	if (ids_[id]==id) {
    		return id;
    	}
    	else {
    		// binary search
    		size_t start = 0;
    		size_t end = ids_.size();

    		while (start<end) {
    			size_t mid = (start+end)/2;
    			if (ids_[mid]==id) {
    				point_index = mid;
    				break;
    			}
    			else if (ids_[mid]<id) {
    				start = mid + 1;
    			}
    			else {
    				end = mid;
    			}
    		}
    	}
    	return point_index;
    }


    void indices_to_ids(const size_t* in, size_t* out, size_t size) const
    {
		if (removed_) {
			for (size_t i=0;i<size;++i) {
				out[i] = ids_[in[i]];
			}
		}
    }

    void setDataset(const Matrix<ElementType>& dataset)
    {
    	size_ = dataset.rows;
    	veclen_ = dataset.cols;
    	last_id_ = 0;

    	ids_.clear();
    	removed_points_.clear();
    	removed_ = false;
    	removed_count_ = 0;

    	points_.resize(size_);
    	for (size_t i=0;i<size_;++i) {
    		points_[i] = dataset[i];
    	}
    }

    void extendDataset(const Matrix<ElementType>& new_points)
    {
    	size_t new_size = size_ + new_points.rows;
    	if (removed_) {
    		removed_points_.resize(new_size);
    		ids_.resize(new_size);
    	}
    	points_.resize(new_size);
    	for (size_t i=size_;i<new_size;++i) {
    		points_[i] = new_points[i-size_];
    		if (removed_) {
    			ids_[i] = last_id_++;
    			removed_points_.reset(i);
    		}
    	}
    	size_ = new_size;
    }


    void cleanRemovedPoints()
    {
    	if (!removed_) return;

    	size_t last_idx = 0;
    	for (size_t i=0;i<size_;++i) {
    		if (!removed_points_.test(i)) {
    			points_[last_idx] = points_[i];
    			ids_[last_idx] = ids_[i];
    			removed_points_.reset(last_idx);
    			++last_idx;
    		}
    	}
    	points_.resize(last_idx);
    	ids_.resize(last_idx);
    	removed_points_.resize(last_idx);
    	size_ = last_idx;
    	removed_count_ = 0;
    }

    void swap(NNIndex& other)
    {
    	std::swap(distance_, other.distance_);
    	std::swap(last_id_, other.last_id_);
    	std::swap(size_, other.size_);
    	std::swap(size_at_build_, other.size_at_build_);
    	std::swap(veclen_, other.veclen_);
    	std::swap(index_params_, other.index_params_);
    	std::swap(removed_, other.removed_);
    	std::swap(removed_points_, other.removed_points_);
    	std::swap(removed_count_, other.removed_count_);
    	std::swap(ids_, other.ids_);
    	std::swap(points_, other.points_);
    	std::swap(data_ptr_, other.data_ptr_);
    }

protected:

    /**
     * The distance functor
     */
    Distance distance_;


    /**
     * Each index point has an associated ID. IDs are assigned sequentially in
     * increasing order. This indicates the ID assigned to the last point added to the
     * index.
     */
    size_t last_id_;

    /**
     * Number of points in the index (and database)
     */
    size_t size_;

    /**
     * Number of features in the dataset when the index was last built.
     */
    size_t size_at_build_;

    /**
     * Size of one point in the index (and database)
     */
    size_t veclen_;

    /**
     * Parameters of the index.
     */
    IndexParams index_params_;

    /**
     * Flag indicating if at least a point was removed from the index
     */
    bool removed_;

    /**
     * Array used to mark points removed from the index
     */
    DynamicBitset removed_points_;

    /**
     * Number of points removed from the index
     */
    size_t removed_count_;

    /**
     * Array of point IDs, returned by nearest-neighbour operations
     */
    std::vector<size_t> ids_;

    /**
     * Point data
     */
    std::vector<ElementType*> points_;

    /**
     * Pointer to dataset memory if allocated by this index, otherwise NULL
     */
    ElementType* data_ptr_;


};


#define USING_BASECLASS_SYMBOLS \
		using NNIndex<Distance>::distance_;\
		using NNIndex<Distance>::size_;\
		using NNIndex<Distance>::size_at_build_;\
		using NNIndex<Distance>::veclen_;\
		using NNIndex<Distance>::index_params_;\
		using NNIndex<Distance>::removed_points_;\
		using NNIndex<Distance>::ids_;\
		using NNIndex<Distance>::removed_;\
		using NNIndex<Distance>::points_;\
		using NNIndex<Distance>::extendDataset;\
		using NNIndex<Distance>::setDataset;\
		using NNIndex<Distance>::cleanRemovedPoints;\
		using NNIndex<Distance>::indices_to_ids;



}


#endif //FLANN_NNINDEX_H
