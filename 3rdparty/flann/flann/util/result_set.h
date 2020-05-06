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

#ifndef FLANN_RESULTSET_H
#define FLANN_RESULTSET_H

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace flann
{

/* This record represents a branch point when finding neighbors in
    the tree.  It contains a record of the minimum distance to the query
    point, as well as the node at which the search resumes.
 */

template <typename T, typename DistanceType>
struct BranchStruct
{
    T node;           /* Tree node at which search resumes */
    DistanceType mindist;     /* Minimum distance to query for all nodes below. */

    BranchStruct() {}
    BranchStruct(const T& aNode, DistanceType dist) : node(aNode), mindist(dist) {}

    bool operator<(const BranchStruct<T, DistanceType>& rhs) const
    {
        return mindist<rhs.mindist;
    }
};


template <typename DistanceType>
struct DistanceIndex
{
    DistanceIndex(DistanceType dist, size_t index) :
        dist_(dist), index_(index)
    {
    }
    bool operator<(const DistanceIndex& dist_index) const
    {
        return (dist_ < dist_index.dist_) || ((dist_ == dist_index.dist_) && index_ < dist_index.index_);
    }
    DistanceType dist_;
    size_t index_;
};


template <typename DistanceType>
class ResultSet
{
public:
    virtual ~ResultSet() {}

    virtual bool full() const = 0;

    virtual void addPoint(DistanceType dist, size_t index) = 0;

    virtual DistanceType worstDist() const = 0;

};

/**
 * KNNSimpleResultSet does not ensure that the element it holds are unique.
 * Is used in those cases where the nearest neighbour algorithm used does not
 * attempt to insert the same element multiple times.
 */
template <typename DistanceType>
class KNNSimpleResultSet : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

	KNNSimpleResultSet(size_t capacity_) :
        capacity_(capacity_)
    {
		// reserving capacity to prevent memory re-allocations
		dist_index_.resize(capacity_, DistIndex(std::numeric_limits<DistanceType>::max(),-1));
    	clear();
    }

    ~KNNSimpleResultSet()
    {
    }

    /**
     * Clears the result set
     */
    void clear()
    {
        worst_distance_ = std::numeric_limits<DistanceType>::max();
        dist_index_[capacity_-1].dist_ = worst_distance_;
        count_ = 0;
    }

    /**
     *
     * @return Number of elements in the result set
     */
    size_t size() const
    {
        return count_;
    }

    /**
     * Radius search result set always reports full
     * @return
     */
    bool full() const
    {
        return count_==capacity_;
    }

    /**
     * Add a point to result set
     * @param dist distance to point
     * @param index index of point
     */
    void addPoint(DistanceType dist, size_t index)
    {
    	if (dist>=worst_distance_) return;

        if (count_ < capacity_) ++count_;
        size_t i;
        for (i=count_-1; i>0; --i) {
#ifdef FLANN_FIRST_MATCH
            if ( (dist_index_[i-1].dist_>dist) || ((dist==dist_index_[i-1].dist_)&&(dist_index_[i-1].index_>index)) )
#else
            if (dist_index_[i-1].dist_>dist)
#endif
            {
            	dist_index_[i] = dist_index_[i-1];
            }
            else break;
        }
        dist_index_[i].dist_ = dist;
        dist_index_[i].index_ = index;
        worst_distance_ = dist_index_[capacity_-1].dist_;
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	size_t n = std::min(count_, num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
    	return worst_distance_;
    }

private:
    size_t capacity_;
    size_t count_;
    DistanceType worst_distance_;
    std::vector<DistIndex> dist_index_;
};

/**
 * K-Nearest neighbour result set. Ensures that the elements inserted are unique
 */
template <typename DistanceType>
class KNNResultSet : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

    KNNResultSet(int capacity) : capacity_(capacity)
    {
		// reserving capacity to prevent memory re-allocations
		dist_index_.resize(capacity_, DistIndex(std::numeric_limits<DistanceType>::max(),-1));
    	clear();
    }

    ~KNNResultSet()
    {

    }

    /**
     * Clears the result set
     */
    void clear()
    {
        worst_distance_ = std::numeric_limits<DistanceType>::max();
        dist_index_[capacity_-1].dist_ = worst_distance_;
        count_ = 0;
    }

    size_t size() const
    {
        return count_;
    }

    bool full() const
    {
        return count_ == capacity_;
    }


    void addPoint(DistanceType dist, size_t index)
    {
        if (dist >= worst_distance_) return;
        size_t i;
        for (i = count_; i > 0; --i) {
#ifdef FLANN_FIRST_MATCH
            if ( (dist_index_[i-1].dist_<=dist) && ((dist!=dist_index_[i-1].dist_)||(dist_index_[i-1].index_<=index)) )
#else
            if (dist_index_[i-1].dist_<=dist)
#endif
            {
                // Check for duplicate indices
                size_t j = i - 1;
                while (dist_index_[j].dist_ == dist) {
                    if (dist_index_[j].index_ == index) {
                        return;
                    }
                    --j;
                }
                break;
            }
        }
        if (count_ < capacity_) ++count_;
        for (size_t j = count_-1; j > i; --j) {
            dist_index_[j] = dist_index_[j-1];
        }
        dist_index_[i].dist_ = dist;
        dist_index_[i].index_ = index;
        worst_distance_ = dist_index_[capacity_-1].dist_;
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	size_t n = std::min(count_, num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
        return worst_distance_;
    }

private:
    size_t capacity_;
    size_t count_;
    DistanceType worst_distance_;
    std::vector<DistIndex> dist_index_;

};



template <typename DistanceType>
class KNNResultSet2 : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

	KNNResultSet2(size_t capacity_) :
        capacity_(capacity_)
    {
		// reserving capacity to prevent memory re-allocations
		dist_index_.reserve(capacity_);
    	clear();
    }

    ~KNNResultSet2()
    {
    }

    /**
     * Clears the result set
     */
    void clear()
    {
        dist_index_.clear();
        worst_dist_ = std::numeric_limits<DistanceType>::max();
        is_full_ = false;
    }

    /**
     *
     * @return Number of elements in the result set
     */
    size_t size() const
    {
        return dist_index_.size();
    }

    /**
     * Radius search result set always reports full
     * @return
     */
    bool full() const
    {
        return is_full_;
    }

    /**
     * Add another point to result set
     * @param dist distance to point
     * @param index index of point
     * Pre-conditions: capacity_>0
     */
    void addPoint(DistanceType dist, size_t index)
    {
    	if (dist>=worst_dist_) return;

    	if (dist_index_.size()==capacity_) {
    		// if result set if filled to capacity, remove farthest element
    		std::pop_heap(dist_index_.begin(), dist_index_.end());
        	dist_index_.pop_back();
    	}

    	// add new element
    	dist_index_.push_back(DistIndex(dist,index));
    	if (is_full_) { // when is_full_==true, we have a heap
    		std::push_heap(dist_index_.begin(), dist_index_.end());
    	}

    	if (dist_index_.size()==capacity_) {
    		if (!is_full_) {
    			std::make_heap(dist_index_.begin(), dist_index_.end());
            	is_full_ = true;
    		}
    		// we replaced the farthest element, update worst distance
        	worst_dist_ = dist_index_[0].dist_;
        }
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	if (sorted) {
    		// std::sort_heap(dist_index_.begin(), dist_index_.end());
    		// sort seems faster here, even though dist_index_ is a heap
    		std::sort(dist_index_.begin(), dist_index_.end());
    	}
    	else {
    		if (num_elements<size()) {
    			std::nth_element(dist_index_.begin(), dist_index_.begin()+num_elements, dist_index_.end());
    		}
    	}

    	size_t n = std::min(dist_index_.size(), num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
    	return worst_dist_;
    }

private:
    size_t capacity_;
    DistanceType worst_dist_;
    std::vector<DistIndex> dist_index_;
    bool is_full_;
};


/**
 * Unbounded radius result set. It will hold as many elements as
 * are added to it.
 */
template <typename DistanceType>
class RadiusResultSet : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

	RadiusResultSet(DistanceType radius_) :
        radius_(radius_)
    {
		// reserving some memory to limit number of re-allocations
		dist_index_.reserve(1024);
    	clear();
    }

    ~RadiusResultSet()
    {
    }

    /**
     * Clears the result set
     */
    void clear()
    {
        dist_index_.clear();
    }

    /**
     *
     * @return Number of elements in the result set
     */
    size_t size() const
    {
        return dist_index_.size();
    }

    /**
     * Radius search result set always reports full
     * @return
     */
    bool full() const
    {
        return true;
    }

    /**
     * Add another point to result set
     * @param dist distance to point
     * @param index index of point
     * Pre-conditions: capacity_>0
     */
    void addPoint(DistanceType dist, size_t index)
    {
    	if (dist<radius_) {
    		// add new element
    		dist_index_.push_back(DistIndex(dist,index));
    	}
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	if (sorted) {
    		// std::sort_heap(dist_index_.begin(), dist_index_.end());
    		// sort seems faster here, even though dist_index_ is a heap
    		std::sort(dist_index_.begin(), dist_index_.end());
    	}
    	else {
    		if (num_elements<size()) {
    			std::nth_element(dist_index_.begin(), dist_index_.begin()+num_elements, dist_index_.end());
    		}
    	}

    	size_t n = std::min(dist_index_.size(), num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
    	return radius_;
    }

private:
    DistanceType radius_;
    std::vector<DistIndex> dist_index_;
};



/**
 * Bounded radius result set. It limits the number of elements
 * it can hold to a preset capacity.
 */
template <typename DistanceType>
class KNNRadiusResultSet : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

	KNNRadiusResultSet(DistanceType radius_, size_t capacity_) :
        radius_(radius_), capacity_(capacity_)
    {
		// reserving capacity to prevent memory re-allocations
		dist_index_.reserve(capacity_);
    	clear();
    }

    ~KNNRadiusResultSet()
    {
    }

    /**
     * Clears the result set
     */
    void clear()
    {
        dist_index_.clear();
        worst_dist_ = radius_;
        is_heap_ = false;
    }

    /**
     *
     * @return Number of elements in the result set
     */
    size_t size() const
    {
        return dist_index_.size();
    }

    /**
     * Radius search result set always reports full
     * @return
     */
    bool full() const
    {
        return true;
    }

    /**
     * Add another point to result set
     * @param dist distance to point
     * @param index index of point
     * Pre-conditions: capacity_>0
     */
    void addPoint(DistanceType dist, size_t index)
    {
    	if (dist>=worst_dist_) return;

    	if (dist_index_.size()==capacity_) {
    		// if result set is filled to capacity, remove farthest element
    		std::pop_heap(dist_index_.begin(), dist_index_.end());
        	dist_index_.pop_back();
    	}

    	// add new element
    	dist_index_.push_back(DistIndex(dist,index));
    	if (is_heap_) {
    		std::push_heap(dist_index_.begin(), dist_index_.end());
    	}

    	if (dist_index_.size()==capacity_) {
    		// when got to full capacity, make it a heap
    		if (!is_heap_) {
    			std::make_heap(dist_index_.begin(), dist_index_.end());
    			is_heap_ = true;
    		}
    		// we replaced the farthest element, update worst distance
        	worst_dist_ = dist_index_[0].dist_;
        }
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	if (sorted) {
    		// std::sort_heap(dist_index_.begin(), dist_index_.end());
    		// sort seems faster here, even though dist_index_ is a heap
    		std::sort(dist_index_.begin(), dist_index_.end());
    	}
    	else {
    		if (num_elements<size()) {
    			std::nth_element(dist_index_.begin(), dist_index_.begin()+num_elements, dist_index_.end());
    		}
    	}

    	size_t n = std::min(dist_index_.size(), num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
    	return worst_dist_;
    }

private:
    bool is_heap_;
    DistanceType radius_;
    size_t capacity_;
    DistanceType worst_dist_;
    std::vector<DistIndex> dist_index_;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * This is a result set that only counts the neighbors within a radius.
 */

template <typename DistanceType>
class CountRadiusResultSet : public ResultSet<DistanceType>
{
    DistanceType radius;
    size_t count;

public:
    CountRadiusResultSet(DistanceType radius_ ) :
        radius(radius_)
    {
        clear();
    }

    ~CountRadiusResultSet()
    {
    }

    void clear()
    {
        count = 0;
    }

    size_t size() const
    {
        return count;
    }

    bool full() const
    {
        return true;
    }

    void addPoint(DistanceType dist, size_t index)
    {
        if (dist<radius) {
            count++;
        }
    }

    DistanceType worstDist() const
    {
        return radius;
    }

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the k NN neighbors
 */
template<typename DistanceType>
class UniqueResultSet : public ResultSet<DistanceType>
{
public:
    struct DistIndex
    {
        DistIndex(DistanceType dist, unsigned int index) :
            dist_(dist), index_(index)
        {
        }
        bool operator<(const DistIndex dist_index) const
        {
            return (dist_ < dist_index.dist_) || ((dist_ == dist_index.dist_) && index_ < dist_index.index_);
        }
        DistanceType dist_;
        unsigned int index_;
    };

    /** Default cosntructor */
    UniqueResultSet() :
        worst_distance_(std::numeric_limits<DistanceType>::max())
    {
    }

    /** Check the status of the set
     * @return true if we have k NN
     */
    inline bool full() const
    {
        return is_full_;
    }

    /** Copy the set to two C arrays
     * @param indices pointer to a C array of indices
     * @param dist pointer to a C array of distances
     * @param n_neighbors the number of neighbors to copy
     */
    void copy(size_t* indices, DistanceType* dist, int n_neighbors, bool sorted = true)
    {
    	if (n_neighbors<0) n_neighbors = dist_indices_.size();
    	int i = 0;
    	typedef typename std::set<DistIndex>::const_iterator Iterator;
    	for (Iterator dist_index = dist_indices_.begin(), dist_index_end =
    			dist_indices_.end(); (dist_index != dist_index_end) && (i < n_neighbors); ++dist_index, ++indices, ++dist, ++i) {
    		*indices = dist_index->index_;
    		*dist = dist_index->dist_;
    	}
    }

    /** The number of neighbors in the set
     * @return
     */
    size_t size() const
    {
        return dist_indices_.size();
    }

    /** The distance of the furthest neighbor
     * If we don't have enough neighbors, it returns the max possible value
     * @return
     */
    inline DistanceType worstDist() const
    {
        return worst_distance_;
    }
protected:
    /** Flag to say if the set is full */
    bool is_full_;

    /** The worst distance found so far */
    DistanceType worst_distance_;

    /** The best candidates so far */
    std::set<DistIndex> dist_indices_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the k NN neighbors
 * Faster than KNNResultSet as it uses a binary heap and does not maintain two arrays
 */
template<typename DistanceType>
class KNNUniqueResultSet : public UniqueResultSet<DistanceType>
{
public:
    /** Constructor
     * @param capacity the number of neighbors to store at max
     */
    KNNUniqueResultSet(unsigned int capacity) : capacity_(capacity)
    {
        this->is_full_ = false;
        this->clear();
    }

    /** Add a possible candidate to the best neighbors
     * @param dist distance for that neighbor
     * @param index index of that neighbor
     */
    inline void addPoint(DistanceType dist, size_t index)
    {
        // Don't do anything if we are worse than the worst
        if (dist >= worst_distance_) return;
        dist_indices_.insert(DistIndex(dist, index));

        if (is_full_) {
            if (dist_indices_.size() > capacity_) {
                dist_indices_.erase(*dist_indices_.rbegin());
                worst_distance_ = dist_indices_.rbegin()->dist_;
            }
        }
        else if (dist_indices_.size() == capacity_) {
            is_full_ = true;
            worst_distance_ = dist_indices_.rbegin()->dist_;
        }
    }

    /** Remove all elements in the set
     */
    void clear()
    {
        dist_indices_.clear();
        worst_distance_ = std::numeric_limits<DistanceType>::max();
        is_full_ = false;
    }

protected:
    typedef typename UniqueResultSet<DistanceType>::DistIndex DistIndex;
    using UniqueResultSet<DistanceType>::is_full_;
    using UniqueResultSet<DistanceType>::worst_distance_;
    using UniqueResultSet<DistanceType>::dist_indices_;

    /** The number of neighbors to keep */
    unsigned int capacity_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the radius nearest neighbors
 * It is more accurate than RadiusResult as it is not limited in the number of neighbors
 */
template<typename DistanceType>
class RadiusUniqueResultSet : public UniqueResultSet<DistanceType>
{
public:
    /** Constructor
     * @param capacity the number of neighbors to store at max
     */
    RadiusUniqueResultSet(DistanceType radius) :
        radius_(radius)
    {
        is_full_ = true;
    }

    /** Add a possible candidate to the best neighbors
     * @param dist distance for that neighbor
     * @param index index of that neighbor
     */
    void addPoint(DistanceType dist, size_t index)
    {
        if (dist < radius_) dist_indices_.insert(DistIndex(dist, index));
    }

    /** Remove all elements in the set
     */
    inline void clear()
    {
        dist_indices_.clear();
    }


    /** Check the status of the set
     * @return alwys false
     */
    inline bool full() const
    {
        return true;
    }

    /** The distance of the furthest neighbor
     * If we don't have enough neighbors, it returns the max possible value
     * @return
     */
    inline DistanceType worstDist() const
    {
        return radius_;
    }
private:
    typedef typename UniqueResultSet<DistanceType>::DistIndex DistIndex;
    using UniqueResultSet<DistanceType>::dist_indices_;
    using UniqueResultSet<DistanceType>::is_full_;

    /** The furthest distance a neighbor can be */
    DistanceType radius_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the k NN neighbors within a radius distance
 */
template<typename DistanceType>
class KNNRadiusUniqueResultSet : public KNNUniqueResultSet<DistanceType>
{
public:
    /** Constructor
     * @param capacity the number of neighbors to store at max
     */
    KNNRadiusUniqueResultSet(DistanceType radius, size_t capacity) : KNNUniqueResultSet<DistanceType>(capacity)
    {
        this->radius_ = radius;
        this->clear();
    }

    /** Remove all elements in the set
     */
    void clear()
    {
        dist_indices_.clear();
        worst_distance_ = radius_;
        is_full_ = true;
    }
private:
    using KNNUniqueResultSet<DistanceType>::dist_indices_;
    using KNNUniqueResultSet<DistanceType>::is_full_;
    using KNNUniqueResultSet<DistanceType>::worst_distance_;

    /** The maximum distance of a neighbor */
    DistanceType radius_;
};
}

#endif //FLANN_RESULTSET_H

