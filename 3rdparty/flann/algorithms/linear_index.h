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

#ifndef FLANN_LINEAR_INDEX_H_
#define FLANN_LINEAR_INDEX_H_

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"

namespace flann
{

struct LinearIndexParams : public IndexParams
{
    LinearIndexParams()
    {
        (* this)["algorithm"] = FLANN_INDEX_LINEAR;
    }
};

template <typename Distance>
class LinearIndex : public NNIndex<Distance>
{
public:

    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    typedef NNIndex<Distance> BaseClass;

    LinearIndex(const IndexParams& params = LinearIndexParams(), Distance d = Distance()) :
    	BaseClass(params, d)
    {
    }

    LinearIndex(const Matrix<ElementType>& input_data, const IndexParams& params = LinearIndexParams(), Distance d = Distance()) :
    	BaseClass(params, d)
    {
        setDataset(input_data);
    }

    LinearIndex(const LinearIndex& other) : BaseClass(other)
    {
    }

    LinearIndex& operator=(LinearIndex other)
    {
    	this->swap(other);
    	return *this;
    }

    virtual ~LinearIndex()
    {
    }

    BaseClass* clone() const
    {
    	return new LinearIndex(*this);
    }

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        assert(points.cols==veclen_);
        extendDataset(points);
    }

    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_LINEAR;
    }


    int usedMemory() const
    {
        return 0;
    }

    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	if (Archive::is_loading::value) {
            index_params_["algorithm"] = getType();
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

    void findNeighbors(ResultSet<DistanceType>& resultSet, const ElementType* vec, const SearchParams& /*searchParams*/) const
    {
    	if (removed_) {
    		for (size_t i = 0; i < points_.size(); ++i) {
    			if (removed_points_.test(i)) continue;
    			DistanceType dist = distance_(points_[i], vec, veclen_);
    			resultSet.addPoint(dist, i);
    		}
    	}
    	else {
    		for (size_t i = 0; i < points_.size(); ++i) {
    			DistanceType dist = distance_(points_[i], vec, veclen_);
    			resultSet.addPoint(dist, i);
    		}
    	}
    }
protected:
    void buildIndexImpl()
    {
        /* nothing to do here for linear search */
    }

    void freeIndex()
    {
        /* nothing to do here for linear search */
    }

private:

    USING_BASECLASS_SYMBOLS
};

}

#endif // FLANN_LINEAR_INDEX_H_
