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

#ifndef FLANN_DATASET_H_
#define FLANN_DATASET_H_

#include "flann/general.h"
#include "flann/util/serialization.h"
#include <stdio.h>

namespace flann
{

typedef unsigned char uchar;

class Matrix_
{
public:

	Matrix_() : rows(0), cols(0), stride(0), type(FLANN_NONE), data(NULL)
	{
	};

    Matrix_(void* data_, size_t rows_, size_t cols_, flann_datatype_t type_, size_t stride_ = 0) :
        rows(rows_), cols(cols_), stride(stride_), type(type_)
    {
    	data = static_cast<uchar*>(data_);

    	if (stride==0) stride = flann_datatype_size(type)*cols;
    }

    /**
     * Operator that returns a (pointer to a) row of the data.
     */
    inline void* operator[](size_t index) const
    {
        return data+index*stride;
    }

    void* ptr() const
    {
        return data;
    }

    size_t rows;
    size_t cols;
    size_t stride;
    flann_datatype_t type;
protected:
    uchar* data;

    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar & rows;
    	ar & cols;
    	ar & stride;
    	ar & type;
    	if (Archive::is_loading::value) {
    		data = new uchar[rows*stride];
    	}
    	ar & serialization::make_binary_object(data, rows*stride);
    }
    friend struct serialization::access;
};


/**
 * Class that implements a simple rectangular matrix stored in a memory buffer and
 * provides convenient matrix-like access using the [] operators.
 *
 * This class has the same memory structure as the un-templated class flann::Matrix_ and
 * it's directly convertible from it.
 */
template <typename T>
class Matrix : public Matrix_
{
public:
    typedef T type;

    Matrix() : Matrix_()
    {
    }

    Matrix(T* data_, size_t rows_, size_t cols_, size_t stride_ = 0) :
    	Matrix_(data_, rows_, cols_, flann_datatype_value<T>::value, stride_)
    {
    	if (stride==0) stride = sizeof(T)*cols;
    }

    /**
     * Operator that returns a (pointer to a) row of the data.
     */
    inline T* operator[](size_t index) const
    {
    	return reinterpret_cast<T*>(data+index*stride);
    }


    T* ptr() const
    {
    	return reinterpret_cast<T*>(data);
    }
};

}

#endif //FLANN_DATASET_H_
