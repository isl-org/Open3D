// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

namespace open3d {
namespace geometry {

/// \class KDTreeSearchParam
///
/// \brief Base class for KDTree search parameters.
class KDTreeSearchParam {
public:
    /// \enum SearchType
    ///
    /// \brief Specifies the search type for the search.
    enum class SearchType {
        Knn = 0,
        Radius = 1,
        Hybrid = 2,
    };

public:
    virtual ~KDTreeSearchParam() {}

protected:
    KDTreeSearchParam(SearchType type) : search_type_(type) {}

public:
    /// Get the search type (KNN, Radius, Hybrid) for the search parameter.
    SearchType GetSearchType() const { return search_type_; }

private:
    SearchType search_type_;
};

/// \class KDTreeSearchParamKNN
///
/// \brief KDTree search parameters for pure KNN search.
class KDTreeSearchParamKNN : public KDTreeSearchParam {
public:
    /// \brief Default Constructor.
    ///
    /// \param knn Specifies the knn neighbors that will searched. Default
    /// is 30.
    KDTreeSearchParamKNN(int knn = 30)
        : KDTreeSearchParam(SearchType::Knn), knn_(knn) {}

public:
    /// Number of the neighbors that will be searched.
    int knn_;
};

/// \class KDTreeSearchParamRadius
///
/// \brief KDTree search parameters for pure radius search.
class KDTreeSearchParamRadius : public KDTreeSearchParam {
public:
    /// \brief Default Constructor.
    ///
    /// \param radius Specifies the radius of the search.
    KDTreeSearchParamRadius(double radius)
        : KDTreeSearchParam(SearchType::Radius), radius_(radius) {}

public:
    /// Search radius.
    double radius_;
};

/// \class KDTreeSearchParamHybrid
///
/// \brief KDTree search parameters for hybrid KNN and radius search.
class KDTreeSearchParamHybrid : public KDTreeSearchParam {
public:
    /// \brief Default Constructor.
    ///
    /// \param radius Specifies the radius of the search.
    /// \param max_nn Specifies the max neighbors to be searched.
    KDTreeSearchParamHybrid(double radius, int max_nn)
        : KDTreeSearchParam(SearchType::Hybrid),
          radius_(radius),
          max_nn_(max_nn) {}

public:
    /// Search radius.
    double radius_;
    /// At maximum, max_nn neighbors will be searched.
    int max_nn_;
};

}  // namespace geometry
}  // namespace open3d
