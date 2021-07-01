// -*- C++ -*-
//===-- convex_hull.cpp ---------------------------------------------------===//
//
// Copyright (C) 2017-2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

/*
    This file contains the Parallel STL-based implementation of quickhull algorithm.
    Quickhull algorithm description:
    1. Initial phase
      1) Find two points that guaranteed to belong to the convex hull. Min and max points in X can be used for it.
      2) Divide the initial set of points in two subsets by the line formed by two points from previous step.
         This subset will be processed recursively.

    2. Iteration Phase
      1) Divide current subset by dividing line [p1,p2] into right and left subsets.
      2) New point (p) of the convex hull is found as farthest point of right subset from the dividing line.
      3) If the right subset has more than 1 point, repeat the iteration phase with the right subset and dividing lines [p1,p] and [p,p2].

    The implementation based on std::copy_if, std::max_element and std::minmax_element algorithms of Parallel STL.
    Each of the algorithms use par_unseq policy. In order to get effect of the policy usage problem size should be big enough.
    By default problem size was set as 5M points. With point set with less than 500K points par_unseq policy could be inefficient.
    Correctness of the convex hull is checked by std::any_of algorithm with counting iterators.
*/

#include <chrono>
#include <vector>
#include <fstream>
#include <pstl/algorithm>
#include <pstl/numeric>
#include <pstl/execution>
#include <pstl/iterators.h>

#include "utils.h"

typedef util::point<double> point_t;
typedef std::vector< point_t > pointVec_t;
const size_t grain_size = 500000;

template<typename Policy, typename Iterator>
void find_hull_points(Policy exec, Iterator first, Iterator last, pointVec_t &H, point_t p1, point_t p2);

//Iteration Phase based on the divide and conquer technique
template<typename Policy, typename Iterator>
void divide_and_conquer(Policy exec, Iterator first, Iterator last, pointVec_t &H, point_t p1, point_t fp, point_t p2) {
    pointVec_t H1;
    //decomposes the work and combines the results
    if (first - last > grain_size){ //for small set size parallel policy could be inefficient
        find_hull_points(exec, first, last, H, p1, fp);
        find_hull_points(exec, first, last, H1, fp, p2);
    }
    else {
        find_hull_points(pstl::execution::unseq, first, last, H, p1, fp);
        find_hull_points(pstl::execution::unseq, first, last, H1, fp, p2);
    }
    H.insert(H.end(), H1.cbegin(), H1.cend());
}

//Find points of the convex hull on right sides of the segments [p1, p2]
template<typename Policy, typename Iterator>
void find_hull_points(Policy exec, Iterator first, Iterator last, pointVec_t &H, point_t p1, point_t p2) {

    pointVec_t P_reduced(last - first);

    //Find points from the range [first, last-1] that are on the right side of the segment [p1,p2]
    Iterator end = std::copy_if(exec, first, last, P_reduced.begin(),
        [&p1, &p2](const point_t& pnt) {
        return cross_product(p1, p2, pnt) > 0;
    });

    if ((end - P_reduced.cbegin()) < 2) {
        //Add points into the hull
        H.push_back(p1);
        H.insert(H.end(), P_reduced.cbegin(), end);
    }
    else {
        //Find the farthest point from the segment [p1,p1], it will be in the convex hull
        auto far_point = *std::max_element(exec, P_reduced.cbegin(), end,
            [&p1, &p2](const point_t & pnt1, const point_t & pnt2) {
            double how_far1 = cross_product(p1, p2, pnt1);
            double how_far2 = cross_product(p1, p2, pnt2);
            return how_far1 == how_far2 ? pnt1 < pnt2 : how_far1 < how_far2;
        });

        //Repeat for segments [p1, far_point] and [far_point, p2] with points from [P_reduced.cbegin(), end-1]
        divide_and_conquer(exec, P_reduced.cbegin(), end, H, p1, far_point, p2);
    }
}

//Quickhull algorithm
//The algorihm based on the divide and conquer technique
void quickhull(const pointVec_t &points, pointVec_t &hull) {
    if (points.size() < 2) {
        hull.insert(hull.end(), points.cbegin(), points.cend());
        return;
    }
    //Find left and right most points, they will be in the convex hull
#if (__INTEL_COMPILER >= 1900 && __INTEL_COMPILER <= 1910) && (PSTL_VERSION >= 200 && PSTL_VERSION <= 204)
    // A workaround for incorrectly working minmax_element
    point_t p1 = *std::min_element(pstl::execution::par_unseq, points.cbegin(), points.cend());
    point_t p2 = *std::max_element(pstl::execution::par_unseq, points.cbegin(), points.cend());
#else
    auto minmaxx = std::minmax_element(pstl::execution::par_unseq, points.cbegin(), points.cend());
    point_t p1 = *minmaxx.first;
    point_t p2 = *minmaxx.second;
#endif

    //Divide the set of points into two subsets, which will be processed recursively
    divide_and_conquer(pstl::execution::par_unseq, points.cbegin(), points.cend(), hull, p1, p2, p1);
}

// Check if a polygon is convex
bool is_convex(const pointVec_t & points) {
    return std::all_of(pstl::execution::par_unseq,
        pstl::counting_iterator<size_t>(size_t(0)),
        pstl::counting_iterator<size_t>(points.size()),
        [&points](size_t i) {
            point_t p0(points[i]);
            point_t p1(points[(i + 1) % points.size()]);
            point_t p2(points[(i + 2) % points.size()]);
            return (cross_product(p0, p1, p2) < 0);
    });
}

int main(int argc, char* argv[]) {

    const size_t numberOfPoints = 5000000;
    const std::string output_file("ConvexHull.csv");

    pointVec_t points(numberOfPoints);
    pointVec_t hull;

    //initialize set of points
    std::generate(pstl::execution::par, points.begin(), points.end(), util::random_point<double>);
    std::cout << "Points were initialized. Number of the points " << points.size() << std::endl;

    using ms = std::chrono::milliseconds;

    auto tm_start = std::chrono::high_resolution_clock::now();
    //execution of the quickhull algorithm
    quickhull(points, hull);
    auto tm_end = std::chrono::high_resolution_clock::now();

    std::cout << "Computational time " << std::chrono::duration_cast<ms> (tm_end - tm_start).count() << "ms"
        << "  Points in the hull: " << hull.size() << "  The convex hull is " << (is_convex(hull) ? "correct" : "incorrect") << std::endl;

    //writing the results
    std::ofstream fout(output_file);
    if (fout.is_open()) {
        for (auto p : hull)
            fout << p << std::endl;
        std::cout << "The convex hull has been stored to a file " << output_file << std::endl;
    }
    else {
        std::cout << "Cannot open a file " << output_file << " to store result" << std::endl;
    }

    return 0;
}
