// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <open3d/Open3D.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "open3d/geometry/IntersectionTest.h"

/* This example code performs a stress test of the various line/ray
 * intersections with the AxisAlignedBoundingBox to highlight the different
 * performance implications of each.
 *
 * The slab method is currently industry standard for ray tracing as it is
 * incredibly fast, however it can degenerate in circumstances when the line or
 * ray being tested lies entirely within one of the planes of the bounding box.
 * In a problem which is not conditioned to have this occur it would be a
 * statistical near-impossibility, and if it did occur it typically have no
 * serious consequences.
 *
 * However, in a problem which involves rays or lines in a single direction
 * on a regularly spaced grid that coincides with bounding box limits, it is
 * feasible to have it occur. If the consequences of a missed intersection are
 * unacceptable, the naive exact solution will return the correct answer at a
 * significant cost to performance.
 *
 * However, if performance is still critical and the problem is conditioned to
 * be degenerate for the slab method, it's likely that the problem is operating
 * in some reduced dimensionality where a full 3D intersection test is
 * unnecessary and even better performance can be gained with a custom
 * implementation.
 *
 * In the following single threaded test on a Xeon E5-2670 v2 from within a VM,
 * for 1 million bounding box/ray intersections, the following performance was
 * observed:
 *
 *      674ms for the exact method
 *       36ms for the slab method
 *       29ms for the slab method with precomputed inverses
 */

#define TRIAL_COUNT 1000000
using namespace open3d;

class Trial {
public:
    Trial(geometry::AxisAlignedBoundingBox box,
          Eigen::ParametrizedLine<double, 3> hit_line,
          Eigen::ParametrizedLine<double, 3> miss_line)
        : box_{box}, hit_line_{hit_line}, miss_line_{miss_line} {
        // Precompute the direction inverses for the optimized slab method
        h_x_inv = 1.0 / hit_line_.direction().x();
        h_y_inv = 1.0 / hit_line_.direction().y();
        h_z_inv = 1.0 / hit_line_.direction().z();

        m_x_inv = 1.0 / miss_line_.direction().x();
        m_y_inv = 1.0 / miss_line_.direction().y();
        m_z_inv = 1.0 / miss_line_.direction().z();
    }

    geometry::AxisAlignedBoundingBox box_;
    Eigen::ParametrizedLine<double, 3> hit_line_;
    Eigen::ParametrizedLine<double, 3> miss_line_;

    double h_x_inv;
    double h_y_inv;
    double h_z_inv;

    double m_x_inv;
    double m_y_inv;
    double m_z_inv;
};

std::vector<Trial> GenerateTrials();

int main() {
    using namespace open3d::geometry;
    using namespace std::chrono;

    std::cout << "Generating " << TRIAL_COUNT << " random bounding boxes\n";
    auto trials = GenerateTrials();

    std::cout << "\nRunning Exact Method" << std::endl;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    int failed_hits = 0;
    int failed_misses = 0;
    for (const auto& trial : trials) {
        if (!IntersectionTest::RayAABBExact(trial.hit_line_, trial.box_)) {
            failed_hits++;
        }

        if (IntersectionTest::RayAABBExact(trial.miss_line_, trial.box_)) {
            failed_misses++;
        }
    }
    auto duration =
            duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << " * took " << static_cast<double>(duration.count()) << " ms\n";
    std::cout << " * " << failed_hits << " failed hits\n";
    std::cout << " * " << failed_misses << " failed misses\n";

    std::cout << "\nRunning Slab Method" << std::endl;
    start = high_resolution_clock::now();
    failed_hits = 0;
    failed_misses = 0;
    for (const auto& trial : trials) {
        if (!IntersectionTest::RayAABBSlab(trial.hit_line_, trial.box_)) {
            failed_hits++;
        }
        if (IntersectionTest::RayAABBSlab(trial.miss_line_, trial.box_)) {
            failed_misses++;
        }
    }
    duration =
            duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << " * took " << static_cast<double>(duration.count()) << " ms\n";
    std::cout << " * " << failed_hits << " failed hits\n";
    std::cout << " * " << failed_misses << " failed misses\n";

    std::cout << "\nRunning Slab Method with Precomputed Inverses" << std::endl;
    start = high_resolution_clock::now();
    failed_hits = 0;
    failed_misses = 0;
    for (const auto& trial : trials) {
        if (!IntersectionTest::RayAABBSlab(trial.hit_line_, trial.box_,
                                           trial.h_x_inv, trial.h_y_inv,
                                           trial.h_z_inv)) {
            failed_hits++;
        }
        if (IntersectionTest::RayAABBSlab(trial.miss_line_, trial.box_,
                                          trial.m_x_inv, trial.m_y_inv,
                                          trial.m_z_inv)) {
            failed_misses++;
        }
    }
    duration =
            duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << " * took " << static_cast<double>(duration.count()) << " ms\n";
    std::cout << " * " << failed_hits << " failed hits\n";
    std::cout << " * " << failed_misses << " failed misses\n";
}

std::vector<Trial> GenerateTrials() {
    std::random_device rd;
    std::mt19937_64 mt(rd());

    std::uniform_real_distribution<double> pos_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> size_dist(1.0, 10.0);
    std::uniform_real_distribution<double> simple_dist(-1.0, 1.0);

    std::vector<Trial> trials;
    for (int i = 0; i < TRIAL_COUNT; ++i) {
        // Generate a random bounding box
        Eigen::Vector3d center{pos_dist(mt), pos_dist(mt), pos_dist(mt)};
        Eigen::Vector3d size{size_dist(mt), size_dist(mt), size_dist(mt)};
        geometry::AxisAlignedBoundingBox box{center - size, center + size};

        // Generate a point inside the box and a normal direction
        Eigen::Vector3d direction{simple_dist(mt), simple_dist(mt),
                                  simple_dist(mt)};
        Eigen::Vector3d focus_shift{size.x() * simple_dist(mt),
                                    size.y() * simple_dist(mt),
                                    size.z() * simple_dist(mt)};
        auto focus = center + focus_shift;
        direction.normalize();

        // Construct a line outside the box pointing inward
        auto exterior_line = Eigen::ParametrizedLine<double, 3>::Through(
                focus - (direction * size.norm() * 1.1),
                focus + (direction * size.norm() * 1.1));

        // Construct a random line outside the box that does not intersect
        Eigen::Hyperplane<double, 3> direction_plane(direction, center);
        Eigen::Vector3d shift_direction{simple_dist(mt), simple_dist(mt),
                                        simple_dist(mt)};
        auto in_plane_shift =
                (direction_plane.projection(center + shift_direction) - center)
                        .normalized();
        auto miss_focus = center + (in_plane_shift * (1.5 * size.norm()));
        auto miss_start = miss_focus - (direction * size.norm());
        auto miss_end = miss_focus + (direction * size.norm());
        auto miss_line = Eigen::ParametrizedLine<double, 3>::Through(miss_start,
                                                                     miss_end);

        // Store this as a trial object
        trials.emplace_back(box, exterior_line, miss_line);
    }

    return trials;
}
