// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <memory>
#include <Eigen/Dense>

#include <Open3D/Open3D.h>

#include <iostream>
#include <iomanip>
using namespace std;

#define WIDTH 20
#define PRECISION 3

// display table header
void DisplayHeader();

// display row of results
void DisplayResults(const string& method,
                    const string& parameters,
                    const double& duration,
                    const int& nr_loops);

// Usage:
// $ ./build/bin/benchmark/benchPointCloud examples/TestData/fragment.ply
int main(int argc, char* argv[]) {
    using namespace open3d;

    int nr_loops = 10;
    utility::Timer timer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseError);

    auto pcd = io::CreatePointCloudFromFile(argv[1]);

    cout << setprecision(PRECISION);
    cout << fixed;
    cout << endl;

    cout << setw(WIDTH) << "Test:";
    cout << setw(WIDTH) << "PointCloud";
    cout << endl;

    cout << setw(WIDTH) << "Nr. of points:";
    cout << setw(WIDTH) << pcd->points_.size() << endl;
    cout << endl;

    DisplayHeader();

    // FPFH estimation with Radius 0.25
    int fpfh_loops = 1;
    timer.Start();
    for (int i = 0; i < fpfh_loops; i++) {
        registration::ComputeFPFHFeature(
                *pcd, open3d::geometry::KDTreeSearchParamRadius(0.25));
    }
    timer.Stop();
    DisplayResults("FPFH estimation", "Radius 0.25", timer.GetDuration(),
                   fpfh_loops);

    // Normal estimation with KNN20
    timer.Start();
    for (int i = 0; i < nr_loops; i++) {
        geometry::EstimateNormals(*pcd,
                                  open3d::geometry::KDTreeSearchParamKNN(20));
    }
    timer.Stop();
    DisplayResults("Normal estimation", "KNN20", timer.GetDuration(), nr_loops);

    // Normal estimation with Radius 0.01666"
    timer.Start();
    for (int i = 0; i < nr_loops; i++) {
        geometry::EstimateNormals(
                *pcd, open3d::geometry::KDTreeSearchParamRadius(0.01666));
    }
    timer.Stop();
    DisplayResults("Normal estimation", "Radius 0.01666", timer.GetDuration(),
                   nr_loops);

    // Normal estimation with Hybrid 0.01666, 60"
    timer.Start();
    for (int i = 0; i < nr_loops; i++) {
        geometry::EstimateNormals(
                *pcd, open3d::geometry::KDTreeSearchParamHybrid(0.01666, 60));
    }
    timer.Stop();
    DisplayResults("Normal estimation", "Hybrid 0.01666, 60",
                   timer.GetDuration(), nr_loops);

    cout << endl;
    return 0;
}

// display table header
void DisplayHeader() {
    cout << setw(WIDTH) << "Method";
    cout << setw(WIDTH) << "Parameters";
    cout << setw(WIDTH) << "Duration (ms)";
    cout << setw(WIDTH) << "Nr. of nr_loops";
    cout << endl;
}

// display row of results
void DisplayResults(const string& method,
                    const string& parameters,
                    const double& duration,
                    const int& nr_loops) {
    cout << setw(WIDTH) << method;
    cout << setw(WIDTH) << parameters;
    cout << setw(WIDTH) << duration / nr_loops;
    cout << setw(WIDTH) << nr_loops;
    cout << endl;
}
