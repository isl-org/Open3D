// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

// FOR TESTING TENSOR ADVANCED INDEXING IDEAS
// ABOUT THIS EXAMPLE:
//      TODO: ADD DETAILS

#include <iostream>
#include <vector>
#include "open3d/Open3D.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/AdvancedIndexing.h"
#include "open3d/t/pipelines/registration/Registration.h"

using namespace open3d;

int main(){

    std::vector<int64_t> src_points {
        1,1,1,
        5,5,5,
        10,10,10,
        15,15,15,
        20,20,20,
        25,25,25,
    };
    std::vector<int64_t> target_points {
        20,20,20,
        2,2,2,
        40,40,40,
        6,6,6,
        50,50,50,
        26,26,26,
    };
    std::vector<int64_t> corres {
        1,
        3,
        -1,
        -1,
        0,
        5
    };
    std::vector<double_t> dist {
        3,
        3,
        0,
        0,
        0,
        3
    };

    core::Dtype dtype = core::Dtype::Float64;
    core::Device device = core::Device("CPU:0");
    core::Tensor src_points_(src_points, {6,3}, core::Dtype::Int64, device);
    core::Tensor target_points_(target_points, {6,3}, core::Dtype::Int64, device);
    core::Tensor corres_(corres, {6}, core::Dtype::Int64, device);
    core::Tensor dist_(dist, {6}, dtype, device);

    std::cout << " Source Points: " << std::endl;
    auto src_pt_vec = src_points_.ToFlatVector<int64_t>();
    for(int i = 0; i < 6; i++){
        for(int j = 0; j < 3; j++){
            std::cout << src_pt_vec[i*3 + j] << " ";
        }
        std::cout << std::endl;
    }

    core::Tensor bool_corres = corres_.Ne(-1);
    std::cout << " Correspondence Points: " << std::endl;
    auto bool_corres_vec = bool_corres.ToFlatVector<bool>();
    for(int i = 0; i < 6; i++){
            std::cout << bool_corres_vec[i] << " ";
    } std::cout << std::endl;

    core::Tensor source_select = src_points_.IndexGet({bool_corres});
    core::Tensor corres_select = corres_.IndexGet({bool_corres});
    std::cout << " Selected Correspondence Points: " << std::endl;
    auto selected_corres_vec = corres_select.ToFlatVector<int64_t>();
    for(int i = 0; i < (int)selected_corres_vec.size(); i++){
            std::cout << selected_corres_vec[i] << " ";
    } std::cout << std::endl;

    std::cout << " Selected Source Points: " << std::endl;
    auto selected_src_vec = source_select.ToFlatVector<int64_t>();
    for(int i = 0; i < (int)selected_src_vec.size()/3; i++){
        for(int j = 0; j < 3; j++){
            std::cout << selected_src_vec[i*3 + j] << " ";
        } std::cout << std::endl;
    }

    core::Tensor target_select = target_points_.IndexGet({corres_select});
    std::cout << " Selected Target Points: " << std::endl;
    auto target_select_vec = target_select.ToFlatVector<int64_t>();
    for(int i = 0; i < (int)target_select_vec.size()/3; i++){
        for(int j = 0; j < 3; j++){
            std::cout << target_select_vec[i*3 + j] << " ";
        } std::cout << std::endl;
    }

    core::Tensor error = (source_select - target_select);
    error.Mul_(error);
    double e2 = (double)error.Sum({0,1}).Item<int64_t>();
    std::cout << "Error: " << e2 << std::endl;

    return 0;

}