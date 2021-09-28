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
//
//    Based on PVCNN Library (MIT License):
//    https://github.com/mit-han-lab/pvcnn
//
// Copyright (c) 2018 Zhijian Liu, Haotian Tang, Yujun Lin
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

/// This function performs trilinear devoxelization operation.
/// It computes aggregated features from the voxel grid for each
/// point passed in the input.
///
/// \param b    The batch size.
/// \param c    Feature dimension of voxel grid.
/// \param n    Number of points per batch.
/// \param r    Resolution of the grid.
/// \param r2   r squared.
/// \param r3   r cubed.
/// \param is_training  Whether model is in training phase.
/// \param coords   Array with the point positions. The shape is
///        [b, 3, n]
/// \param feat    Aray with the voxel grid. The shape is
///        [b, c, r, r, r]
/// \param inds    The voxel coordinates of point cube [b, 8, n]
/// \param wgts    weight for trilinear interpolation [b, 8, n]
/// \param outs    Outputs, FloatTensor[b, c, n]
///
void TrilinearDevoxelize(int b,
                         int c,
                         int n,
                         int r,
                         int r2,
                         int r3,
                         bool is_training,
                         const float *coords,
                         const float *feat,
                         int *inds,
                         float *wgts,
                         float *outs);

/// This function computes gradient for trilinear devoxelization op.
/// It computes gradient for the input voxelgrid.
///
/// \param b    The batch size.
/// \param c    Feature dimension of voxel grid.
/// \param n    Number of points per batch.
/// \param r3   resolution cubed.
/// \param inds    The voxel coordinates of point cube [b, 8, n]
/// \param wgts    weight for trilinear interpolation [b, 8, n]
/// \param grad_y    The gradient passed from top.
/// \param grad_x   The computed gradient for voxelgrid.
///
void TrilinearDevoxelizeGrad(int b,
                             int c,
                             int n,
                             int r3,
                             const int *inds,
                             const float *wgts,
                             const float *grad_y,
                             float *grad_x);
