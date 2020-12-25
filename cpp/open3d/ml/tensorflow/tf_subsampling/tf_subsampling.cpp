// Source code from: https://github.com/HuguesTHOMAS/KPConv.
//
// MIT License
//
// Copyright (c) 2019 HuguesTHOMAS
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

#include "open3d/ml/contrib/GridSubsampling.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace open3d::ml::contrib;

REGISTER_OP("Open3DGridSubsampling")
        .Input("points: float")
        .Input("dl: float")
        .Output("sub_points: float")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle input;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
            c->set_output(0, input);
            return Status::OK();
        });

class GridSubsamplingOp : public OpKernel {
public:
    explicit GridSubsamplingOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& points_tensor = context->input(0);
        const Tensor& dl_tensor = context->input(1);

        // check shapes of input and weights
        const TensorShape& points_shape = points_tensor.shape();

        // check input are [N x 3] matrices
        DCHECK_EQ(points_shape.dims(), 2);
        DCHECK_EQ(points_shape.dim_size(1), 3);

        // Dimensions
        int N = (int)points_shape.dim_size(0);

        // get the data as std vector of points
        float sampleDl = dl_tensor.flat<float>().data()[0];
        std::vector<PointXYZ> original_points = std::vector<PointXYZ>(
                (PointXYZ*)points_tensor.flat<float>().data(),
                (PointXYZ*)points_tensor.flat<float>().data() + N);

        // Unsupported label and features
        std::vector<float> original_features;
        std::vector<int> original_classes;

        // Create result containers
        std::vector<PointXYZ> subsampled_points;
        std::vector<float> subsampled_features;
        std::vector<int> subsampled_classes;

        // Compute results
        grid_subsampling(original_points, subsampled_points, original_features,
                         subsampled_features, original_classes,
                         subsampled_classes, sampleDl, 0);

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(subsampled_points.size());
        output_shape.AddDim(3);

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &output));
        auto output_tensor = output->matrix<float>();

        // Fill output tensor
        for (int i = 0; i < output->shape().dim_size(0); i++) {
            output_tensor(i, 0) = subsampled_points[i].x;
            output_tensor(i, 1) = subsampled_points[i].y;
            output_tensor(i, 2) = subsampled_points[i].z;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DGridSubsampling").Device(DEVICE_CPU),
                        GridSubsamplingOp);
