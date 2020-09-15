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

REGISTER_OP("Open3DBatchGridSubsampling")
        .Input("points: float")
        .Input("batches: int32")
        .Input("dl: float")
        .Output("sub_points: float")
        .Output("sub_batches: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle input0_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input0_shape));
            c->set_output(0, input0_shape);
            c->set_output(1, c->input(1));
            return Status::OK();
        });

class BatchGridSubsamplingOp : public OpKernel {
public:
    explicit BatchGridSubsamplingOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& points_tensor = context->input(0);
        const Tensor& batches_tensor = context->input(1);
        const Tensor& dl_tensor = context->input(2);

        // check shapes of input and weights
        const TensorShape& points_shape = points_tensor.shape();
        const TensorShape& batches_shape = batches_tensor.shape();

        // check input is a [N x 3] matrix
        DCHECK_EQ(points_shape.dims(), 2);
        DCHECK_EQ(points_shape.dim_size(1), 3);

        // Check that Batch lengths is a vector
        DCHECK_EQ(batches_shape.dims(), 1);

        // Dimensions
        int N = (int)points_shape.dim_size(0);

        // Number of batches
        int Nb = (int)batches_shape.dim_size(0);

        // get the data as std vector of points
        float sampleDl = dl_tensor.flat<float>().data()[0];
        std::vector<PointXYZ> original_points = std::vector<PointXYZ>(
                (PointXYZ*)points_tensor.flat<float>().data(),
                (PointXYZ*)points_tensor.flat<float>().data() + N);

        // Batches lengths
        std::vector<int> batches =
                std::vector<int>((int*)batches_tensor.flat<int>().data(),
                                 (int*)batches_tensor.flat<int>().data() + Nb);

        // Unsupported label and features
        std::vector<float> original_features;
        std::vector<int> original_classes;

        // Create result containers
        std::vector<PointXYZ> subsampled_points;
        std::vector<float> subsampled_features;
        std::vector<int> subsampled_classes;
        std::vector<int> subsampled_batches;

        // Compute results
        batch_grid_subsampling(original_points, subsampled_points,
                               original_features, subsampled_features,
                               original_classes, subsampled_classes, batches,
                               subsampled_batches, sampleDl, 0);

        // Sub_points output
        // *****************

        // create output shape
        TensorShape sub_points_shape;
        sub_points_shape.AddDim(subsampled_points.size());
        sub_points_shape.AddDim(3);

        // create output tensor
        Tensor* sub_points_output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, sub_points_shape,
                                                         &sub_points_output));
        auto sub_points_tensor = sub_points_output->matrix<float>();

        // Fill output tensor
        for (int i = 0; i < (signed)subsampled_points.size(); i++) {
            sub_points_tensor(i, 0) = subsampled_points[i].x;
            sub_points_tensor(i, 1) = subsampled_points[i].y;
            sub_points_tensor(i, 2) = subsampled_points[i].z;
        }

        // Batch length output
        // *******************

        // create output shape
        TensorShape sub_batches_shape;
        sub_batches_shape.AddDim(subsampled_batches.size());

        // create output tensor
        Tensor* sub_batches_output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, sub_batches_shape,
                                                         &sub_batches_output));
        auto sub_batches_tensor = sub_batches_output->flat<int>();

        // Fill output tensor
        for (int i = 0; i < (signed)subsampled_batches.size(); i++)
            sub_batches_tensor(i) = subsampled_batches[i];
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DBatchGridSubsampling").Device(DEVICE_CPU),
                        BatchGridSubsamplingOp);
