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

#include "open3d/ml/contrib/neighbors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace open3d::ml::contrib;

REGISTER_OP("Open3DBatchOrderedNeighbors")
        .Input("queries: float")
        .Input("supports: float")
        .Input("q_batches: int32")
        .Input("s_batches: int32")
        .Input("radius: float")
        .Output("neighbors: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            // Create input shape container
            ::tensorflow::shape_inference::ShapeHandle input;

            // Check inputs rank
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input));

            // Create the output shape
            c->set_output(0, c->UnknownShapeOfRank(2));

            return Status::OK();
        });

class BatchOrderedNeighborsOp : public OpKernel {
public:
    explicit BatchOrderedNeighborsOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& queries_tensor = context->input(0);
        const Tensor& supports_tensor = context->input(1);
        const Tensor& q_batches_tensor = context->input(2);
        const Tensor& s_batches_tensor = context->input(3);
        const Tensor& radius_tensor = context->input(4);

        // check shapes of input and weights
        const TensorShape& queries_shape = queries_tensor.shape();
        const TensorShape& supports_shape = supports_tensor.shape();
        const TensorShape& q_batches_shape = q_batches_tensor.shape();
        const TensorShape& s_batches_shape = s_batches_tensor.shape();

        // check input are [N x 3] matrices
        DCHECK_EQ(queries_shape.dims(), 2);
        DCHECK_EQ(queries_shape.dim_size(1), 3);
        DCHECK_EQ(supports_shape.dims(), 2);
        DCHECK_EQ(supports_shape.dim_size(1), 3);

        // Check that Batch lengths are vectors and same number of batch for
        // both query and support
        DCHECK_EQ(q_batches_shape.dims(), 1);
        DCHECK_EQ(s_batches_shape.dims(), 1);
        DCHECK_EQ(q_batches_shape.dim_size(0), s_batches_shape.dim_size(0));

        // Points Dimensions
        int Nq = (int)queries_shape.dim_size(0);
        int Ns = (int)supports_shape.dim_size(0);

        // Number of batches
        int Nb = (int)q_batches_shape.dim_size(0);

        // get the data as std vector of points
        float radius = radius_tensor.flat<float>().data()[0];
        std::vector<PointXYZ> queries = std::vector<PointXYZ>(
                (PointXYZ*)queries_tensor.flat<float>().data(),
                (PointXYZ*)queries_tensor.flat<float>().data() + Nq);
        std::vector<PointXYZ> supports = std::vector<PointXYZ>(
                (PointXYZ*)supports_tensor.flat<float>().data(),
                (PointXYZ*)supports_tensor.flat<float>().data() + Ns);

        // Batches lengths
        std::vector<int> q_batches = std::vector<int>(
                (int*)q_batches_tensor.flat<int>().data(),
                (int*)q_batches_tensor.flat<int>().data() + Nb);
        std::vector<int> s_batches = std::vector<int>(
                (int*)s_batches_tensor.flat<int>().data(),
                (int*)s_batches_tensor.flat<int>().data() + Nb);

        // Create result containers
        std::vector<int> neighbors_indices;

        // Compute results
        batch_nanoflann_neighbors(queries, supports, q_batches, s_batches,
                                  neighbors_indices, radius);

        // Maximal number of neighbors
        int max_neighbors = neighbors_indices.size() / Nq;

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(Nq);
        output_shape.AddDim(max_neighbors);

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &output));
        auto output_tensor = output->matrix<int>();

        // Fill output tensor
        for (int i = 0; i < output->shape().dim_size(0); i++) {
            for (int j = 0; j < output->shape().dim_size(1); j++) {
                output_tensor(i, j) = neighbors_indices[max_neighbors * i + j];
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Open3DBatchOrderedNeighbors").Device(DEVICE_CPU),
                        BatchOrderedNeighborsOp);
