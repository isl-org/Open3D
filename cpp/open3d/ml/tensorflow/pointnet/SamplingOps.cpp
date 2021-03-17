/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017.
 */
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Open3DFurthestPointSampling")
        .Input("points: float32")
        .Attr("sample_size: int")
        .Output("out: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * npoint * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            int npoint;
            TF_RETURN_IF_ERROR(c->GetAttr("sample_size", &npoint));
            ::tensorflow::shape_inference::ShapeHandle output =
                    c->MakeShape({c->Dim(dims1, 0), npoint});
            c->set_output(0, output);
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DGatherPoints")
        .Input("points: float32")
        .Input("idx: int32")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * ndataset * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle
                    dims2;  // batch_size * npoints
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &dims2));
            // batch_size * npoints * 3
            ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape(
                    {c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
            c->set_output(0, output);
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DGatherPointsGrad")
        .Input("grad_out: float32")
        .Input("idx: int32")
        .Attr("C: int")
        .Attr("N: int")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");
