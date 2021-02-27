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

REGISTER_OP("Open3DGroupPoints")
        .Input("points: float32")
        .Input("idx: int32")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle
                    dims2;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));

            ::tensorflow::shape_inference::ShapeHandle output =
                    c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1),
                                  c->Dim(dims2, 1), c->Dim(dims2, 2)});
            c->set_output(0, output);
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DGroupPointsGrad")
        .Input("grad_out: float32")
        .Input("idx: int32")
        .Attr("N: int")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // batch_size * nsample * 3
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            int N;
            TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
            ::tensorflow::shape_inference::ShapeHandle output =
                    c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), N});
            c->set_output(0, output);
            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");
