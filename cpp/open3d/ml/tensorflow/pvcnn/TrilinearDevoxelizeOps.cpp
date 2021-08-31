#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Open3DTrilinearDevoxelize")
        .Attr("r: int")
        .Attr("is_training: bool")
        .Input("coords: float32")
        .Input("features: float32")
        .Output("outs: float32")
        .Output("inds: int32")
        .Output("wgts: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // (batch_size, 3, N)
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle
                    dims2;  // (batch_size, C, R, R, R)
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &dims2));

            ::tensorflow::shape_inference::ShapeHandle out1, out2;
            out1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1),
                                 c->Dim(dims1, 2)});  // (batch_size, C, N)
            out2 = c->MakeShape({c->Dim(dims2, 0), 8,
                                 c->Dim(dims1, 2)});  // (batch_size, 8, N)

            c->set_output(0, out1);
            c->set_output(1, out2);
            c->set_output(2, out2);

            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");

REGISTER_OP("Open3DTrilinearDevoxelizeGrad")
        .Input("grad_y: float32")
        .Input("indices: int32")
        .Input("weights: float32")
        .Attr("r: int")
        .Output("grad_x: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle
                    dims1;  // (batch_size, C, N)
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
            ::tensorflow::shape_inference::ShapeHandle
                    dims2;  // (batch_size, 8, N)
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));

            ::tensorflow::shape_inference::ShapeHandle out;
            int r_;
            TF_RETURN_IF_ERROR(c->GetAttr("r", &r_));
            out = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), r_, r_,
                                r_});  // (batch_size, C, R, R, R)

            c->set_output(0, out);

            return Status::OK();
        })
        .Doc(R"doc( TODO )doc");
