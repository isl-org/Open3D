//#include "torch/serialize/tensor.h"
//#include "torch/extension.h"
//#include "torch/script.h"

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/contrib/RoIPoolKernel.h"


void roipool3dLauncher(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num, 
                       const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag);


std::tuple<at::Tensor, at::Tensor> roipool3d(at::Tensor xyz, at::Tensor boxes3d, at::Tensor pts_feature, const int64_t sampled_pts_num){
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int boxes_num = boxes3d.size(1);
    int feature_in_len = pts_feature.size(2);

    auto device = xyz.device();
    torch::Tensor features = at::zeros(
        {batch_size, boxes_num, sampled_pts_num, 3 + feature_in_len}, 
        at::dtype(ToTorchDtype<float>()).device(device)); 

    torch::Tensor empty_flag = at::zeros(
        {batch_size, boxes_num}, at::dtype(ToTorchDtype<int>()).device(device)); 

    const float * xyz_data = xyz.data<float>();
    const float * boxes3d_data = boxes3d.data<float>();
    const float * pts_feature_data = pts_feature.data<float>();
    float * pooled_features_data = features.data<float>();
    int * pooled_empty_flag_data = empty_flag.data<int>();

    roipool3dLauncher(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, 
                       xyz_data, boxes3d_data, pts_feature_data, pooled_features_data, pooled_empty_flag_data);
    
    return std::tuple<at::Tensor, at::Tensor>(features, empty_flag);
}


int pt_in_box3d_cpu(float x, float y, float z, float cx, float bottom_y, float cz, float h, float w, float l, float angle){
    float max_dis = 10.0, x_rot, z_rot, cosa, sina, cy;
    int in_flag;
    cy = bottom_y - h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) || (fabsf(z - cz) > max_dis)){
        return 0;
    }
    cosa = cos(angle); sina = sin(angle);
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;

    in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) & (z_rot <= w / 2.0);
    return in_flag;
}

int pts_in_boxes3d_cpu(at::Tensor pts_flag, at::Tensor pts, at::Tensor boxes3d){
    long boxes_num = boxes3d.size(0);
    long pts_num = pts.size(0);

    long * pts_flag_flat = pts_flag.data<long>();
    float * pts_flat = pts.data<float>();
    float * boxes3d_flat = boxes3d.data<float>();

    memset(pts_flag_flat, 0, boxes_num * pts_num * sizeof(long));

    int i, j, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                          boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                          boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);
            pts_flag_flat[i * pts_num + j] = cur_in_flag;
        }
    }
    return 1;
}

int roipool3d_cpu(at::Tensor pts, at::Tensor boxes3d, at::Tensor pts_feature, at::Tensor pooled_pts,
                  at::Tensor pooled_features, at::Tensor pooled_empty_flag){
    long boxes_num = boxes3d.size(0);
    long pts_num = pts.size(0);
    long feature_len = pts_feature.size(1);
    long sampled_pts_num = pooled_pts.size(1);

    float * pts_flat = pts.data<float>();
    float * boxes3d_flat = boxes3d.data<float>();
    float * pts_feature_flat = pts_feature.data<float>();
    float * pooled_pts_flat = pooled_pts.data<float>();
    float * pooled_features_flat = pooled_features.data<float>();
    long * pooled_empty_flag_flat = pooled_empty_flag.data<long>();

    memset(pooled_empty_flag_flat, 0, boxes_num * sizeof(long));

    int i, j, k, cnt, temp_idx, duplicate_idx, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        cnt = 0;
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                       boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                       boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);

            if (cur_in_flag){
                if (cnt < sampled_pts_num){
                    temp_idx = i * sampled_pts_num * 3 + cnt * 3;
                    for (k = 0; k < 3; k++) pooled_pts_flat[temp_idx + k] = pts_flat[j * 3 + k];
                    temp_idx = i * sampled_pts_num * feature_len + cnt * feature_len;
                    for (k = 0; k < feature_len; k++) pooled_features_flat[temp_idx + k] = pts_feature_flat[j * feature_len + k];
                    cnt++;
                }
                else break;
            }
        }

        if (cnt == 0){
            // no points in this box
            pooled_empty_flag_flat[i] = 1;
        }
        else if (cnt < sampled_pts_num){
            // duplicate same points
            duplicate_idx = 0;
            for (j = cnt; j < sampled_pts_num; j++){
                temp_idx = i * sampled_pts_num * 3 + j * 3;
                duplicate_idx = i * sampled_pts_num * 3 + (j % cnt) * 3;
                for (k = 0; k < 3; k++) pooled_pts_flat[temp_idx + k] = pooled_pts_flat[duplicate_idx + k];
                temp_idx = i * sampled_pts_num * feature_len + j * feature_len;
                duplicate_idx = i * sampled_pts_num * feature_len + (j % cnt) * feature_len;
                for (k = 0; k < feature_len; k++){
                    pooled_features_flat[temp_idx + k] = pooled_features_flat[duplicate_idx + k];
                }
            }
        }
    }
    return 1;
}

static auto registry = torch::RegisterOperators(
        "open3d::roipool3d(Tensor xyz, Tensor boxes3d,"
        "Tensor pts_feature, int sampled_pts_num)"
        " -> (Tensor features, Tensor flags)",
        &roipool3d);
