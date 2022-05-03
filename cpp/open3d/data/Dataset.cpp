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

#include "open3d/data/Dataset.h"

#include <string>

#include "open3d/utility/Download.h"
#include "open3d/utility/Extract.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

std::string LocateDataRoot() {
    std::string data_root = "";
    if (const char* env_p = std::getenv("OPEN3D_DATA_ROOT")) {
        data_root = std::string(env_p);
    }
    if (data_root.empty()) {
        data_root = utility::filesystem::GetHomeDirectory() + "/open3d_data";
    }
    return data_root;
}

Dataset::Dataset(const std::string& prefix, const std::string& data_root)
    : prefix_(prefix) {
    if (data_root.empty()) {
        data_root_ = LocateDataRoot();
    } else {
        data_root_ = data_root;
    }
    if (prefix_.empty()) {
        utility::LogError("prefix cannot be empty.");
    }
}

SingleDownloadDataset::SingleDownloadDataset(
        const std::string& prefix,
        const std::vector<std::string>& urls,
        const std::string& md5,
        const bool no_extract,
        const std::string& data_root)
    : Dataset(prefix, data_root) {
    const std::string filename =
            utility::filesystem::GetFileNameWithoutDirectory(urls[0]);

    const bool is_extract_present =
            utility::filesystem::DirectoryExists(Dataset::GetExtractDir());

    if (!is_extract_present) {
        // `download_dir` is relative path from `${data_root}`.
        const std::string download_dir = "download/" + GetPrefix();
        const std::string download_file_path =
                utility::DownloadFromURL(urls, md5, download_dir, data_root_);

        // Extract / Copy data.
        if (!no_extract) {
            utility::Extract(download_file_path, Dataset::GetExtractDir());
        } else {
            utility::filesystem::MakeDirectoryHierarchy(
                    Dataset::GetExtractDir());
            utility::filesystem::Copy(download_file_path,
                                      Dataset::GetExtractDir());
        }
    }
}

DemoICPPointClouds::DemoICPPointClouds(const std::string& data_root)
    : SingleDownloadDataset(
              "DemoICPPointClouds",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/DemoICPPointClouds.zip"},
              "596cffe5f9c587045e7397ad70754de9",
              /*no_extract =*/false,
              data_root) {
    for (int i = 0; i < 3; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".pcd");
    }
    transformation_log_path_ = Dataset::GetExtractDir() + "/init.log";
}

std::string DemoICPPointClouds::GetPaths(size_t index) const {
    if (index > 2) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 2 but got {}.",
                index);
    }
    return paths_[index];
}

DemoColoredICPPointClouds::DemoColoredICPPointClouds(
        const std::string& data_root)
    : SingleDownloadDataset(
              "DemoColoredICPPointClouds",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/DemoColoredICPPointClouds.zip"},
              "bf8d469e892d76f2e69e1213207c0e30",
              /*no_extract =*/false,
              data_root) {
    paths_.push_back(Dataset::GetExtractDir() + "/frag_115.ply");
    paths_.push_back(Dataset::GetExtractDir() + "/frag_116.ply");
}

std::string DemoColoredICPPointClouds::GetPaths(size_t index) const {
    if (index > 1) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 1 but got {}.",
                index);
    }
    return paths_[index];
}

DemoCropPointCloud::DemoCropPointCloud(const std::string& data_root)
    : SingleDownloadDataset(
              "DemoCropPointCloud",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/DemoCropPointCloud.zip"},
              "12dbcdddd3f0865d8312929506135e23",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    point_cloud_path_ = extract_dir + "/fragment.ply";
    cropped_json_path_ = extract_dir + "/cropped.json";
}

DemoFeatureMatchingPointClouds::DemoFeatureMatchingPointClouds(
        const std::string& data_root)
    : SingleDownloadDataset(
              "DemoFeatureMatchingPointClouds",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/DemoFeatureMatchingPointClouds.zip"},
              "02f0703ce0cbf4df78ce2602ae33fc79",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    point_cloud_paths_ = {extract_dir + "/cloud_bin_0.pcd",
                          extract_dir + "/cloud_bin_1.pcd"};
    fpfh_feature_paths_ = {extract_dir + "/cloud_bin_0.fpfh.bin",
                           extract_dir + "/cloud_bin_1.fpfh.bin"};
    l32d_feature_paths_ = {extract_dir + "/cloud_bin_0.d32.bin",
                           extract_dir + "/cloud_bin_1.d32.bin"};
}

DemoPoseGraphOptimization::DemoPoseGraphOptimization(
        const std::string& data_root)
    : SingleDownloadDataset(
              "DemoPoseGraphOptimization",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/DemoPoseGraphOptimization.zip"},
              "af085b28d79dea7f0a50aef50c96b62c",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    pose_graph_fragment_path_ =
            extract_dir + "/pose_graph_example_fragment.json";
    pose_graph_global_path_ = extract_dir + "/pose_graph_example_global.json";
}

DemoCustomVisualization::DemoCustomVisualization(const std::string& data_root)
    : SingleDownloadDataset(
              "DemoCustomVisualization",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/DemoCustomVisualization.zip"},
              "04cb716145c51d0119b59c7876249891",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    point_cloud_path_ = extract_dir + "/fragment.ply";
    camera_trajectory_path_ = extract_dir + "/camera_trajectory.json";
    render_option_path_ = extract_dir + "/renderoption.json";
}

PCDPointCloud::PCDPointCloud(const std::string& data_root)
    : SingleDownloadDataset(
              "PCDPointCloud",
              {"https://github.com/isl-org/open3d_downloads/releases/"
               "download/20220201-data/fragment.pcd"},
              "f3a613fd2bdecd699aabdd858fb29606",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/fragment.pcd";
}

PLYPointCloud::PLYPointCloud(const std::string& data_root)
    : SingleDownloadDataset(
              "PLYPointCloud",
              {"https://github.com/isl-org/open3d_downloads/releases/"
               "download/20220201-data/fragment.ply"},
              "831ecffd4d7cbbbe02494c5c351aa6e5",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/fragment.ply";
}

PTSPointCloud::PTSPointCloud(const std::string& data_root)
    : SingleDownloadDataset(
              "PTSPointCloud",
              {"https://github.com/isl-org/open3d_downloads/releases/"
               "download/20220301-data/point_cloud_sample1.pts"},
              "5c2c618b703d0161e6e333fcbf55a1e9",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/point_cloud_sample1.pts";
}

SampleNYURGBDImage::SampleNYURGBDImage(const std::string& data_root)
    : SingleDownloadDataset(
              "SampleNYURGBDImage",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/SampleNYURGBDImage.zip"},
              "b0baaf892c7ff9b202eb5fb40c5f7b58",
              /*no_extract =*/false,
              data_root) {
    color_path_ = Dataset::GetExtractDir() + "/NYU_color.ppm";
    depth_path_ = Dataset::GetExtractDir() + "/NYU_depth.pgm";
}

SampleSUNRGBDImage::SampleSUNRGBDImage(const std::string& data_root)
    : SingleDownloadDataset(
              "SampleSUNRGBDImage",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/SampleSUNRGBDImage.zip"},
              "b1a430586547c8986bdf8b36179a8e67",
              /*no_extract =*/false,
              data_root) {
    color_path_ = Dataset::GetExtractDir() + "/SUN_color.jpg";
    depth_path_ = Dataset::GetExtractDir() + "/SUN_depth.png";
}

SampleTUMRGBDImage::SampleTUMRGBDImage(const std::string& data_root)
    : SingleDownloadDataset(
              "SampleTUMRGBDImage",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/SampleTUMRGBDImage.zip"},
              "91758d42b142dbad7b0d90e857ad47a8",
              /*no_extract =*/false,
              data_root) {
    color_path_ = Dataset::GetExtractDir() + "/TUM_color.png";
    depth_path_ = Dataset::GetExtractDir() + "/TUM_depth.png";
}

SampleRedwoodRGBDImages::SampleRedwoodRGBDImages(const std::string& data_root)
    : SingleDownloadDataset(
              "SampleRedwoodRGBDImages",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/SampleRedwoodRGBDImages.zip"},
              "43971c5f690c9cfc52dda8c96a0140ee",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();

    color_paths_ = {
            extract_dir + "/color/00000.jpg", extract_dir + "/color/00001.jpg",
            extract_dir + "/color/00002.jpg", extract_dir + "/color/00003.jpg",
            extract_dir + "/color/00004.jpg"};

    depth_paths_ = {
            extract_dir + "/depth/00000.png", extract_dir + "/depth/00001.png",
            extract_dir + "/depth/00002.png", extract_dir + "/depth/00003.png",
            extract_dir + "/depth/00004.png"};

    trajectory_log_path_ = extract_dir + "/trajectory.log";
    odometry_log_path_ = extract_dir + "/odometry.log";
    rgbd_match_path_ = extract_dir + "/rgbd.match";
    reconstruction_path_ = extract_dir + "/example_tsdf_pcd.ply";
    camera_intrinsic_path_ = extract_dir + "/camera_primesense.json";
}

SampleFountainRGBDImages::SampleFountainRGBDImages(const std::string& data_root)
    : SingleDownloadDataset(
              "SampleFountainRGBDImages",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/SampleFountainRGBDImages.zip"},
              "c6c1b2171099f571e2a78d78675df350",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    color_paths_ = {extract_dir + "/image/0000010-000001228920.jpg",
                    extract_dir + "/image/0000031-000004096400.jpg",
                    extract_dir + "/image/0000044-000005871507.jpg",
                    extract_dir + "/image/0000064-000008602440.jpg",
                    extract_dir + "/image/0000110-000014883587.jpg",
                    extract_dir + "/image/0000156-000021164733.jpg",
                    extract_dir + "/image/0000200-000027172787.jpg",
                    extract_dir + "/image/0000215-000029220987.jpg",
                    extract_dir + "/image/0000255-000034682853.jpg",
                    extract_dir + "/image/0000299-000040690907.jpg",
                    extract_dir + "/image/0000331-000045060400.jpg",
                    extract_dir + "/image/0000368-000050112627.jpg",
                    extract_dir + "/image/0000412-000056120680.jpg",
                    extract_dir + "/image/0000429-000058441973.jpg",
                    extract_dir + "/image/0000474-000064586573.jpg",
                    extract_dir + "/image/0000487-000066361680.jpg",
                    extract_dir + "/image/0000526-000071687000.jpg",
                    extract_dir + "/image/0000549-000074827573.jpg",
                    extract_dir + "/image/0000582-000079333613.jpg",
                    extract_dir + "/image/0000630-000085887853.jpg",
                    extract_dir + "/image/0000655-000089301520.jpg",
                    extract_dir + "/image/0000703-000095855760.jpg",
                    extract_dir + "/image/0000722-000098450147.jpg",
                    extract_dir + "/image/0000771-000105140933.jpg",
                    extract_dir + "/image/0000792-000108008413.jpg",
                    extract_dir + "/image/0000818-000111558627.jpg",
                    extract_dir + "/image/0000849-000115791573.jpg",
                    extract_dir + "/image/0000883-000120434160.jpg",
                    extract_dir + "/image/0000896-000122209267.jpg",
                    extract_dir + "/image/0000935-000127534587.jpg",
                    extract_dir + "/image/0000985-000134361920.jpg",
                    extract_dir + "/image/0001028-000140233427.jpg",
                    extract_dir + "/image/0001061-000144739467.jpg"};

    depth_paths_ = {extract_dir + "/depth/0000038-000001234662.png",
                    extract_dir + "/depth/0000124-000004104418.png",
                    extract_dir + "/depth/0000177-000005872988.png",
                    extract_dir + "/depth/0000259-000008609267.png",
                    extract_dir + "/depth/0000447-000014882686.png",
                    extract_dir + "/depth/0000635-000021156105.png",
                    extract_dir + "/depth/0000815-000027162570.png",
                    extract_dir + "/depth/0000877-000029231463.png",
                    extract_dir + "/depth/0001040-000034670651.png",
                    extract_dir + "/depth/0001220-000040677116.png",
                    extract_dir + "/depth/0001351-000045048488.png",
                    extract_dir + "/depth/0001503-000050120614.png",
                    extract_dir + "/depth/0001683-000056127079.png",
                    extract_dir + "/depth/0001752-000058429557.png",
                    extract_dir + "/depth/0001937-000064602868.png",
                    extract_dir + "/depth/0001990-000066371438.png",
                    extract_dir + "/depth/0002149-000071677149.png",
                    extract_dir + "/depth/0002243-000074813859.png",
                    extract_dir + "/depth/0002378-000079318707.png",
                    extract_dir + "/depth/0002575-000085892450.png",
                    extract_dir + "/depth/0002677-000089296113.png",
                    extract_dir + "/depth/0002874-000095869855.png",
                    extract_dir + "/depth/0002951-000098439288.png",
                    extract_dir + "/depth/0003152-000105146507.png",
                    extract_dir + "/depth/0003238-000108016262.png",
                    extract_dir + "/depth/0003344-000111553403.png",
                    extract_dir + "/depth/0003471-000115791298.png",
                    extract_dir + "/depth/0003610-000120429623.png",
                    extract_dir + "/depth/0003663-000122198194.png",
                    extract_dir + "/depth/0003823-000127537274.png",
                    extract_dir + "/depth/0004028-000134377970.png",
                    extract_dir + "/depth/0004203-000140217589.png",
                    extract_dir + "/depth/0004339-000144755807.png"};

    keyframe_poses_log_path_ = extract_dir + "/scene/key.log";
    reconstruction_path_ = extract_dir + "/scene/integrated.ply";
}

SampleL515Bag::SampleL515Bag(const std::string& data_root)
    : SingleDownloadDataset(
              "SampleL515Bag",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/SampleL515Bag.zip"},
              "9770eeb194c78103037dbdbec78b9c8c",
              /*no_extract =*/false,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/L515_test_s.bag";
}

EaglePointCloud::EaglePointCloud(const std::string& data_root)
    : SingleDownloadDataset(
              "EaglePointCloud",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/EaglePointCloud.ply"},
              "e4e6c77bc548e7eb7548542a0220ad78",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/EaglePointCloud.ply";
}

ArmadilloMesh::ArmadilloMesh(const std::string& data_root)
    : SingleDownloadDataset(
              "ArmadilloMesh",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/ArmadilloMesh.ply"},
              "9e68ff1b1cc914ed88cd84f6a8235021",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/ArmadilloMesh.ply";
}

BunnyMesh::BunnyMesh(const std::string& data_root)
    : SingleDownloadDataset(
              "BunnyMesh",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/BunnyMesh.ply"},
              "568f871d1a221ba6627569f1e6f9a3f2",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/BunnyMesh.ply";
}

KnotMesh::KnotMesh(const std::string& data_root)
    : SingleDownloadDataset(
              "KnotMesh",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/KnotMesh.ply"},
              "bfc9f132ecdfb7f9fdc42abf620170fc",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/KnotMesh.ply";
}

MonkeyModel::MonkeyModel(const std::string& data_root)
    : SingleDownloadDataset(
              "MonkeyModel",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/MonkeyModel.zip"},
              "fc330bf4fd8e022c1e5ded76139785d4",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/albedo.png"},
            {"ao", extract_dir + "/ao.png"},
            {"metallic", extract_dir + "/metallic.png"},
            {"monkey_material", extract_dir + "/monkey.mtl"},
            {"monkey_model", extract_dir + "/monkey.obj"},
            {"monkey_solid_material", extract_dir + "/monkey_solid.mtl"},
            {"monkey_solid_model", extract_dir + "/monkey_solid.obj"},
            {"normal", extract_dir + "/normal.png"},
            {"roughness", extract_dir + "/roughness.png"}};
}

SwordModel::SwordModel(const std::string& data_root)
    : SingleDownloadDataset(
              "SwordModel",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/SwordModel.zip"},
              "eb7df358b5c31c839f03c4b3b4157c04",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"sword_material", extract_dir + "/UV.mtl"},
            {"sword_model", extract_dir + "/UV.obj"},
            {"base_color", extract_dir + "/UV_blinn1SG_BaseColor.png"},
            {"metallic", extract_dir + "/UV_blinn1SG_Metallic.png"},
            {"normal", extract_dir + "/UV_blinn1SG_Normal.png"},
            {"roughness", extract_dir + "/UV_blinn1SG_Roughness.png"}};
}

CrateModel::CrateModel(const std::string& data_root)
    : SingleDownloadDataset(
              "CrateModel",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/CrateModel.zip"},
              "20413eada103969bb3ca5df9aebc2034",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {{"crate_material", extract_dir + "/crate.mtl"},
                             {"crate_model", extract_dir + "/crate.obj"},
                             {"texture_image", extract_dir + "/crate.jpg"}};
}

FlightHelmetModel::FlightHelmetModel(const std::string& data_root)
    : SingleDownloadDataset(
              "FlightHelmetModel",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/FlightHelmetModel.zip"},
              "597c3aa8b46955fff1949a8baa768bb4",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"flight_helmet", extract_dir + "/FlightHelmet.gltf"},
            {"flight_helmet_bin", extract_dir + "/FlightHelmet.bin"},
            {"mat_glass_plastic_base",
             extract_dir +
                     "/FlightHelmet_Materials_GlassPlasticMat_BaseColor.png"},
            {"mat_glass_plastic_normal",
             extract_dir +
                     "/FlightHelmet_Materials_GlassPlasticMat_Normal.png"},
            {"mat_glass_plastic_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_GlassPlasticMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_leather_parts_base",
             extract_dir +
                     "/FlightHelmet_Materials_LeatherPartsMat_BaseColor.png"},
            {"mat_leather_parts_normal",
             extract_dir +
                     "/FlightHelmet_Materials_LeatherPartsMat_Normal.png"},
            {"mat_leather_parts_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_LeatherPartsMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_lenses_base",
             extract_dir + "/FlightHelmet_Materials_LensesMat_BaseColor.png"},
            {"mat_lenses_normal",
             extract_dir + "/FlightHelmet_Materials_LensesMat_Normal.png"},
            {"mat_lenses_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_LensesMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_metal_parts_base",
             extract_dir +
                     "/FlightHelmet_Materials_MetalPartsMat_BaseColor.png"},
            {"mat_metal_parts_normal",
             extract_dir + "/FlightHelmet_Materials_MetalPartsMat_Normal.png"},
            {"mat_metal_parts_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_MetalPartsMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_rubber_wood_base",
             extract_dir +
                     "/FlightHelmet_Materials_RubberWoodMat_BaseColor.png"},
            {"mat_rubber_wood_normal",
             extract_dir + "/FlightHelmet_Materials_RubberWoodMat_Normal.png"},
            {"mat_rubber_wood_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_RubberWoodMat_"
                           "OcclusionRoughMetal.png"}};
}

MetalTexture::MetalTexture(const std::string& data_root)
    : SingleDownloadDataset(
              "MetalTexture",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/MetalTexture.zip"},
              "2b6a17e41157138868a2cd2926eedcc7",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Metal008_Color.jpg"},
            {"normal", extract_dir + "/Metal008_NormalDX.jpg"},
            {"roughness", extract_dir + "/Metal008_Roughness.jpg"},
            {"metallic", extract_dir + "/Metal008_Metalness.jpg"}};
}

PaintedPlasterTexture::PaintedPlasterTexture(const std::string& data_root)
    : SingleDownloadDataset(
              "PaintedPlasterTexture",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/PaintedPlasterTexture.zip"},
              "344096b29b06f14aac58f9ad73851dc2",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/PaintedPlaster017_Color.jpg"},
            {"normal", extract_dir + "/PaintedPlaster017_NormalDX.jpg"},
            {"roughness", extract_dir + "/noiseTexture.png"}};
}

TilesTexture::TilesTexture(const std::string& data_root)
    : SingleDownloadDataset(
              "TilesTexture",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/TilesTexture.zip"},
              "23f47f1e8e1799216724eb0c837c274d",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Tiles074_Color.jpg"},
            {"normal", extract_dir + "/Tiles074_NormalDX.jpg"},
            {"roughness", extract_dir + "/Tiles074_Roughness.jpg"}};
}

TerrazzoTexture::TerrazzoTexture(const std::string& data_root)
    : SingleDownloadDataset(
              "TerrazzoTexture",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/TerrazzoTexture.zip"},
              "8d67f191fb5d80a27d8110902cac008e",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Terrazzo018_Color.jpg"},
            {"normal", extract_dir + "/Terrazzo018_NormalDX.jpg"},
            {"roughness", extract_dir + "/Terrazzo018_Roughness.jpg"}};
}

WoodTexture::WoodTexture(const std::string& data_root)
    : SingleDownloadDataset(
              "WoodTexture",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/WoodTexture.zip"},
              "28788c7ecc42d78d4d623afbab2301e9",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Wood049_Color.jpg"},
            {"normal", extract_dir + "/Wood049_NormalDX.jpg"},
            {"roughness", extract_dir + "/Wood049_Roughness.jpg"}};
}

WoodFloorTexture::WoodFloorTexture(const std::string& data_root)
    : SingleDownloadDataset(
              "WoodFloorTexture",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220301-data/WoodFloorTexture.zip"},
              "f11b3e50208095e87340049b9ac3c319",
              /*no_extract =*/false,
              data_root) {
    const std::string extract_dir = Dataset::GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/WoodFloor050_Color.jpg"},
            {"normal", extract_dir + "/WoodFloor050_NormalDX.jpg"},
            {"roughness", extract_dir + "/WoodFloor050_Roughness.jpg"}};
}

JuneauImage::JuneauImage(const std::string& data_root)
    : SingleDownloadDataset(
              "JuneauImage",
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "20220201-data/JuneauImage.jpg"},
              "a090f6342893bdf0caefd83c6debbecd",
              /*no_extract =*/true,
              data_root) {
    path_ = Dataset::GetExtractDir() + "/JuneauImage.jpg";
}

LivingRoomPointClouds::LivingRoomPointClouds(const std::string& data_root)
    : SingleDownloadDataset(
              "LivingRoomPointClouds",
              {"http://redwood-data.org/indoor/data/"
               "livingroom1-fragments-ply.zip",
               "https://github.com/isl-org/open3d_downloads/releases/"
               "download/redwood/livingroom1-fragments-ply.zip"},
              "36e0eb23a66ccad6af52c05f8390d33e",
              /*no_extract =*/false,
              data_root) {
    paths_.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".ply");
    }
}

std::string LivingRoomPointClouds::GetPaths(size_t index) const {
    if (index > 56) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 56 but got {}.",
                index);
    }
    return paths_[index];
}

OfficePointClouds::OfficePointClouds(const std::string& data_root)
    : SingleDownloadDataset(
              "OfficePointClouds",
              {"http://redwood-data.org/indoor/data/"
               "office1-fragments-ply.zip",
               "https://github.com/isl-org/open3d_downloads/releases/"
               "download/redwood/office1-fragments-ply.zip"},
              "c519fe0495b3c731ebe38ae3a227ac25",
              /*no_extract =*/false,
              data_root) {
    paths_.reserve(53);
    for (int i = 0; i < 53; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".ply");
    }
}

std::string OfficePointClouds::GetPaths(size_t index) const {
    if (index > 52) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 52 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
