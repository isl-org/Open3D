// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::visualization;
using namespace open3d::t::pipelines::registration;

const int WIDTH = 1024;
const int HEIGHT = 768;

const Eigen::Vector3f CENTER_OFFSET(0.0f, 0.0f, 30.0f);

const std::string SRC_CLOUD = "source_pointcloud";
const std::string DST_CLOUD = "target_pointcloud";

// Initial transformation guess for registation.
// std::vector<float> initial_transform_flat = {
//         0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
//         0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};
std::vector<float> initial_transform_flat = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                             0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                             0.0, 0.0, 0.0, 1.0};

class MultipleWindowsApp {
public:
    MultipleWindowsApp(const std::string& path_config,
                       const core::Device& device)
        : result_(device),
          device_(device),
          host_(core::Device("CPU:0")),
          dtype_(core::Dtype::Float32) {
        ReadConfigFile(path_config);
        pointclouds_device_ = LoadTensorPointClouds();

        transformation_ =
                core::Tensor(initial_transform_flat, {4, 4}, dtype_, device_);

        // Warm Up.
        std::vector<ICPConvergenceCriteria> warm_up_criteria = {
                ICPConvergenceCriteria(0.01, 0.01, 1)};
        result_ = RegistrationMultiScaleICP(
                pointclouds_device_[0], pointclouds_device_[0], {1.0},
                warm_up_criteria, {1.5}, core::Tensor::Eye(4, dtype_, device_),
                *estimation_);

        std::cout << " [Debug] Warm up transformation: "
                  << result_.transformation_.ToString() << std::endl;
        is_done_ = false;
        visualize_output_ = true;

        gui::Application::GetInstance().Initialize();
    }

    void Run() {
        main_vis_ = std::make_shared<visualizer::O3DVisualizer>(
                "Open3D - Multi-Window Demo", WIDTH, HEIGHT);

        main_vis_->SetOnClose([this]() { return this->OnMainWindowClosing(); });

        gui::Application::GetInstance().AddWindow(main_vis_);

        std::thread read_thread([this]() { this->MultiScaleICPMappingDemo(); });
        gui::Application::GetInstance().Run();
        read_thread.join();
    }

private:
    bool OnMainWindowClosing() {
        // Ensure object is free so Filament can clean up without crashing.
        // Also signals to the "reading" thread that it is finished.
        main_vis_.reset();
        return true;  // false would cancel the close
    }

private:
    void MultiScaleICPMappingDemo() {
        // This is NOT the UI thread, need to call PostToMainThread() to
        // update the scene or any part of the UI.

        geometry::AxisAlignedBoundingBox bounds;
        // Eigen::Vector3d extent;
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            legacy_output_ = std::make_shared<open3d::geometry::PointCloud>();
            *legacy_output_ = pointclouds_device_[0].ToLegacyPointCloud();
            bounds = legacy_output_->GetAxisAlignedBoundingBox();
            // extent = bounds.GetExtent();
        }

        auto mat = rendering::Material();
        mat.shader = "defaultUnlit";

        gui::Application::GetInstance().PostToMainThread(
                main_vis_.get(), [this, bounds, mat]() {
                    std::lock_guard<std::mutex> lock(cloud_lock_);
                    main_vis_->AddGeometry(filenames_[0], legacy_output_, &mat);
                    main_vis_->ResetCameraToDefault();
                    Eigen::Vector3f center = bounds.GetCenter().cast<float>();
                    main_vis_->SetupCamera(100, center, center + CENTER_OFFSET,
                                           {0.0f, -1.0f, 0.0f});
                    main_vis_->SetBackground({0.0f, 0.0f, 0.0f, 1.0f});
                });

        core::Tensor initial_transform = core::Tensor::Eye(4, dtype_, device_);
        core::Tensor cumulative_transform = initial_transform.Clone();

        double total_processing_time = 0;
        for (int i = 0; i < end_range_ - 1; i++) {
            utility::Timer time_icp_odom_loop;
            time_icp_odom_loop.Start();
            auto source = pointclouds_device_[i];
            auto target = pointclouds_device_[i + 1];

            auto result = RegistrationMultiScaleICP(
                    source, target, voxel_sizes_, criterias_, search_radius_,
                    initial_transform, *estimation_);

            cumulative_transform = cumulative_transform.Matmul(
                    result.transformation_.Inverse());

            time_icp_odom_loop.Stop();
            total_processing_time += time_icp_odom_loop.GetDuration();
            // if (visualize_output_) {
                auto pcd_transformed = target.Transform(cumulative_transform);
                {
                    std::lock_guard<std::mutex> lock(cloud_lock_);
                    *legacy_output_ = pcd_transformed.ToLegacyPointCloud();
                    // legacy_output_->PaintUniformColor({0.0, 0.0, 1.0});
                }

                if (!main_vis_) {  // might have changed while sleeping
                    break;
                }

                gui::Application::GetInstance().PostToMainThread(
                        main_vis_.get(), [this, i, mat]() {
                            std::lock_guard<std::mutex> lock(cloud_lock_);
                            // main_vis_->RemoveGeometry(filenames_[i]);
                            main_vis_->AddGeometry(filenames_[i + 1], legacy_output_,
                                                   &mat);
                            auto bounds = legacy_output_->GetAxisAlignedBoundingBox();
                            // auto extent = bounds.GetExtent();
                            main_vis_->ResetCameraToDefault();
                            Eigen::Vector3f center = bounds.GetCenter().cast<float>();
                            main_vis_->SetupCamera(100, center, center + CENTER_OFFSET,
                                           {0.0f, -1.0f, 0.0f});
                            // auto bbox = main_vis_->GetScene()->GetBoundingBox();
                            // auto center = bbox.GetCenter().cast<float>();
                            // main_vis_->SetupCamera(60, center,
                            //                        center + CENTER_OFFSET,
                            //                        {0.0f, -1.0f, 0.0f});
                        });
            // }
            utility::LogDebug(" Registraion took: {}",
                              time_icp_odom_loop.GetDuration());
            utility::LogDebug(" Cumulative Transformation: \n{}\n",
                              cumulative_transform.ToString());
        }
    }

private:
    // To read parameters from config file.
    void ReadConfigFile(const std::string& path_config) {
        std::ifstream cFile(path_config);
        std::vector<double> relative_fitness;
        std::vector<double> relative_rmse;
        std::vector<int> max_iterations;
        std::string verb;

        if (cFile.is_open()) {
            std::string line;
            while (getline(cFile, line)) {
                line.erase(std::remove_if(line.begin(), line.end(), isspace),
                           line.end());
                if (line[0] == '#' || line.empty()) continue;

                auto delimiterPos = line.find("=");
                auto name = line.substr(0, delimiterPos);
                auto value = line.substr(delimiterPos + 1);

                if (name == "dataset_path") {
                    path_dataset = value;
                } else if (name == "end_range") {
                    std::istringstream is(value);
                    end_range_ = std::stoi(value);
                } else if (name == "registration_method") {
                    registration_method_ = value;
                } else if (name == "criteria.relative_fitness") {
                    std::istringstream is(value);
                    relative_fitness.push_back(std::stod(value));
                } else if (name == "criteria.relative_rmse") {
                    std::istringstream is(value);
                    relative_rmse.push_back(std::stod(value));
                } else if (name == "criteria.max_iterations") {
                    std::istringstream is(value);
                    max_iterations.push_back(std::stoi(value));
                } else if (name == "voxel_size") {
                    std::istringstream is(value);
                    voxel_sizes_.push_back(std::stod(value));
                } else if (name == "search_radii") {
                    std::istringstream is(value);
                    search_radius_.push_back(std::stod(value));
                } else if (name == "verbosity") {
                    std::istringstream is(value);
                    verb = value;
                } else if (name == "ground_truth_tx") {
                    std::istringstream is(value);
                    gt_tx_ = std::stod(value);
                } else if (name == "ground_truth_ty") {
                    std::istringstream is(value);
                    gt_ty_ = std::stod(value);
                }
            }
        } else {
            std::cerr << "Couldn't open config file for reading.\n";
        }

        utility::LogInfo(" Dataset path: {}", path_dataset);
        if (end_range_ > 500) {
            utility::LogWarning(" Too large range. Memory might exceed.");
        }
        utility::LogInfo(" Range: 0 to {} pointcloud files in sequence.",
                         end_range_ - 1);
        utility::LogInfo(" Registrtion method: {}", registration_method_);
        std::cout << std::endl;

        std::cout << " Voxel Sizes: ";
        for (auto voxel_size : voxel_sizes_) std::cout << voxel_size << " ";
        std::cout << std::endl;

        std::cout << " Search Radius Sizes: ";
        for (auto search_radii : search_radius_)
            std::cout << search_radii << " ";
        std::cout << std::endl;

        std::cout << " ICPCriteria: " << std::endl;
        std::cout << "   Max Iterations: ";
        for (auto iteration : max_iterations) std::cout << iteration << " ";
        std::cout << std::endl;
        std::cout << "   Relative Fitness: ";
        for (auto fitness : relative_fitness) std::cout << fitness << " ";
        std::cout << std::endl;
        std::cout << "   Relative RMSE: ";
        for (auto rmse : relative_rmse) std::cout << rmse << " ";
        std::cout << std::endl;

        size_t length = voxel_sizes_.size();
        if (search_radius_.size() != length ||
            max_iterations.size() != length ||
            relative_fitness.size() != length ||
            relative_rmse.size() != length) {
            utility::LogError(
                    " Length of vector: voxel_sizes, search_sizes, "
                    "max_iterations, "
                    "relative_fitness, relative_rmse must be same.");
        }

        for (int i = 0; i < (int)length; i++) {
            auto criteria = ICPConvergenceCriteria(
                    relative_fitness[i], relative_rmse[i], max_iterations[i]);
            criterias_.push_back(criteria);
        }

        if (registration_method_ == "PointToPoint") {
            estimation_ =
                    std::make_shared<TransformationEstimationPointToPoint>();
        } else if (registration_method_ == "PointToPlane") {
            estimation_ =
                    std::make_shared<TransformationEstimationPointToPlane>();
        } else {
            utility::LogError(" Registration method {}, not implemented.",
                              registration_method_);
        }

        if (verb == "Debug") {
            verbosity_ = utility::VerbosityLevel::Debug;
        } else {
            verbosity_ = utility::VerbosityLevel::Info;
        }

        std::cout << " Config file read complete. " << std::endl;
    }

    // To perform required dtype conversion, normal estimation and device
    // transfer.
    std::vector<t::geometry::PointCloud> LoadTensorPointClouds() {
        for (int i = 0; i < end_range_; i++) {
            filenames_.push_back(path_dataset + std::to_string(i) +
                                 std::string(".pcd"));
        }

        std::vector<t::geometry::PointCloud> pointclouds_device(
                filenames_.size());

        try {
            int i = 0;
            t::geometry::PointCloud pointcloud_local;
            for (auto& path : filenames_) {
                t::io::ReadPointCloud(path, pointcloud_local,
                                      {"auto", false, false, true});
                // Device transfer.
                pointcloud_local = pointcloud_local.To(device_);
                // Dtype conversion to Float32. Currently only Float32
                // pointcloud is supported.
                for (std::string attr : {"points", "colors", "normals"}) {
                    if (pointcloud_local.HasPointAttr(attr)) {
                        pointcloud_local.SetPointAttr(
                                attr,
                                pointcloud_local.GetPointAttr(attr).To(dtype_));
                    }
                }
                // Normal Estimation. Currenly Normal Estimation is not
                // supported by Tensor Pointcloud.
                if (registration_method_ == "PointToPoint" &&
                    !pointcloud_local.HasPointNormals()) {
                    auto pointcloud_legacy =
                            pointcloud_local.ToLegacyPointCloud();
                    pointcloud_legacy.EstimateNormals(
                            open3d::geometry::KDTreeSearchParamKNN(), false);
                    core::Tensor pointcloud_normals =
                            t::geometry::PointCloud::FromLegacyPointCloud(
                                    pointcloud_legacy)
                                    .GetPointNormals()
                                    .To(device_, dtype_);
                    pointcloud_local.SetPointNormals(pointcloud_normals);
                }
                // Adding it to our vector of pointclouds.
                pointclouds_device[i++] = pointcloud_local.Clone();
            }
        } catch (...) {
            utility::LogError(
                    " Failed to read pointcloud in sequence. Ensure pointcloud "
                    "files are present in the given dataset path in continuous "
                    "sequence from 0 to {}. Also, in case of large range, the "
                    "system might be going out-of-memory. ",
                    end_range_);
        }
        return pointclouds_device;
    }

private:
    std::mutex cloud_lock_;
    std::shared_ptr<open3d::geometry::PointCloud> legacy_output_;

    std::atomic<bool> is_done_;
    std::shared_ptr<visualizer::O3DVisualizer> main_vis_;

private:
    std::vector<open3d::t::geometry::PointCloud> pointclouds_device_;

private:
    std::string path_dataset;
    std::string registration_method_;
    std::vector<std::string> filenames_;
    utility::VerbosityLevel verbosity_;
    int end_range_;
    bool visualize_output_;

private:
    std::vector<double> voxel_sizes_;
    std::vector<double> search_radius_;
    std::vector<ICPConvergenceCriteria> criterias_;
    std::shared_ptr<TransformationEstimation> estimation_;

private:
    core::Tensor transformation_;
    t::pipelines::registration::RegistrationResult result_;

private:
    core::Device device_;
    core::Device host_;
    core::Dtype dtype_;

private:
    // Ground Truth value of odometry after (end_range - 1)th iteration.
    double gt_tx_;
    double gt_ty_;
};

int main(int argc, char* argv[]) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    const std::string path_config = std::string(argv[2]);
    MultipleWindowsApp(path_config, core::Device(argv[1])).Run();
    return 0;
}
