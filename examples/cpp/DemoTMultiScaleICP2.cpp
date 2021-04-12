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
const std::string DATA_PATH = "../../../examples/test_data/ICP/cloud_bin_0.pcd";
const Eigen::Vector3f CENTER_OFFSET(0.0f, 0.0f, -3.0f);
const std::string CLOUD_NAME = "points";

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
                core::Tensor(initial_transform_flat, {4, 4}, dtype_, host_);

        // Warm Up.
        std::vector<ICPConvergenceCriteria> warm_up_criteria = {
                ICPConvergenceCriteria(0.01, 0.01, 1)};
        result_ = RegistrationMultiScaleICP(
                pointclouds_device_[0].To(device_), pointclouds_device_[0].To(device_), {1.0},
                warm_up_criteria, {1.5}, core::Tensor::Eye(4, dtype_, device_),
                *estimation_);

        std::cout << " [Debug] Warm up transformation: "
                  << result_.transformation_.ToString() << std::endl;
        is_done_ = false;

        gui::Application::GetInstance().Initialize();
    }

    void Run() {
        main_vis_ = std::make_shared<visualizer::O3DVisualizer>(
                "Open3D - Multi-Window Demo", WIDTH, HEIGHT);

        main_vis_->SetOnClose([this]() { return this->OnMainWindowClosing(); });

        gui::Application::GetInstance().AddWindow(main_vis_);
        auto r = main_vis_->GetOSFrame();
        snapshot_pos_ = gui::Point(r.x, r.y);

        std::thread read_thread([this]() { this->MultiScaleICPDemo(); });
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
    void MultiScaleICPDemo() {
        // This is NOT the UI thread, need to call PostToMainThread() to
        // update the scene or any part of the UI.

        geometry::AxisAlignedBoundingBox bounds;
        Eigen::Vector3d extent;
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            lsource_ = std::make_shared<geometry::PointCloud>();
            
            auto pcd = pointclouds_device_[0].Clone();
            *lsource_ = pcd.ToLegacyPointCloud();
            // io::ReadPointCloud(path_source_, *lsource_);
            ltarget_ = std::make_shared<geometry::PointCloud>();
            auto pcd2 = pointclouds_device_[1].Clone();
            *ltarget_ = pcd2.ToLegacyPointCloud();
            // io::ReadPointCloud(path_target_, *ltarget_);
            bounds = lsource_->GetAxisAlignedBoundingBox();
            extent = bounds.GetExtent();
        }

        auto mat = rendering::Material();
        mat.shader = "defaultUnlit";

        gui::Application::GetInstance().PostToMainThread(
                main_vis_.get(), [this, bounds, mat]() {
                    std::lock_guard<std::mutex> lock(cloud_lock_);
                    main_vis_->AddGeometry(SRC_CLOUD, lsource_, &mat);
                    main_vis_->AddGeometry(DST_CLOUD, ltarget_, &mat);
                    main_vis_->ResetCameraToDefault();
                    Eigen::Vector3f center = bounds.GetCenter().cast<float>();
                    main_vis_->SetupCamera(60, center, center + CENTER_OFFSET,
                                           {0.0f, -1.0f, 0.0f});
                });

        utility::SetVerbosityLevel(verbosity_);

        // while (main_vis_) {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // source_ and target_ are tensor pointcloud on host.
        auto transformation_device = transformation_.To(device_, true);
        auto source_device = pointclouds_device_[0].To(device_, true);
        auto target_device = pointclouds_device_[1].To(device_, true);

        utility::Timer time_icp;

        time_icp.Start();
        int64_t num_iterations = int64_t(criterias_.size());

        // Creating pointcloud pyramid with different voxel scale.
        std::vector<t::geometry::PointCloud> source_down_pyramid(
                num_iterations);
        std::vector<t::geometry::PointCloud> target_down_pyramid(
                num_iterations);

        if (voxel_sizes_[num_iterations - 1] == -1) {
            source_down_pyramid[num_iterations - 1] = source_device;
            target_down_pyramid[num_iterations - 1] = target_device;
        } else {
            source_down_pyramid[num_iterations - 1] =
                    source_device.VoxelDownSample(
                            voxel_sizes_[num_iterations - 1]);
            target_down_pyramid[num_iterations - 1] =
                    target_device.VoxelDownSample(
                            voxel_sizes_[num_iterations - 1]);
        }
        for (int k = num_iterations - 2; k >= 0; k--) {
            source_down_pyramid[k] =
                    source_down_pyramid[k + 1].VoxelDownSample(voxel_sizes_[k]);
            target_down_pyramid[k] =
                    target_down_pyramid[k + 1].VoxelDownSample(voxel_sizes_[k]);
        }

        RegistrationResult result_device(transformation_device);

        for (int64_t i = 0; i < num_iterations; i++) {
            source_down_pyramid[i].Transform(transformation_device);

            core::nns::NearestNeighborSearch target_nns(
                    target_down_pyramid[i].GetPoints());

            result_device = GetRegistrationResultAndCorrespondences(
                    source_down_pyramid[i], target_down_pyramid[i], target_nns,
                    search_radius_[i], transformation_device);

            for (int j = 0; j < criterias_[i].max_iteration_; j++) {
                utility::LogInfo(
                        " ICP Scale #{:d} Iteration #{:d}: Fitness {:.4f}, "
                        "RMSE {:.4f}",
                        i + 1, j, result_device.fitness_,
                        result_device.inlier_rmse_);

                core::Tensor update = estimation_->ComputeTransformation(
                        source_down_pyramid[i], target_down_pyramid[i],
                        result_device.correspondence_set_);

                utility::LogDebug(" Delta Transformation: {}",
                                  update.ToString());

                // Multiply the delta transform [n-1 to n] to the cumulative
                // transformation [0 to n-1] to update cumulative [0 to n].
                transformation_device = update.Matmul(transformation_device);
                // Apply the transform on source pointcloud.
                source_down_pyramid[i].Transform(update);

                auto temp =
                        source_device.Transform(transformation_device).Clone();

                // UPDATE VISUALIZATION!
                {
                    std::lock_guard<std::mutex> lock(cloud_lock_);
                    *lsource_ = temp.ToLegacyPointCloud();
                }

                if (!main_vis_) {  // might have changed while sleeping
                    break;
                }

                gui::Application::GetInstance().PostToMainThread(
                        main_vis_.get(), [this, mat]() {
                            std::lock_guard<std::mutex> lock(cloud_lock_);
                            main_vis_->RemoveGeometry(SRC_CLOUD);
                            main_vis_->AddGeometry(SRC_CLOUD, lsource_, &mat);
                        });

                double prev_fitness_ = result_device.fitness_;
                double prev_inliner_rmse_ = result_device.inlier_rmse_;

                result_device = GetRegistrationResultAndCorrespondences(
                        source_down_pyramid[i], target_down_pyramid[i],
                        target_nns, search_radius_[i], transformation_device);

                // ICPConvergenceCriteria, to terminate iteration.
                if (j != 0 &&
                    std::abs(prev_fitness_ - result_device.fitness_) <
                            criterias_[i].relative_fitness_ &&
                    std::abs(prev_inliner_rmse_ - result_device.inlier_rmse_) <
                            criterias_[i].relative_rmse_) {
                    break;
                }
            }
        }
        time_icp.Stop();
        utility::LogInfo(" Time [ICP + Visualization update]: {}",
                         time_icp.GetDuration());
        // }
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
                    path_dataset_ = value;
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
                // } else if (name == "ground_truth_tx") {
                //     std::istringstream is(value);
                //     gt_tx_ = std::stod(value);
                // } else if (name == "ground_truth_ty") {
                //     std::istringstream is(value);
                //     gt_ty_ = std::stod(value);
                }
            }
        } else {
            std::cerr << "Couldn't open config file for reading.\n";
        }

        utility::LogInfo(" Dataset path: {}", path_dataset_);
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
    // To perform required dtype conversion, normal estimation and device
    // transfer.
    std::vector<t::geometry::PointCloud> LoadTensorPointClouds() {
        for (int i = 0; i < end_range_; i++) {
            filenames_.push_back(path_dataset_ + std::to_string(i) +
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
                                pointcloud_local.GetPointAttr(attr).To(dtype_, true));
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
                                    .To(device_, dtype_, true);
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

    RegistrationResult GetRegistrationResultAndCorrespondences(
            const t::geometry::PointCloud& source,
            const t::geometry::PointCloud& target,
            open3d::core::nns::NearestNeighborSearch& target_nns,
            double max_correspondence_distance,
            const core::Tensor& transformation) {
        core::Device device = source.GetDevice();
        core::Dtype dtype = core::Dtype::Float32;
        source.GetPoints().AssertDtype(dtype);
        target.GetPoints().AssertDtype(dtype);
        if (target.GetDevice() != device) {
            utility::LogError(
                    "Target Pointcloud device {} != Source Pointcloud's device "
                    "{}.",
                    target.GetDevice().ToString(), device.ToString());
        }
        transformation.AssertShape({4, 4});
        transformation.AssertDtype(dtype);

        core::Tensor transformation_device = transformation.To(device);

        RegistrationResult result(transformation_device);
        if (max_correspondence_distance <= 0.0) {
            return result;
        }

        bool check = target_nns.HybridIndex(max_correspondence_distance);
        if (!check) {
            utility::LogError(
                    "[Tensor: EvaluateRegistration: "
                    "GetRegistrationResultAndCorrespondences: "
                    "NearestNeighborSearch::HybridSearch] "
                    "Index is not set.");
        }

        core::Tensor distances;
        std::tie(result.correspondence_set_.first,
                 result.correspondence_set_.second, distances) =
                target_nns.Hybrid1NNSearch(source.GetPoints(),
                                           max_correspondence_distance);

        // Number of good correspondences (C).
        int num_correspondences = result.correspondence_set_.first.GetLength();

        // Reduction sum of "distances" for error.
        double squared_error =
                static_cast<double>(distances.Sum({0}).Item<float>());
        result.fitness_ = static_cast<double>(num_correspondences) /
                          static_cast<double>(source.GetPoints().GetLength());
        result.inlier_rmse_ = std::sqrt(
                squared_error / static_cast<double>(num_correspondences));
        result.transformation_ = transformation;

        return result;
    }

private:
    std::mutex cloud_lock_;
    std::shared_ptr<geometry::PointCloud> cloud_;

    std::atomic<bool> is_done_;
    std::shared_ptr<visualizer::O3DVisualizer> main_vis_;
    int n_snapshots_ = 0;
    gui::Point snapshot_pos_;

private:
    std::vector<open3d::t::geometry::PointCloud> pointclouds_device_;
    t::geometry::PointCloud source_;
    t::geometry::PointCloud target_;

    // Source PointCloud on CPU, used for visualization.
    std::shared_ptr<geometry::PointCloud> lsource_;
    // Target PointCloud on CPU, used for visualization.
    std::shared_ptr<geometry::PointCloud> ltarget_;

private:
    std::string path_source_;
    std::string path_target_;
    std::string path_dataset_;
    std::string registration_method_;
    utility::VerbosityLevel verbosity_;
    std::vector<std::string> filenames_;
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
};

int main(int argc, char* argv[]) {
    const std::string path_config = std::string(argv[2]);
    MultipleWindowsApp(path_config, core::Device(argv[1])).Run();
    return 0;
}
