// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

const int WIDTH = 1280;
const int HEIGHT = 800;
float verticalFoV = 25;

const Eigen::Vector3f CENTER_OFFSET(-10.0f, 0.0f, 30.0f);
// const Eigen::Vector3f CENTER_OFFSET(0.5f, -0.5f, -5.0f);
const std::string CLOUD_NAME = "points";

const std::string SRC_CLOUD = "source_pointcloud";
const std::string DST_CLOUD = "target_pointcloud";
const std::string LINE_SET = "correspondences_lines";
const std::string SRC_CORRES = "source_correspondences_idx";
const std::string TAR_CORRES = "target_correspondences_idx";

std::vector<double> initial_transform_flat = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                              0.0, 0.0, 0.0, 1.0};

class ReconstructionWindow : public gui::Window {
    using Super = gui::Window;

public:
    ReconstructionWindow() : gui::Window("Open3D - Reconstruction", 1600, 900) {
        widget3d_ = std::make_shared<gui::SceneWidget>();
        AddChild(widget3d_);
        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~ReconstructionWindow() {}

protected:
    std::shared_ptr<gui::SceneWidget> widget3d_;
};

//------------------------------------------------------------------------------
class ExampleWindow : public ReconstructionWindow {
public:
    ExampleWindow(const std::string& path_config, const core::Device& device)
        : device_(device),
          host_(core::Device("CPU:0")),
          dtype_(core::Dtype::Float32) {
        ReadConfigFile(path_config);
        std::tie(source_, target_) = LoadTensorPointClouds();

        transformation_ = core::Tensor(initial_transform_flat, {4, 4},
                                       core::Dtype::Float64, host_);

        // Warm Up.
        std::vector<ICPConvergenceCriteria> warm_up_criteria = {
                ICPConvergenceCriteria(0.01, 0.01, 1)};
        result_ = RegistrationMultiScaleICP(
                source_.To(device_), target_.To(device_), {1.0},
                warm_up_criteria, {1.5},
                core::Tensor::Eye(4, core::Dtype::Float64, host_),
                *estimation_);

        std::cout << " [Debug] Warm up transformation: "
                  << result_.transformation_.ToString() << std::endl;
        is_done_ = false;

        // --------------------- VISUALIZER ---------------------
        gui::Application::GetInstance().Initialize();

        src_cloud_mat_ = rendering::Material();
        src_cloud_mat_.shader = "defaultUnlit";
        // src_cloud_mat_.base_color = Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f);

        tar_cloud_mat_ = rendering::Material();
        tar_cloud_mat_.shader = "defaultUnlit";
        // tar_cloud_mat_.base_color = Eigen::Vector4f(0.7f, 0.7f, 0.7f, 1.0f);

        src_corres_mat_ = rendering::Material();
        src_corres_mat_.shader = "defaultUnlit";
        src_corres_mat_.base_color = Eigen::Vector4f(0.f, 1.0f, 0.0f, 1.0f);
        src_corres_mat_.point_size = 3.0f;

        tar_corres_mat_ = rendering::Material();
        tar_corres_mat_.shader = "defaultUnlit";
        tar_corres_mat_.base_color = Eigen::Vector4f(1.f, 0.0f, 0.0f, 1.0f);
        tar_corres_mat_.point_size = 3.0f;
        // ------------------------------------------------------

        SetOnClose([this]() {
            is_done_ = true;
            return true;  // false would cancel the close
        });
        update_thread_ = std::thread([this]() { this->UpdateMain(); });
    }

    ~ExampleWindow() { update_thread_.join(); }

private:
    std::thread update_thread_;

    void UpdateMain() {
        core::Tensor initial_transform = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0"));
        core::Tensor cumulative_transform = initial_transform.Clone();

        // ------------------------ VISUALIZER ------------------------------
        // Intialize visualizer.
        {
            // lock to protect `source_` and `target_` before modifying the
            // value, ensuring the visualizer thread doesn't read the data,
            // while we are modifying it.
            std::lock_guard<std::mutex> lock(pcd_.lock_);

            // Copying the pointcloud on CPU, as required by the visualizer.
            pcd_.source_ = source_.CPU();
            pcd_.target_ = target_.CPU();
        }

        gui::Application::GetInstance().PostToMainThread(this, [this]() {
            // lock to protect `pcd_.source_` and `pcd.target_`
            // before passing it to the visualizer, ensuring we don't
            // modify the value, when visualizer is reading it.
            std::lock_guard<std::mutex> lock(pcd_.lock_);

            // Setting the background.
            this->widget3d_->GetScene()->SetBackground({0, 0, 0, 1});

            // Adding the target pointcloud.
            this->widget3d_->GetScene()->AddGeometry(DST_CLOUD, &pcd_.target_,
                                                     tar_cloud_mat_);

            // Adding the source pointcloud, and correspondences pointclouds.
            // This works as a pointcloud container, i.e. reserves the
            // resources. Later we will just use `UpdateGeometry` which is
            // efficient when the number of points in the updated pointcloud
            // are same or less than the geometry added initially.
            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    SRC_CLOUD, pcd_.source_, src_cloud_mat_);
            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    SRC_CORRES, pcd_.source_, src_corres_mat_);
            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    TAR_CORRES, pcd_.source_, tar_corres_mat_);

            // Getting bounding box and center to setup camera view.
            auto bbox = this->widget3d_->GetScene()->GetBoundingBox();
            auto center = bbox.GetCenter().cast<float>();
            this->widget3d_->SetupCamera(18, bbox, center);
            this->widget3d_->LookAt(center, center - Eigen::Vector3f{-10, 5, 8},
                                    {0.0f, -1.0f, 0.0f});
        });
        // -----------------------------------------------------

        // ----- Class members passed to function arguments
        // ----- in t::pipeline::registration::RegistrationMultiScaleICP
        const t::geometry::PointCloud source = source_.To(device_);
        const t::geometry::PointCloud target = target_.To(device_);
        const std::vector<double> voxel_sizes = voxel_sizes_;
        const std::vector<ICPConvergenceCriteria> criterias = criterias_;
        const std::vector<double> max_correspondence_distances =
                max_correspondence_distances_;
        const core::Tensor init = transformation_;
        auto& estimation = *estimation_;

        // ----- RegistrationMultiScaleICP Function directly taken from
        // ----- t::pipelines::registration, and added O3DVisualizer to it.
        core::Device device = source.GetDevice();
        core::Dtype dtype = core::Dtype::Float32;

        source.GetPoints().AssertDtype(
                dtype,
                " RegistrationICP: Only Float32 Point cloud "
                "are supported currently.");
        target.GetPoints().AssertDtype(
                dtype,
                " RegistrationICP: Only Float32 Point cloud "
                "are supported currently.");

        if (target.GetDevice() != device) {
            utility::LogError(
                    "Target Pointcloud device {} != Source Pointcloud's device "
                    "{}.",
                    target.GetDevice().ToString(), device.ToString());
        }

        int64_t num_iterations = int64_t(criterias.size());
        if (!(criterias.size() == voxel_sizes.size() &&
              criterias.size() == max_correspondence_distances.size())) {
            utility::LogError(
                    " [RegistrationMultiScaleICP]: Size of criterias, "
                    "voxel_size,"
                    " max_correspondence_distances vectors must be same.");
        }

        if ((estimation.GetTransformationEstimationType() ==
                     TransformationEstimationType::PointToPlane ||
             estimation.GetTransformationEstimationType() ==
                     TransformationEstimationType::ColoredICP) &&
            (!target.HasPointNormals())) {
            utility::LogError(
                    "TransformationEstimationPointToPlane and "
                    "TransformationEstimationColoredICP "
                    "require pre-computed normal vectors for target "
                    "PointCloud.");
        }

        if (max_correspondence_distances[0] <= 0.0) {
            utility::LogError(
                    " Max correspondence distance must be greater than 0, but"
                    " got {} in scale: {}.",
                    max_correspondence_distances[0], 0);
        }

        for (int64_t i = 1; i < num_iterations; i++) {
            if (voxel_sizes[i] >= voxel_sizes[i - 1]) {
                utility::LogError(
                        " [MultiScaleICP] Voxel sizes must be in strictly "
                        "decreasing order.");
            }
            if (max_correspondence_distances[i] <= 0.0) {
                utility::LogError(
                        " Max correspondence distance must be greater than 0, "
                        "but got {} in scale: {}.",
                        max_correspondence_distances[i], i);
            }
        }

        init.AssertShape({4, 4});

        core::Tensor transformation =
                init.To(core::Device("CPU:0"), core::Dtype::Float64);

        std::vector<t::geometry::PointCloud> source_down_pyramid(
                num_iterations);
        std::vector<t::geometry::PointCloud> target_down_pyramid(
                num_iterations);

        if (voxel_sizes[num_iterations - 1] == -1) {
            source_down_pyramid[num_iterations - 1] = source.Clone();
            target_down_pyramid[num_iterations - 1] = target;
        } else {
            source_down_pyramid[num_iterations - 1] =
                    source.Clone().VoxelDownSample(
                            voxel_sizes[num_iterations - 1]);
            target_down_pyramid[num_iterations - 1] =
                    target.Clone().VoxelDownSample(
                            voxel_sizes[num_iterations - 1]);
        }

        for (int k = num_iterations - 2; k >= 0; k--) {
            source_down_pyramid[k] =
                    source_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
            target_down_pyramid[k] =
                    target_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
        }

        RegistrationResult result(transformation);

        for (int64_t i = 0; i < num_iterations; i++) {
            source_down_pyramid[i].Transform(transformation.To(device, dtype));

            core::nns::NearestNeighborSearch target_nns(
                    target_down_pyramid[i].GetPoints());

            result = GetRegistrationResultAndCorrespondences(
                    source_down_pyramid[i], target_down_pyramid[i], target_nns,
                    max_correspondence_distances[i], transformation);

            for (int j = 0; j < criterias[i].max_iteration_; j++) {
                utility::LogDebug(
                        " ICP Scale #{:d} Iteration #{:d}: Fitness {:.4f}, "
                        "RMSE "
                        "{:.4f}",
                        i + 1, j, result.fitness_, result.inlier_rmse_);

                // ComputeTransformation returns transformation matrix of
                // dtype Float64.
                core::Tensor update = estimation.ComputeTransformation(
                        source_down_pyramid[i], target_down_pyramid[i],
                        result.correspondence_set_);

                // Multiply the transform to the cumulative transformation
                // (update).
                transformation = update.Matmul(transformation);
                // Apply the transform on source pointcloud.
                source_down_pyramid[i].Transform(update.To(device, dtype));

                double prev_fitness_ = result.fitness_;
                double prev_inliner_rmse_ = result.inlier_rmse_;

                result = GetRegistrationResultAndCorrespondences(
                        source_down_pyramid[i], target_down_pyramid[i],
                        target_nns, max_correspondence_distances[i],
                        transformation);

                // -------------------- VISUALIZER ----------------------
                {
                    std::lock_guard<std::mutex> lock(pcd_.lock_);
                    pcd_.correspondence_src_.SetPoints(
                            source_down_pyramid[i]
                                    .GetPoints()
                                    .IndexGet(
                                            {result.correspondence_set_.first})
                                    .To(host_));
                    pcd_.correspondence_tar_.SetPoints(
                            target_down_pyramid[i]
                                    .GetPoints()
                                    .IndexGet(
                                            {result.correspondence_set_.second})
                                    .To(host_));

                    pcd_.source_ = source_.Clone().Transform(
                            transformation.To(dtype_));
                }

                // To update visualizer, we go to the `main thread`,
                // bring the data on the `main thread`, ensure there is no race
                // condition with the data, and pass it to the visualizer for
                // rendering, using `AddGeometry`, or update an existing
                // pointcloud using `UpdateGeometry`, then setup camera.
                gui::Application::GetInstance().PostToMainThread(this, [this,
                                                                        i]() {
                    // Locking to protect: pcd_.source_,
                    // pcd_.correspondence_src_, pcd_correpondece_tar_.
                    std::lock_guard<std::mutex> lock(pcd_lock_);

                    this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                            SRC_CLOUD, pcd_.source_,
                            rendering::Scene::kUpdatePointsFlag |
                                    rendering::Scene::kUpdateColorsFlag);
                    this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                            SRC_CORRES, pcd_.correspondence_src_,
                            rendering::Scene::kUpdatePointsFlag |
                                    rendering::Scene::kUpdateColorsFlag);
                    this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                            TAR_CORRES, pcd_.correspondence_tar_,
                            rendering::Scene::kUpdatePointsFlag |
                                    rendering::Scene::kUpdateColorsFlag);
                });
                // -------------------------------------------------------

                // ICPConvergenceCriteria, to terminate iteration.
                if (j != 0 &&
                    std::abs(prev_fitness_ - result.fitness_) <
                            criterias_[i].relative_fitness_ &&
                    std::abs(prev_inliner_rmse_ - result.inlier_rmse_) <
                            criterias_[i].relative_rmse_) {
                    break;
                }

                // Delays each iteration to allow clear visualization of
                // each iteration.
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        // ------------------ VISUALIZER ----------------------------
        // Clearing up the correspondences representation,
        // after all the iterations are completed.
        gui::Application::GetInstance().PostToMainThread(this, [this]() {
            // Locking before removing correspondence_src_ and
            // correspondence_tar_.
            std::lock_guard<std::mutex> lock(pcd_.lock_);

            this->widget3d_->GetScene()->GetScene()->RemoveGeometry(SRC_CORRES);
            this->widget3d_->GetScene()->GetScene()->RemoveGeometry(TAR_CORRES);
        });
        // ----------------------------------------------------------
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

                if (name == "source_path") {
                    path_source_ = value;
                } else if (name == "target_path") {
                    path_target_ = value;
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
                    max_correspondence_distances_.push_back(std::stod(value));
                } else if (name == "verbosity") {
                    std::istringstream is(value);
                    verb = value;
                }
            }
        } else {
            std::cerr << "Couldn't open config file for reading.\n";
        }

        utility::LogInfo(" Source path: {}", path_source_);
        utility::LogInfo(" Target path: {}", path_target_);
        utility::LogInfo(" Registrtion method: {}", registration_method_);
        std::cout << std::endl;

        std::cout << " Initial Transformation Guess: " << std::endl;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << " " << initial_transform_flat[i * 4 + j];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << " Voxel Sizes: ";
        for (auto voxel_size : voxel_sizes_) std::cout << voxel_size << " ";
        std::cout << std::endl;

        std::cout << " Search Radius Sizes: ";
        for (auto search_radii : max_correspondence_distances_)
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
        if (max_correspondence_distances_.size() != length ||
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
    std::tuple<t::geometry::PointCloud, t::geometry::PointCloud>
    LoadTensorPointClouds() {
        t::geometry::PointCloud source(host_), target(host_);

        // t::io::ReadPointCloud copies the pointcloud to CPU.
        t::io::ReadPointCloud(path_source_, source,
                              {"auto", false, false, true});
        t::io::ReadPointCloud(path_target_, target,
                              {"auto", false, false, true});

        // Cpnverting attributes to Floar32 and currently only
        // Float32 pointcloud is supported by the tensor registration module.
        for (std::string attr : {"points", "colors", "normals"}) {
            if (source.HasPointAttr(attr)) {
                source.SetPointAttr(attr, source.GetPointAttr(attr).To(dtype_));
            }
        }
        for (std::string attr : {"points", "colors", "normals"}) {
            if (target.HasPointAttr(attr)) {
                target.SetPointAttr(attr, target.GetPointAttr(attr).To(dtype_));
            }
        }

        // Normals are required for `PointToPlane` type registration method.
        // Currenly Normal Estimation is not supported by Tensor Pointcloud.
        if (registration_method_ == "PointToPlane" &&
            !target.HasPointNormals()) {
            auto target_legacy = target.ToLegacyPointCloud();
            target_legacy.EstimateNormals(geometry::KDTreeSearchParamKNN(),
                                          false);
            core::Tensor target_normals =
                    t::geometry::PointCloud::FromLegacyPointCloud(target_legacy)
                            .GetPointNormals()
                            .To(dtype_);
            target.SetPointNormals(target_normals);
        }

        return std::make_tuple(source, target);
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

        core::Tensor transformation_host =
                transformation.To(core::Device("CPU:0"), core::Dtype::Float64);

        RegistrationResult result(transformation_host);
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
        std::tie(result.correspondence_set_.second, distances) =
                target_nns.HybridSearch(source.GetPoints(),
                                        max_correspondence_distance, 1);

        core::Tensor valid =
                result.correspondence_set_.second.Ne(-1).Reshape({-1});
        // correpondence_set : (i, corres[i]).
        // source[i] and target[corres[i]] is a correspondence.
        result.correspondence_set_.first =
                core::Tensor::Arange(0, source.GetPoints().GetShape()[0], 1,
                                     core::Dtype::Int64, device)
                        .IndexGet({valid});
        // Only take valid indices.
        result.correspondence_set_.second =
                result.correspondence_set_.second.IndexGet({valid}).Reshape(
                        {-1});

        // Number of good correspondences (C).
        int num_correspondences = result.correspondence_set_.first.GetLength();

        // Reduction sum of "distances" for error.
        double squared_error =
                static_cast<double>(distances.Sum({0}).Item<float>());
        result.fitness_ = static_cast<double>(num_correspondences) /
                          static_cast<double>(source.GetPoints().GetLength());
        result.inlier_rmse_ = std::sqrt(
                squared_error / static_cast<double>(num_correspondences));

        return result;
    }

private:
    core::Device device_;
    core::Device host_;
    core::Dtype dtype_;

private:
    std::mutex pcd_lock_;

    std::atomic<bool> is_done_;
    open3d::visualization::rendering::Material src_cloud_mat_;
    open3d::visualization::rendering::Material tar_cloud_mat_;
    open3d::visualization::rendering::Material src_corres_mat_;
    open3d::visualization::rendering::Material tar_corres_mat_;

    // For Visualization.
    // The members of this structure can be protected by the mutex lock,
    // to avoid the case, when we are trying to modify the values,
    // while visualizer is tring to access it.
    struct {
        std::mutex lock_;
        t::geometry::PointCloud correspondence_src_;
        t::geometry::PointCloud correspondence_tar_;
        t::geometry::PointCloud source_;
        t::geometry::PointCloud target_;
    } pcd_;

    t::geometry::PointCloud source_;
    t::geometry::PointCloud target_;

private:
    std::string path_source_;
    std::string path_target_;
    std::string registration_method_;
    utility::VerbosityLevel verbosity_;
    bool visualize_output_;

private:
    std::vector<double> voxel_sizes_;
    std::vector<double> max_correspondence_distances_;
    std::vector<ICPConvergenceCriteria> criterias_;
    std::shared_ptr<TransformationEstimation> estimation_;

private:
    core::Tensor transformation_;
    t::pipelines::registration::RegistrationResult result_;
};

//------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    if (argc < 3) {
        utility::LogError("Expected dataset path as input");
    }
    const std::string path_config = std::string(argv[2]);

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, argv);
    app.AddWindow(std::make_shared<ExampleWindow>(path_config,
                                                  core::Device(argv[1])));
    app.Run();
    return 0;
}
