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

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"
#include "open3d/t/geometry/Utility.h"

using namespace open3d;
using namespace open3d::visualization;
using namespace open3d::t::pipelines::registration;

#define DONT_RESAMPLE -1

const int WIDTH = 1280;
const int HEIGHT = 800;
float verticalFoV = 25;

const Eigen::Vector3f CENTER_OFFSET(-10.0f, 0.0f, 30.0f);
const std::string CURRENT_CLOUD = "current_scan";
std::string window_name = "Open3D - ICP Frame to Frame Odometry";
std::string device_string = "CPU:0";
std::string widget_string = "Average FPS on ";

//------------------------------------------------------------------------------
// Creating GUI Layout
//------------------------------------------------------------------------------
class ReconstructionWindow : public gui::Window {
    using Super = gui::Window;

public:
    ReconstructionWindow() : gui::Window(window_name, WIDTH, HEIGHT) {
        auto& theme = GetTheme();
        int em = theme.font_size;
        int spacing = int(std::round(0.5f * float(em)));
        gui::Margins margins(int(std::round(0.5f * float(em))));

        widget3d_ = std::make_shared<gui::SceneWidget>();
        output_panel_ = std::make_shared<gui::Vert>(spacing, margins);

        // For displaying pointcloud data.
        AddChild(widget3d_);

        // For output text information, such as FPS and total number of points.
        AddChild(output_panel_);

        output_ = std::make_shared<gui::Label>("");
        const char* label = widget_string.c_str();
        output_panel_->AddChild(std::make_shared<gui::Label>(label));
        output_panel_->AddChild(output_);

        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~ReconstructionWindow() {}

    void Layout(const gui::LayoutContext& context) override {
        int em = context.theme.font_size;
        int panel_width = 15 * em;
        int panel_height = 100;
        // The usable part of the window may not be the full size if there
        // is a menu.
        auto content_rect = GetContentRect();

        output_panel_->SetFrame(gui::Rect(content_rect.GetRight() - panel_width,
                                          content_rect.y, panel_width,
                                          panel_height));
        int x = content_rect.x;
        widget3d_->SetFrame(gui::Rect(x, content_rect.y, content_rect.width,
                                      content_rect.height));
        Super::Layout(context);
    }

protected:
    std::shared_ptr<gui::Vert> output_panel_;
    std::shared_ptr<gui::Label> output_;
    std::shared_ptr<gui::SceneWidget> widget3d_;

    void SetOutput(const std::string& output) {
        output_->SetText(output.c_str());
    }
};
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Class containing the MultiScaleICP based Frame to Frame Odometry function
// integrated with visualizer.
//------------------------------------------------------------------------------
class ExampleWindow : public ReconstructionWindow {
public:
    ExampleWindow(const std::string& path_config, const core::Device& device)
        : device_(device),
          host_(core::Device("CPU:0")),
          dtype_(core::Dtype::Float32) {
        ReadConfigFile(path_config);

        // Loads the pointcloud, converts to Float32 if required (currently
        // only Float32 dtype pointcloud is supported by tensor registration
        // pipeline), estimates normals if required (PointToPlane Registration),
        // sets the "__visualization_scalar" parameter and its min max values.
        LoadTensorPointClouds();

        // Rendering Material used for `current frame`.
        mat_ = rendering::MaterialRecord();
        mat_.shader = "defaultUnlit";
        mat_.base_color = Eigen::Vector4f(0.72f, 0.45f, 0.69f, 1.0f);
        mat_.point_size = 3.0f;

        // Rendering Material used for `cummulative pointcloud`.
        pointcloud_mat_ = GetPointCloudMaterial();

        // When window is closed, it will stop the execution of the code.
        is_done_ = false;
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
        // --------------------- VISUALIZATION ------------------------------
        // Initialize visualizer.
        if (visualize_output_) {
            {
                // lock to protect `curren_scan_` and `bbox_` before modifying
                // the value, ensuring the visualizer thread doesn't read the
                // data, while we are modifying it.
                std::lock_guard<std::mutex> lock(pcd_and_bbox_.lock_);

                // Copying the pointcloud to pcd_and_bbox_.current_scan_ on the
                // `main thread` on CPU, which is later passed to the visualizer
                // for rendering.
                pcd_and_bbox_.current_scan_ =
                        pointclouds_device_[0].To(core::Device("CPU:0"));

                // Removing `normal` attribute before passing it to
                // the visualizer might give us some performance benifits.
                pcd_and_bbox_.current_scan_.RemovePointAttr("normals");
            }

            gui::Application::GetInstance().PostToMainThread(this, [&]() {
                std::lock_guard<std::mutex> lock(pcd_and_bbox_.lock_);

                // Setting background for the visualizer. [In this case: Black].
                this->widget3d_->GetScene()->SetBackground({0, 0, 0, 1.0});

                // Adding the first frame of the sequence to the visualizer,
                // and rendering it using the material set for `current scan`.
                this->widget3d_->GetScene()->AddGeometry(
                        filenames_[0], &pcd_and_bbox_.current_scan_, mat_);

                // Getting bounding box and center to setup camera view.
                pcd_and_bbox_.bbox_ =
                        this->widget3d_->GetScene()->GetBoundingBox();
                auto center = pcd_and_bbox_.bbox_.GetCenter().cast<float>();
                this->widget3d_->SetupCamera(verticalFoV, pcd_and_bbox_.bbox_,
                                             center);
            });
        }
        // ------------------------------------------------------------------

        // Initial transfrom from source to target, to initialize ICP.
        core::Tensor initial_transform = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0"));

        // Cumulative transform or frame to model transform
        // from pcd[i] (current frame) to pcd[0] (initial or reference frame).
        core::Tensor cumulative_transform = initial_transform.Clone();

        // Final scale level downsampling is already performed while loading the
        // data. -1 avoids re-downsampling for the last scale level.
        voxel_sizes_[icp_scale_levels_ - 1] = DONT_RESAMPLE;

        // ---------------- Warm up -----------------------
        auto result = MultiScaleICP(pointclouds_device_[0].To(device_),
                                    pointclouds_device_[1].To(device_),
                                    voxel_sizes_, criterias_, search_radius_,
                                    initial_transform, *estimation_);
        // ------------------------------------------------

        utility::SetVerbosityLevel(verbosity_);

        // Global variables required for calculating avergage FPS till
        // i-th iteration.
        double total_time_i = 0;
        int64_t total_points_in_frame = 0;
        int i = 0;

        int total_frames = end_index_ - start_index_;

        // --------------------- Main Compute Function ----------------------
        for (i = 0; i < total_frames - 1 && !is_done_; i++) {
            utility::Timer time_total;
            time_total.Start();

            // NOTE:
            // IN CASE THE DATASET IS TOO LARGE FOR YOUR MEMORY, AVOID
            // PREFETCHING THE DATA IN THE FUNCTION `LoadTensorPointClouds()`
            // AND READ IT HERE DIRECTLY. REFER TO THE FUNCTION
            // `LoadTensorPointClouds` TO UNDERSTAND THE PRE-PROCESSING
            // REQUIREMENTS.

            // Reads the pre-fetched and pre-processed pointcloud frames.
            auto source = pointclouds_device_[i].To(device_);
            auto target = pointclouds_device_[i + 1].To(device_);

            // Computes the transformation from pcd_[i] to pcd_[i + 1], for
            // `Frame to Frame Odometry`.
            auto result = MultiScaleICP(source, target, voxel_sizes_,
                                        criterias_, search_radius_,
                                        initial_transform, *estimation_);

            // `cumulative_transform` before update is from `i to 0`.
            // `result.transformation_` is from i to i + 1.
            // so, `cumulative_transform @ (result.transformation_).Inverse`
            // gives `transformation of [i + 1]th frame to 0` [reference or
            // initial] frame. So, pose of the ego-vehicle / sensor
            // till this frame w.r.t. the inital frame, or `global_pose`
            // or `frame to model transform` is given by `cumulative_transform.`
            cumulative_transform = cumulative_transform.Matmul(
                    t::geometry::InverseTransformation(
                            result.transformation_.Contiguous()));

            // -------------------- VISUALIZATION ----------------------------
            if (visualize_output_) {
                // Output stream to our visualizer, in this case we update the
                // Average FPS and Total Points values.
                std::stringstream out_;

                {
                    // lock `current_scan_` and `bbox_` before modifying the
                    // value, to protect the case, when visualizer is accessing
                    // it at the same time we are modifying it.
                    std::lock_guard<std::mutex> lock(pcd_and_bbox_.lock_);

                    // For visualization it is required that the pointcloud
                    // must be on CPU device.
                    // The `target` pointcloud is transformed to it's global
                    // position in the model by it's `frame to model transform`.
                    pcd_and_bbox_.current_scan_ =
                            target.Transform(cumulative_transform)
                                    .To(core::Device("CPU:0"));

                    // Translate bounding box to current scan frame to model
                    // transform.
                    pcd_and_bbox_.bbox_ = pcd_and_bbox_.bbox_.Translate(
                            core::eigen_converter::TensorToEigenMatrixXd(
                                    cumulative_transform.Clone()
                                            .Slice(0, 0, 3)
                                            .Slice(1, 3, 4)),
                            /*relative = */ false);

                    total_points_in_frame +=
                            pcd_and_bbox_.current_scan_.GetPointPositions()
                                    .GetLength();

                    // Removing `normal` attribute before passing it to
                    // the visualizer might give us some performance benifits.
                    pcd_and_bbox_.current_scan_.RemovePointAttr("normals");
                }

                if (i != 0) {
                    out_ << std::setprecision(4) << 1000.0 * i / total_time_i
                         << " FPS " << std::endl
                         << std::endl
                         << "Total Points: " << total_points_in_frame;
                }

                // To update visualizer, we go to the `main thread`,
                // bring the data on the `main thread`, ensure there is no race
                // condition with the data, and pass it to the visualizer for
                // rendering, using `AddGeometry`, or update an existing
                // pointcloud using `UpdateGeometry`, then setup camera.
                gui::Application::GetInstance().PostToMainThread(
                        this, [this, i, out_ = out_.str()]() {
                            // Note. We are getting `i` and `out_` by value
                            // instead of by reference, therefore the data is
                            // locally copied on the `main thread` itself,
                            // so, we don't need to use locks for such cases.
                            this->SetOutput(out_);

                            std::lock_guard<std::mutex> lock(
                                    pcd_and_bbox_.lock_);

                            // We render the `source` or the previous
                            // "current scan" pointcloud by using the material
                            // we set for the entire model.
                            this->widget3d_->GetScene()->ModifyGeometryMaterial(
                                    filenames_[i], pointcloud_mat_);

                            // To highlight the `current scan` we render using
                            // a different material. In next iteration we will
                            // change the material to the `model` material.
                            this->widget3d_->GetScene()->AddGeometry(
                                    filenames_[i + 1],
                                    &pcd_and_bbox_.current_scan_, mat_);

                            // Setup camera.
                            auto center = pcd_and_bbox_.bbox_.GetCenter()
                                                  .cast<float>();
                            this->widget3d_->SetupCamera(
                                    verticalFoV, pcd_and_bbox_.bbox_, center);
                        });
            }
            // --------------------------------------------------------------

            time_total.Stop();
            total_time_i += time_total.GetDuration();
        }
        // ------------------------------------------------------------------
        utility::LogInfo(" Total Average FPS: {}", 1000 * i / total_time_i);
    }

private:
    // To read parameters from the config file (.txt).
    void ReadConfigFile(const std::string& path_config) {
        std::ifstream cFile(path_config);
        std::vector<double> relative_fitness;
        std::vector<double> relative_rmse;
        std::vector<int> max_iterations;
        std::string verb, visualize;

        // ---------------------- Reading Configuration File ----------------
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
                } else if (name == "visualization") {
                    visualize = value;
                } else if (name == "start_index") {
                    std::istringstream is(value);
                    start_index_ = std::stoi(value);
                } else if (name == "end_index") {
                    std::istringstream is(value);
                    end_index_ = std::stoi(value);
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
                } else if (name == "visualization_min") {
                    std::istringstream is(value);
                    min_visualization_scalar_ = std::stod(value);
                } else if (name == "visualization_max") {
                    std::istringstream is(value);
                    max_visualization_scalar_ = std::stod(value);
                }
            }
        } else {
            std::cerr << "Couldn't open config file for reading.\n";
        }
        //-------------------------------------------------------------------

        //-------- Prining values and intilising class data members ---------

        if (end_index_ < start_index_ + 1) {
            utility::LogError(
                    " End index must be greater than the start index. Please "
                    "recheck the configuration file.");
        }

        utility::LogInfo(" Dataset path: {}", path_dataset);

        // The dataset might be too large for your memory. If that is the case,
        // one may directly read the pointcloud frame inside
        if (end_index_ - start_index_ > 500 &&
            device_.GetType() == core::Device::DeviceType::CUDA) {
            utility::LogWarning(
                    "The range of data might exceed memory. "
                    "You might want to avoid pre-fetching the data to your "
                    "device, for large datasets. "
                    "Refer the example's documentation.");
        }
        utility::LogInfo(" Range: {} to {} pointcloud files in sequence.",
                         start_index_, end_index_ - 1);
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

        icp_scale_levels_ = voxel_sizes_.size();
        if (search_radius_.size() != icp_scale_levels_ ||
            max_iterations.size() != icp_scale_levels_ ||
            relative_fitness.size() != icp_scale_levels_ ||
            relative_rmse.size() != icp_scale_levels_) {
            utility::LogError(
                    "Length of vector: voxel_sizes, search_sizes, "
                    "max_iterations, "
                    "relative_fitness, relative_rmse must be same.");
        }

        for (int i = 0; i < (int)icp_scale_levels_; i++) {
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

        if (visualize == "ON" || visualize == "on" || visualize == "On") {
            visualize_output_ = true;
        } else {
            visualize_output_ = false;
        }

        //-------------------------------------------------------------------
        std::cout << " Config file read complete. " << std::endl;
    }

    // To perform required dtype conversion, normal estimation.
    void LoadTensorPointClouds() {
        // Reading all the filenames in the given dataset path
        // with supported extensions. [.ply and .pcd].
        std::vector<std::string> all_pcd_files;

        utility::filesystem::ListFilesInDirectoryWithExtension(
                path_dataset, "pcd", all_pcd_files);
        if (all_pcd_files.size() == 0) {
            utility::filesystem::ListFilesInDirectoryWithExtension(
                    path_dataset, "ply", all_pcd_files);
        }

        if (static_cast<int>(all_pcd_files.size()) < end_index_) {
            utility::LogError(
                    "Pointcloud files in the directory {}, must be more than "
                    "the defined end index: {}, but only {} found.",
                    path_dataset, end_index_, all_pcd_files.size());
        }

        // Sorting the filenames to get the data in sequence.
        std::sort(all_pcd_files.begin(), all_pcd_files.end());

        filenames_ =
                std::vector<std::string>(all_pcd_files.begin() + start_index_,
                                         all_pcd_files.begin() + end_index_);
        utility::LogInfo(" Number of frames: {}", filenames_.size());

        int total_frames = filenames_.size();
        pointclouds_device_.reserve(total_frames);

        try {
            t::geometry::PointCloud pointcloud_local;
            // counts frames loaded, to show the progress %.
            int count = 0;
            for (auto& path : filenames_) {
                std::cout << " \rPre-fetching Data... "
                          << count * 100 / total_frames << "%"
                          << " " << std::flush;

                t::io::ReadPointCloud(path, pointcloud_local,
                                      {"auto", false, false, true});

                // registration module.
                for (std::string attr : {"positions", "colors", "normals"}) {
                    if (pointcloud_local.HasPointAttr(attr)) {
                        pointcloud_local.SetPointAttr(
                                attr,
                                pointcloud_local.GetPointAttr(attr).To(dtype_));
                    }
                }

                // `__visualization_scalar` attribute in a tensor pointcloud
                // is used by the visualizer when shader is set to
                // `unlitGradient`. `unlitGradient` assigns each point a
                // color based on this value. More about this is described in
                // the `GetPointCloudMaterial` function.
                // Here `z` value of a `x y z` point is used as
                // `__visualization_scalar`.
                pointcloud_local.SetPointAttr(
                        "__visualization_scalar",
                        pointcloud_local.GetPointPositions()
                                .Slice(0, 0, -1)
                                .Slice(1, 2, 3)
                                .To(dtype_, false));

                // Normals are required by `PointToPlane` registration method.
                // Currenly Normal Estimation is not supported by
                // Tensor Pointcloud.
                if (registration_method_ == "PointToPlane" &&
                    !pointcloud_local.HasPointNormals()) {
                    auto pointcloud_legacy = pointcloud_local.ToLegacy();
                    pointcloud_legacy.EstimateNormals(
                            open3d::geometry::KDTreeSearchParamKNN(), false);
                    core::Tensor pointcloud_normals =
                            t::geometry::PointCloud::FromLegacy(
                                    pointcloud_legacy)
                                    .GetPointNormals()
                                    .To(dtype_);
                    pointcloud_local.SetPointNormals(pointcloud_normals);
                }
                // Adding it to our vector of pointclouds.
                // We save the pointcloud downsampled by the highest
                // resolution voxel size, during data pre-fetching,
                // to same memory.
                pointclouds_device_.push_back(
                        pointcloud_local.To(device_).VoxelDownSample(
                                voxel_sizes_[icp_scale_levels_ - 1]));

                count = count + 1;
            }
            std::cout << std::endl;
        } catch (const std::bad_alloc& e) {
            utility::LogError(
                    "Memory allocation failed: {}"
                    "\nTo use large dataset, it is advised to avoid "
                    "pre-fetching data to device, and read the "
                    "pointcloud directly from inside the computation "
                    "loop. Please refer the example documentation. ",
                    e.what());
        }
    }

    rendering::MaterialRecord GetPointCloudMaterial() {
        auto pointcloud_mat = rendering::MaterialRecord();
        pointcloud_mat.shader = "unlitGradient";

        // The values of `__visualization_scalar` for each point is mapped to
        // [0, 1] such that value <= scalar_min are mapped to 0,
        // value >= scalar_max are mapped to 1, and the values in between are
        // linearly mapped. [Windowed normalisation method].
        pointcloud_mat.scalar_min = min_visualization_scalar_;
        pointcloud_mat.scalar_max = max_visualization_scalar_;

        pointcloud_mat.point_size = 0.3f;

        // This defines the color gradient scheme for rending the material.
        // The values of `__visualization_scalar` is mapped to the
        // color gradient, such that the points <= scalar_min are assigned
        // the color {0.0f, 0.25f, 0.0f, 1.0f}, and the points >= scalar_max
        // are assigned the color {1.0f, 0.0f, 0.0f, 1.0f}. The points
        // between this range are assigned colors accordingly.
        //
        // For example:
        // let's say the points {0, 1, 2, 3, 4, 5} have the following
        // `__visualization_scalar` values: {-20.5, -1.0, -0.0, 1, 3.5, 500}.
        // if we set `scalar_min` = -1, `scalar_max` = 3.
        // The windowed_normalized values will be: {0, 0, 0.25, 0.50, 1.0, 1.0}.
        // Therefore the color assigned to the points according to the following
        // scheme will be:
        // {{0.0f, 0.25f, 0.0f}, {0.0f, 0.25f, 0.0f}, {0.0f, 1.0f, 1.0f},
        //  {0.0f, 1.0f, 0.0f},  {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}.
        pointcloud_mat.gradient = std::make_shared<
                rendering::Gradient>(std::vector<rendering::Gradient::Point>{
                rendering::Gradient::Point{0.000f, {0.0f, 0.25f, 0.0f, 1.0f}},
                rendering::Gradient::Point{0.125f, {0.0f, 0.5f, 1.0f, 1.0f}},
                rendering::Gradient::Point{0.250f, {0.0f, 1.0f, 1.0f, 1.0f}},
                rendering::Gradient::Point{0.375f, {0.0f, 1.0f, 0.5f, 1.0f}},
                rendering::Gradient::Point{0.500f, {0.0f, 1.0f, 0.0f, 1.0f}},
                rendering::Gradient::Point{0.625f, {0.5f, 1.0f, 0.0f, 1.0f}},
                rendering::Gradient::Point{0.750f, {1.0f, 1.0f, 0.0f, 1.0f}},
                rendering::Gradient::Point{0.875f, {1.0f, 0.5f, 0.0f, 1.0f}},
                rendering::Gradient::Point{1.000f, {1.0f, 0.0f, 0.0f, 1.0f}}});

        return pointcloud_mat;
    }

private:
    // lock to protect `current_scan_` and  `bbox_` before modifying
    // the value, ensuring the visualizer thread doesn't read the
    // data, while we are modifying it.
    struct {
        // Mutex lock to protect data memeber current_scan_.
        std::mutex lock_;
        // Pointcloud to store the "current scan", used for visualization.
        t::geometry::PointCloud current_scan_;
        // Bounding box. It is translated by the translation component of the
        // cumulative transformation.
        geometry::AxisAlignedBoundingBox bbox_;
    } pcd_and_bbox_;

    // Checks if the GUI is closed, and if so, stop the code.
    std::atomic<bool> is_done_;

    // Material for model pointcloud and current scan pointcloud.
    open3d::visualization::rendering::MaterialRecord pointcloud_mat_;
    open3d::visualization::rendering::MaterialRecord mat_;

    // Stores the vector of pre-processed pointclouds on device.
    std::vector<open3d::t::geometry::PointCloud> pointclouds_device_;

    // Used for gradient shader color scaling.
    double min_visualization_scalar_;
    double max_visualization_scalar_;

private:
    // Path of the dataset having pointcloud frames.
    std::string path_dataset;
    // Registration estimation method type. ["PointToPoint" or "PointToPlane"].
    std::string registration_method_;
    // List of filenames of the pointcloud frames.
    std::vector<std::string> filenames_;
    // Verbosity level ["Debug" or "Info"].
    utility::VerbosityLevel verbosity_;
    // To set end index from the frame sequence.
    int end_index_;
    // To set start index from the frame sequence.
    int start_index_;
    // If `True` GUI is enabled.
    bool visualize_output_;

private:
    // MultiScaleICP parameters.
    std::vector<double> voxel_sizes_;
    std::vector<double> search_radius_;
    std::vector<ICPConvergenceCriteria> criterias_;
    std::shared_ptr<TransformationEstimation> estimation_;

    size_t icp_scale_levels_;

private:
    core::Device device_;
    core::Device host_;
    core::Dtype dtype_;
};

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TICPOdometry [device] [config_file_path]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    if (argc != 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    const std::string path_config = std::string(argv[2]);

    device_string = std::string(argv[1]);
    window_name = window_name + " [" + device_string + "]";
    widget_string = widget_string + device_string;

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, (const char**)argv);
    app.AddWindow(std::make_shared<ExampleWindow>(path_config,
                                                  core::Device(device_string)));
    app.Run();
    return 0;
}
