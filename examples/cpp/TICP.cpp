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

using namespace open3d;
using namespace open3d::visualization;
using namespace open3d::t::pipelines::registration;

float verticalFoV = 25;

const Eigen::Vector3f CENTER_OFFSET(-10.0f, 0.0f, 30.0f);

const std::string SRC_CLOUD = "source_pointcloud";
const std::string DST_CLOUD = "target_pointcloud";

//------------------------------------------------------------------------------
class RegistrationWindow : public gui::Window {
    using Super = gui::Window;

public:
    RegistrationWindow(const core::Device& device)
        : gui::Window("Open3D - Registration", 1280, 800),
          device_(device),
          host_(core::Device("CPU:0")),
          dtype_(core::Dtype::Float32) {
        data::DemoICPPointClouds demo_icp_data;

        // Read Point-Clouds.
        // t::io::ReadPointCloud copies the pointcloud to CPU.
        t::io::ReadPointCloud(demo_icp_data.GetPaths()[0], source_,
                              {"auto", false, false, true});
        t::io::ReadPointCloud(demo_icp_data.GetPaths()[1], target_,
                              {"auto", false, false, true});
        // Device transfer.
        source_ = source_.To(device_, false);
        target_ = target_.To(device_, false);
        // For Colored-ICP `colors` attribute must be of the same dtype as
        // `positions` and `normals` attribute.
        source_.SetPointColors(
                source_.GetPointColors()
                        .To(core::Float32)
                        .Div(static_cast<double>(
                                std::numeric_limits<uint8_t>::max())));
        target_.SetPointColors(
                target_.GetPointColors()
                        .To(core::Float32)
                        .Div(static_cast<double>(
                                std::numeric_limits<uint8_t>::max())));

        // Load other parameters.
        estimation_ = std::make_shared<TransformationEstimationPointToPlane>();
        utility::LogInfo(" Registrtion method: PointToPlane");

        voxel_sizes_ = {0.05, 0.025, 0.0125};
        utility::LogInfo("Voxel Sizes: ");
        for (auto voxel_size : voxel_sizes_) std::cout << voxel_size << " ";

        max_correspondence_distances_ = {0.07, 0.07, 0.07};
        utility::LogInfo("Search Radius Sizes: ");
        for (auto search_radii : max_correspondence_distances_)
            std::cout << search_radii << " ";

        transformation_ =
                core::Tensor::Init<double>({{0.862, 0.011, -0.507, 0.5},
                                            {-0.139, 0.967, -0.215, 0.7},
                                            {0.487, 0.255, 0.835, -1.4},
                                            {0.0, 0.0, 0.0, 1.0}},
                                           host_);
        utility::LogInfo("Initial Transformation Guess: \n {} \n",
                         transformation_.ToString());

        verbosity_ = utility::VerbosityLevel::Debug;
        criterias_.emplace_back(0.0000001, 0.0000001, 50);
        criterias_.emplace_back(0.0000001, 0.0000001, 30);
        criterias_.emplace_back(0.0000001, 0.0000001, 14);

        is_done_ = false;

        // --------------------- VISUALIZER ---------------------
        gui::Application::GetInstance().Initialize();

        src_cloud_mat_ = rendering::MaterialRecord();
        src_cloud_mat_.shader = "defaultUnlit";
        src_cloud_mat_.base_color = Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);

        tar_cloud_mat_ = rendering::MaterialRecord();
        tar_cloud_mat_.shader = "defaultUnlit";
        tar_cloud_mat_.base_color = Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f);
        // ------------------------------------------------------

        SetOnClose([this]() {
            is_done_ = true;
            return true;  // false would cancel the close
        });
        update_thread_ = std::thread([this]() { this->UpdateMain(); });

        gui::Margins margins(int(
                std::round(0.5f * static_cast<float>(GetTheme().font_size))));
        widget3d_ = std::make_shared<gui::SceneWidget>();
        AddChild(widget3d_);

        // ----------------- VISUALIZER -----------------
        // Initialize visualizer.
        {
            // lock to protect `source_` and `target_`
            // before modifying the value, ensuring the
            // visualizer thread doesn't read the data,
            // while we are modifying it.
            std::lock_guard<std::mutex> lock(pcd_.lock_);

            // Copying the pointcloud on CPU, as required by
            // the visualizer.
            pcd_.source_ = source_.To(core::Device("CPU:0"));
            pcd_.target_ = target_.To(core::Device("CPU:0"));
        }

        gui::Application::GetInstance().PostToMainThread(this, [this]() {
            // lock to protect `pcd_.source_` and
            // `pcd.target_` before passing it to
            // the visualizer, ensuring we don't
            // modify the value, when visualizer is
            // reading it.
            std::lock_guard<std::mutex> lock(pcd_.lock_);

            // Setting the background.
            this->widget3d_->GetScene()->SetBackground({0, 0, 0, 1});

            // Adding the target pointcloud.
            this->widget3d_->GetScene()->AddGeometry(DST_CLOUD, &pcd_.target_,
                                                     tar_cloud_mat_);
            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    SRC_CLOUD, pcd_.source_, src_cloud_mat_);

            // Getting bounding box and center to
            // setup camera view.
            auto bbox = this->widget3d_->GetScene()->GetBoundingBox();
            Eigen::Vector3f center = bbox.GetCenter().cast<float>();
            this->widget3d_->SetupCamera(18, bbox, center);
            this->widget3d_->LookAt(center, center - Eigen::Vector3f{-10, 5, 8},
                                    {0.0f, -1.0f, 0.0f});
        });
        // -----------------------------------------------------
        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~RegistrationWindow() { update_thread_.join(); }

    void Layout(const gui::LayoutContext& context) override {
        auto content_rect = GetContentRect();
        widget3d_->SetFrame(gui::Rect(content_rect.x, content_rect.y,
                                      content_rect.GetRight(),
                                      content_rect.height));
        Super::Layout(context);
    }

protected:
    void UpdateMain() {
        auto callback_after_iteration = [&](const std::unordered_map<
                                                std::string, core::Tensor>&
                                                    updated_information) {
            core::Tensor transformation;

            // Delays each iteration to allow clear visualization of
            // each iteration.
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            {
                std::lock_guard<std::mutex> lock(pcd_.lock_);
                pcd_.source_ = source_.To(host_, true)
                                       .Transform(updated_information.at(
                                               "transformation"));
            }

            // To update visualizer, we go to the `main thread`,
            // bring the data on the `main thread`, ensure there is no race
            // condition with the data, and pass it to the visualizer for
            // rendering, using `AddGeometry`, or update an existing
            // pointcloud using `UpdateGeometry`, then setup camera.
            gui::Application::GetInstance().PostToMainThread(this, [this]() {
                // Locking to protect: pcd_.source_,
                std::lock_guard<std::mutex> lock(pcd_.lock_);
                this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                        SRC_CLOUD, pcd_.source_,
                        rendering::Scene::kUpdatePointsFlag |
                                rendering::Scene::kUpdateColorsFlag);
            });
        };

        result_ = t::pipelines::registration::MultiScaleICP(
                source_.To(device_), target_.To(device_), voxel_sizes_,
                criterias_, max_correspondence_distances_, transformation_,
                *estimation_, callback_after_iteration);
    }

private:
    std::shared_ptr<gui::SceneWidget> widget3d_;
    // General logic
    std::atomic<bool> is_done_;
    std::thread update_thread_;

    open3d::visualization::rendering::MaterialRecord src_cloud_mat_;
    open3d::visualization::rendering::MaterialRecord tar_cloud_mat_;
    // For Visualization.
    // The members of this structure can be protected by the mutex lock,
    // to avoid the case, when we are trying to modify the values,
    // while visualizer is trying to access it.
    struct {
        std::mutex lock_;
        t::geometry::PointCloud source_;
        t::geometry::PointCloud target_;
    } pcd_;

    // Input Parameters.
    utility::VerbosityLevel verbosity_;
    core::Device device_;
    core::Device host_;
    core::Dtype dtype_;
    t::geometry::PointCloud source_;
    t::geometry::PointCloud target_;
    std::vector<double> voxel_sizes_;
    std::vector<double> max_correspondence_distances_;
    std::vector<ICPConvergenceCriteria> criterias_;
    std::shared_ptr<TransformationEstimation> estimation_;

    // Output
    core::Tensor transformation_;
    t::pipelines::registration::RegistrationResult result_;
};

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TICP [device]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, (const char**)argv);
    app.AddWindow(std::make_shared<RegistrationWindow>(core::Device(argv[1])));
    app.Run();
    return 0;
}
