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

const Eigen::Vector3f CENTER_OFFSET(-10.0f, 0.0f, 100.0f);
const std::string CURRENT_CLOUD = "current_scan";

std::vector<float> initial_transform_flat = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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

    // void Layout(const gui::Theme& theme) override {
    //     Super::Layout(theme);

    //     int em = theme.font_size;
    //     int panel_width = 20 * em;
    //     // The usable part of the window may not be the full size if there
    //     // is a menu.
    //     auto content_rect = GetContentRect();

    //     int x = panel_->GetFrame().GetRight();
    //     widget3d_->SetFrame(gui::Rect(x, content_rect.y,
    //                                   output_panel_->GetFrame().x - x,
    //                                   content_rect.height));
    // }

protected:
    std::shared_ptr<gui::SceneWidget> widget3d_;
};

//------------------------------------------------------------------------------
class ExampleWindow : public ReconstructionWindow {
public:
    ExampleWindow(const std::string& path_config, const core::Device& device)
        : result_(device),
          device_(device),
          host_(core::Device("CPU:0")),
          dtype_(core::Dtype::Float32) {
        ReadConfigFile(path_config);
        pointclouds_host_ = LoadTensorPointClouds();

        transformation_ =
                core::Tensor(initial_transform_flat, {4, 4}, dtype_, device_);

        // Warm Up.
        std::vector<ICPConvergenceCriteria> warm_up_criteria = {
                ICPConvergenceCriteria(0.01, 0.01, 1)};
        result_ = RegistrationMultiScaleICP(
                pointclouds_host_[0].To(device_),
                pointclouds_host_[1].To(device_), voxel_sizes_, criterias_,
                search_radius_, transformation_, *estimation_);

        std::cout << " [Debug] Warm up transformation: "
                  << result_.transformation_.ToString() << std::endl;
        is_done_ = false;
        visualize_output_ = true;

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
        t::geometry::PointCloud pcd;

        core::Tensor initial_transform = core::Tensor::Eye(4, dtype_, device_);
        core::Tensor cumulative_transform = initial_transform.Clone();

        geometry::AxisAlignedBoundingBox bounds;
        auto mat = rendering::Material();
        mat.shader = "defaultUnlit";

        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            pcd = pointclouds_host_[0].Clone();
        }

        if (visualize_output_) {
            gui::Application::GetInstance().PostToMainThread(this, [this,
                                                                    bounds, mat,
                                                                    pcd]() {
                std::lock_guard<std::mutex> lock(cloud_lock_);

                this->widget3d_->GetScene()->GetScene()->AddGeometry(
                        filenames_[0], pcd, mat);
                // pcd->->PaintUniformColor({1.0, 0.0, 0.0});
                // this->widget3d_->GetScene()->GetScene()->AddGeometry(CURRENT_CLOUD,
                // legacy_output_, &mat);
                auto bbox = this->widget3d_->GetScene()->GetBoundingBox();
                auto center = bbox.GetCenter().cast<float>();

                this->widget3d_->SetupCamera(60, bbox, center);
                this->widget3d_->GetScene()->SetBackground({0, 0, 0, 1});
            });
        }

        for (int i = 0; i < end_range_ - 1; i++) {
            utility::Timer time_icp_odom_loop;
            time_icp_odom_loop.Start();
            auto source = pointclouds_host_[i].To(device_);
            auto target = pointclouds_host_[i + 1].To(device_);

            auto result = RegistrationMultiScaleICP(
                    source, target, voxel_sizes_, criterias_, search_radius_,
                    initial_transform, *estimation_);

            cumulative_transform = cumulative_transform.Matmul(
                    result.transformation_.Inverse());

            time_icp_odom_loop.Stop();
            double total_processing_time = time_icp_odom_loop.GetDuration();
            utility::LogDebug(" Registraion took: {}", total_processing_time);
            utility::LogDebug(" Cumulative Transformation: \n{}\n",
                              cumulative_transform.ToString());

            {
                std::lock_guard<std::mutex> lock(cloud_lock_);
                pcd = target.Transform(cumulative_transform)
                              .To(core::Device("CPU:0"), true);
            }

            gui::Application::GetInstance().PostToMainThread(this, [this, pcd,
                                                                    &mat, i]() {
                std::lock_guard<std::mutex> lock(cloud_lock_);
                utility::Timer timer;
                timer.Start();

                this->widget3d_->GetScene()->GetScene()->AddGeometry(
                        filenames_[i + 1], pcd, mat);

                auto bbox = this->widget3d_->GetScene()->GetBoundingBox();
                auto center = bbox.GetCenter().cast<float>();

                this->widget3d_->SetupCamera(60, bbox, center);

                timer.Stop();
                utility::LogInfo("Update geometry takes {}",
                                 timer.GetDuration());
                // is_scene_updated = false;
            });
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

        std::vector<t::geometry::PointCloud> pointclouds_host(
                filenames_.size());

        try {
            int i = 0;
            t::geometry::PointCloud pointcloud_local;
            for (auto& path : filenames_) {
                t::io::ReadPointCloud(path, pointcloud_local,
                                      {"auto", false, false, true});

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
                                    .To(dtype_);
                    pointcloud_local.SetPointNormals(pointcloud_normals);
                }
                // Adding it to our vector of pointclouds.
                pointclouds_host[i++] = pointcloud_local.Clone();
            }
        } catch (...) {
            utility::LogError(
                    " Failed to read pointcloud in sequence. Ensure pointcloud "
                    "files are present in the given dataset path in continuous "
                    "sequence from 0 to {}. Also, in case of large range, the "
                    "system might be going out-of-memory. ",
                    end_range_);
        }
        return pointclouds_host;
    }

private:
    std::mutex cloud_lock_;

    std::atomic<bool> is_done_;
    // std::shared_ptr<visualizer::O3DVisualizer> main_vis_;

    std::vector<open3d::t::geometry::PointCloud> pointclouds_host_;

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

    double gt_tx_;
    double gt_ty_;
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
