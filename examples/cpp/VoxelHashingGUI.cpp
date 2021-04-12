#include <atomic>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::visualization;
std::shared_ptr<geometry::LineSet> CreateCameraFrustum(
        int view_width_px,
        int view_height_px,
        const Eigen::Matrix3d& intrinsic,
        const Eigen::Matrix4d& extrinsic) {
    Eigen::Matrix4d intrinsic4;
    intrinsic4 << intrinsic(0, 0), intrinsic(0, 1), intrinsic(0, 2), 0.0,
            intrinsic(1, 0), intrinsic(1, 1), intrinsic(1, 2), 0.0,
            intrinsic(2, 0), intrinsic(2, 1), intrinsic(2, 2), 0.0, 0.0, 0.0,
            0.0, 1.0;
    Eigen::Matrix4d m = (intrinsic4 * extrinsic).inverse();
    auto lines = std::make_shared<geometry::LineSet>();

    auto mult = [](const Eigen::Matrix4d& m,
                   const Eigen::Vector3d& v) -> Eigen::Vector3d {
        Eigen::Vector4d v4(v.x(), v.y(), v.z(), 1.0);
        auto result = m * v4;
        return Eigen::Vector3d{result.x() / result.w(), result.y() / result.w(),
                               result.z() / result.w()};
    };
    double dist = 0.2;
    double w = double(view_width_px);
    double h = double(view_height_px);
    // Matrix m transforms from homogenous pixel coordinates to world
    // coordinates so x and y need to be multiplied by z. In the case of the
    // first point, the eye point, z=0, so x and y will be zero, too regardless
    // of their initial values as the center.
    lines->points_.push_back(mult(m, Eigen::Vector3d{0.0, 0.0, 0.0}));
    lines->points_.push_back(mult(m, Eigen::Vector3d{0.0, 0.0, dist}));
    lines->points_.push_back(mult(m, Eigen::Vector3d{w * dist, 0.0, dist}));
    lines->points_.push_back(
            mult(m, Eigen::Vector3d{w * dist, h * dist, dist}));
    lines->points_.push_back(mult(m, Eigen::Vector3d{0.0, h * dist, dist}));

    lines->lines_.push_back({0, 1});
    lines->lines_.push_back({0, 2});
    lines->lines_.push_back({0, 3});
    lines->lines_.push_back({0, 4});
    lines->lines_.push_back({1, 2});
    lines->lines_.push_back({2, 3});
    lines->lines_.push_back({3, 4});
    lines->lines_.push_back({4, 1});
    lines->PaintUniformColor({0.0f, 0.0f, 1.0f});

    return lines;
}
//------------------------------------------------------------------------------
class PropertyPanel : public gui::VGrid {
public:
    PropertyPanel(int spacing) : gui::VGrid(2, spacing) {}

    void AddBool(const std::string& name,
                 std::atomic<bool>* bool_addr,
                 bool default_val) {
        auto cb = std::make_shared<gui::Checkbox>(name.c_str());
        cb->SetChecked(default_val);
        *bool_addr = default_val;
        cb->SetOnChecked([bool_addr, this](bool is_checked) {
            *bool_addr = is_checked;
            this->NotifyChanged();
        });
        AddChild(std::make_shared<gui::Label>(""));  // checkbox has name in it
        AddChild(cb);
    }

    void AddNumber(const std::string& name,
                   std::atomic<double>* num_addr,
                   double default_val,
                   double min_val,
                   double max_val) {
        auto s = std::make_shared<gui::Slider>(gui::Slider::DOUBLE);
        s->SetLimits(min_val, max_val);
        s->SetValue(default_val);
        *num_addr = default_val;
        s->SetOnValueChanged([num_addr, this](double new_val) {
            *num_addr = new_val;
            this->NotifyChanged();
        });
        AddChild(std::make_shared<gui::Label>(name.c_str()));
        AddChild(s);
    }

    void AddValues(const std::string& name,
                   std::atomic<int>* idx_addr,
                   int default_idx,
                   std::vector<std::string> values) {
        auto combo = std::make_shared<gui::Combobox>();
        for (auto& v : values) {
            combo->AddItem(v.c_str());
        }
        combo->SetSelectedIndex(default_idx);
        *idx_addr = default_idx;
        combo->SetOnValueChanged(
                [idx_addr, this](const char* new_value, int new_idx) {
                    *idx_addr = new_idx;
                    this->NotifyChanged();
                });
        AddChild(std::make_shared<gui::Label>(name.c_str()));
        AddChild(combo);
    }

    void SetOnChanged(std::function<void()> f) { on_changed_ = f; }

private:
    std::function<void()> on_changed_;

    void NotifyChanged() {
        if (on_changed_) {
            on_changed_();
        }
    }
};

//------------------------------------------------------------------------------
class ReconstructionWindow : public gui::Window {
    using Super = gui::Window;

public:
    ReconstructionWindow() : gui::Window("Open3D - Reconstruction", 1600, 900) {
        auto& theme = GetTheme();
        int em = theme.font_size;
        int spacing = int(std::round(0.5f * float(em)));
        gui::Margins margins(int(std::round(0.5f * float(em))));
        panel_ = std::make_shared<gui::Vert>(spacing, margins);
        widget3d_ = std::make_shared<gui::SceneWidget>();
        output_panel_ = std::make_shared<gui::Vert>(spacing, margins);
        AddChild(panel_);
        AddChild(widget3d_);
        AddChild(output_panel_);

        props_ = std::make_shared<PropertyPanel>(
                int(std::round(0.25f * float(em))));
        panel_->AddChild(std::make_shared<gui::Label>("Control"));
        panel_->AddChild(props_);
        panel_->AddStretch();

        panel_->AddChild(std::make_shared<gui::Label>("Input image(s)"));
        rgb_image_ = std::make_shared<gui::ImageWidget>();
        depth_image_ = std::make_shared<gui::ImageWidget>();
        panel_->AddChild(rgb_image_);
        panel_->AddChild(depth_image_);

        output_ = std::make_shared<gui::Label>("");
        raycast_color_image_ = std::make_shared<gui::ImageWidget>();
        raycast_depth_image_ = std::make_shared<gui::ImageWidget>();
        output_panel_->AddChild(std::make_shared<gui::Label>("Tracking"));
        output_panel_->AddChild(output_);
        output_panel_->AddStretch();

        output_panel_->AddChild(
                std::make_shared<gui::Label>("Ray casted image(s)"));
        output_panel_->AddChild(raycast_color_image_);
        output_panel_->AddChild(raycast_depth_image_);

        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~ReconstructionWindow() {}

    void Layout(const gui::Theme& theme) override {
        Super::Layout(theme);

        int em = theme.font_size;
        int panel_width = 20 * em;
        // The usable part of the window may not be the full size if there
        // is a menu.
        auto content_rect = GetContentRect();
        panel_->SetFrame(gui::Rect(content_rect.x, content_rect.y, panel_width,
                                   content_rect.height));
        output_panel_->SetFrame(gui::Rect(content_rect.GetRight() - panel_width,
                                          content_rect.y, panel_width,
                                          content_rect.height));
        int x = panel_->GetFrame().GetRight();
        widget3d_->SetFrame(gui::Rect(x, content_rect.y,
                                      output_panel_->GetFrame().x - x,
                                      content_rect.height));
    }

protected:
    std::shared_ptr<gui::Vert> panel_;
    std::shared_ptr<gui::Vert> output_panel_;
    std::shared_ptr<gui::Label> output_;
    std::shared_ptr<gui::SceneWidget> widget3d_;
    std::shared_ptr<PropertyPanel> props_;
    std::shared_ptr<gui::ImageWidget> rgb_image_;
    std::shared_ptr<gui::ImageWidget> depth_image_;

    std::shared_ptr<gui::ImageWidget> raycast_color_image_;
    std::shared_ptr<gui::ImageWidget> raycast_depth_image_;

    void SetOutput(const std::string& output) {
        output_->SetText(output.c_str());
    }
};

//------------------------------------------------------------------------------
class ExampleWindow : public ReconstructionWindow {
public:
    ExampleWindow(const std::string& dataset_path) {
        dataset_path_ = dataset_path;
        props_->AddNumber("Surface update", &prop_values_.surface_interval,
                          50.0, 1.0, 100.);

        props_->AddNumber("Depth scale", &prop_values_.depth_scale, 1000.0, 1.0,
                          1500.0);
        props_->AddNumber("Depth max", &prop_values_.depth_max, 3.0, 0.0, 5.0);
        props_->AddNumber("Depth diff", &prop_values_.depth_diff, 0.07, 0.03,
                          0.5);
        props_->AddBool("Color points", &prop_values_.color_points, true);

        is_done_ = false;
        SetOnClose([this]() {
            is_done_ = true;
            return true;  // false would cancel the close
        });
        update_thread_ = std::thread([this]() { this->UpdateMain(); });
    }

    ~ExampleWindow() { update_thread_.join(); }

private:
    std::string dataset_path_;

    struct {
        std::atomic<double> surface_interval;
        std::atomic<double> depth_scale;
        std::atomic<double> depth_max;
        std::atomic<double> depth_diff;
        std::atomic<bool> color_points;
    } prop_values_;
    std::atomic<bool> is_done_;
    std::thread update_thread_;

    void UpdateMain() {
        // Note that we cannot update the GUI on this thread, we must post to
        // the main thread!
        const std::string rgb_dir = dataset_path_ + "/color";
        const std::string depth_dir = dataset_path_ + "/depth";
        std::vector<std::string> rgb_files;
        std::vector<std::string> depth_files;
        utility::filesystem::ListFilesInDirectoryWithExtension(rgb_dir, "png",
                                                               rgb_files);
        std::sort(rgb_files.begin(), rgb_files.end());
        utility::filesystem::ListFilesInDirectoryWithExtension(depth_dir, "png",
                                                               depth_files);
        std::sort(depth_files.begin(), depth_files.end());

        // Only set at initialization
        float voxel_size = 3.0 / 512;
        int block_resolution = 16;
        int block_count = 80000;
        float depth_scale = prop_values_.depth_scale;

        // Can be changed at runtime
        float sdf_trunc = 0.04f;
        // float depth_max = prop_values_.depth_max;
        // float depth_diff = prop_values_.depth_diff;

        core::Tensor T_frame_to_model = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0"));
        core::Tensor intrinsic_t = core::Tensor::Init<float>(
                {{525.0, 0, 319.5}, {0, 525.0, 239.5}, {0, 0, 1}});
        core::Device device("CUDA:0");

        t::geometry::Image ref_depth =
                *t::io::CreateImageFromFile(depth_files[0]);
        t::pipelines::voxelhashing::Frame input_frame(
                ref_depth.GetRows(), ref_depth.GetCols(), intrinsic_t, device);
        t::pipelines::voxelhashing::Frame raycast_frame(
                ref_depth.GetRows(), ref_depth.GetCols(), intrinsic_t, device);
        t::pipelines::voxelhashing::Model model(voxel_size, sdf_trunc,
                                                block_resolution, block_count,
                                                T_frame_to_model, device);

        bool is_scene_updated = false;
        bool is_initialized = false;
        size_t idx = 0;

        // Odom
        auto traj = std::make_shared<geometry::LineSet>();

        std::shared_ptr<open3d::geometry::PointCloud> pcd;
        while (!is_done_) {
            // Input
            t::geometry::Image input_depth =
                    *t::io::CreateImageFromFile(depth_files[idx]);
            t::geometry::Image input_color =
                    *t::io::CreateImageFromFile(rgb_files[idx]);
            input_frame.SetDataFromImage("depth", input_depth);
            input_frame.SetDataFromImage("color", input_color);

            if (idx > 0) {
                utility::LogInfo("Frame-to-model for the frame {}", idx);

                core::Tensor delta_frame_to_model = model.TrackFrameToModel(
                        input_frame, raycast_frame, depth_scale,
                        prop_values_.depth_max, prop_values_.depth_diff);
                T_frame_to_model =
                        T_frame_to_model.Matmul(delta_frame_to_model);
            }

            // Integrate
            model.UpdateFramePose(idx, T_frame_to_model);
            model.Integrate(input_frame, depth_scale, prop_values_.depth_max);
            model.SynthesizeModelFrame(raycast_frame);

            idx++;
            is_done_ = (idx >= depth_files.size());

            auto K_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(
                    intrinsic_t);
            auto T_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(
                    T_frame_to_model);
            std::stringstream out;
            out << "Frame " << idx << "\n";
            out << T_eigen;

            traj->points_.push_back(T_eigen.block<3, 1>(0, 3));
            if (traj->points_.size() > 1) {
                int n = traj->points_.size();
                traj->lines_.push_back({n - 1, n - 2});
                traj->colors_.push_back(Eigen::Vector3d(0, 0, 1));
            }

            auto frustum =
                    CreateCameraFrustum(640, 480, K_eigen, T_eigen.inverse());

            // TODO: update support for timages
            // image conversion
            auto color = std::make_shared<open3d::geometry::Image>(
                    input_frame.GetDataAsImage("color").ToLegacyImage());
            auto depth = std::make_shared<open3d::geometry::Image>(
                    input_frame.GetDataAsImage("depth")
                            .To(core::Dtype::Float32, false, 1.0f)
                            .ToLegacyImage());
            auto depth8 = ConvertDepthToNormalizedGrey8(*depth);

            auto raycast_color = std::make_shared<open3d::geometry::Image>(
                    raycast_frame.GetDataAsImage("color")
                            .To(core::Dtype::UInt8, false, 255.0f)
                            .ToLegacyImage());
            auto raycast_depth = std::make_shared<open3d::geometry::Image>(
                    raycast_frame.GetDataAsImage("depth").ToLegacyImage());
            auto raycast_depth8 = ConvertDepthToNormalizedGrey8(*raycast_depth);

            // Extract surface on demand
            if (idx % static_cast<int>(prop_values_.surface_interval) == 0) {
                gui::Application::GetInstance().RunInThread([&]() {
                    pcd = std::make_shared<open3d::geometry::PointCloud>(
                            model.ExtractPointCloud().ToLegacyPointCloud());
                    is_scene_updated = true;
                });
            }

            gui::Application::GetInstance().PostToMainThread(
                    this, [this, color, depth8, raycast_color, raycast_depth8,
                           pcd, traj, frustum, &is_initialized,
                           &is_scene_updated, out = out.str()]() {
                        this->SetOutput(out);
                        this->rgb_image_->UpdateImage(color);
                        this->depth_image_->UpdateImage(depth8);

                        this->raycast_color_image_->UpdateImage(raycast_color);
                        this->raycast_depth_image_->UpdateImage(raycast_depth8);

                        this->widget3d_->GetScene()->RemoveGeometry("frustum");
                        auto mat = rendering::Material();
                        mat.shader = "unlitLine";
                        mat.line_width = 5.0f;
                        this->widget3d_->GetScene()->AddGeometry(
                                "frustum", frustum.get(), mat);

                        if (traj->points_.size() > 1) {
                            this->widget3d_->GetScene()->RemoveGeometry(
                                    "trajectory");
                            auto mat = rendering::Material();
                            mat.shader = "unlitLine";
                            mat.line_width = 2.0f;
                            this->widget3d_->GetScene()->AddGeometry(
                                    "trajectory", traj.get(), mat);
                        }

                        if (is_scene_updated) {
                            this->widget3d_->GetScene()->RemoveGeometry(
                                    "points");
                            auto mat = rendering::Material();
                            mat.shader = "defaultUnlit";
                            this->widget3d_->GetScene()->AddGeometry(
                                    "points", pcd.get(), mat);
                            is_scene_updated = false;
                        }
                        if (!is_initialized) {
                            auto bbox = this->widget3d_->GetScene()
                                                ->GetBoundingBox();
                            auto center = bbox.GetCenter().cast<float>();
                            this->widget3d_->SetupCamera(60, bbox, center);
                            this->widget3d_->LookAt(
                                    center, center - Eigen::Vector3f{0, 1, 3},
                                    {0.0f, -1.0f, 0.0f});
                            is_initialized = true;
                        }
                    });
        }
    }

    // The renderer can only use 8-bit channels currently. Also, we need to
    // convert to RGB because the renderer will display one-channel images
    // in red. Normalize because otherwise it can be hard to see the image.
    std::shared_ptr<geometry::Image> ConvertDepthToNormalizedGrey8(
            const geometry::Image& depth) {
        float* data = depth.PointerAs<float>();
        float max_val = 0.0f;
        int n_pixels = depth.width_ * depth.height_;
        for (int i = 0; i < n_pixels; ++i) {
            max_val = std::max(max_val, data[i]);
        }

        auto img888 = std::make_shared<geometry::Image>();
        img888->width_ = depth.width_;
        img888->height_ = depth.height_;
        img888->num_of_channels_ = 3;
        img888->bytes_per_channel_ = 1;
        img888->data_.reserve(img888->width_ * img888->height_ *
                              img888->num_of_channels_ *
                              img888->bytes_per_channel_);
        for (int i = 0; i < n_pixels; ++i) {
            float val = data[i] / max_val * 255.0f;
            uint8_t px = uint8_t(val);
            img888->data_.push_back(px);
            img888->data_.push_back(px);
            img888->data_.push_back(px);
        }

        return img888;
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    if (argc < 2) {
        utility::LogError("Expected dataset path as input");
    }
    std::string dataset_path = argv[1];
    if (!utility::filesystem::DirectoryExists(dataset_path)) {
        utility::LogError("Expected color/ and depth/ directories in {}",
                          dataset_path);
    }

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, argv);
    app.AddWindow(std::make_shared<ExampleWindow>(dataset_path));
    app.Run();
}
