#include <atomic>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::visualization;

// Tanglo colorscheme (see https://en.wikipedia.org/wiki/Tango_Desktop_Project)
static const Eigen::Vector3d kTangoOrange(0.961, 0.475, 0.000);
static const Eigen::Vector3d kTangoSkyBlueDark(0.125, 0.290, 0.529);

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

    lines->PaintUniformColor(kTangoOrange);

    return lines;
}

//------------------------------------------------------------------------------
class PropertyPanel : public gui::VGrid {
    using Super = gui::VGrid;

public:
    PropertyPanel(int spacing, int left_margin)
        : gui::VGrid(2, spacing, gui::Margins(left_margin, 0, 0, 0)) {
        default_label_color_ =
                std::make_shared<gui::Label>("temp")->GetTextColor();
    }

    void AddBool(const std::string& name,
                 std::atomic<bool>* bool_addr,
                 bool default_val) {
        auto cb = std::make_shared<gui::Checkbox>("");
        cb->SetChecked(default_val);
        *bool_addr = default_val;
        cb->SetOnChecked([bool_addr, this](bool is_checked) {
            *bool_addr = is_checked;
            this->NotifyChanged();
        });
        AddChild(std::make_shared<gui::Label>(name.c_str()));
        AddChild(cb);
    }

    void AddFloatSlider(const std::string& name,
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

    void AddIntSlider(const std::string& name,
                      std::atomic<int>* num_addr,
                      int default_val,
                      int min_val,
                      int max_val) {
        auto s = std::make_shared<gui::Slider>(gui::Slider::INT);
        s->SetLimits(min_val, max_val);
        s->SetValue(default_val);
        *num_addr = default_val;
        s->SetOnValueChanged([num_addr, this](int new_val) {
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

    void SetEnabled(bool enable) override {
        Super::SetEnabled(enable);
        for (auto child : GetChildren()) {
            child->SetEnabled(enable);
            auto label = std::dynamic_pointer_cast<gui::Label>(child);
            if (label) {
                if (enable) {
                    label->SetTextColor(default_label_color_);
                } else {
                    label->SetTextColor(gui::Color(0.5f, 0.5f, 0.5f, 1.0f));
                }
            }
        }
    }

    void SetOnChanged(std::function<void()> f) { on_changed_ = f; }

private:
    gui::Color default_label_color_;
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
    ReconstructionWindow()
        : gui::Window("Open3D - Reconstruction", 1600, 900),
          is_running_(false) {
        auto& theme = GetTheme();
        int em = theme.font_size;
        int spacing = int(std::round(0.25f * float(em)));
        int left_margin = em;
        int vspacing = int(std::round(0.5f * float(em)));
        gui::Margins margins(int(std::round(0.5f * float(em))));
        panel_ = std::make_shared<gui::Vert>(spacing, margins);
        widget3d_ = std::make_shared<gui::SceneWidget>();
        AddChild(panel_);
        AddChild(widget3d_);

        fixed_props_ = std::make_shared<PropertyPanel>(spacing, left_margin);
        adjustable_props_ =
                std::make_shared<PropertyPanel>(spacing, left_margin);

        panel_->AddChild(std::make_shared<gui::Label>("Starting settings"));
        panel_->AddChild(fixed_props_);

        panel_->AddFixed(vspacing);
        panel_->AddChild(
                std::make_shared<gui::Label>("Reconstruction settings"));
        panel_->AddChild(adjustable_props_);
        panel_->SetEnabled(false);

        auto b = std::make_shared<gui::ToggleSwitch>("Resume/Pause");
        b->SetOnClicked([b, this](bool is_on) {
            this->is_running_ = !(this->is_running_);
            this->adjustable_props_->SetEnabled(true);
        });
        panel_->AddChild(b);
        panel_->AddFixed(vspacing);

        panel_->AddStretch();

        gui::Margins tab_margins(0, int(std::round(0.5f * float(em))), 0, 0);
        auto tabs = std::make_shared<gui::TabControl>();
        panel_->AddChild(tabs);
        auto tab1 = std::make_shared<gui::Vert>(0, tab_margins);
        input_color_image_ = std::make_shared<gui::ImageWidget>();
        input_depth_image_ = std::make_shared<gui::ImageWidget>();
        tab1->AddChild(input_color_image_);
        tab1->AddFixed(vspacing);
        tab1->AddChild(input_depth_image_);
        tabs->AddTab("Input images", tab1);

        auto tab2 = std::make_shared<gui::Vert>(0, tab_margins);
        output_ = std::make_shared<gui::Label>("");
        raycast_color_image_ = std::make_shared<gui::ImageWidget>();
        raycast_depth_image_ = std::make_shared<gui::ImageWidget>();

        tab2->AddChild(raycast_color_image_);
        tab2->AddFixed(vspacing);
        tab2->AddChild(raycast_depth_image_);
        tabs->AddTab("Raycast images", tab2);

        auto tab3 = std::make_shared<gui::Vert>(0, tab_margins);
        tab3->AddChild(output_);
        tabs->AddTab("Tracking", tab3);

        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~ReconstructionWindow() {}

    void Layout(const gui::LayoutContext& context) override {
        int em = context.theme.font_size;
        int panel_width = 20 * em;
        // The usable part of the window may not be the full size if there
        // is a menu.
        auto content_rect = GetContentRect();
        panel_->SetFrame(gui::Rect(content_rect.x, content_rect.y, panel_width,
                                   content_rect.height));
        int x = panel_->GetFrame().GetRight();
        widget3d_->SetFrame(gui::Rect(x, content_rect.y,
                                      content_rect.GetRight() - x,
                                      content_rect.height));

        // Now that all the children are sized correctly, we can super to
        // layout all their children.
        Super::Layout(context);
    }

protected:
    std::atomic<bool> is_running_;

    std::shared_ptr<gui::Vert> panel_;
    std::shared_ptr<gui::Label> output_;
    std::shared_ptr<gui::SceneWidget> widget3d_;
    std::shared_ptr<PropertyPanel> fixed_props_;
    std::shared_ptr<PropertyPanel> adjustable_props_;

    std::shared_ptr<gui::ImageWidget> input_color_image_;
    std::shared_ptr<gui::ImageWidget> input_depth_image_;
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

        // Fixed
        fixed_props_->AddIntSlider("Depth scale", &prop_values_.depth_scale,
                                   1000, 1, 5000);
        fixed_props_->AddIntSlider("Estimated points",
                                   &prop_values_.pointcloud_size, 6000000,
                                   500000, 8000000);

        // Adjustable
        adjustable_props_->AddIntSlider(
                "Update interval", &prop_values_.surface_interval, 50, 1, 100);

        adjustable_props_->AddFloatSlider("Depth max", &prop_values_.depth_max,
                                          3.0, 0.0, 5.0);
        adjustable_props_->AddFloatSlider(
                "Depth diff", &prop_values_.depth_diff, 0.07, 0.03, 0.5);
        adjustable_props_->AddBool("Raycast color", &prop_values_.raycast_color,
                                   true);

        // Set adjustable disabled to make the Start button clearer
        adjustable_props_->SetEnabled(false);

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
        std::atomic<int> surface_interval;
        std::atomic<int> pointcloud_size;
        std::atomic<int> depth_scale;
        std::atomic<double> depth_max;
        std::atomic<double> depth_diff;
        std::atomic<bool> raycast_color;
    } prop_values_;
    std::atomic<bool> is_done_;
    std::thread update_thread_;
    struct {
        std::mutex lock;
        t::geometry::PointCloud pcd;
    } surface_;
    std::atomic<bool> is_scene_updated_;

    void UpdateMain() {
        // Note that we cannot update the GUI on this thread, we must post to
        // the main thread!
        const std::string rgb_dir = dataset_path_ + "/color";
        const std::string depth_dir = dataset_path_ + "/depth";
        std::vector<std::string> rgb_files;
        std::vector<std::string> depth_files;
        utility::filesystem::ListFilesInDirectoryWithExtension(rgb_dir, "jpg",
                                                               rgb_files);
        if (rgb_files.size() == 0) {
            utility::filesystem::ListFilesInDirectoryWithExtension(
                    rgb_dir, "png", rgb_files);
        }
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
        t::geometry::Image ref_color =
                *t::io::CreateImageFromFile(rgb_files[0]);

        t::pipelines::voxelhashing::Frame input_frame(
                ref_depth.GetRows(), ref_depth.GetCols(), intrinsic_t, device);
        t::pipelines::voxelhashing::Frame raycast_frame(
                ref_depth.GetRows(), ref_depth.GetCols(), intrinsic_t, device);
        t::pipelines::voxelhashing::Model model(voxel_size, sdf_trunc,
                                                block_resolution, block_count,
                                                T_frame_to_model, device);

        size_t idx;
        idx = 0;

        // Odometry
        auto traj = std::make_shared<geometry::LineSet>();
        auto frustum = std::make_shared<geometry::LineSet>();
        auto color = std::make_shared<geometry::Image>();
        auto depth_colored = std::make_shared<geometry::Image>();

        auto raycast_color = std::make_shared<geometry::Image>();
        auto raycast_depth_colored = std::make_shared<geometry::Image>();

        is_scene_updated_ = false;

        color = std::make_shared<open3d::geometry::Image>(
                ref_color.ToLegacyImage());
        depth_colored = std::make_shared<open3d::geometry::Image>(
                ref_depth
                        .ColorizeDepth(depth_scale, 0.3, prop_values_.depth_max)
                        .ToLegacyImage());

        raycast_color = std::make_shared<geometry::Image>(
                t::geometry::Image(
                        core::Tensor::Zeros(
                                {ref_depth.GetRows(), ref_depth.GetCols(), 3},
                                core::Dtype::UInt8, core::Device("CPU:0")))
                        .ToLegacyImage());
        raycast_depth_colored = std::make_shared<geometry::Image>(
                t::geometry::Image(
                        core::Tensor::Zeros(
                                {ref_depth.GetRows(), ref_depth.GetCols(), 3},
                                core::Dtype::UInt8, core::Device("CPU:0")))
                        .ToLegacyImage());

        // Render once to refresh
        gui::Application::GetInstance().PostToMainThread(
                this, [this, color, depth_colored, raycast_color,
                       raycast_depth_colored]() {
                    this->input_color_image_->UpdateImage(color);
                    this->input_depth_image_->UpdateImage(depth_colored);
                    this->raycast_color_image_->UpdateImage(raycast_color);
                    this->raycast_depth_image_->UpdateImage(
                            raycast_depth_colored);
                    this->SetNeedsLayout();  // size of image changed

                    int max_points = prop_values_.pointcloud_size;
                    t::geometry::PointCloud pcd_placeholder(
                            core::Tensor({max_points, 3}, core::Dtype::Float32,
                                         core::Device("CPU:0")));
                    pcd_placeholder.SetPointColors(
                            core::Tensor({max_points, 3}, core::Dtype::Float32,
                                         core::Device("CPU:0")));

                    auto mat = rendering::Material();
                    mat.shader = "defaultUnlit";
                    this->widget3d_->GetScene()->GetScene()->AddGeometry(
                            "points", pcd_placeholder, mat);

                    geometry::AxisAlignedBoundingBox bbox(
                            Eigen::Vector3d(-5, -5, -5),
                            Eigen::Vector3d(5, 5, 5));
                    auto center = bbox.GetCenter().cast<float>();
                    this->widget3d_->SetupCamera(60, bbox, center);
                    this->widget3d_->LookAt(center,
                                            center - Eigen::Vector3f{0, 1, 3},
                                            {0.0f, -1.0f, 0.0f});
                });

        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

        utility::Timer timer;
        timer.Start();
        while (!is_done_) {
            float depth_scale = prop_values_.depth_scale;

            if (!is_running_) {
                // If we aren't running, sleep a little bit so that we don't
                // use 100% of the CPU just checking if we need to run.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            timer.Stop();
            float elapsed_time = timer.GetDuration();
            timer.Start();

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

                core::Tensor trans =
                        delta_frame_to_model.Slice(0, 0, 3).Slice(1, 3, 4);
                double trans_norm =
                        std::sqrt((trans * trans).Sum({0, 1}).Item<double>());
                utility::LogInfo("trans norm = {}", trans_norm);
                if (trans_norm < 0.15) {
                    T_frame_to_model =
                            T_frame_to_model.Matmul(delta_frame_to_model);
                }
            }

            // Integrate
            model.UpdateFramePose(idx, T_frame_to_model);
            model.Integrate(input_frame, depth_scale, prop_values_.depth_max);
            model.SynthesizeModelFrame(raycast_frame, depth_scale, 0.1,
                                       prop_values_.depth_max);

            auto K_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(
                    intrinsic_t);
            auto T_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(
                    T_frame_to_model);
            std::stringstream out;
            out << "Frame " << idx << "\n";
            out << T_eigen.format(CleanFmt) << "\n";
            out << "Active voxel blocks: " << model.GetHashmapSize() << "\n";
            std::cout << out.str() << "\n";

            {
                std::lock_guard<std::mutex> locker(surface_.lock);
                int64_t len = surface_.pcd.HasPoints()
                                      ? surface_.pcd.GetPoints().GetLength()
                                      : 0;
                out << "Surface points: " << len << "\n";

                out << "FPS: " << 1000.0 / elapsed_time << "\n";
            }

            traj->points_.push_back(T_eigen.block<3, 1>(0, 3));
            if (traj->points_.size() > 1) {
                int n = traj->points_.size();
                traj->lines_.push_back({n - 1, n - 2});
                traj->colors_.push_back(kTangoSkyBlueDark);
            }

            frustum = CreateCameraFrustum(color->width_, color->height_,
                                          K_eigen, T_eigen.inverse());

            // TODO: update support for timages-image conversion
            color = std::make_shared<open3d::geometry::Image>(
                    input_frame.GetDataAsImage("color").ToLegacyImage());
            depth_colored = std::make_shared<open3d::geometry::Image>(
                    input_frame.GetDataAsImage("depth")
                            .ColorizeDepth(depth_scale, 0.3,
                                           prop_values_.depth_max)
                            .ToLegacyImage());

            if (prop_values_.raycast_color) {
                raycast_color = std::make_shared<open3d::geometry::Image>(
                        raycast_frame.GetDataAsImage("color")
                                .To(core::Dtype::UInt8, false, 255.0f)
                                .ToLegacyImage());
            }

            raycast_depth_colored = std::make_shared<open3d::geometry::Image>(
                    raycast_frame.GetDataAsImage("depth")
                            .ColorizeDepth(depth_scale, 0.3,
                                           prop_values_.depth_max)
                            .ToLegacyImage());

            // Extract surface on demand (do before we increment idx, so that
            // we see something immediately, on interation 0)
            if (idx % static_cast<int>(prop_values_.surface_interval) == 0 ||
                idx == depth_files.size() - 1) {
                std::lock_guard<std::mutex> locker(surface_.lock);
                printf("lock acquired for surface\n");
                surface_.pcd =
                        model.ExtractPointCloud(prop_values_.pointcloud_size,
                                                std::min<float>(idx, 3.0f))
                                .CPU();
                is_scene_updated_ = true;
            }

            gui::Application::GetInstance().PostToMainThread(
                    this,
                    [this, color, depth_colored, raycast_color,
                     raycast_depth_colored, traj, frustum, out = out.str()]() {
                        // Disable depth_scale and pcd buffer size change
                        this->fixed_props_->SetEnabled(false);

                        this->raycast_color_image_->SetVisible(
                                this->prop_values_.raycast_color);

                        this->SetOutput(out);
                        this->input_color_image_->UpdateImage(color);
                        this->input_depth_image_->UpdateImage(depth_colored);

                        if (prop_values_.raycast_color) {
                            this->raycast_color_image_->UpdateImage(
                                    raycast_color);
                        }
                        this->raycast_depth_image_->UpdateImage(
                                raycast_depth_colored);

                        this->widget3d_->GetScene()->RemoveGeometry("frustum");
                        auto mat = rendering::Material();
                        mat.shader = "unlitLine";
                        mat.line_width = 5.0f;
                        this->widget3d_->GetScene()->AddGeometry(
                                "frustum", frustum.get(), mat);

                        if (traj->points_.size() > 1) {
                            // 1) Add geometry once w/ max size
                            // 2) Update geometry
                            // TPointCloud
                            this->widget3d_->GetScene()->RemoveGeometry(
                                    "trajectory");
                            auto mat = rendering::Material();
                            mat.shader = "unlitLine";
                            mat.line_width = 5.0f;
                            this->widget3d_->GetScene()->AddGeometry(
                                    "trajectory", traj.get(), mat);
                        }

                        if (is_scene_updated_) {
                            using namespace rendering;
                            std::lock_guard<std::mutex> locker(surface_.lock);
                            printf("lock acquired for rendering\n");
                            if (surface_.pcd.HasPoints() &&
                                surface_.pcd.HasPointColors()) {
                                printf("obtain scene\n");
                                auto* scene =
                                        this->widget3d_->GetScene()->GetScene();
                                printf("update geometry\n");
                                scene->UpdateGeometry(
                                        "points", surface_.pcd,
                                        Scene::kUpdatePointsFlag |
                                                Scene::kUpdateColorsFlag);
                            }
                            is_scene_updated_ = false;
                        }
                    });

            // Note that the user might have closed the window, in which case we
            // want to maintain a value of true.
            idx++;
            is_done_ = is_done_ | (idx >= depth_files.size());
        }
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
