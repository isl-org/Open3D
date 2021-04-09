#include <atomic>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::visualization;

const std::string TEST_DIR = "../../../examples/test_data/RGBD";

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
    ReconstructionWindow() : gui::Window("Open3D - Reconstruction", 1200, 768) {
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
        panel_->AddChild(props_);
        panel_->AddStretch();

        rgb_image_ = std::make_shared<gui::ImageWidget>();
        depth_image_ = std::make_shared<gui::ImageWidget>();
        panel_->AddChild(rgb_image_);
        panel_->AddChild(depth_image_);

        output_ = std::make_shared<gui::Label>("");
        output_panel_->AddChild(std::make_shared<gui::Label>("Output"));
        output_panel_->AddChild(output_);

        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~ReconstructionWindow() {}

    void Layout(const gui::Theme& theme) override {
        Super::Layout(theme);

        int em = theme.font_size;
        int panel_width = 15 * em;
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

    void SetOutput(const std::string& output) {
        output_->SetText(output.c_str());
    }
};

//------------------------------------------------------------------------------
class ExampleWindow : public ReconstructionWindow {
public:
    ExampleWindow() {
        props_->AddNumber("Depth scale", &prop_values_.depth_scale, 1000.0, 1.0,
                          1500.0);
        props_->AddNumber("Depth trunc", &prop_values_.depth_trunc, 3.0, 0.0,
                          10.0);
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
    struct {
        std::atomic<double> depth_scale;
        std::atomic<double> depth_trunc;
        std::atomic<bool> color_points;
    } prop_values_;
    std::atomic<bool> is_done_;
    std::thread update_thread_;

    void UpdateMain() {
        // Note that we cannot update the GUI on this thread, we must post to
        // the main thread!

        const std::string rgb_dir = TEST_DIR + "/color";
        const std::string depth_dir = TEST_DIR + "/depth";
        std::vector<std::string> rgb_files;
        std::vector<std::string> depth_files;
        utility::filesystem::ListFilesInDirectoryWithExtension(rgb_dir, "jpg",
                                                               rgb_files);
        std::sort(rgb_files.begin(), rgb_files.end());
        utility::filesystem::ListFilesInDirectoryWithExtension(depth_dir, "png",
                                                               depth_files);
        std::sort(depth_files.begin(), depth_files.end());

        bool is_initialized = false;
        size_t idx = 0;
        while (!is_done_) {
            std::stringstream out;
            auto color = std::make_shared<geometry::Image>();
            auto tcolor = std::make_shared<t::geometry::Image>(
                    t::geometry::Image::FromLegacyImage(*color));
            auto depth = std::make_shared<geometry::Image>();
            io::ReadImage(rgb_files[idx], *color);
            io::ReadImage(depth_files[idx], *depth);
            auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
                    *color, *depth, prop_values_.depth_scale,
                    prop_values_.depth_trunc, !prop_values_.color_points);
            auto depth8 = ConvertDepthToNormalizedGrey8(rgbd->depth_);

            camera::PinholeCameraIntrinsic intrinsic =
                    camera::PinholeCameraIntrinsic(
                            camera::PinholeCameraIntrinsicParameters::
                                    PrimeSenseDefault);
            auto pcd =
                    geometry::PointCloud::CreateFromRGBDImage(*rgbd, intrinsic);
            auto tpcd = std::make_shared<t::geometry::PointCloud>(
                    t::geometry::PointCloud::FromLegacyPointCloud(*pcd));

            out << pcd->points_.size() << " points" << std::endl;
            out << "intrinsic matrix:" << std::endl;
            out << intrinsic.intrinsic_matrix_ << std::endl;

            idx++;
            if (idx >= depth_files.size()) {
                idx = 0;
            }

            gui::Application::GetInstance().PostToMainThread(
                    this, [this, tcolor, color, depth8, pcd, tpcd,
                           is_initialized, out = out.str()]() {
                        this->SetOutput(out);
                        this->rgb_image_->UpdateImage(color);
                        this->depth_image_->UpdateImage(depth8);
                        this->widget3d_->GetScene()->RemoveGeometry("points");
                        auto mat = rendering::Material();
                        mat.shader = "defaultUnlit";
                        this->widget3d_->GetScene()->AddGeometry(
                                "points", tpcd.get(), mat);
                        if (!is_initialized) {
                            auto bbox = this->widget3d_->GetScene()
                                                ->GetBoundingBox();
                            auto center = bbox.GetCenter().cast<float>();
                            this->widget3d_->SetupCamera(60, bbox, center);
                            this->widget3d_->LookAt(
                                    center, center - Eigen::Vector3f{0, 1, 3},
                                    {0.0f, -1.0f, 0.0f});
                        }
                    });

            is_initialized = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // The renderer can only use 8-bit channels currently. Also, we need to
    // convert to RGB because the renderer will display one-channel images in
    // red. Normalize because otherwise it can be hard to see the image.
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
    if (!utility::filesystem::DirectoryExists(TEST_DIR)) {
        utility::LogError(
                "This example needs to be run from the <build>/bin/examples "
                "directory");
    }

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, argv);
    app.AddWindow(std::make_shared<ExampleWindow>());
    app.Run();
}
