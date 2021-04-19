#include <atomic>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::visualization;

std::mutex pcd_mutex;

// Tanglo colorscheme (see https://en.wikipedia.org/wiki/Tango_Desktop_Project)
static const Eigen::Vector3d kTangoOrange(0.961, 0.475, 0.000);
static const Eigen::Vector3d kTangoSkyBlueDark(0.125, 0.290, 0.529);

// Turbo colormap
//   https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
static const unsigned char turbo_srgb_bytes[256][3] = {
        {48, 18, 59},   {50, 21, 67},   {51, 24, 74},    {52, 27, 81},
        {53, 30, 88},   {54, 33, 95},   {55, 36, 102},   {56, 39, 109},
        {57, 42, 115},  {58, 45, 121},  {59, 47, 128},   {60, 50, 134},
        {61, 53, 139},  {62, 56, 145},  {63, 59, 151},   {63, 62, 156},
        {64, 64, 162},  {65, 67, 167},  {65, 70, 172},   {66, 73, 177},
        {66, 75, 181},  {67, 78, 186},  {68, 81, 191},   {68, 84, 195},
        {68, 86, 199},  {69, 89, 203},  {69, 92, 207},   {69, 94, 211},
        {70, 97, 214},  {70, 100, 218}, {70, 102, 221},  {70, 105, 224},
        {70, 107, 227}, {71, 110, 230}, {71, 113, 233},  {71, 115, 235},
        {71, 118, 238}, {71, 120, 240}, {71, 123, 242},  {70, 125, 244},
        {70, 128, 246}, {70, 130, 248}, {70, 133, 250},  {70, 135, 251},
        {69, 138, 252}, {69, 140, 253}, {68, 143, 254},  {67, 145, 254},
        {66, 148, 255}, {65, 150, 255}, {64, 153, 255},  {62, 155, 254},
        {61, 158, 254}, {59, 160, 253}, {58, 163, 252},  {56, 165, 251},
        {55, 168, 250}, {53, 171, 248}, {51, 173, 247},  {49, 175, 245},
        {47, 178, 244}, {46, 180, 242}, {44, 183, 240},  {42, 185, 238},
        {40, 188, 235}, {39, 190, 233}, {37, 192, 231},  {35, 195, 228},
        {34, 197, 226}, {32, 199, 223}, {31, 201, 221},  {30, 203, 218},
        {28, 205, 216}, {27, 208, 213}, {26, 210, 210},  {26, 212, 208},
        {25, 213, 205}, {24, 215, 202}, {24, 217, 200},  {24, 219, 197},
        {24, 221, 194}, {24, 222, 192}, {24, 224, 189},  {25, 226, 187},
        {25, 227, 185}, {26, 228, 182}, {28, 230, 180},  {29, 231, 178},
        {31, 233, 175}, {32, 234, 172}, {34, 235, 170},  {37, 236, 167},
        {39, 238, 164}, {42, 239, 161}, {44, 240, 158},  {47, 241, 155},
        {50, 242, 152}, {53, 243, 148}, {56, 244, 145},  {60, 245, 142},
        {63, 246, 138}, {67, 247, 135}, {70, 248, 132},  {74, 248, 128},
        {78, 249, 125}, {82, 250, 122}, {85, 250, 118},  {89, 251, 115},
        {93, 252, 111}, {97, 252, 108}, {101, 253, 105}, {105, 253, 102},
        {109, 254, 98}, {113, 254, 95}, {117, 254, 92},  {121, 254, 89},
        {125, 255, 86}, {128, 255, 83}, {132, 255, 81},  {136, 255, 78},
        {139, 255, 75}, {143, 255, 73}, {146, 255, 71},  {150, 254, 68},
        {153, 254, 66}, {156, 254, 64}, {159, 253, 63},  {161, 253, 61},
        {164, 252, 60}, {167, 252, 58}, {169, 251, 57},  {172, 251, 56},
        {175, 250, 55}, {177, 249, 54}, {180, 248, 54},  {183, 247, 53},
        {185, 246, 53}, {188, 245, 52}, {190, 244, 52},  {193, 243, 52},
        {195, 241, 52}, {198, 240, 52}, {200, 239, 52},  {203, 237, 52},
        {205, 236, 52}, {208, 234, 52}, {210, 233, 53},  {212, 231, 53},
        {215, 229, 53}, {217, 228, 54}, {219, 226, 54},  {221, 224, 55},
        {223, 223, 55}, {225, 221, 55}, {227, 219, 56},  {229, 217, 56},
        {231, 215, 57}, {233, 213, 57}, {235, 211, 57},  {236, 209, 58},
        {238, 207, 58}, {239, 205, 58}, {241, 203, 58},  {242, 201, 58},
        {244, 199, 58}, {245, 197, 58}, {246, 195, 58},  {247, 193, 58},
        {248, 190, 57}, {249, 188, 57}, {250, 186, 57},  {251, 184, 56},
        {251, 182, 55}, {252, 179, 54}, {252, 177, 54},  {253, 174, 53},
        {253, 172, 52}, {254, 169, 51}, {254, 167, 50},  {254, 164, 49},
        {254, 161, 48}, {254, 158, 47}, {254, 155, 45},  {254, 153, 44},
        {254, 150, 43}, {254, 147, 42}, {254, 144, 41},  {253, 141, 39},
        {253, 138, 38}, {252, 135, 37}, {252, 132, 35},  {251, 129, 34},
        {251, 126, 33}, {250, 123, 31}, {249, 120, 30},  {249, 117, 29},
        {248, 114, 28}, {247, 111, 26}, {246, 108, 25},  {245, 105, 24},
        {244, 102, 23}, {243, 99, 21},  {242, 96, 20},   {241, 93, 19},
        {240, 91, 18},  {239, 88, 17},  {237, 85, 16},   {236, 83, 15},
        {235, 80, 14},  {234, 78, 13},  {232, 75, 12},   {231, 73, 12},
        {229, 71, 11},  {228, 69, 10},  {226, 67, 10},   {225, 65, 9},
        {223, 63, 8},   {221, 61, 8},   {220, 59, 7},    {218, 57, 7},
        {216, 55, 6},   {214, 53, 6},   {212, 51, 5},    {210, 49, 5},
        {208, 47, 5},   {206, 45, 4},   {204, 43, 4},    {202, 42, 4},
        {200, 40, 3},   {197, 38, 3},   {195, 37, 3},    {193, 35, 2},
        {190, 33, 2},   {188, 32, 2},   {185, 30, 2},    {183, 29, 2},
        {180, 27, 1},   {178, 26, 1},   {175, 24, 1},    {172, 23, 1},
        {169, 22, 1},   {167, 20, 1},   {164, 19, 1},    {161, 18, 1},
        {158, 16, 1},   {155, 15, 1},   {152, 14, 1},    {149, 13, 1},
        {146, 11, 1},   {142, 10, 1},   {139, 9, 2},     {136, 8, 2},
        {133, 7, 2},    {129, 6, 2},    {126, 5, 2},     {122, 4, 3}};

// The renderer can only use 8-bit channels currently. Also, we need to
// convert to RGB because the renderer will display one-channel images
// in red. Normalize because otherwise it can be hard to see the image.
std::shared_ptr<geometry::Image> ColorizeDepth(const geometry::Image& depth,
                                               float depth_scale = 1000.0,
                                               float depth_min = 0.3,
                                               float depth_max = 3.0) {
    float* data = depth.PointerAs<float>();
    int n_pixels = depth.width_ * depth.height_;

    auto img888 = std::make_shared<geometry::Image>();
    img888->width_ = depth.width_;
    img888->height_ = depth.height_;
    img888->num_of_channels_ = 3;
    img888->bytes_per_channel_ = 1;
    img888->data_.reserve(img888->width_ * img888->height_ *
                          img888->num_of_channels_ *
                          img888->bytes_per_channel_);
    for (int i = 0; i < n_pixels; ++i) {
        float val = data[i] / depth_scale;
        val = std::max<float>(depth_min, val);
        val = std::min<float>(depth_max, val);
        val = (val - depth_min) / (depth_max - depth_min) * 255.0;
        uint8_t px = uint8_t(val);
        img888->data_.push_back(turbo_srgb_bytes[px][0]);
        img888->data_.push_back(turbo_srgb_bytes[px][1]);
        img888->data_.push_back(turbo_srgb_bytes[px][2]);
    }

    return img888;
}

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
        default_label_color_ = std::make_shared<gui::Label>("temp")->GetTextColor();
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
        : gui::Window("Open3D - Reconstruction", 1600, 900), is_running_(false) {
        auto& theme = GetTheme();
        int em = theme.font_size;
        int spacing = int(std::round(0.25f * float(em)));
        int left_margin = em;
        int vspacing = int(std::round(0.5f * float(em)));
        gui::Margins margins(int(std::round(0.5f * float(em))));
        panel_ = std::make_shared<gui::Vert>(spacing, margins);
        widget3d_ = std::make_shared<gui::SceneWidget>();
        output_panel_ = std::make_shared<gui::Vert>(spacing, margins);
        AddChild(panel_);
        AddChild(widget3d_);
        AddChild(output_panel_);

        fixed_props_ = std::make_shared<PropertyPanel>(spacing, left_margin);
        adjustable_props_ = std::make_shared<PropertyPanel>(spacing,
                                                            left_margin);

        panel_->AddChild(std::make_shared<gui::Label>("Starting settings"));
        panel_->AddChild(fixed_props_);

        panel_->AddFixed(vspacing);
        panel_->AddChild(std::make_shared<gui::Label>("Reconstruction settings"));
        panel_->AddChild(adjustable_props_);
        panel_->SetEnabled(false);

        auto b = std::make_shared<gui::Button>(" Start ");
        b->SetOnClicked([b, this]() {
            this->is_running_ = !(this->is_running_);
            if (this->is_running_) {
                b->SetText("Pause");
            } else {
                b->SetText("Resume");
            }
            this->adjustable_props_->SetEnabled(true);
        });
        auto h = std::make_shared<gui::Horiz>();
        h->AddStretch();
        h->AddChild(b);
        h->AddStretch();
        panel_->AddFixed(vspacing);
        panel_->AddChild(h);

        panel_->AddStretch();

        panel_->AddChild(std::make_shared<gui::Label>("Input image(s)"));
        input_color_image_ = std::make_shared<gui::ImageWidget>();
        input_depth_image_ = std::make_shared<gui::ImageWidget>();
        panel_->AddChild(input_color_image_);
        panel_->AddChild(input_depth_image_);

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
    std::atomic<bool> is_running_;

    std::shared_ptr<gui::Vert> panel_;
    std::shared_ptr<gui::Vert> output_panel_;
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
        fixed_props_->AddIntSlider("Pointcloud size estimate",
                                   &prop_values_.pointcloud_size, 3000000,
                                   500000, 8000000);

        // Adjustable
        adjustable_props_->AddIntSlider(
                "Surface update", &prop_values_.surface_interval, 50, 1, 100);

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
        core::Device device("CPU:0");

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

        bool is_scene_updated = false;
        size_t idx = 0;

        // Odometry
        auto traj = std::make_shared<geometry::LineSet>();
        auto frustum = std::make_shared<geometry::LineSet>();
        auto color = std::make_shared<geometry::Image>();
        auto depth = std::make_shared<geometry::Image>();
        auto depth8 = std::make_shared<geometry::Image>();

        auto raycast_color = std::make_shared<geometry::Image>();
        auto raycast_depth = std::make_shared<geometry::Image>();
        auto raycast_depth8 = std::make_shared<geometry::Image>();

        t::geometry::PointCloud pcd;

        color = std::make_shared<open3d::geometry::Image>(
                ref_color.ToLegacyImage());
        depth = std::make_shared<open3d::geometry::Image>(
                ref_depth.To(core::Dtype::Float32, false, 1.0f)
                        .ToLegacyImage());
        depth8 =
                ColorizeDepth(*depth, depth_scale, 0.3, prop_values_.depth_max);
        raycast_color = std::make_shared<geometry::Image>(
                t::geometry::Image(
                        core::Tensor::Zeros(
                                {ref_depth.GetRows(), ref_depth.GetCols(), 3},
                                core::Dtype::UInt8, core::Device("CPU:0")))
                        .ToLegacyImage());
        raycast_depth8 = std::make_shared<geometry::Image>(
                t::geometry::Image(
                        core::Tensor::Zeros(
                                {ref_depth.GetRows(), ref_depth.GetCols(), 3},
                                core::Dtype::UInt8, core::Device("CPU:0")))
                        .ToLegacyImage());

        // Render once to refresh
        gui::Application::GetInstance().PostToMainThread(
                this, [this, color, depth8, raycast_color, raycast_depth8]() {
                    this->input_color_image_->UpdateImage(color);
                    this->input_depth_image_->UpdateImage(depth8);
                    this->raycast_color_image_->UpdateImage(raycast_color);
                    this->raycast_depth_image_->UpdateImage(raycast_depth8);

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
        while (!is_done_) {
            if (!is_running_) {
                // If we aren't running, sleep a little bit so that we don't
                // use 100% of the CPU just checking if we need to run.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

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
            model.SynthesizeModelFrame(raycast_frame, depth_scale);

            idx++;
            is_done_ = (idx >= depth_files.size());

            auto K_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(
                    intrinsic_t);
            auto T_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(
                    T_frame_to_model);
            std::stringstream out;
            out << "Frame " << idx << "\n";
            out << T_eigen.format(CleanFmt) << "\n";
            out << "Active voxel blocks: " << model.GetHashmapSize() << "\n";
            std::cout << out.str() << "\n";

            int64_t len = pcd.HasPoints() ? pcd.GetPoints().GetLength() : 0;
            out << "Surface points: " << len << "\n";

            traj->points_.push_back(T_eigen.block<3, 1>(0, 3));
            if (traj->points_.size() > 1) {
                int n = traj->points_.size();
                traj->lines_.push_back({n - 1, n - 2});
                traj->colors_.push_back(kTangoSkyBlueDark);
            }

            frustum = CreateCameraFrustum(640, 480, K_eigen, T_eigen.inverse());

            // TODO: update support for timages-image conversion
            color = std::make_shared<open3d::geometry::Image>(
                    input_frame.GetDataAsImage("color").ToLegacyImage());
            depth = std::make_shared<open3d::geometry::Image>(
                    input_frame.GetDataAsImage("depth")
                            .To(core::Dtype::Float32, false, 1.0f)
                            .ToLegacyImage());
            depth8 = ColorizeDepth(*depth, depth_scale, 0.3,
                                   prop_values_.depth_max);

            if (prop_values_.raycast_color) {
                raycast_color = std::make_shared<open3d::geometry::Image>(
                        raycast_frame.GetDataAsImage("color")
                                .To(core::Dtype::UInt8, false, 255.0f)
                                .ToLegacyImage());
            }

            raycast_depth = std::make_shared<open3d::geometry::Image>(
                    raycast_frame.GetDataAsImage("depth").ToLegacyImage());
            raycast_depth8 = ColorizeDepth(*raycast_depth, depth_scale, 0.3,
                                           prop_values_.depth_max);

            // Extract surface on demand
            if (idx % static_cast<int>(prop_values_.surface_interval) == 0 ||
                idx == depth_files.size() - 1) {
                pcd = model.ExtractPointCloud(prop_values_.pointcloud_size,
                                              std::min<float>(idx, 3.0f))
                              .CPU();
                is_scene_updated = true;
            }

            gui::Application::GetInstance().PostToMainThread(
                    this,
                    [this, color, depth8, raycast_color, raycast_depth8, pcd,
                     traj, frustum, &is_scene_updated, out = out.str()]() {
                        // Disable depth_scale and pcd buffer size change
                        this->fixed_props_->SetEnabled(false);

                        this->raycast_color_image_->SetVisible(this->prop_values_.raycast_color);

                        this->SetOutput(out);
                        this->input_color_image_->UpdateImage(color);
                        this->input_depth_image_->UpdateImage(depth8);

                        if (prop_values_.raycast_color) {
                            this->raycast_color_image_->UpdateImage(
                                    raycast_color);
                        }
                        this->raycast_depth_image_->UpdateImage(raycast_depth8);

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

                        if (is_scene_updated && pcd.HasPoints() &&
                            pcd.HasPointColors()) {
                            this->widget3d_->GetScene()
                                    ->GetScene()
                                    ->UpdateGeometry(
                                            "points", pcd,
                                            rendering::Scene::
                                                            kUpdatePointsFlag |
                                                    rendering::Scene ::
                                                            kUpdateColorsFlag);
                            is_scene_updated = false;
                        }
                    });
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
