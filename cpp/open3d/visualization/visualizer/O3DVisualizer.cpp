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

#include "open3d/visualization/visualizer/O3DVisualizer.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "open3d/Open3DConfig.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/Octree.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/io/ImageIO.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Checkbox.h"
#include "open3d/visualization/gui/ColorEdit.h"
#include "open3d/visualization/gui/Combobox.h"
#include "open3d/visualization/gui/Dialog.h"
#include "open3d/visualization/gui/FileDialog.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/ListView.h"
#include "open3d/visualization/gui/NumberEdit.h"
#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/gui/Slider.h"
#include "open3d/visualization/gui/StackedWidget.h"
#include "open3d/visualization/gui/TabControl.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/TreeView.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/visualizer/GuiWidgets.h"
#include "open3d/visualization/visualizer/O3DVisualizerSelections.h"
#include "open3d/visualization/visualizer/Receiver.h"

#define GROUPS_USE_TREE 1

using namespace open3d::visualization::gui;
using namespace open3d::visualization::rendering;

namespace open3d {
namespace visualization {
namespace visualizer {

namespace {
static const std::string kShaderLit = "defaultLit";
static const std::string kShaderUnlit = "defaultUnlit";
static const std::string kShaderUnlitLines = "unlitLine";

static const std::string kDefaultIBL = "default";

enum MenuId {
    MENU_HELP_ABOUT = 0,
    MENU_HELP_CONTACT_US,
    MENU_HELP_SHOW_CONTROLS,
    MENU_HELP_SHOW_CAMERA_INFO,
    MENU_EXPORT_RGB,
    MENU_CLOSE,
    MENU_SETTINGS,
    MENU_ACTIONS_BASE = 1000 /* this should be last */
};

template <typename T>
std::shared_ptr<T> GiveOwnership(T *ptr) {
    return std::shared_ptr<T>(ptr);
}

class ButtonList : public Widget {
public:
    explicit ButtonList(int spacing) : spacing_(spacing) {}

    void SetWidth(int width) { width_ = width; }

    Size CalcPreferredSize(const LayoutContext &context,
                           const Constraints &constraints) const override {
        auto frames = CalcFrames(context, constraints);
        if (!frames.empty()) {
            // Add spacing on the bottom to look like the start of a new row
            return Size(width_,
                        frames.back().GetBottom() - frames[0].y + spacing_);
        } else {
            return Size(width_, 0);
        }
    }

    void Layout(const LayoutContext &context) override {
        auto frames = CalcFrames(context, Constraints());
        auto &children = GetChildren();
        for (size_t i = 0; i < children.size(); ++i) {
            children[i]->SetFrame(frames[i]);
        }
    }

    size_t size() const { return GetChildren().size(); }

private:
    int spacing_;
    int width_ = 10000;

    std::vector<Rect> CalcFrames(const LayoutContext &context,
                                 const Widget::Constraints &constraints) const {
        auto &f = GetFrame();
        std::vector<Rect> frames;
        int x = f.x;
        int y = f.y;
        int lineHeight = 0;
        for (auto child : GetChildren()) {
            auto pref = child->CalcPreferredSize(context, constraints);
            if (x > f.x && x + pref.width > f.x + width_) {
                y = y + lineHeight + spacing_;
                x = f.x;
                lineHeight = 0;
            }
            frames.emplace_back(x, y, pref.width, pref.height);
            x += pref.width + spacing_;
            lineHeight = std::max(lineHeight, pref.height);
        }
        return frames;
    }
};

class EmptyIfHiddenVert : public CollapsableVert {
    using Super = CollapsableVert;

public:
    EmptyIfHiddenVert(const char *text) : CollapsableVert(text) {}
    EmptyIfHiddenVert(const char *text,
                      int spacing,
                      const Margins &margins = Margins())
        : CollapsableVert(text, spacing, margins) {}

    void SetVisible(bool vis) override {
        Super::SetVisible(vis);
        Super::SetIsOpen(vis);
        needsLayout_ = true;
    }

    Size CalcPreferredSize(const LayoutContext &context,
                           const Constraints &constraints) const override {
        if (IsVisible()) {
            return Super::CalcPreferredSize(context, constraints);
        } else {
            return Size(0, 0);
        }
    }

    Widget::DrawResult Draw(const DrawContext &context) override {
        auto result = Super::Draw(context);
        if (needsLayout_) {
            needsLayout_ = false;
            return Widget::DrawResult::RELAYOUT;
        } else {
            return result;
        }
    }

private:
    bool needsLayout_ = false;
};

class DrawObjectTreeCell : public Widget {
    using Super = Widget;

public:
    enum { FLAG_NONE = 0, FLAG_GROUP = (1 << 0), FLAG_TIME = (1 << 1) };

    DrawObjectTreeCell(const char *name,
                       const char *group,
                       double time,
                       bool is_checked,
                       int flags,
                       std::function<void(bool)> on_toggled) {
        flags_ = flags;

        std::string time_str;
        if (flags & FLAG_TIME) {
            char buf[32];
            if (time == double(int(time))) {
                snprintf(buf, sizeof(buf), "t=%d", int(time));
            } else {
                snprintf(buf, sizeof(buf), "t=%g", time);
            }
            time_str = std::string(buf);
        }

        // We don't want any text in the checkbox, but passing "" seems to make
        // it not toggle, so we need to pass in something. This way it will
        // just be extra spacing.
        checkbox_ = std::make_shared<Checkbox>(" ");
        checkbox_->SetChecked(is_checked);
        checkbox_->SetOnChecked(on_toggled);
        name_ = std::make_shared<Label>(name);
        group_ = std::make_shared<Label>((flags & FLAG_GROUP) ? group : "");
        time_ = std::make_shared<Label>(time_str.c_str());
        AddChild(checkbox_);
        AddChild(name_);
        AddChild(group_);
        AddChild(time_);
    }

    ~DrawObjectTreeCell() {}

    std::shared_ptr<Checkbox> GetCheckbox() { return checkbox_; }
    std::shared_ptr<Label> GetName() { return name_; }

    Size CalcPreferredSize(const LayoutContext &context,
                           const Constraints &constraints) const override {
        auto check_pref = checkbox_->CalcPreferredSize(context, constraints);
        auto name_pref = name_->CalcPreferredSize(context, constraints);
        int w = check_pref.width + name_pref.width + GroupWidth(context.theme) +
                TimeWidth(context.theme);
        return Size(w, std::max(check_pref.height, name_pref.height));
    }

    void Layout(const LayoutContext &context) override {
        auto &frame = GetFrame();
        auto check_width =
                checkbox_->CalcPreferredSize(context, Constraints()).width;
        checkbox_->SetFrame(Rect(frame.x, frame.y, check_width, frame.height));
        auto group_width = GroupWidth(context.theme);
        auto time_width = TimeWidth(context.theme);
        auto x = checkbox_->GetFrame().GetRight();
        auto name_width = frame.GetRight() - group_width - time_width - x;
        name_->SetFrame(Rect(x, frame.y, name_width, frame.height));
        x += name_width;
        group_->SetFrame(Rect(x, frame.y, group_width, frame.height));
        x += group_width;
        time_->SetFrame(Rect(x, frame.y, time_width, frame.height));
    }

private:
    int flags_;
    std::shared_ptr<Checkbox> checkbox_;
    std::shared_ptr<Label> name_;
    std::shared_ptr<Label> group_;
    std::shared_ptr<Label> time_;

    int GroupWidth(const Theme &theme) const {
        if (flags_ & FLAG_GROUP) {
            return 5 * theme.font_size;
        } else {
            return 0;
        }
    }

    int TimeWidth(const Theme &theme) const {
        if (flags_ & FLAG_TIME) {
            return 3 * theme.font_size;
        } else {
            return 0;
        }
    }
};

class CameraInfo : public VGrid {
public:
    CameraInfo(const Theme &theme) : VGrid(2, 0, Margins(theme.font_size)) {
        SetBackgroundColor(Color(0, 0, 0, 0.5));

        auto MakeLabel = [](const char *text) {
            auto label = std::make_shared<Label>(text);
            label->SetTextColor(Color(1, 1, 1));
            return label;
        };

        position_ = MakeLabel("[0, 0, 0]");
        forward_ = MakeLabel("[0, 0, 0]");
        left_ = MakeLabel("[0, 0, 0]");
        up_ = MakeLabel("[0, 0, 0]");

        AddChild(MakeLabel("Position:"));
        AddChild(position_);
        AddChild(MakeLabel("Forward:"));
        AddChild(forward_);
        AddChild(MakeLabel("Left:"));
        AddChild(left_);
        AddChild(MakeLabel("Up:"));
        AddChild(up_);

        Eigen::Vector3f zero(0, 0, 0);
        SetFromVectors(zero, zero, zero, zero);
    }

    void SetFromCamera(rendering::Camera *camera) {
        SetFromVectors(camera->GetPosition(), camera->GetForwardVector(),
                       camera->GetLeftVector(), camera->GetUpVector());
    }

    void SetFromVectors(const Eigen::Vector3f &position,
                        const Eigen::Vector3f &forward,
                        const Eigen::Vector3f &left,
                        const Eigen::Vector3f &up) {
        auto SetText = [](const Eigen::Vector3f &v,
                          std::shared_ptr<Label> label) {
            label->SetText(
                    fmt::format("[{:.2f}, {:.2f}, {:.2f}]", v.x(), v.y(), v.z())
                            .c_str());
        };
        SetText(position, position_);
        SetText(forward, forward_);
        SetText(left, left_);
        SetText(up, up_);
    }

private:
    std::shared_ptr<Label> position_;
    std::shared_ptr<Label> forward_;
    std::shared_ptr<Label> left_;
    std::shared_ptr<Label> up_;
};

struct LightingProfile {
    std::string name;
    Open3DScene::LightingProfile profile;
};

static const char *kCustomName = "Custom";
static const std::vector<LightingProfile> gLightingProfiles = {
        {"Hard shadows", Open3DScene::LightingProfile::HARD_SHADOWS},
        {"Dark shadows", Open3DScene::LightingProfile::DARK_SHADOWS},
        {"Medium shadows", Open3DScene::LightingProfile::MED_SHADOWS},
        {"Soft shadows", Open3DScene::LightingProfile::SOFT_SHADOWS},
        {"No shadows", Open3DScene::LightingProfile::NO_SHADOWS}};

struct MaterialProfile {
    std::string name;
    Material material;
};
std::vector<MaterialProfile> InitMaterialProfiles();

static const std::vector<MaterialProfile> kMaterialProfiles =
        InitMaterialProfiles();
std::vector<MaterialProfile> InitMaterialProfiles() {
    std::vector<MaterialProfile> profs;
    profs.push_back({"Default", Material()});
    profs.back().material.shader = "defaultLit";
    profs.push_back({"Metal (rougher)", Material()});
    profs.back().material.shader = "defaultLit";
    profs.back().material.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
    profs.back().material.base_metallic = 1.0f;
    profs.back().material.base_roughness = 0.5;
    profs.back().material.base_reflectance = 0.5f;
    profs.back().material.base_clearcoat = 0.0f;
    profs.back().material.base_clearcoat_roughness = 0.0f;
    profs.back().material.base_anisotropy = 0.0f;
    profs.push_back({"Metal (smoother)", Material()});
    profs.back().material.shader = "defaultLit";
    profs.back().material.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
    profs.back().material.base_metallic = 1.0f;
    profs.back().material.base_roughness = 0.2;
    profs.back().material.base_reflectance = 0.5f;
    profs.back().material.base_clearcoat = 0.0f;
    profs.back().material.base_clearcoat_roughness = 0.0f;
    profs.back().material.base_anisotropy = 0.0f;
    profs.push_back({"Plastic", Material()});
    profs.back().material.shader = "defaultLit";
    profs.back().material.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
    profs.back().material.base_metallic = 0.0f;
    profs.back().material.base_roughness = 0.5;
    profs.back().material.base_reflectance = 0.5f;
    profs.back().material.base_clearcoat = 0.5f;
    profs.back().material.base_clearcoat_roughness = 0.8f;
    profs.back().material.base_anisotropy = 0.0f;
    profs.push_back({"Glazed ceramic", Material()});
    profs.back().material.shader = "defaultLit";
    profs.back().material.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
    profs.back().material.base_metallic = 0.0f;
    profs.back().material.base_roughness = 0.5;
    profs.back().material.base_reflectance = 0.5f;
    profs.back().material.base_clearcoat = 1.0f;
    profs.back().material.base_clearcoat_roughness = 0.2f;
    profs.back().material.base_anisotropy = 0.0f;
    profs.push_back({"Clay", Material()});
    profs.back().material.shader = "defaultLit";
    profs.back().material.base_color = {0.7725f, 0.7725f, 0.7725f, 1.0f};
    profs.back().material.base_metallic = 0.0f;
    profs.back().material.base_roughness = 1.0;
    profs.back().material.base_reflectance = 0.35f;
    profs.back().material.base_clearcoat = 0.0f;
    profs.back().material.base_clearcoat_roughness = 0.f;
    profs.back().material.base_anisotropy = 0.0f;
    return profs;
}

struct PropertyPanel {
    int panel_idx;
    VGrid *panel;

    void SetEnabled(bool enabled) { this->panel->SetEnabled(enabled); }
};

}  // namespace

struct O3DVisualizer::Impl {
    std::set<std::string> added_names_;
    std::set<std::string> added_groups_;
    std::vector<DrawObject> objects_;
    std::shared_ptr<O3DVisualizerSelections> selections_;
    bool polygon_selection_unselects_ = false;
    bool selections_need_update_ = true;
    std::function<void(double)> on_animation_;
    std::function<bool()> on_animation_tick_;
    std::shared_ptr<Receiver> receiver_;

    UIState ui_state_;
    bool can_auto_show_settings_ = true;

    double min_time_ = 0.0;
    double max_time_ = 0.0;
    double start_animation_clock_time_ = 0.0;
    double next_animation_tick_clock_time_ = 0.0;
    double last_animation_tick_clock_time_ = 0.0;

    MenuCustomization app_menu_;
    MenuCustomization file_menu_;

    Window *window_ = nullptr;
    SceneWidget *widget3d_ = nullptr;

    struct {
        // We only keep pointers here because that way we don't have to release
        // all the shared_ptrs at destruction just to ensure that the gui gets
        // destroyed before the Window, because the Window will do that for us.

        Vert *panel;
        CollapsableVert *mouse_panel;
        TabControl *mouse_tab;
        Vert *view_panel;
        SceneWidget::Controls view_mouse_mode;
        std::map<SceneWidget::Controls, Button *> mouse_buttons;
        Vert *pick_panel;
        Horiz *polygon_selection_panel;
        Button *new_selection_set;
        Button *delete_selection_set;
        ListView *selection_sets;

        struct {
            std::unordered_map<int, std::function<void(O3DVisualizer &)>>
                    menuid2action;

            EmptyIfHiddenVert *panel;
            Menu *menu;
            ButtonList *buttons;
        } actions;

        struct {
            CollapsableVert *panel;
            ColorEdit *bg_color;
            Slider *point_size;
            Combobox *shader;
            Combobox *lighting;
        } global;

        struct {
            CollapsableVert *panel;
            TreeView *entities;

            std::unordered_map<TreeView::ItemId, int> id2panelidx;
        } scene;

        struct {
            Checkbox *show;
            struct : public PropertyPanel {
            } properties;
        } skybox;

        struct {
            Checkbox *show;
            struct : public PropertyPanel {
            } properties;
        } axes;

        struct {
            Checkbox *show;
            struct : public PropertyPanel {
                Combobox *ground_plane;
            } properties;
        } ground;

        struct {
            Checkbox *show;
            struct : public PropertyPanel {
                Combobox *names;
                Slider *intensity;
            } properties;
        } ibl;

        struct {
            Checkbox *show;
            struct : public PropertyPanel {
                Slider *intensity;
                VectorEdit *dir;
                ColorEdit *color;
                Checkbox *follows_camera;
            } properties;
        } sun;

        CollapsableVert *property_parent_panel;
        StackedWidget *property_panels;
        int empty_panel_idx;

#if GROUPS_USE_TREE
        std::map<std::string, TreeView::ItemId> group2itemid;
#endif  // GROUPS_USE_TREE
        std::map<std::string, TreeView::ItemId> object2itemid;

#if !GROUPS_USE_TREE
        EmptyIfHiddenVert *groups_panel;
        TreeView *groups;
#endif  // !GROUPS_USE_TREE

        EmptyIfHiddenVert *time_panel;
        Slider *time_slider;
        NumberEdit *time_edit;
        SmallToggleButton *play;

        struct : public PropertyPanel {
            Label *info;
            Combobox *shader;
            Combobox *material;
            ColorEdit *color;
        } geom_properties;
    } settings;

    Widget *controls_help_panel_;
    CameraInfo *camera_info_panel_;

    void Construct(O3DVisualizer *w) {
        if (window_) {
            return;
        }

        window_ = w;
        widget3d_ = new SceneWidget();
        selections_ = std::make_shared<O3DVisualizerSelections>(*widget3d_);
        widget3d_->SetScene(std::make_shared<Open3DScene>(w->GetRenderer()));
        widget3d_->EnableSceneCaching(true);  // smoother UI with large geometry
        widget3d_->SetOnCameraChanged(
                [this](rendering::Camera *cam) { this->OnCameraChanged(cam); });
        widget3d_->SetOnPointsPicked(
                [this](const std::map<
                               std::string,
                               std::vector<std::pair<size_t, Eigen::Vector3d>>>
                               &indices,
                       int keymods) {
                    if ((keymods & int(KeyModifier::SHIFT)) ||
                        polygon_selection_unselects_) {
                        selections_->UnselectIndices(indices);
                    } else {
                        selections_->SelectIndices(indices);
                    }
                    polygon_selection_unselects_ = false;
                });
        w->AddChild(GiveOwnership(widget3d_));

        auto o3dscene = widget3d_->GetScene();
        o3dscene->SetBackground(ui_state_.bg_color);

        controls_help_panel_ = CreateControlsHelp();
        controls_help_panel_->SetVisible(false);
        window_->AddChild(GiveOwnership(controls_help_panel_));

        camera_info_panel_ = new CameraInfo(window_->GetTheme());
        camera_info_panel_->SetVisible(false);
        window_->AddChild(GiveOwnership(camera_info_panel_));

        MakeSettingsUI();
        SetMouseMode(SceneWidget::Controls::ROTATE_CAMERA);
        SetLightingProfile(gLightingProfiles[2]);  // med shadows
        SetPointSize(ui_state_.point_size);  // sync selections_' point size
    }

    void MakeSettingsUI() {
        auto em = window_->GetTheme().font_size;
        auto half_em = int(std::round(0.5f * float(em)));
        auto v_spacing = int(std::round(0.25 * float(em)));

        settings.panel = new Vert(0.75 * em);
        window_->AddChild(GiveOwnership(settings.panel));

        Margins margins(em, 0, half_em, 0);
        Margins tabbed_margins(0, half_em, 0, 0);

        // Custom actions
        settings.actions.panel =
                new EmptyIfHiddenVert("Custom Actions", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.actions.panel));
        settings.actions.panel->SetVisible(false);

        settings.actions.buttons = new ButtonList(v_spacing);
        settings.actions.panel->AddChild(
                GiveOwnership(settings.actions.buttons));

        // Mouse countrols
        settings.mouse_panel =
                new CollapsableVert("Mouse Controls", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.mouse_panel));

        settings.mouse_tab = new TabControl();
        settings.mouse_tab->SetPreferredSizeMode(
                TabControl::PreferredSize::CURRENT_TAB);
        settings.mouse_panel->AddChild(GiveOwnership(settings.mouse_tab));

        settings.view_panel = new Vert(v_spacing, tabbed_margins);
        settings.pick_panel = new Vert(v_spacing, tabbed_margins);
        settings.mouse_tab->AddTab("Scene", GiveOwnership(settings.view_panel));
        settings.mouse_tab->AddTab("Selection",
                                   GiveOwnership(settings.pick_panel));
        settings.mouse_tab->SetOnSelectedTabChanged([this](int tab_idx) {
            if (tab_idx == 0) {
                SetMouseMode(settings.view_mouse_mode);
            } else {
                SetPicking();
            }
        });

        auto MakeMouseButton = [this](const char *name,
                                      SceneWidget::Controls type) {
            auto button = new SmallToggleButton(name);
            button->SetOnClicked([this, type]() { this->SetMouseMode(type); });
            this->settings.mouse_buttons[type] = button;
            return button;
        };
        auto *h = new Horiz(v_spacing);
        h->AddStretch();
        h->AddChild(GiveOwnership(MakeMouseButton(
                "Arcball", SceneWidget::Controls::ROTATE_CAMERA)));
        h->AddChild(GiveOwnership(
                MakeMouseButton("Fly", SceneWidget::Controls::FLY)));
        h->AddChild(GiveOwnership(
                MakeMouseButton("Model", SceneWidget::Controls::ROTATE_MODEL)));
        h->AddStretch();
        settings.view_panel->AddChild(GiveOwnership(h));

        h = new Horiz(v_spacing);
        h->AddStretch();
        h->AddChild(GiveOwnership(MakeMouseButton(
                "Sun Direction", SceneWidget::Controls::ROTATE_SUN)));
        h->AddChild(GiveOwnership(MakeMouseButton(
                "Environment", SceneWidget::Controls::ROTATE_IBL)));
        h->AddStretch();
        settings.view_panel->AddChild(GiveOwnership(h));
        settings.view_panel->AddFixed(half_em);

        auto *reset = new SmallButton("Reset Camera");
        reset->SetOnClicked([this]() { this->ResetCameraToDefault(); });

        h = new Horiz(v_spacing);
        h->AddStretch();
        h->AddChild(GiveOwnership(reset));
        h->AddStretch();
        settings.view_panel->AddChild(GiveOwnership(h));

        // Selection sets controls
        settings.new_selection_set = new SmallButton(" + ");
        settings.new_selection_set->SetOnClicked(
                [this]() { NewSelectionSet(); });
        settings.delete_selection_set = new SmallButton(" - ");
        settings.delete_selection_set->SetOnClicked([this]() {
            int idx = settings.selection_sets->GetSelectedIndex();
            RemoveSelectionSet(idx);
        });
        settings.selection_sets = new ListView();
        settings.selection_sets->SetOnValueChanged([this](const char *, bool) {
            SelectSelectionSet(settings.selection_sets->GetSelectedIndex());
        });

#if __APPLE__
        const char *selection_help =
                "Cmd-click to select a point\nCmd-ctrl-click to polygon select";
#else
        const char *selection_help =
                "Ctrl-click to select a point\nCmd-alt-click to polygon select";
#endif  // __APPLE__
        h = new Horiz();
        h->AddStretch();
        h->AddChild(std::make_shared<Label>(selection_help));
        h->AddStretch();
        settings.pick_panel->AddChild(GiveOwnership(h));

        h = new Horiz(int(std::round(0.25f * float(em))));
        settings.polygon_selection_panel = h;
        h->AddStretch();
        auto b = std::make_shared<SmallButton>("Select");
        b->SetOnClicked([this]() {
            widget3d_->DoPolygonPick(SceneWidget::PolygonPickAction::SELECT);
            settings.polygon_selection_panel->SetVisible(false);
        });
        h->AddChild(b);
        b = std::make_shared<SmallButton>("Unselect");
        b->SetOnClicked([this]() {
            polygon_selection_unselects_ = true;
            widget3d_->DoPolygonPick(SceneWidget::PolygonPickAction::SELECT);
            settings.polygon_selection_panel->SetVisible(false);
        });
        h->AddChild(b);
        b = std::make_shared<SmallButton>("Cancel");
        b->SetOnClicked([this]() {
            widget3d_->DoPolygonPick(SceneWidget::PolygonPickAction::CANCEL);
            settings.polygon_selection_panel->SetVisible(false);
        });
        h->AddChild(b);
        h->AddStretch();
        h->SetVisible(false);
        settings.pick_panel->AddChild(GiveOwnership(h));

        h = new Horiz(v_spacing);
        h->AddChild(std::make_shared<Label>("Selection Sets"));
        h->AddStretch();
        h->AddChild(GiveOwnership(settings.new_selection_set));
        h->AddChild(GiveOwnership(settings.delete_selection_set));
        settings.pick_panel->AddChild(GiveOwnership(h));
        settings.pick_panel->AddChild(GiveOwnership(settings.selection_sets));

        // Global scene settings
        settings.global.panel =
                new CollapsableVert("Global", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.global.panel));

        settings.global.bg_color = new ColorEdit();
        settings.global.bg_color->SetValue(ui_state_.bg_color.x(),
                                           ui_state_.bg_color.y(),
                                           ui_state_.bg_color.z());
        settings.global.bg_color->SetOnValueChanged([this](const Color &c) {
            this->SetBackground({c.GetRed(), c.GetGreen(), c.GetBlue(), 1.0f},
                                nullptr);
        });

        settings.global.point_size = new Slider(Slider::INT);
        settings.global.point_size->SetLimits(1, 10);
        settings.global.point_size->SetValue(ui_state_.point_size);
        settings.global.point_size->SetOnValueChanged(
                [this](const double newValue) {
                    this->SetPointSize(int(newValue));
                });

        settings.global.shader = new Combobox();
        settings.global.shader->AddItem("Standard");
        settings.global.shader->AddItem("Unlit");
        settings.global.shader->AddItem("Normal map");
        settings.global.shader->AddItem("Depth");
        settings.global.shader->SetOnValueChanged(
                [this](const char *item, int idx) {
                    if (idx == 1) {
                        this->SetShader(O3DVisualizer::Shader::UNLIT);
                    } else if (idx == 2) {
                        this->SetShader(O3DVisualizer::Shader::NORMALS);
                    } else if (idx == 3) {
                        this->SetShader(O3DVisualizer::Shader::DEPTH);
                    } else {
                        this->SetShader(O3DVisualizer::Shader::STANDARD);
                    }
                });

        settings.global.lighting = new Combobox();
        for (auto &profile : gLightingProfiles) {
            settings.global.lighting->AddItem(profile.name.c_str());
        }
        settings.global.lighting->AddItem(kCustomName);
        settings.global.lighting->SetOnValueChanged(
                [this](const char *, int index) {
                    if (index < int(gLightingProfiles.size())) {
                        this->SetLightingProfile(gLightingProfiles[index]);
                    }
                });

        auto *grid = new VGrid(2, v_spacing);
        settings.global.panel->AddChild(GiveOwnership(grid));

        grid->AddChild(std::make_shared<Label>("BG Color"));
        grid->AddChild(GiveOwnership(settings.global.bg_color));
        grid->AddChild(std::make_shared<Label>("PointSize"));
        grid->AddChild(GiveOwnership(settings.global.point_size));
        grid->AddChild(std::make_shared<Label>("Shader"));
        grid->AddChild(GiveOwnership(settings.global.shader));
        grid->AddChild(std::make_shared<Label>("Lighting"));
        grid->AddChild(GiveOwnership(settings.global.lighting));

        // Properties part 1: creation
        settings.property_panels = new StackedWidget();

        settings.empty_panel_idx = 0;
        auto *no_properties_panel = new VGrid(2);
        settings.property_panels->AddChild(GiveOwnership(no_properties_panel));

        // Geometry properties
        settings.geom_properties.info = new Label("");
        settings.geom_properties.shader =
                new Combobox({"Lit", "Unlit", "Normal map", "Depth"});
        settings.geom_properties.shader->SetOnValueChanged(
                [this](const char *, int) {
                    SetCurrentObjectToUserMaterial(kSetShader);
                });
        settings.geom_properties.material = new Combobox();
        for (auto &m : kMaterialProfiles) {
            settings.geom_properties.material->AddItem(m.name.c_str());
        }
        settings.geom_properties.material->SetOnValueChanged(
                [this](const char *, int) {
                    SetCurrentObjectToUserMaterial(kSetMaterial);
                });
        settings.geom_properties.color = new ColorEdit();
        settings.geom_properties.color->SetOnValueChanged(
                [this](const Color &) {
                    SetCurrentObjectToUserMaterial(kSetColor);
                });

        grid = new VGrid(2, v_spacing);
        grid->AddChild(std::make_shared<Label>("Info"));
        grid->AddChild(GiveOwnership(settings.geom_properties.info));
        grid->AddChild(std::make_shared<Label>("Shader"));
        grid->AddChild(GiveOwnership(settings.geom_properties.shader));
        grid->AddChild(std::make_shared<Label>("Material"));
        grid->AddChild(GiveOwnership(settings.geom_properties.material));
        grid->AddChild(std::make_shared<Label>("Color"));
        grid->AddChild(GiveOwnership(settings.geom_properties.color));

        int panel_idx = int(settings.property_panels->GetChildren().size());
        settings.geom_properties.panel_idx = panel_idx;
        settings.geom_properties.panel = grid;
        settings.property_panels->AddChild(GiveOwnership(grid));

        // Scene tree
        // All panels need to have an panel index registered, as panels without
        // one will get the geometry panel.
        settings.scene.panel = new CollapsableVert("Scene", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.scene.panel));
        settings.scene.entities = new TreeView();
        settings.scene.panel->AddChild(GiveOwnership(settings.scene.entities));

        settings.scene.entities->SetOnSelectionChanged(
                [this](TreeView::ItemId item_id) {
                    UpdatePropertyPanel(item_id);
                });

        // ... local functions to add properties
        auto add_no_properties = [this, no_properties_panel](
                                         PropertyPanel &props,
                                         TreeView::ItemId item_id) {
            auto panel_idx = settings.empty_panel_idx;
            props.panel_idx = panel_idx;
            props.panel = no_properties_panel;
            settings.scene.id2panelidx[item_id] = panel_idx;
        };
        auto add_properties = [this](PropertyPanel &props,
                                     TreeView::ItemId item_id,
                                     std::shared_ptr<VGrid> grid) {
            auto panel_idx =
                    int(settings.property_panels->GetChildren().size());
            props.panel_idx = panel_idx;
            props.panel = grid.get();                  // does not own
            settings.property_panels->AddChild(grid);  // takes ownership
            settings.scene.id2panelidx[item_id] = panel_idx;
        };

        auto root = settings.scene.entities->GetRootItem();
        // ... skybox
        auto scene_item = std::make_shared<CheckableTextTreeCell>(
                "Skybox", false,
                [this](bool is_checked) { this->ShowSkybox(is_checked); });
        settings.skybox.show = scene_item->GetCheckbox().get();
        auto item_id = settings.scene.entities->AddItem(root, scene_item);
        add_no_properties(settings.skybox.properties, item_id);

        // ... axes
        scene_item = std::make_shared<CheckableTextTreeCell>(
                "Axes", false,
                [this](bool is_checked) { this->ShowAxes(is_checked); });
        settings.axes.show = scene_item->GetCheckbox().get();
        item_id = settings.scene.entities->AddItem(root, scene_item);
        add_no_properties(settings.axes.properties, item_id);

        // ... ground
        scene_item = std::make_shared<CheckableTextTreeCell>(
                "Ground", false,
                [this](bool is_checked) { this->ShowGround(is_checked); });
        settings.ground.show = scene_item->GetCheckbox().get();
        item_id = settings.scene.entities->AddItem(root, scene_item);

        // ... ground properties
        settings.ground.properties.ground_plane = new Combobox();
        settings.ground.properties.ground_plane->AddItem("XZ");
        settings.ground.properties.ground_plane->AddItem("XY");
        settings.ground.properties.ground_plane->AddItem("YZ");
        settings.ground.properties.ground_plane->SetOnValueChanged(
                [this](const char *item, int idx) {
                    if (idx == 1) {
                        ui_state_.ground_plane =
                                rendering::Scene::GroundPlane::XY;
                    } else if (idx == 2) {
                        ui_state_.ground_plane =
                                rendering::Scene::GroundPlane::YZ;
                    } else {
                        ui_state_.ground_plane =
                                rendering::Scene::GroundPlane::XZ;
                    }
                    this->ShowGround(ui_state_.show_ground);
                });

        grid = new VGrid(2, v_spacing);
        grid->AddChild(std::make_shared<Label>("Ground plane"));
        grid->AddChild(GiveOwnership(settings.ground.properties.ground_plane));

        add_properties(settings.ground.properties, item_id,
                       GiveOwnership(grid));

        // ... ibl
        scene_item = std::make_shared<CheckableTextTreeCell>(
                "Environmental/HDR lighting", false, [this](bool is_checked) {
                    this->ui_state_.use_ibl = is_checked;
                    this->SetUIState(ui_state_);
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        settings.ibl.show = scene_item->GetCheckbox().get();
        settings.ibl.show->SetChecked(ui_state_.use_ibl);
        item_id = settings.scene.entities->AddItem(root, scene_item);

        // ... ibl properties
        grid = new VGrid(2, v_spacing);

        settings.ibl.properties.names = new Combobox();
        for (auto &name : GetListOfIBLs()) {
            settings.ibl.properties.names->AddItem(name.c_str());
        }
        settings.ibl.properties.names->SetSelectedValue(kDefaultIBL.c_str());
        settings.ibl.properties.names->SetOnValueChanged([this](const char *val,
                                                                int idx) {
            std::string resource_path =
                    Application::GetInstance().GetResourcePath();
            this->SetIBL(resource_path + std::string("/") + std::string(val));
            this->settings.global.lighting->SetSelectedValue(kCustomName);
        });
        grid->AddChild(std::make_shared<Label>("HDR map"));
        grid->AddChild(GiveOwnership(settings.ibl.properties.names));

        settings.ibl.properties.intensity = new Slider(Slider::INT);
        settings.ibl.properties.intensity->SetLimits(0.0, 150000.0);
        settings.ibl.properties.intensity->SetValue(ui_state_.ibl_intensity);
        settings.ibl.properties.intensity->SetOnValueChanged(
                [this](double new_value) {
                    this->ui_state_.ibl_intensity = int(new_value);
                    this->SetUIState(ui_state_);
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        grid->AddChild(std::make_shared<Label>("Intensity"));
        grid->AddChild(GiveOwnership(settings.ibl.properties.intensity));

        add_properties(settings.ibl.properties, item_id, GiveOwnership(grid));

        // ... sun
        scene_item = std::make_shared<CheckableTextTreeCell>(
                "Sun (Directional lighting)", false, [this](bool is_checked) {
                    this->ui_state_.use_sun = is_checked;
                    this->SetUIState(ui_state_);
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        settings.sun.show = scene_item->GetCheckbox().get();
        settings.sun.show->SetChecked(ui_state_.use_sun);
        item_id = settings.scene.entities->AddItem(root, scene_item);

        // ... sun properties
        grid = new VGrid(2, v_spacing);

        settings.sun.properties.intensity = new Slider(Slider::INT);
        settings.sun.properties.intensity->SetLimits(0.0, 200000.0);
        settings.sun.properties.intensity->SetValue(ui_state_.ibl_intensity);
        settings.sun.properties.intensity->SetOnValueChanged(
                [this](double new_value) {
                    this->ui_state_.sun_intensity = int(new_value);
                    this->SetUIState(ui_state_);
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        grid->AddChild(std::make_shared<Label>("Intensity"));
        grid->AddChild(GiveOwnership(settings.sun.properties.intensity));

        settings.sun.properties.dir = new VectorEdit();
        settings.sun.properties.dir->SetValue(ui_state_.sun_dir);
        settings.sun.properties.dir->SetOnValueChanged(
                [this](const Eigen::Vector3f &dir) {
                    this->ui_state_.sun_dir = dir;
                    this->SetUIState(ui_state_);
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        widget3d_->SetOnSunDirectionChanged(
                [this](const Eigen::Vector3f &new_dir) {
                    this->ui_state_.sun_dir = new_dir;
                    this->settings.sun.properties.dir->SetValue(new_dir);
                    // Don't need to call SetUIState(), the SceneWidget already
                    // modified the scene.
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        grid->AddChild(std::make_shared<Label>("Direction"));
        grid->AddChild(GiveOwnership(settings.sun.properties.dir));

        settings.sun.properties.follows_camera = new Checkbox("Follows camera");
        settings.sun.properties.follows_camera->SetChecked(
                ui_state_.sun_follows_camera);
        settings.sun.properties.follows_camera->SetOnChecked(
                [this](bool checked) {
                    auto new_state =
                            this->ui_state_;  // need to copy or SetUIState
                    new_state.sun_follows_camera =
                            checked;  // will detect a change
                    this->SetUIState(new_state);
                });
        grid->AddChild(std::make_shared<Label>(" "));
        grid->AddChild(GiveOwnership(settings.sun.properties.follows_camera));

        settings.sun.properties.color = new ColorEdit();
        settings.sun.properties.color->SetValue(ui_state_.sun_color);
        settings.sun.properties.color->SetOnValueChanged(
                [this](const Color &new_color) {
                    this->ui_state_.sun_color = {new_color.GetRed(),
                                                 new_color.GetGreen(),
                                                 new_color.GetBlue()};
                    this->SetUIState(ui_state_);
                    this->settings.global.lighting->SetSelectedValue(
                            kCustomName);
                });
        grid->AddChild(std::make_shared<Label>("Color"));
        grid->AddChild(GiveOwnership(settings.sun.properties.color));

        add_properties(settings.sun.properties, item_id, GiveOwnership(grid));

        // Properties part 2: add to the UI
        settings.property_parent_panel =
                new CollapsableVert("Properties", v_spacing, margins);
        settings.property_parent_panel->AddChild(
                GiveOwnership(settings.property_panels));
        settings.panel->AddChild(GiveOwnership(settings.property_parent_panel));

#if !GROUPS_USE_TREE
        // Groups
        settings.groups_panel =
                new EmptyIfHiddenVert("Groups", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.groups_panel));

        settings.groups = new TreeView();
        settings.groups_panel->AddChild(GiveOwnership(settings.groups));

        settings.groups_panel->SetVisible(false);
#endif  // !GROUPS_USE_TREE

        // Time controls
        settings.time_panel = new EmptyIfHiddenVert("Time", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.time_panel));

        settings.time_slider = new Slider(Slider::DOUBLE);
        settings.time_slider->SetOnValueChanged([this](double new_value) {
            this->ui_state_.current_time = new_value;
            this->UpdateTimeUI();
            this->SetCurrentTime(new_value);
        });

        settings.time_edit = new NumberEdit(NumberEdit::DOUBLE);
        settings.time_edit->SetOnValueChanged([this](double new_value) {
            this->ui_state_.current_time = new_value;
            this->UpdateTimeUI();
            this->SetCurrentTime(new_value);
        });

        settings.play = new SmallToggleButton("Play");
        settings.play->SetOnClicked(
                [this]() { this->SetAnimating(settings.play->GetIsOn()); });

        h = new Horiz(v_spacing);
        h->AddChild(GiveOwnership(settings.time_slider));
        h->AddChild(GiveOwnership(settings.time_edit));
        h->AddChild(GiveOwnership(settings.play));
        settings.time_panel->AddChild(GiveOwnership(h));

        settings.time_panel->SetVisible(false);  // hide until we add a
                                                 // geometry with time

        // Picking callbacks
        widget3d_->SetOnStartedPolygonPicking([this]() {
            settings.polygon_selection_panel->SetVisible(true);
        });
    }

    void UpdatePropertyPanel(TreeView::ItemId selected_id) {
        auto it = settings.scene.id2panelidx.find(selected_id);
        if (it != settings.scene.id2panelidx.end()) {
            settings.property_panels->SetSelectedIndex(it->second);
            auto item = settings.scene.entities->GetItem(selected_id);
            auto checkable =
                    std::dynamic_pointer_cast<CheckableTextTreeCell>(item);
            if (checkable) {
                bool enabled = checkable->GetCheckbox()->IsChecked();
                settings.property_panels->GetChildren()[it->second]->SetEnabled(
                        enabled);
            }
        } else {
            settings.property_panels->SetSelectedIndex(
                    settings.geom_properties.panel_idx);
            UpdateGeometryPropertyPanel();
        }
    }

    void UpdatePropertyPanelEnabled(PropertyPanel &props, bool enabled) {
        if (settings.property_panels->GetSelectedIndex() == props.panel_idx) {
            // This property is currently showing, so updated enabledness
            props.panel->SetEnabled(enabled);
        }
    }

    /*    void AddModel(const std::string& model_name,
                      std::shared_ptr<rendering::TriangleMeshModel> model,
                      const std::string& group,
                      double time,
                      bool is_visible) {
            std::string new_group;
            if (model->meshes_.size() == 1) {
                auto &m = model->meshes_[0];
                auto debug = model->materials_[m.material_idx];
                std::cout << "[o3d] shader: " << debug.shader << ", base_color:
       {" << debug.base_color.x() << ", " << debug.base_color.y() << ", " <<
       debug.base_color.z() << ", " << debug.base_color.w() << "}" << std::endl;
                AddGeometry(model_name, m.mesh, nullptr,
                            &model->materials_[m.material_idx], group, time,
                            is_visible);
            } else {
                new_group = group;
                if (!new_group.empty()) {
                    new_group += "/";
                }
                new_group += model_name;
                for (auto &m : model->meshes_) {
                    AddGeometry(m.mesh_name, m.mesh, nullptr,
                                &model->materials_[m.material_idx], new_group,
       time, is_visible);
                }
            }
        }
    */
    void AddGeometry(const std::string &name,
                     std::shared_ptr<geometry::Geometry3D> geom,
                     std::shared_ptr<t::geometry::Geometry> tgeom,
                     std::shared_ptr<rendering::TriangleMeshModel> model,
                     const rendering::Material *material,
                     const std::string &group,
                     double time,
                     bool is_visible) {
        std::string group_name = group;
        if (group_name == "") {
            group_name = "default";
        }
        bool is_default_color = false;
        bool no_shadows = false;
        bool has_colors = false;
        bool has_normals = false;

        auto cloud = std::dynamic_pointer_cast<geometry::PointCloud>(geom);
        auto lines = std::dynamic_pointer_cast<geometry::LineSet>(geom);
        auto obb =
                std::dynamic_pointer_cast<geometry::OrientedBoundingBox>(geom);
        auto aabb = std::dynamic_pointer_cast<geometry::AxisAlignedBoundingBox>(
                geom);
        auto mesh = std::dynamic_pointer_cast<geometry::MeshBase>(geom);
        auto voxel_grid = std::dynamic_pointer_cast<geometry::VoxelGrid>(geom);
        auto octree = std::dynamic_pointer_cast<geometry::Octree>(geom);

        auto t_cloud =
                std::dynamic_pointer_cast<t::geometry::PointCloud>(tgeom);
        auto t_mesh =
                std::dynamic_pointer_cast<t::geometry::TriangleMesh>(tgeom);

        if (cloud) {
            has_colors = !cloud->colors_.empty();
            has_normals = !cloud->normals_.empty();
        } else if (t_cloud) {
            has_colors = t_cloud->HasPointColors();
            has_normals = t_cloud->HasPointNormals();
        } else if (lines) {
            has_colors = !lines->colors_.empty();
            no_shadows = true;
        } else if (obb) {
            has_colors = (obb->color_ != Eigen::Vector3d{0.0, 0.0, 0.0});
            no_shadows = true;
        } else if (aabb) {
            has_colors = (aabb->color_ != Eigen::Vector3d{0.0, 0.0, 0.0});
            no_shadows = true;
        } else if (mesh) {
            has_normals = !mesh->vertex_normals_.empty();
            has_colors = true;  // always want base_color as white
        } else if (t_mesh) {
            has_normals = !t_mesh->HasVertexNormals();
            has_colors = true;  // always want base_color as white
        } else if (voxel_grid) {
            has_normals = false;
            has_colors = voxel_grid->HasColors();
        } else if (octree) {
            has_normals = false;
            has_colors = true;
        }

        Material mat;
        if (material) {
            mat = *material;
            is_default_color = false;
            no_shadows = false;
        } else if (model) {
            mat.shader = "defaultLit";
        } else {
            mat.base_color = CalcDefaultUnlitColor();
            mat.shader = kShaderUnlit;
            if (lines || obb || aabb) {
                mat.shader = kShaderUnlitLines;
                mat.line_width = ui_state_.line_width * window_->GetScaling();
            }
            is_default_color = true;
            if (has_colors) {
                mat.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
                is_default_color = false;
            }
            if (has_normals) {
                mat.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
                mat.shader = kShaderLit;
                is_default_color = false;
            }
            mat.point_size = ConvertToScaledPixels(ui_state_.point_size);
        }

        // We assume that the caller isn't setting a group or time (and in any
        // case we don't know beforehand what they will do). So if they do,
        // we need to update the geometry tree accordingly. This needs to happen
        // before we add the object to the list, otherwise when we regenerate
        // the object will already be added in the list and then get added again
        // below.
        AddGroup(group_name);  // regenerates if necessary
        bool update_for_time = (min_time_ == max_time_ && time != max_time_);
        min_time_ = std::min(min_time_, time);
        max_time_ = std::max(max_time_, time);
        if (time != 0.0) {
            UpdateTimeUIRange();
            settings.time_panel->SetVisible(true);
        }
        if (update_for_time) {
            UpdateObjectTree();
        }
        // Auto-open the settings panel if we set anything fancy that would
        // imply using the UI.
        if (can_auto_show_settings_ &&
            (added_groups_.size() == 2 || update_for_time)) {
            ShowSettings(true);
        }

        objects_.push_back({name, geom, tgeom, model, mat, group_name, time,
                            is_visible, has_normals, has_colors,
                            is_default_color});
        AddObjectToTree(objects_.back());

        auto scene = widget3d_->GetScene();
        // Do we have a geometry, tgeometry or model?
        if (geom) {
            scene->AddGeometry(name, geom.get(), mat);
        } else if (t_cloud) {
            scene->AddGeometry(name, t_cloud.get(), mat);
        } else if (model) {
            scene->AddModel(name, *model);
        } else {
            utility::LogWarning(
                    "No valid geometry specified to O3DVisualizer. Only "
                    "supported "
                    "geometries are Geometry3D and TGeometry PointClouds.");
        }

        if (no_shadows) {
            scene->GetScene()->GeometryShadows(name, false, false);
        }
        UpdateGeometryVisibility(objects_.back());

        // Bounds have changed, so update the selection point size, since they
        // depend on the bounds.
        SetPointSize(ui_state_.point_size);

        widget3d_->ForceRedraw();
    }

    void RemoveGeometry(const std::string &name) {
        std::string group;
        for (size_t i = 0; i < objects_.size(); ++i) {
            if (objects_[i].name == name) {
                group = objects_[i].group;
                objects_.erase(objects_.begin() + i);
                settings.object2itemid.erase(objects_[i].name);
                break;
            }
        }

        // Need to check group membership in case this was the last item in its
        // group. As long as we're doing that, recompute the min/max time, too.
        std::set<std::string> groups;
        min_time_ = max_time_ = 0.0;
        for (size_t i = 0; i < objects_.size(); ++i) {
            auto &o = objects_[i];
            min_time_ = std::min(min_time_, o.time);
            max_time_ = std::max(max_time_, o.time);
            groups.insert(o.group);
        }
        if (min_time_ == max_time_) {
            SetAnimating(false);
        }
        UpdateTimeUIRange();

        added_groups_ = groups;
        std::set<std::string> enabled;
        for (auto &g : ui_state_.enabled_groups) {  // remove deleted groups
            if (groups.find(g) != groups.end()) {
                enabled.insert(g);
            }
        }
        ui_state_.enabled_groups = enabled;
        UpdateObjectTree();
        widget3d_->GetScene()->RemoveGeometry(name);

        // Bounds have changed, so update the selection point size, since they
        // depend on the bounds.
        SetPointSize(ui_state_.point_size);

        widget3d_->ForceRedraw();
    }

    void ClearGeometry() {
        widget3d_->GetScene()->ClearGeometry();
        objects_.clear();
        added_names_.clear();
        added_groups_.clear();
        ui_state_.enabled_groups.clear();
        UpdateObjectTree();
        SetAnimating(false);
        UpdateTimeUIRange();
        int nsets = selections_->GetNumberOfSets();
        for (int i = 0; i < nsets; ++i) {
            selections_->RemoveSet(nsets - 1 - i);
        }
    }

    void ShowGeometry(const std::string &name, bool show) {
        for (auto &o : objects_) {
            if (o.name == name) {
                if (show != o.is_visible) {
                    o.is_visible = show;

                    auto id = settings.object2itemid[o.name];
                    auto cell = settings.scene.entities->GetItem(id);
                    auto obj_cell =
                            std::dynamic_pointer_cast<DrawObjectTreeCell>(cell);
                    if (obj_cell) {
                        obj_cell->GetCheckbox()->SetChecked(show);
                    }

                    UpdateGeometryPropertyPanel();
                    UpdateGeometryVisibility(o);  // calls ForceRedraw()
                    window_->PostRedraw();

                    if (selections_->IsActive()) {
                        UpdateSelectableGeometry();
                    } else {
                        selections_need_update_ = true;
                    }
                }
                break;
            }
        }
    }

    O3DVisualizer::DrawObject GetGeometry(const std::string &name) const {
        for (auto &o : objects_) {
            if (o.name == name) {
                return o;
            }
        }
        return DrawObject();
    }

    enum SetMaterialMode { kSetShader, kSetMaterial, kSetColor };
    void SetCurrentObjectToUserMaterial(SetMaterialMode mode) {
        auto selected_id = settings.scene.entities->GetSelectedItemId();
        auto cell = settings.scene.entities->GetItem(selected_id);
        auto obj_cell = std::dynamic_pointer_cast<DrawObjectTreeCell>(cell);
        if (!obj_cell) {
            return;
        }
        auto obj_name = obj_cell->GetName()->GetText();
        for (auto &o : objects_) {
            if (o.name == obj_name) {
                if (mode == kSetShader) {
                    auto shader_idx =
                            settings.geom_properties.shader->GetSelectedIndex();
                    switch (shader_idx) {
                        case 0:
                            o.material.shader = "defaultLit";
                            break;
                        case 1:
                        case 2:
                        case 3:
                            o.material.shader =
                                    GetShaderString(Shader(shader_idx));
                            break;
                        default:
                            break;
                    }
                } else if (mode == kSetMaterial) {
                    int idx = settings.geom_properties.material
                                      ->GetSelectedIndex();
                    if (idx < 0 || size_t(idx) >= kMaterialProfiles.size()) {
                        return;
                    }
                    o.material = kMaterialProfiles[idx].material;
                    settings.geom_properties.color->SetValue(Color(
                            o.material.base_color[0], o.material.base_color[1],
                            o.material.base_color[2], 1.0f));
                } else {
                    auto c = settings.geom_properties.color->GetValue();
                    o.material.base_color = {c.GetRed(), c.GetGreen(),
                                             c.GetBlue(), 1.0f};
                }
                widget3d_->GetScene()->GetScene()->OverrideMaterial(obj_name,
                                                                    o.material);
                widget3d_->ForceRedraw();
                break;
            }
        }
    }

    void UpdateGeometryPropertyPanel() {
        auto selected_id = settings.scene.entities->GetSelectedItemId();
        auto cell = settings.scene.entities->GetItem(selected_id);
        auto obj_cell = std::dynamic_pointer_cast<DrawObjectTreeCell>(cell);
        if (!obj_cell) {
            return;
        }
        auto obj_name = obj_cell->GetName()->GetText();
        for (auto &o : objects_) {
            if (o.name == obj_name) {
                if (o.model) {
                    auto n_meshes = o.model->meshes_.size();
                    auto info = std::string("Model with ") +
                                std::to_string(n_meshes) + " mesh";
                    if (n_meshes != 1) {
                        info += "es";
                    }
                    settings.geom_properties.info->SetText(info.c_str());
                } else {
                    std::string info;
                    if (o.has_normals) {
                        info += "Has ";
                    } else {
                        info += "No ";
                    }
                    info += "normals, ";
                    if (o.has_colors) {
                        info += "has ";
                    } else {
                        info += "no ";
                    }
                    info += "colors";
                    settings.geom_properties.info->SetText(info.c_str());
                }

                if (o.material.shader == "defaultLit") {
                    settings.geom_properties.shader->SetSelectedIndex(0);
                } else if (o.material.shader == "defaultUnlit") {
                    settings.geom_properties.shader->SetSelectedIndex(1);
                } else if (o.material.shader == "normals") {
                    settings.geom_properties.shader->SetSelectedIndex(2);
                } else if (o.material.shader == "depth") {
                    settings.geom_properties.shader->SetSelectedIndex(3);
                } else {
                    utility::LogWarning("TODO: handle shaders like lines");
                }
                auto &c = o.material.base_color;
                settings.geom_properties.color->SetValue(
                        Color(c[0], c[1], c[2], 1.0f));
                settings.geom_properties.panel->SetEnabled(o.is_visible);
                // If the geometry has colors, setting the material modulates
                // the vertex colors, which seems counter-intuitive.
                settings.geom_properties.color->SetEnabled(!o.has_colors &&
                                                           o.is_visible);
                break;
            }
        }
    }

    void Add3DLabel(const Eigen::Vector3f &pos, const char *text) {
        widget3d_->AddLabel(pos, text);
    }

    void Clear3DLabels() { widget3d_->ClearLabels(); }

    void SetupCamera(float fov,
                     const Eigen::Vector3f &center,
                     const Eigen::Vector3f &eye,
                     const Eigen::Vector3f &up) {
        widget3d_->LookAt(center, eye, up);
        widget3d_->ForceRedraw();
    }

    void SetupCamera(const camera::PinholeCameraIntrinsic &intrinsic,
                     const Eigen::Matrix4d &extrinsic) {
        widget3d_->SetupCamera(intrinsic, extrinsic,
                               widget3d_->GetScene()->GetBoundingBox());
        widget3d_->ForceRedraw();
    }

    void SetupCamera(const Eigen::Matrix3d &intrinsic,
                     const Eigen::Matrix4d &extrinsic,
                     int intrinsic_width_px,
                     int intrinsic_height_px) {
        widget3d_->SetupCamera(intrinsic, extrinsic, intrinsic_width_px,
                               intrinsic_height_px,
                               widget3d_->GetScene()->GetBoundingBox());
        widget3d_->ForceRedraw();
    }

    void ResetCameraToDefault() {
        auto scene = widget3d_->GetScene();
        widget3d_->SetupCamera(60.0f, scene->GetBoundingBox(),
                               {0.0f, 0.0f, 0.0f});
        widget3d_->ForceRedraw();
    }

    void SetBackground(const Eigen::Vector4f &bg_color,
                       std::shared_ptr<geometry::Image> bg_image) {
        auto old_default_color = CalcDefaultUnlitColor();
        ui_state_.bg_color = bg_color;
        auto scene = widget3d_->GetScene();
        scene->SetBackground(ui_state_.bg_color, bg_image);

        auto new_default_color = CalcDefaultUnlitColor();
        if (new_default_color != old_default_color) {
            for (auto &o : objects_) {
                if (o.is_color_default) {
                    o.material.base_color = new_default_color;
                    OverrideMaterial(o.name, o.material,
                                     ui_state_.scene_shader);
                }
            }
        }

        widget3d_->ForceRedraw();
    }

    void ShowSettings(bool show, bool cancel_auto_show = true) {
        if (cancel_auto_show) {
            can_auto_show_settings_ = false;
        }
        ui_state_.show_settings = show;
        settings.panel->SetVisible(show);
        auto menubar = Application::GetInstance().GetMenubar();
        if (menubar) {  // might not have been created yet
            menubar->SetChecked(MENU_SETTINGS, show);
        }
        window_->SetNeedsLayout();
    }

    void ShowSkybox(bool show) {
        ui_state_.show_skybox = show;
        settings.skybox.show->SetChecked(show);  // in case called manually
        UpdatePropertyPanelEnabled(settings.skybox.properties, show);
        widget3d_->GetScene()->ShowSkybox(show);
        widget3d_->ForceRedraw();
    }

    void ShowAxes(bool show) {
        ui_state_.show_axes = show;
        settings.axes.show->SetChecked(show);  // in case called manually
        UpdatePropertyPanelEnabled(settings.axes.properties, show);
        widget3d_->GetScene()->ShowAxes(show);
        widget3d_->ForceRedraw();
    }

    void ShowGround(bool show) {
        ui_state_.show_ground = show;
        settings.ground.show->SetChecked(show);  // in case called manually
        UpdatePropertyPanelEnabled(settings.ground.properties, show);
        widget3d_->GetScene()->ShowGroundPlane(show, ui_state_.ground_plane);
        widget3d_->ForceRedraw();
    }

    void SetGroundPlane(rendering::Scene::GroundPlane plane) {
        ui_state_.ground_plane = plane;
        if (plane == rendering::Scene::GroundPlane::XZ) {
            settings.ground.properties.ground_plane->SetSelectedIndex(0);
        } else if (plane == rendering::Scene::GroundPlane::XY) {
            settings.ground.properties.ground_plane->SetSelectedIndex(1);
        } else {
            settings.ground.properties.ground_plane->SetSelectedIndex(2);
        }
        // Update ground plane if it is currently showing
        if (ui_state_.show_ground) {
            widget3d_->GetScene()->ShowGroundPlane(ui_state_.show_ground,
                                                   plane);
            widget3d_->ForceRedraw();
        }
    }

    void SetPointSize(int px) {
        ui_state_.point_size = px;
        settings.global.point_size->SetValue(double(px));

        px = int(ConvertToScaledPixels(px));
        for (auto &o : objects_) {
            o.material.point_size = float(px);
            OverrideMaterial(o.name, o.material, ui_state_.scene_shader);
        }
        auto bbox = widget3d_->GetScene()->GetBoundingBox();
        auto xdim = bbox.max_bound_.x() - bbox.min_bound_.x();
        auto ydim = bbox.max_bound_.y() - bbox.min_bound_.z();
        auto zdim = bbox.max_bound_.z() - bbox.min_bound_.y();
        auto psize = double(std::max(5, px)) * 0.000666 *
                     std::max(xdim, std::max(ydim, zdim));
        selections_->SetPointSize(psize);

        widget3d_->SetPickablePointSize(px);
        widget3d_->ForceRedraw();
    }

    void SetLineWidth(int px) {
        ui_state_.line_width = px;

        px = int(ConvertToScaledPixels(px));
        for (auto &o : objects_) {
            o.material.line_width = float(px);
            OverrideMaterial(o.name, o.material, ui_state_.scene_shader);
        }
        widget3d_->ForceRedraw();
    }

    void SetShader(O3DVisualizer::Shader shader) {
        ui_state_.scene_shader = shader;
        for (auto &o : objects_) {
            OverrideMaterial(o.name, o.material, shader);
        }
        widget3d_->ForceRedraw();
    }

    void OverrideMaterial(const std::string &name,
                          const Material &original_material,
                          O3DVisualizer::Shader shader) {
        bool is_lines = (original_material.shader == "unlitLine");
        auto scene = widget3d_->GetScene();
        // Lines are already unlit, so keep using the original shader when in
        // unlit mode so that we can keep the wide lines.
        if (shader == Shader::STANDARD ||
            (shader == Shader::UNLIT && is_lines)) {
            scene->GetScene()->OverrideMaterial(name, original_material);
        } else {
            Material m = original_material;
            m.shader = GetShaderString(shader);
            scene->GetScene()->OverrideMaterial(name, m);
        }
    }

    float ConvertToScaledPixels(int px) {
        return std::round(px * window_->GetScaling());
    }

    const char *GetShaderString(O3DVisualizer::Shader shader) {
        switch (shader) {
            case Shader::STANDARD:
                return nullptr;
            case Shader::UNLIT:
                return "defaultUnlit";
            case Shader::NORMALS:
                return "normals";
            case Shader::DEPTH:
                return "depth";
            default:
                utility::LogWarning(
                        "O3DVisualizer::GetShaderString(): unhandled Shader "
                        "value");
                return nullptr;
        }
    }

    void SetIBL(std::string path) {
        if (path == "") {
            path = std::string(Application::GetInstance().GetResourcePath()) +
                   std::string("/") + std::string(kDefaultIBL);
        }
        if (utility::filesystem::FileExists(path + "_ibl.ktx")) {
            widget3d_->GetScene()->GetScene()->SetIndirectLight(path);
            widget3d_->ForceRedraw();
            ui_state_.ibl_path = path;
        } else if (utility::filesystem::FileExists(path)) {
            if (path.find("_ibl.ktx") == path.size() - 8) {
                ui_state_.ibl_path = path.substr(0, path.size() - 8);
                widget3d_->GetScene()->GetScene()->SetIndirectLight(
                        ui_state_.ibl_path);
                widget3d_->ForceRedraw();
            } else {
                utility::LogWarning(
                        "Could not load IBL path. Filename must be of the form "
                        "'name_ibl.ktx' and be paired with 'name_skybox.ktx'");
            }
        }
    }

    void SetLightingProfile(const LightingProfile &profile) {
        Eigen::Vector3f sun_dir = {0.577f, -0.577f, -0.577f};
        auto scene = widget3d_->GetScene();
        scene->SetLighting(profile.profile, sun_dir);
        ui_state_.use_ibl =
                (profile.profile != Open3DScene::LightingProfile::HARD_SHADOWS);
        ui_state_.use_sun =
                (profile.profile != Open3DScene::LightingProfile::NO_SHADOWS);
        ui_state_.ibl_intensity =
                int(scene->GetScene()->GetIndirectLightIntensity());
        ui_state_.sun_intensity =
                int(scene->GetScene()->GetSunLightIntensity());
        ui_state_.sun_dir = sun_dir;
        ui_state_.sun_color = {1.0f, 1.0f, 1.0f};
        SetUIState(ui_state_);
        // SetUIState will set the combobox to "Custom", so undo that.
        this->settings.global.lighting->SetSelectedValue(profile.name.c_str());
    }

    void SetMouseMode(SceneWidget::Controls mode) {
        if (selections_->IsActive()) {
            selections_->MakeInactive();
        }

        widget3d_->SetViewControls(mode);
        ui_state_.mouse_mode = mode;
        settings.view_mouse_mode = mode;
        for (const auto &t_b : settings.mouse_buttons) {
            t_b.second->SetOn(false);
        }
        auto it = settings.mouse_buttons.find(mode);
        if (it != settings.mouse_buttons.end()) {
            it->second->SetOn(true);
        }
    }

    void SetPicking() {
        if (selections_->GetNumberOfSets() == 0) {
            NewSelectionSet();
        }
        if (selections_need_update_) {
            UpdateSelectableGeometry();
        }
        selections_->MakeActive();
    }

    std::vector<O3DVisualizerSelections::SelectionSet> GetSelectionSets()
            const {
        return selections_->GetSets();
    }

    void SetCurrentTime(double t) {
        ui_state_.current_time = t;
        if (ui_state_.current_time > max_time_) {
            ui_state_.current_time = min_time_;
        }
        for (auto &o : objects_) {
            UpdateGeometryVisibility(o);
        }
        UpdateTimeUI();

        if (on_animation_) {
            on_animation_(ui_state_.current_time);
        }
    }

    void SetAnimating(bool is_animating) {
        if (is_animating == ui_state_.is_animating) {
            return;
        }

        ui_state_.is_animating = is_animating;
        if (is_animating) {
            ui_state_.current_time = max_time_;
            auto now = Application::GetInstance().Now();
            start_animation_clock_time_ = now;
            last_animation_tick_clock_time_ = now;
            if (on_animation_tick_) {
                window_->SetOnTickEvent(on_animation_tick_);
            } else {
                window_->SetOnTickEvent(
                        [this]() -> bool { return this->OnAnimationTick(); });
            }
        } else {
            window_->SetOnTickEvent(nullptr);
            SetCurrentTime(0.0);
            next_animation_tick_clock_time_ = 0.0;
        }
        settings.time_slider->SetEnabled(!is_animating);
        settings.time_edit->SetEnabled(!is_animating);
    }

    void SetOnAnimationTick(
            O3DVisualizer &o3dvis,
            std::function<TickResult(O3DVisualizer &, double, double)> cb) {
        if (cb) {
            on_animation_tick_ = [this, &o3dvis, cb]() -> bool {
                auto now = Application::GetInstance().Now();
                auto dt = now - this->last_animation_tick_clock_time_;
                auto total_time = now - this->start_animation_clock_time_;
                this->last_animation_tick_clock_time_ = now;

                auto result = cb(o3dvis, dt, total_time);

                if (result == TickResult::REDRAW) {
                    this->widget3d_->ForceRedraw();
                    return true;
                } else {
                    return false;
                }
            };
        } else {
            on_animation_tick_ = nullptr;
        }
    }

    void SetUIState(const UIState &new_state) {
        int point_size_changed = (new_state.point_size != ui_state_.point_size);
        int line_width_changed = (new_state.line_width != ui_state_.line_width);
        bool ibl_path_changed = (new_state.ibl_path != ui_state_.ibl_path);
        auto old_enabled_groups = ui_state_.enabled_groups;
        bool old_is_animating = ui_state_.is_animating;
        bool new_is_animating = new_state.is_animating;
        bool sun_follows_cam_changed =
                (new_state.sun_follows_camera != ui_state_.sun_follows_camera);
        bool is_new_lighting =
                (ibl_path_changed || new_state.use_ibl != ui_state_.use_ibl ||
                 new_state.use_sun != ui_state_.use_sun ||
                 new_state.ibl_intensity != ui_state_.ibl_intensity ||
                 new_state.sun_intensity != ui_state_.sun_intensity ||
                 new_state.sun_dir != ui_state_.sun_dir ||
                 new_state.sun_color != ui_state_.sun_color ||
                 sun_follows_cam_changed);

        if (&new_state != &ui_state_) {
            ui_state_ = new_state;
        }

        if (ibl_path_changed) {
            SetIBL(ui_state_.ibl_path);
        }

        ShowSettings(ui_state_.show_settings, false);
        SetShader(ui_state_.scene_shader);
        SetBackground(ui_state_.bg_color, nullptr);
        ShowSkybox(ui_state_.show_skybox);
        ShowAxes(ui_state_.show_axes);
        ShowGround(ui_state_.show_ground);

        if (point_size_changed) {
            SetPointSize(ui_state_.point_size);
        }
        if (line_width_changed) {
            SetLineWidth(ui_state_.line_width);
        }

        settings.ibl.show->SetChecked(ui_state_.use_ibl);
        UpdatePropertyPanelEnabled(settings.ibl.properties, ui_state_.use_ibl);
        settings.sun.show->SetChecked(ui_state_.use_sun);
        UpdatePropertyPanelEnabled(settings.sun.properties, ui_state_.use_sun);
        settings.ibl.properties.intensity->SetValue(ui_state_.ibl_intensity);
        settings.sun.properties.intensity->SetValue(ui_state_.sun_intensity);
        settings.sun.properties.dir->SetValue(ui_state_.sun_dir);
        settings.sun.properties.color->SetValue(ui_state_.sun_color);
        // Re-assign intensity in case it was out of range.
        ui_state_.ibl_intensity =
                settings.ibl.properties.intensity->GetIntValue();
        ui_state_.sun_intensity =
                settings.sun.properties.intensity->GetIntValue();

        if (is_new_lighting) {
            settings.global.lighting->SetSelectedValue(kCustomName);
        }

        if (sun_follows_cam_changed) {
            if (ui_state_.sun_follows_camera) {
                if (settings.view_mouse_mode ==
                    SceneWidget::Controls::ROTATE_SUN) {
                    SetMouseMode(SceneWidget::Controls::ROTATE_CAMERA);
                }
                auto cam = widget3d_->GetScene()->GetCamera();
                auto rscene = widget3d_->GetScene()->GetScene();
                rscene->SetSunLightDirection(cam->GetForwardVector());
            }
            settings.mouse_buttons[SceneWidget::Controls::ROTATE_SUN]
                    ->SetEnabled(!ui_state_.sun_follows_camera);
        }

        auto *raw_scene = widget3d_->GetScene()->GetScene();
        raw_scene->EnableIndirectLight(ui_state_.use_ibl);
        raw_scene->SetIndirectLightIntensity(float(ui_state_.ibl_intensity));
        raw_scene->EnableSunLight(ui_state_.use_sun);
        if (!ui_state_.sun_follows_camera) {
            raw_scene->SetSunLight(ui_state_.sun_dir, ui_state_.sun_color,
                                   float(ui_state_.sun_intensity));
        }

        if (old_enabled_groups != ui_state_.enabled_groups) {
            for (auto &group : added_groups_) {
                bool enabled = (ui_state_.enabled_groups.find(group) !=
                                ui_state_.enabled_groups.end());
                EnableGroup(group, enabled);
            }
        }

        if (old_is_animating != new_is_animating) {
            ui_state_.is_animating = old_is_animating;
            SetAnimating(new_is_animating);
        }

        widget3d_->ForceRedraw();
    }

    void AddGroup(const std::string &group) {
#if GROUPS_USE_TREE
        if (added_groups_.find(group) == added_groups_.end()) {
            added_groups_.insert(group);
            ui_state_.enabled_groups.insert(group);
        }
        if (added_groups_.size() == 2) {
            UpdateObjectTree();
        }
#else
        if (added_groups_.find(group) == added_groups_.end()) {
            added_groups_.insert(group);
            ui_state_.enabled_groups.insert(group);

            auto cell = std::make_shared<CheckableTextTreeCell>(
                    group.c_str(), true, [this, group](bool is_on) {
                        this->EnableGroup(group, is_on);
                    });
            auto root = settings.groups->GetRootItem();
            settings.groups->AddItem(root, cell);
        }
        if (added_groups_.size() >= 2) {
            settings.groups_panel->SetVisible(true);
        }
#endif  // GROUPS_USE_TREE
    }

    void EnableGroup(const std::string &group, bool enable) {
#if GROUPS_USE_TREE
        auto group_it = settings.group2itemid.find(group);
        if (group_it != settings.group2itemid.end()) {
            auto cell = settings.scene.entities->GetItem(group_it->second);
            auto group_cell =
                    std::dynamic_pointer_cast<CheckableTextTreeCell>(cell);
            if (group_cell) {
                group_cell->GetCheckbox()->SetChecked(enable);
            }
        }
#endif  // GROUPS_USE_TREE
        if (enable) {
            ui_state_.enabled_groups.insert(group);
        } else {
            ui_state_.enabled_groups.erase(group);
        }
        for (auto &o : objects_) {
            UpdateGeometryVisibility(o);
        }
    }

    void AddObjectToTree(const DrawObject &o) {
        TreeView::ItemId parent = settings.scene.entities->GetRootItem();
#if GROUPS_USE_TREE
        if (added_groups_.size() >= 2) {
            auto it = settings.group2itemid.find(o.group);
            if (it != settings.group2itemid.end()) {
                parent = it->second;
            } else {
                auto cell = std::make_shared<CheckableTextTreeCell>(
                        o.group.c_str(), true,
                        [this, group = o.group](bool is_on) {
                            this->EnableGroup(group, is_on);
                        });
                parent = settings.scene.entities->AddItem(parent, cell);
                settings.group2itemid[o.group] = parent;
            }
        }
#endif  // GROUPS_USE_TREE

        int flag = DrawObjectTreeCell::FLAG_NONE;
#if !GROUPS_USE_TREE
        flag |= (added_groups_.size() >= 2 ? DrawObjectTreeCell::FLAG_GROUP
                                           : 0);
#endif  // !GROUPS_USE_TREE
        flag |= (min_time_ != max_time_ ? DrawObjectTreeCell::FLAG_TIME : 0);
        auto cell = std::make_shared<DrawObjectTreeCell>(
                o.name.c_str(), o.group.c_str(), o.time, o.is_visible, flag,
                [this, name = o.name](bool is_on) {
                    ShowGeometry(name, is_on);
                });
        auto id = settings.scene.entities->AddItem(parent, cell);
        settings.object2itemid[o.name] = id;
    }

    void UpdateObjectTree() {
        for (auto &kv : settings.object2itemid) {
            settings.scene.entities->RemoveItem(kv.second);
        }

#if GROUPS_USE_TREE
        settings.group2itemid.clear();
#endif  // GROUPS_USE_TREE
        settings.object2itemid.clear();

        for (auto &o : objects_) {
            AddObjectToTree(o);
        }
    }

    void UpdateTimeUIRange() {
        bool enabled = (min_time_ < max_time_);
        settings.time_slider->SetEnabled(enabled);
        settings.time_edit->SetEnabled(enabled);
        settings.play->SetEnabled(enabled);

        settings.time_slider->SetLimits(min_time_, max_time_);
        ui_state_.current_time = std::min(
                max_time_, std::max(min_time_, ui_state_.current_time));
        UpdateTimeUI();
    }

    void UpdateTimeUI() {
        settings.time_slider->SetValue(ui_state_.current_time);
        settings.time_edit->SetValue(ui_state_.current_time);
    }

    void UpdateGeometryVisibility(const DrawObject &o) {
        widget3d_->GetScene()->ShowGeometry(o.name, IsGeometryVisible(o));
        widget3d_->ForceRedraw();
    }

    bool IsGeometryVisible(const DrawObject &o) {
        bool is_current =
                (o.time >= ui_state_.current_time &&
                 o.time < ui_state_.current_time + ui_state_.time_step);
        bool is_group_enabled = (ui_state_.enabled_groups.find(o.group) !=
                                 ui_state_.enabled_groups.end());
        bool is_visible = o.is_visible;
        return (is_visible & is_current & is_group_enabled);
    }

    void NewSelectionSet() {
        selections_->NewSet();
        UpdateSelectionSetList();
        SelectSelectionSet(int(selections_->GetNumberOfSets()) - 1);
    }

    void RemoveSelectionSet(int index) {
        selections_->RemoveSet(index);
        if (selections_->GetNumberOfSets() == 0) {
            // You can remove the last set, but there must always be one
            // set, so we re-create one. (So removing the last set has the
            // effect of clearing it.)
            selections_->NewSet();
        }
        UpdateSelectionSetList();
    }

    void SelectSelectionSet(int index) {
        settings.selection_sets->SetSelectedIndex(index);
        selections_->SelectSet(index);
        widget3d_->ForceRedraw();  // redraw with new selection highlighted
    }

    void UpdateSelectionSetList() {
        size_t n = selections_->GetNumberOfSets();
        int idx = settings.selection_sets->GetSelectedIndex();
        idx = std::max(0, idx);
        idx = std::min(idx, int(n) - 1);

        std::vector<std::string> items;
        items.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            std::stringstream s;
            s << "Set " << (i + 1);
            items.push_back(s.str());
        }
        settings.selection_sets->SetItems(items);
        SelectSelectionSet(idx);
        window_->PostRedraw();
    }

    void UpdateSelectableGeometry() {
        std::vector<SceneWidget::PickableGeometry> pickable;
        pickable.reserve(objects_.size());
        for (auto &o : objects_) {
            if (!IsGeometryVisible(o)) {
                continue;
            }
            pickable.emplace_back(o.name, o.geometry.get(), o.tgeometry.get());
        }
        selections_->SetSelectableGeometry(pickable);
    }

    void OnCameraChanged(rendering::Camera *camera) {
        if (ui_state_.sun_follows_camera) {
            auto rendering_scene = widget3d_->GetScene()->GetScene();
            rendering_scene->SetSunLightDirection(camera->GetForwardVector());
        }
        if (camera_info_panel_->IsVisible()) {
            camera_info_panel_->SetFromCamera(camera);
        }
    }

    bool OnAnimationTick() {
        auto now = Application::GetInstance().Now();
        if (now >= next_animation_tick_clock_time_) {
            SetCurrentTime(ui_state_.current_time + ui_state_.time_step);
            UpdateAnimationTickClockTime(now);

            return true;
        }
        return false;
    }

    void UpdateAnimationTickClockTime(double now) {
        next_animation_tick_clock_time_ = now + ui_state_.frame_delay;
    }

    void ExportCurrentImage(const std::string &path) {
        widget3d_->EnableSceneCaching(false);
        widget3d_->GetScene()->GetScene()->RenderToImage(
                [this, path](std::shared_ptr<geometry::Image> image) mutable {
                    if (!io::WriteImage(path, *image)) {
                        this->window_->ShowMessageBox(
                                "Error",
                                (std::string("Could not write image to ") +
                                 path + ".")
                                        .c_str());
                    }
                    widget3d_->EnableSceneCaching(true);
                });
    }

    void OnExportRGB() {
        auto dlg = std::make_shared<gui::FileDialog>(
                gui::FileDialog::Mode::SAVE, "Save File", window_->GetTheme());
        dlg->AddFilter(".png", "PNG images (.png)");
        dlg->AddFilter("", "All files");
        dlg->SetOnCancel([this]() { this->window_->CloseDialog(); });
        dlg->SetOnDone([this](const char *path) {
            this->window_->CloseDialog();
            this->ExportCurrentImage(path);
        });
        window_->ShowDialog(dlg);
    }

    void OnClose() { window_->Close(); }

    void OnToggleSettings() { ShowSettings(!ui_state_.show_settings); }

    void OnAbout() {
        auto &theme = window_->GetTheme();
        auto dlg = std::make_shared<gui::Dialog>("About");

        auto title = std::make_shared<gui::Label>(
                (std::string("Open3D ") + OPEN3D_VERSION).c_str());
        auto text = std::make_shared<gui::Label>(
                "The MIT License (MIT)\n"
                "Copyright (c) 2018 - 2020 www.open3d.org\n\n"

                "Permission is hereby granted, free of charge, to any person "
                "obtaining a copy of this software and associated "
                "documentation "
                "files (the \"Software\"), to deal in the Software without "
                "restriction, including without limitation the rights to use, "
                "copy, modify, merge, publish, distribute, sublicense, and/or "
                "sell copies of the Software, and to permit persons to whom "
                "the Software is furnished to do so, subject to the following "
                "conditions:\n\n"

                "The above copyright notice and this permission notice shall "
                "be included in all copies or substantial portions of the "
                "Software.\n\n"

                "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY "
                "KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE "
                "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
                "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS "
                "OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR "
                "OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR "
                "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE "
                "SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.");
        auto ok = std::make_shared<gui::Button>("OK");
        ok->SetOnClicked([this]() { this->window_->CloseDialog(); });

        gui::Margins margins(theme.font_size);
        auto layout = std::make_shared<gui::Vert>(0, margins);
        layout->AddChild(gui::Horiz::MakeCentered(title));
        layout->AddFixed(theme.font_size);
        layout->AddChild(text);
        layout->AddFixed(theme.font_size);
        layout->AddChild(gui::Horiz::MakeCentered(ok));
        dlg->AddChild(layout);

        window_->ShowDialog(dlg);
    }

    void OnContactUs() {
        auto &theme = window_->GetTheme();
        auto em = theme.font_size;
        auto dlg = std::make_shared<gui::Dialog>("Contact Us");

        auto title = std::make_shared<gui::Label>("Contact Us");
        auto left_col = std::make_shared<gui::Label>(
                "Web site:\n"
                "Code:\n"
                "Mailing list:\n"
                "Discord channel:");
        auto right_col = std::make_shared<gui::Label>(
                "http://www.open3d.org\n"
                "http://github.org/intel-isl/Open3D\n"
                "http://www.open3d.org/index.php/subscribe/\n"
                "https://discord.gg/D35BGvn");
        auto ok = std::make_shared<gui::Button>("OK");
        ok->SetOnClicked([this]() { window_->CloseDialog(); });

        gui::Margins margins(em);
        auto layout = std::make_shared<gui::Vert>(0, margins);
        layout->AddChild(gui::Horiz::MakeCentered(title));
        layout->AddFixed(em);

        auto columns = std::make_shared<gui::Horiz>(em, gui::Margins());
        columns->AddChild(left_col);
        columns->AddChild(right_col);
        layout->AddChild(columns);

        layout->AddFixed(em);
        layout->AddChild(gui::Horiz::MakeCentered(ok));
        dlg->AddChild(layout);

        window_->ShowDialog(dlg);
    }

    VGrid *CreateControlsHelp() {
        auto &theme = window_->GetTheme();

        gui::Margins margins(theme.font_size);
        auto layout = new VGrid(2, 0, margins);
        layout->SetBackgroundColor(gui::Color(0, 0, 0, 0.5));

        auto AddLabel = [layout](const char *text) {
            auto label = std::make_shared<gui::Label>(text);
            label->SetTextColor(gui::Color(1, 1, 1));
            layout->AddChild(label);
        };
        auto AddRow = [&AddLabel](const char *left, const char *right) {
            AddLabel(left);
            AddLabel(right);
        };

        AddRow("Arcball mode", " ");
        AddRow("Left-drag", "Rotate camera");
        AddRow("Shift + left-drag", "Forward/backward");

#if defined(__APPLE__)
        AddLabel("Cmd + left-drag");
#else
        AddLabel("Ctrl + left-drag");
#endif  // __APPLE__
        AddLabel("Pan camera");

#if defined(__APPLE__)
        AddLabel("Opt + left-drag (up/down)  ");
#else
        AddLabel("Win + left-drag (up/down)  ");
#endif  // __APPLE__
        AddLabel("Rotate around forward axis");

        // GNOME3 uses Win/Meta as a shortcut to move windows around, so we
        // need another way to rotate around the forward axis.
        AddLabel("Ctrl + Shift + left-drag");
        AddLabel("Rotate around forward axis");

#if defined(__APPLE__)
        AddLabel("Ctrl + left-drag");
#else
        AddLabel("Alt + left-drag");
#endif  // __APPLE__
        AddLabel("Rotate directional light");

        AddRow("Right-drag", "Pan camera");
        AddRow("Middle-drag", "Rotate directional light");
        AddRow("Wheel", "Forward/backward");
        AddRow("Shift + Wheel", "Change field of view");
        AddRow("", "");

        AddRow("Fly mode", " ");
        AddRow("Left-drag", "Rotate camera");
#if defined(__APPLE__)
        AddLabel("Opt + left-drag");
#else
        AddLabel("Win + left-drag");
#endif  // __APPLE__
        AddLabel("Rotate around forward axis");
        AddRow("W", "Forward");
        AddRow("S", "Backward");
        AddRow("A", "Step left");
        AddRow("D", "Step right");
        AddRow("Q", "Step up");
        AddRow("Z", "Step down");
        AddRow("E", "Roll left");
        AddRow("R", "Roll right");
        AddRow("Up", "Look up");
        AddRow("Down", "Look down");
        AddRow("Left", "Look left");
        AddRow("Right", "Look right");

        return layout;
    }

    std::string UniquifyName(const std::string &name) {
        if (added_names_.find(name) == added_names_.end()) {
            return name;
        }

        int n = 0;
        std::string unique;
        do {
            n += 1;
            std::stringstream s;
            s << name << "_" << n;
            unique = s.str();
        } while (added_names_.find(unique) != added_names_.end());

        return unique;
    }

    Eigen::Vector4f CalcDefaultUnlitColor() {
        float luminosity = 0.21f * ui_state_.bg_color.x() +
                           0.72f * ui_state_.bg_color.y() +
                           0.07f * ui_state_.bg_color.z();
        if (luminosity >= 0.5f) {
            return {0.0f, 0.0f, 0.0f, 1.0f};
        } else {
            return {1.0f, 1.0f, 1.0f, 1.0f};
        }
    }

    std::vector<std::string> GetListOfIBLs() {
        std::vector<std::string> ibls;
        std::vector<std::string> resource_files;
        std::string resource_path =
                Application::GetInstance().GetResourcePath();
        utility::filesystem::ListFilesInDirectory(resource_path,
                                                  resource_files);
        std::sort(resource_files.begin(), resource_files.end());
        for (auto &f : resource_files) {
            if (f.find("_ibl.ktx") == f.size() - 8) {
                auto name = utility::filesystem::GetFileNameWithoutDirectory(f);
                name = name.substr(0, name.size() - 8);
                ibls.push_back(name);
            }
        }
        return ibls;
    }
};

// ----------------------------------------------------------------------------
O3DVisualizer::O3DVisualizer(const std::string &title, int width, int height)
    : Window(title, width, height), impl_(new O3DVisualizer::Impl()) {
    impl_->Construct(this);

    // Create the app menu. We will take over the existing menubar (if any)
    // since a) we need to cache a pointer, and b) we should be the only
    // window, since the whole point of this class is to have an easy way to
    // visualize something with a blocking call to draw().
    auto menu = std::make_shared<Menu>();
#if defined(__APPLE__)
    // The first menu item to be added on macOS becomes the application
    // menu (no matter its name)
    auto app_menu = std::make_shared<Menu>();
    app_menu->AddItem("About...", MENU_HELP_ABOUT);
    impl_->app_menu_.menu = app_menu.get();
    impl_->app_menu_.insertion_idx = app_menu->GetNumberOfItems();
    menu->AddMenu("Open3D", app_menu);
#endif  // __APPLE__
    if (Application::GetInstance().UsingNativeWindows()) {
        auto file_menu = std::make_shared<Menu>();
        impl_->file_menu_.menu = file_menu.get();
        impl_->file_menu_.insertion_idx = file_menu->GetNumberOfItems();
        file_menu->AddItem("Export Current Image...", MENU_EXPORT_RGB);
        file_menu->AddSeparator();
        file_menu->AddItem("Close Window", MENU_CLOSE, KeyName::KEY_W);
        menu->AddMenu("File", file_menu);
    }

    auto actions_menu = std::make_shared<Menu>();
    actions_menu->AddItem("Show Settings", MENU_SETTINGS);
    actions_menu->SetChecked(MENU_SETTINGS, false);
    menu->AddMenu("Actions", actions_menu);
    impl_->settings.actions.menu = actions_menu.get();

    auto help_menu = std::make_shared<Menu>();
#if !defined(__APPLE__)
    help_menu->AddItem("About...", MENU_HELP_ABOUT);  // in app menu on macOS
#endif                                                // !__APPLE__
    help_menu->AddItem("Contact Us...", MENU_HELP_CONTACT_US);
    help_menu->AddSeparator();
    help_menu->AddItem("Show Controls", MENU_HELP_SHOW_CONTROLS);
    help_menu->AddItem("Show Camera Info", MENU_HELP_SHOW_CAMERA_INFO);
    // macOS auto-creates a fancy help menu if a menu is named "Help", but we
    // do not support all those features, so make it a string that looks the
    // same but does not trigger the auto-creation.
    menu->AddMenu("Help ", help_menu);

    Application::GetInstance().SetMenubar(menu);

    SetOnMenuItemActivated(MENU_HELP_ABOUT,
                           [this]() { this->impl_->OnAbout(); });
    SetOnMenuItemActivated(MENU_HELP_CONTACT_US,
                           [this]() { this->impl_->OnContactUs(); });
    SetOnMenuItemActivated(MENU_HELP_SHOW_CONTROLS, [this]() {
        bool vis = !this->impl_->controls_help_panel_->IsVisible();
        impl_->controls_help_panel_->SetVisible(vis);
        auto menubar = Application::GetInstance().GetMenubar();
        if (menubar) {  // might not have been created yet
            menubar->SetChecked(MENU_HELP_SHOW_CONTROLS, vis);
        }
    });
    SetOnMenuItemActivated(MENU_HELP_SHOW_CAMERA_INFO, [this]() {
        bool vis = !this->impl_->camera_info_panel_->IsVisible();
        impl_->camera_info_panel_->SetVisible(vis);
        if (vis) {
            // Update to current camera info
            impl_->OnCameraChanged(impl_->widget3d_->GetScene()->GetCamera());
        }
        auto menubar = Application::GetInstance().GetMenubar();
        if (menubar) {  // might not have been created yet
            menubar->SetChecked(MENU_HELP_SHOW_CAMERA_INFO, vis);
        }
    });
    SetOnMenuItemActivated(MENU_EXPORT_RGB, [this]() { impl_->OnExportRGB(); });
    SetOnMenuItemActivated(MENU_CLOSE, [this]() { impl_->OnClose(); });
    SetOnMenuItemActivated(MENU_SETTINGS,
                           [this]() { impl_->OnToggleSettings(); });

    impl_->ShowSettings(false, false);
}

O3DVisualizer::~O3DVisualizer() {}

/*void O3DVisualizer::AddItemsToAppMenu(
        const std::vector<std::pair<std::string, gui::Menu::ItemId>> &items) {
#if !defined(__APPLE__)
    return;  // application menu only exists on macOS
#endif

    if (impl_->app_menu_ && impl_->app_menu_custom_items_index_ >= 0) {
        impl_->app_menu_->InsertSeparator(impl_->app_menu_custom_items_index_++);
        for (auto &it : items) {
            if (it.first != "") {
                impl_->app_menu_->InsertItem(impl_->app_menu_custom_items_index_++,
                                             it.first.c_str(), it.second);
            } else {
                impl_->app_menu_->InsertSeparator(impl_->app_menu_custom_items_index_++);
            }
        }
//        impl_->app_menu_->InsertSeparator(
//                impl_->app_menu_custom_items_index_++);
    }
}
*/
O3DVisualizer::MenuCustomization &O3DVisualizer::GetAppMenu() {
    return impl_->app_menu_;
}

O3DVisualizer::MenuCustomization &O3DVisualizer::GetFileMenu() {
    return impl_->file_menu_;
}

Open3DScene *O3DVisualizer::GetScene() const {
    return impl_->widget3d_->GetScene().get();
}

void O3DVisualizer::StartRPCInterface(const std::string &address, int timeout) {
#ifdef BUILD_RPC_INTERFACE
    auto on_geometry = [this](std::shared_ptr<geometry::Geometry3D> geom,
                              const std::string &path, int time,
                              const std::string &layer) {
        impl_->AddGeometry(path, geom, nullptr, nullptr, nullptr, layer, time,
                           true);
        if (impl_->objects_.size() == 1) {
            impl_->ResetCameraToDefault();
        }
    };

    impl_->receiver_ =
            std::make_shared<Receiver>(address, timeout, this, on_geometry);
    try {
        utility::LogInfo("Starting to listen on {}", address);
        impl_->receiver_->Start();
    } catch (std::exception &e) {
        utility::LogWarning("Failed to start RPC interface: {}", e.what());
    }
#else
    utility::LogWarning(
            "O3DVisualizer::StartRPCInterface: RPC interface not built");
#endif
}

void O3DVisualizer::StopRPCInterface() {
#ifdef BUILD_RPC_INTERFACE
    if (impl_->receiver_) {
        utility::LogInfo("Stopping RPC interface");
    }
    impl_->receiver_.reset();
#else
    utility::LogWarning(
            "O3DVisualizer::StopRPCInterface: RPC interface not built");
#endif
}

void O3DVisualizer::AddAction(const std::string &name,
                              std::function<void(O3DVisualizer &)> callback) {
    // Add button to the "Custom Actions" segment in the UI
    SmallButton *button = new SmallButton(name.c_str());
    button->SetOnClicked([this, callback]() { callback(*this); });
    impl_->settings.actions.buttons->AddChild(GiveOwnership(button));

    SetNeedsLayout();
    impl_->settings.actions.panel->SetVisible(true);
    impl_->settings.actions.panel->SetIsOpen(true);

    if (impl_->can_auto_show_settings_ &&
        impl_->settings.actions.buttons->size() == 1) {
        impl_->ShowSettings(true);
    }

    // Add menu item
    if (impl_->settings.actions.menuid2action.empty()) {
        impl_->settings.actions.menu->AddSeparator();
    }
    int id = MENU_ACTIONS_BASE +
             int(impl_->settings.actions.menuid2action.size());
    impl_->settings.actions.menu->AddItem(name.c_str(), id);
    impl_->settings.actions.menuid2action[id] = callback;
    SetOnMenuItemActivated(id, [this, callback]() { callback(*this); });
}

void O3DVisualizer::SetBackground(
        const Eigen::Vector4f &bg_color,
        std::shared_ptr<geometry::Image> bg_image /*= nullptr*/) {
    impl_->SetBackground(bg_color, bg_image);
}

void O3DVisualizer::SetShader(Shader shader) { impl_->SetShader(shader); }

void O3DVisualizer::AddGeometry(
        const std::string &name,
        std::shared_ptr<geometry::Geometry3D> geom,
        const rendering::Material *material /*=nullptr*/,
        const std::string &group /*= ""*/,
        double time /*= 0.0*/,
        bool is_visible /*= true*/) {
    impl_->AddGeometry(name, geom, nullptr, nullptr, material, group, time,
                       is_visible);
}

void O3DVisualizer::AddGeometry(
        const std::string &name,
        std::shared_ptr<t::geometry::Geometry> tgeom,
        const rendering::Material *material /*=nullptr*/,
        const std::string &group /*= ""*/,
        double time /*= 0.0*/,
        bool is_visible /*= true*/) {
    impl_->AddGeometry(name, nullptr, tgeom, nullptr, material, group, time,
                       is_visible);
}

void O3DVisualizer::AddGeometry(
        const std::string &name,
        std::shared_ptr<rendering::TriangleMeshModel> model,
        const rendering::Material *material /*=nullptr*/,
        const std::string &group /*= ""*/,
        double time /*= 0.0*/,
        bool is_visible /*= true*/) {
    if (model->meshes_.size() == 1) {
        auto &mesh = model->meshes_[0];
        impl_->AddGeometry(name, mesh.mesh, nullptr, nullptr,
                           &model->materials_[mesh.material_idx], group, time,
                           is_visible);
    } else {
        impl_->AddGeometry(name, nullptr, nullptr, model, material, group, time,
                           is_visible);
    }
}

void O3DVisualizer::Add3DLabel(const Eigen::Vector3f &pos, const char *text) {
    impl_->Add3DLabel(pos, text);
}

void O3DVisualizer::Clear3DLabels() { impl_->Clear3DLabels(); }

void O3DVisualizer::RemoveGeometry(const std::string &name) {
    return impl_->RemoveGeometry(name);
}

void O3DVisualizer::ClearGeometry() { return impl_->ClearGeometry(); }

void O3DVisualizer::ShowGeometry(const std::string &name, bool show) {
    return impl_->ShowGeometry(name, show);
}

O3DVisualizer::DrawObject O3DVisualizer::GetGeometry(
        const std::string &name) const {
    return impl_->GetGeometry(name);
}

void O3DVisualizer::ShowSettings(bool show) { impl_->ShowSettings(show); }

void O3DVisualizer::ShowSkybox(bool show) { impl_->ShowSkybox(show); }

void O3DVisualizer::ShowAxes(bool show) { impl_->ShowAxes(show); }

void O3DVisualizer::ShowGround(bool show) { impl_->ShowGround(show); }

void O3DVisualizer::SetGroundPlane(rendering::Scene::GroundPlane plane) {
    impl_->SetGroundPlane(plane);
}

void O3DVisualizer::SetPointSize(int point_size) {
    impl_->SetPointSize(point_size);
}

void O3DVisualizer::SetLineWidth(int line_width) {
    impl_->SetLineWidth(line_width);
}

void O3DVisualizer::SetMouseMode(SceneWidget::Controls mode) {
    impl_->SetMouseMode(mode);
}

void O3DVisualizer::EnableGroup(const std::string &group, bool enable) {
    impl_->EnableGroup(group, enable);
}

std::vector<O3DVisualizerSelections::SelectionSet>
O3DVisualizer::GetSelectionSets() const {
    return impl_->GetSelectionSets();
}

double O3DVisualizer::GetAnimationFrameDelay() const {
    return impl_->ui_state_.frame_delay;
}

void O3DVisualizer::SetAnimationFrameDelay(double secs) {
    impl_->ui_state_.frame_delay = secs;
}

double O3DVisualizer::GetAnimationTimeStep() const {
    return impl_->ui_state_.time_step;
}

void O3DVisualizer::SetAnimationTimeStep(double time_step) {
    impl_->ui_state_.time_step = time_step;
    SetAnimationFrameDelay(time_step);
}

double O3DVisualizer::GetAnimationDuration() const {
    return impl_->max_time_ - impl_->min_time_ + GetAnimationTimeStep();
}

void O3DVisualizer::SetAnimationDuration(double sec) {
    impl_->max_time_ = impl_->min_time_ + sec - GetAnimationTimeStep();
    impl_->UpdateTimeUIRange();
    impl_->settings.time_panel->SetVisible(impl_->min_time_ < impl_->max_time_);
}

double O3DVisualizer::GetCurrentTime() const {
    return impl_->ui_state_.current_time;
}

void O3DVisualizer::SetCurrentTime(double t) { impl_->SetCurrentTime(t); }

bool O3DVisualizer::GetIsAnimating() const {
    return impl_->ui_state_.is_animating;
}

void O3DVisualizer::SetAnimating(bool is_animating) {
    impl_->SetAnimating(is_animating);
}

void O3DVisualizer::SetupCamera(float fov,
                                const Eigen::Vector3f &center,
                                const Eigen::Vector3f &eye,
                                const Eigen::Vector3f &up) {
    impl_->SetupCamera(fov, center, eye, up);
}

void O3DVisualizer::SetupCamera(const camera::PinholeCameraIntrinsic &intrinsic,
                                const Eigen::Matrix4d &extrinsic) {
    impl_->SetupCamera(intrinsic, extrinsic);
}

void O3DVisualizer::SetupCamera(const Eigen::Matrix3d &intrinsic,
                                const Eigen::Matrix4d &extrinsic,
                                int intrinsic_width_px,
                                int intrinsic_height_px) {
    impl_->SetupCamera(intrinsic, extrinsic, intrinsic_width_px,
                       intrinsic_height_px);
}

void O3DVisualizer::ResetCameraToDefault() {
    return impl_->ResetCameraToDefault();
}

O3DVisualizer::UIState O3DVisualizer::GetUIState() const {
    return impl_->ui_state_;
}

void O3DVisualizer::SetOnAnimationFrame(
        std::function<void(O3DVisualizer &, double)> cb) {
    if (cb) {
        impl_->on_animation_ = [this, cb](double t) { cb(*this, t); };
    } else {
        impl_->on_animation_ = nullptr;
    }
}

void O3DVisualizer::SetOnAnimationTick(
        std::function<TickResult(O3DVisualizer &, double, double)> cb) {
    impl_->SetOnAnimationTick(*this, cb);
}

void O3DVisualizer::ExportCurrentImage(const std::string &path) {
    impl_->ExportCurrentImage(path);
}

void O3DVisualizer::Layout(const gui::LayoutContext &context) {
    auto em = context.theme.font_size;
    int settings_width = 15 * context.theme.font_size;
#if !GROUPS_USE_TREE
    if (impl_->added_groups_.size() >= 2) {
        settings_width += 5 * context.theme.font_size;
    }
#endif  // !GROUPS_USE_TREE
    if (impl_->min_time_ != impl_->max_time_) {
        settings_width += 3 * context.theme.font_size;
    }

    auto f = GetContentRect();
    impl_->settings.actions.buttons->SetWidth(settings_width -
                                              int(std::round(1.5 * em)));
    if (impl_->settings.panel->IsVisible()) {
        impl_->widget3d_->SetFrame(
                Rect(f.x, f.y, f.width - settings_width, f.height));
        impl_->settings.panel->SetFrame(Rect(f.GetRight() - settings_width, f.y,
                                             settings_width, f.height));
    } else {
        impl_->widget3d_->SetFrame(f);
    }

    // Controls help panel goes in upper left
    auto pref = impl_->controls_help_panel_->CalcPreferredSize(
            context, Widget::Constraints());
    impl_->controls_help_panel_->SetFrame(
            Rect(f.x, f.y, pref.width, pref.height));

    Super::Layout(context);

    // Camera info panel goes in lower left
    pref = impl_->camera_info_panel_->CalcPreferredSize(context,
                                                        Widget::Constraints());
    impl_->camera_info_panel_->SetFrame(
            Rect(f.x, f.GetBottom() - pref.height, pref.width, pref.height));
}

}  // namespace visualizer
}  // namespace visualization
}  // namespace open3d
