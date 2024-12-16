// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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
#include "open3d/io/rpc/ZMQReceiver.h"
#include "open3d/t/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
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
#include "open3d/visualization/gui/TabControl.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/TreeView.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/visualizer/GuiWidgets.h"
#include "open3d/visualization/visualizer/MessageProcessor.h"
#include "open3d/visualization/visualizer/O3DVisualizerSelections.h"

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
    MENU_ABOUT = 0,
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

struct LightingProfile {
    std::string name;
    Open3DScene::LightingProfile profile;
};

static const char *kCustomName = "Custom";
static const std::vector<LightingProfile> gLightingProfiles = {
        {"Dark shadows", Open3DScene::LightingProfile::DARK_SHADOWS},
        {"Medium shadows", Open3DScene::LightingProfile::MED_SHADOWS},
        {"Soft shadows", Open3DScene::LightingProfile::SOFT_SHADOWS},
        {"No shadows", Open3DScene::LightingProfile::NO_SHADOWS}};

}  // namespace

struct O3DVisualizer::Impl {
    std::set<std::string> added_names_;
    std::set<std::string> added_groups_;
    std::vector<DrawObject> objects_;
    std::vector<DrawObject> inspection_objects_;
    std::vector<DrawObject> wireframe_objects_;
    std::shared_ptr<O3DVisualizerSelections> selections_;
    bool polygon_selection_unselects_ = false;
    bool selections_need_update_ = true;
    std::function<void(double)> on_animation_;
    std::function<bool()> on_animation_tick_;
    std::shared_ptr<io::rpc::ZMQReceiver> receiver_;
    std::shared_ptr<MessageProcessor> message_processor_;

    UIState ui_state_;
    bool can_auto_show_settings_ = true;
    bool was_using_sun_follows_cam_ = false;

    double min_time_ = 0.0;
    double max_time_ = 0.0;
    double start_animation_clock_time_ = 0.0;
    double next_animation_tick_clock_time_ = 0.0;
    double last_animation_tick_clock_time_ = 0.0;

    Window *window_ = nullptr;
    SceneWidget *scene_ = nullptr;

    struct {
        // We only keep pointers here because that way we don't have to release
        // all the shared_ptrs at destruction just to ensure that the gui gets
        // destroyed before the Window, because the Window will do that for us.
        Menu *actions_menu;
        std::unordered_map<int, std::function<void(O3DVisualizer &)>>
                menuid2action;

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

        CollapsableVert *scene_panel;
        Checkbox *show_skybox;
        Checkbox *show_axes;
        Checkbox *show_ground;
        Checkbox *basic_mode;
        Checkbox *wireframe_mode;
        Combobox *ground_plane;
        ColorEdit *bg_color;
        Slider *point_size;
        Combobox *shader;
        Combobox *lighting;

        CollapsableVert *light_panel;
        Checkbox *use_ibl;
        Checkbox *use_sun;
        Combobox *ibl_names;
        Slider *ibl_intensity;
        Slider *sun_intensity;
        Checkbox *sun_follows_camera;
        VectorEdit *sun_dir;
        ColorEdit *sun_color;

        CollapsableVert *geometries_panel;
        TreeView *geometries;
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

        EmptyIfHiddenVert *actions_panel;
        ButtonList *actions;
    } settings;

    void Construct(O3DVisualizer *w) {
        if (window_) {
            return;
        }

        window_ = w;
        scene_ = new SceneWidget();
        selections_ = std::make_shared<O3DVisualizerSelections>(*scene_);
        scene_->SetScene(std::make_shared<Open3DScene>(w->GetRenderer()));
        scene_->EnableSceneCaching(true);  // smoother UI with large geometry
        scene_->SetOnPointsPicked(
                [this](const std::map<
                               std::string,
                               std::vector<std::pair<size_t, Eigen::Vector3d>>>
                               &indices,
                       int keymods) {
                    bool unselect_mode_requested =
                            keymods & int(KeyModifier::SHIFT);
                    if (unselect_mode_requested ||
                        polygon_selection_unselects_) {
                        selections_->UnselectIndices(indices);
                    } else {
                        selections_->SelectIndices(indices);
                    }
                    polygon_selection_unselects_ = false;
                });
        w->AddChild(GiveOwnership(scene_));

        auto o3dscene = scene_->GetScene();
        o3dscene->SetBackground(ui_state_.bg_color);

        MakeSettingsUI();
        SetMouseMode(SceneWidget::Controls::ROTATE_CAMERA);
        SetLightingProfile(gLightingProfiles[2]);  // soft shadows
        SetPointSize(ui_state_.point_size);  // sync selections_' point size
        EnableSunFollowsCamera(true);
    }

    void MakeSettingsUI() {
        auto em = window_->GetTheme().font_size;
        auto half_em = int(std::round(0.5f * float(em)));
        auto v_spacing = int(std::round(0.25 * float(em)));

        settings.panel = new Vert(half_em);
        window_->AddChild(GiveOwnership(settings.panel));

        Margins margins(em, 0, half_em, 0);
        Margins tabbed_margins(0, half_em, 0, 0);

        settings.mouse_panel =
                new CollapsableVert("Mouse Controls", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.mouse_panel));

        settings.mouse_tab = new TabControl();
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

        // Mouse countrols
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
        const char *selection_help = R"(Cmd-click to select a point
Cmd-shift-click to deselect a point
Cmd-alt-click to polygon select)";
#else
        const char *selection_help = R"(Ctrl-click to select a point
Ctrl-shift-click to deselect a point
Ctrl-alt-click to polygon select)";
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
            scene_->DoPolygonPick(SceneWidget::PolygonPickAction::SELECT);
            settings.polygon_selection_panel->SetVisible(false);
        });
        h->AddChild(b);
        b = std::make_shared<SmallButton>("Unselect");
        b->SetOnClicked([this]() {
            polygon_selection_unselects_ = true;
            scene_->DoPolygonPick(SceneWidget::PolygonPickAction::SELECT);
            settings.polygon_selection_panel->SetVisible(false);
        });
        h->AddChild(b);
        b = std::make_shared<SmallButton>("Cancel");
        b->SetOnClicked([this]() {
            scene_->DoPolygonPick(SceneWidget::PolygonPickAction::CANCEL);
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

        // Scene controls
        settings.scene_panel = new CollapsableVert("Scene", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.scene_panel));

        settings.show_skybox = new Checkbox("Show Skybox");
        settings.show_skybox->SetOnChecked(
                [this](bool is_checked) { this->ShowSkybox(is_checked); });

        settings.show_axes = new Checkbox("Show Axis");
        settings.show_axes->SetOnChecked(
                [this](bool is_checked) { this->ShowAxes(is_checked); });

        settings.show_ground = new Checkbox("Show Ground");
        settings.show_ground->SetOnChecked(
                [this](bool is_checked) { this->ShowGround(is_checked); });

        settings.ground_plane = new Combobox();
        settings.ground_plane->AddItem("XZ");
        settings.ground_plane->AddItem("XY");
        settings.ground_plane->AddItem("YZ");
        settings.ground_plane->SetOnValueChanged([this](const char *item,
                                                        int idx) {
            if (idx == 1) {
                ui_state_.ground_plane = rendering::Scene::GroundPlane::XY;
            } else if (idx == 2) {
                ui_state_.ground_plane = rendering::Scene::GroundPlane::YZ;
            } else {
                ui_state_.ground_plane = rendering::Scene::GroundPlane::XZ;
            }
            this->ShowGround(ui_state_.show_ground);
        });

        h = new Horiz(v_spacing);
        h->AddChild(GiveOwnership(settings.show_axes));
        h->AddFixed(em);
        h->AddChild(GiveOwnership(settings.show_skybox));
        settings.scene_panel->AddChild(GiveOwnership(h));
        settings.scene_panel->AddChild(GiveOwnership(settings.show_ground));
        settings.scene_panel->AddChild(GiveOwnership(settings.ground_plane));

        settings.bg_color = new ColorEdit();
        settings.bg_color->SetValue(ui_state_.bg_color.x(),
                                    ui_state_.bg_color.y(),
                                    ui_state_.bg_color.z());
        settings.bg_color->SetOnValueChanged([this](const Color &c) {
            this->SetBackground({c.GetRed(), c.GetGreen(), c.GetBlue(), 1.0f},
                                nullptr);
        });

        settings.point_size = new Slider(Slider::INT);
        settings.point_size->SetLimits(1, 10);
        settings.point_size->SetValue(ui_state_.point_size);
        settings.point_size->SetOnValueChanged([this](const double newValue) {
            this->SetPointSize(int(newValue));
        });

        settings.shader = new Combobox();
        settings.shader->AddItem("Standard");
        settings.shader->AddItem("Unlit");
        settings.shader->AddItem("Normal Map");
        settings.shader->AddItem("Depth");
        settings.shader->SetOnValueChanged([this](const char *item, int idx) {
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

        settings.lighting = new Combobox();
        for (auto &profile : gLightingProfiles) {
            settings.lighting->AddItem(profile.name.c_str());
        }
        settings.lighting->AddItem(kCustomName);
        settings.lighting->SetOnValueChanged([this](const char *, int index) {
            if (index < int(gLightingProfiles.size())) {
                this->SetLightingProfile(gLightingProfiles[index]);
            }
        });

        settings.basic_mode = new Checkbox("");
        settings.basic_mode->SetOnChecked(
                [this](bool enable) { this->EnableBasicMode(enable); });
        settings.wireframe_mode = new Checkbox("");
        settings.wireframe_mode->SetOnChecked(
                [this](bool enable) { this->EnableWireframeMode(enable); });

        auto *grid = new VGrid(2, v_spacing);
        settings.scene_panel->AddChild(GiveOwnership(grid));

        grid->AddChild(std::make_shared<Label>("BG Color"));
        grid->AddChild(GiveOwnership(settings.bg_color));
        grid->AddChild(std::make_shared<Label>("PointSize"));
        grid->AddChild(GiveOwnership(settings.point_size));
        grid->AddChild(std::make_shared<Label>("Shader"));
        grid->AddChild(GiveOwnership(settings.shader));
        grid->AddChild(std::make_shared<Label>("Lighting"));
        grid->AddChild(GiveOwnership(settings.lighting));
        grid->AddChild(std::make_shared<Label>("Raw Mode"));
        grid->AddChild(GiveOwnership(settings.basic_mode));
        grid->AddChild(std::make_shared<Label>("Wireframe"));
        grid->AddChild(GiveOwnership(settings.wireframe_mode));

        // Light list
        settings.light_panel = new CollapsableVert("Lighting", 0, margins);
        settings.light_panel->SetIsOpen(false);
        settings.panel->AddChild(GiveOwnership(settings.light_panel));

        h = new Horiz(v_spacing);
        settings.use_ibl = new Checkbox("HDR map");
        settings.use_ibl->SetChecked(ui_state_.use_ibl);
        settings.use_ibl->SetOnChecked([this](bool checked) {
            this->ui_state_.use_ibl = checked;
            this->SetUIState(ui_state_);
            this->settings.lighting->SetSelectedValue(kCustomName);
        });

        settings.use_sun = new Checkbox("Sun");
        settings.use_sun->SetChecked(settings.use_sun);
        settings.use_sun->SetOnChecked([this](bool checked) {
            this->ui_state_.use_sun = checked;
            this->SetUIState(ui_state_);
            this->settings.lighting->SetSelectedValue(kCustomName);
        });

        h->AddChild(GiveOwnership(settings.use_ibl));
        h->AddFixed(int(std::round(
                1.4 * em)));  // align with Show Skybox checkbox above
        h->AddChild(GiveOwnership(settings.use_sun));

        settings.light_panel->AddChild(
                std::make_shared<Label>("Light sources"));
        settings.light_panel->AddChild(GiveOwnership(h));
        settings.light_panel->AddFixed(half_em);

        grid = new VGrid(2, v_spacing);

        settings.ibl_names = new Combobox();
        for (auto &name : GetListOfIBLs()) {
            settings.ibl_names->AddItem(name.c_str());
        }
        settings.ibl_names->SetSelectedValue(kDefaultIBL.c_str());
        settings.ibl_names->SetOnValueChanged([this](const char *val, int idx) {
            std::string resource_path =
                    Application::GetInstance().GetResourcePath();
            this->SetIBL(resource_path + std::string("/") + std::string(val));
            this->settings.lighting->SetSelectedValue(kCustomName);
        });
        grid->AddChild(std::make_shared<Label>("HDR map"));
        grid->AddChild(GiveOwnership(settings.ibl_names));

        settings.ibl_intensity = new Slider(Slider::INT);
        settings.ibl_intensity->SetLimits(0.0, 150000.0);
        settings.ibl_intensity->SetValue(ui_state_.ibl_intensity);
        settings.ibl_intensity->SetOnValueChanged([this](double new_value) {
            this->ui_state_.ibl_intensity = int(new_value);
            this->SetUIState(ui_state_);
            this->settings.lighting->SetSelectedValue(kCustomName);
        });
        grid->AddChild(std::make_shared<Label>("Intensity"));
        grid->AddChild(GiveOwnership(settings.ibl_intensity));

        settings.light_panel->AddChild(std::make_shared<Label>("Environment"));
        settings.light_panel->AddChild(GiveOwnership(grid));
        settings.light_panel->AddFixed(half_em);

        grid = new VGrid(2, v_spacing);

        settings.sun_intensity = new Slider(Slider::INT);
        settings.sun_intensity->SetLimits(0.0, 250000.0);
        settings.sun_intensity->SetValue(ui_state_.sun_intensity);
        settings.sun_intensity->SetOnValueChanged([this](double new_value) {
            this->ui_state_.sun_intensity = int(new_value);
            this->SetUIState(ui_state_);
            this->settings.lighting->SetSelectedValue(kCustomName);
        });
        grid->AddChild(std::make_shared<Label>("Intensity"));
        grid->AddChild(GiveOwnership(settings.sun_intensity));

        settings.sun_dir = new VectorEdit();
        settings.sun_dir->SetValue(ui_state_.sun_dir);
        settings.sun_dir->SetOnValueChanged([this](const Eigen::Vector3f &dir) {
            this->ui_state_.sun_dir = dir;
            this->SetUIState(ui_state_);
            this->settings.lighting->SetSelectedValue(kCustomName);
        });
        scene_->SetOnSunDirectionChanged(
                [this](const Eigen::Vector3f &new_dir) {
                    this->ui_state_.sun_dir = new_dir;
                    this->settings.sun_dir->SetValue(new_dir);
                    // Don't need to call SetUIState(), the SceneWidget already
                    // modified the scene.
                    this->settings.lighting->SetSelectedValue(kCustomName);
                });
        grid->AddChild(std::make_shared<Label>("Direction"));
        grid->AddChild(GiveOwnership(settings.sun_dir));

        settings.sun_follows_camera = new gui::Checkbox(" ");
        settings.sun_follows_camera->SetChecked(true);
        settings.sun_follows_camera->SetOnChecked([this](bool checked) {
            ui_state_.sun_follows_camera = checked;
            this->SetUIState(ui_state_);
            EnableSunFollowsCamera(checked);
        });
        grid->AddChild(GiveOwnership(settings.sun_follows_camera));
        grid->AddChild(std::make_shared<gui::Label>("Sun Follows Camera"));

        settings.sun_color = new ColorEdit();
        settings.sun_color->SetValue(ui_state_.sun_color);
        settings.sun_color->SetOnValueChanged([this](const Color &new_color) {
            this->ui_state_.sun_color = {new_color.GetRed(),
                                         new_color.GetGreen(),
                                         new_color.GetBlue()};
            this->SetUIState(ui_state_);
            this->settings.lighting->SetSelectedValue(kCustomName);
        });
        grid->AddChild(std::make_shared<Label>("Color"));
        grid->AddChild(GiveOwnership(settings.sun_color));

        settings.light_panel->AddChild(
                std::make_shared<Label>("Sun (Directional light)"));
        settings.light_panel->AddChild(GiveOwnership(grid));

        // Geometry list
        settings.geometries_panel =
                new CollapsableVert("Geometries", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.geometries_panel));

        settings.geometries = new TreeView();
        settings.geometries_panel->AddChild(GiveOwnership(settings.geometries));

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

        // Custom actions
        settings.actions_panel =
                new EmptyIfHiddenVert("Custom Actions", v_spacing, margins);
        settings.panel->AddChild(GiveOwnership(settings.actions_panel));
        settings.actions_panel->SetVisible(false);

        settings.actions = new ButtonList(v_spacing);
        settings.actions_panel->AddChild(GiveOwnership(settings.actions));

        // Picking callbacks
        scene_->SetOnStartedPolygonPicking([this]() {
            settings.polygon_selection_panel->SetVisible(true);
        });
    }

    void AddGeometry(const std::string &name,
                     std::shared_ptr<geometry::Geometry3D> geom,
                     std::shared_ptr<t::geometry::Geometry> tgeom,
                     std::shared_ptr<rendering::TriangleMeshModel> model,
                     const rendering::MaterialRecord *material,
                     const std::string &group,
                     double time,
                     bool is_visible) {
        std::string group_name = group;
        if (group_name == "") {
            group_name = "default";
        }
        bool is_default_color = false;
        bool no_shadows = false;
        MaterialRecord mat;

        if (material) {
            mat = *material;
            is_default_color = false;
            auto t_cloud =
                    std::dynamic_pointer_cast<t::geometry::PointCloud>(tgeom);
            ui_state_.point_size = static_cast<int>(mat.point_size);
        } else if (model) {
            // Adding a triangle mesh model. Shader needs to be set to
            // defaultLit for O3D shader handling logic to work.
            mat.shader = kShaderLit;
        } else {  // branch only applies to geometries
            bool has_colors = false;
            bool has_normals = false;

            auto cloud = std::dynamic_pointer_cast<geometry::PointCloud>(geom);
            auto lines = std::dynamic_pointer_cast<geometry::LineSet>(geom);
            auto obb = std::dynamic_pointer_cast<geometry::OrientedBoundingBox>(
                    geom);
            auto aabb =
                    std::dynamic_pointer_cast<geometry::AxisAlignedBoundingBox>(
                            geom);
            auto mesh = std::dynamic_pointer_cast<geometry::MeshBase>(geom);
            auto tmesh =
                    std::dynamic_pointer_cast<geometry::TriangleMesh>(geom);
            auto voxel_grid =
                    std::dynamic_pointer_cast<geometry::VoxelGrid>(geom);
            auto octree = std::dynamic_pointer_cast<geometry::Octree>(geom);

            auto t_cloud =
                    std::dynamic_pointer_cast<t::geometry::PointCloud>(tgeom);
            auto t_mesh =
                    std::dynamic_pointer_cast<t::geometry::TriangleMesh>(tgeom);
            auto t_lines =
                    std::dynamic_pointer_cast<t::geometry::LineSet>(tgeom);

            if (cloud) {
                has_colors = !cloud->colors_.empty();
                has_normals = !cloud->normals_.empty();
            } else if (t_cloud) {
                has_colors = t_cloud->HasPointColors();
                has_normals = t_cloud->HasPointNormals();
            } else if (lines) {
                has_colors = !lines->colors_.empty();
                no_shadows = true;
            } else if (t_lines) {
                has_colors = t_lines->HasLineColors();
                no_shadows = true;
            } else if (obb) {
                has_colors = (obb->color_ != Eigen::Vector3d{0.0, 0.0, 0.0});
                no_shadows = true;
            } else if (aabb) {
                has_colors = (aabb->color_ != Eigen::Vector3d{0.0, 0.0, 0.0});
                no_shadows = true;
            } else if (mesh) {
                has_normals = !mesh->vertex_normals_.empty() ||
                              (tmesh && !tmesh->triangle_normals_.empty());
                has_colors = true;  // always want base_color as white
            } else if (t_mesh) {
                has_normals = t_mesh->HasVertexNormals() ||
                              t_mesh->HasTriangleNormals();
                has_colors = true;  // always want base_color as white
            } else if (voxel_grid) {
                has_normals = false;
                has_colors = voxel_grid->HasColors();
            } else if (octree) {
                has_normals = false;
                has_colors = true;
            }

            mat.base_color = CalcDefaultUnlitColor();
            mat.shader = kShaderUnlit;
            if (lines || obb || aabb || t_lines) {
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

            // If T Geometry has a valid material convert it to MaterialRecord
            if (t_mesh && t_mesh->HasMaterial()) {
                t_mesh->GetMaterial().ToMaterialRecord(mat);
            } else if (t_cloud && t_cloud->HasMaterial()) {
                t_cloud->GetMaterial().ToMaterialRecord(mat);
            } else if (t_lines && t_lines->HasMaterial()) {
                t_lines->GetMaterial().ToMaterialRecord(mat);
            }

            // Finally assign material properties if geometry is a triangle mesh
            if (tmesh && tmesh->materials_.size() > 0) {
                std::size_t material_index;
                if (tmesh->HasTriangleMaterialIds()) {
                    auto minmax_it = std::minmax_element(
                            tmesh->triangle_material_ids_.begin(),
                            tmesh->triangle_material_ids_.end());
                    if (*minmax_it.first != *minmax_it.second) {
                        utility::LogWarning(
                                "Only a single material is "
                                "supported for TriangleMesh visualization, "
                                "only the first referenced material will be "
                                "used. Use TriangleMeshModel if more than one "
                                "material is required.");
                    }
                    material_index = *minmax_it.first;
                } else {
                    material_index = 0;
                }
                auto &mesh_material = tmesh->materials_[material_index].second;
                mat.base_color = {mesh_material.baseColor.r(),
                                  mesh_material.baseColor.g(),
                                  mesh_material.baseColor.b(),
                                  mesh_material.baseColor.a()};
                mat.base_metallic = mesh_material.baseMetallic;
                mat.base_roughness = mesh_material.baseRoughness;
                mat.base_reflectance = mesh_material.baseReflectance;
                mat.base_clearcoat = mesh_material.baseClearCoat;
                mat.base_clearcoat_roughness =
                        mesh_material.baseClearCoatRoughness;
                mat.base_anisotropy = mesh_material.baseAnisotropy;
                mat.albedo_img = mesh_material.albedo;
                mat.normal_img = mesh_material.normalMap;
                mat.ao_img = mesh_material.ambientOcclusion;
                mat.metallic_img = mesh_material.metallic;
                mat.roughness_img = mesh_material.roughness;
                mat.reflectance_img = mesh_material.reflectance;
                mat.clearcoat_img = mesh_material.clearCoat;
                mat.clearcoat_roughness_img = mesh_material.clearCoatRoughness;
                mat.anisotropy_img = mesh_material.anisotropy;
            }
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
                            is_visible, is_default_color});
        AddObjectToTree(objects_.back());

        auto scene = scene_->GetScene();
        // Do we have a geometry, tgeometry or model?
        if (geom) {
            scene->AddGeometry(name, geom.get(), mat);
        } else if (tgeom) {
            scene->AddGeometry(name, tgeom.get(), mat);
        } else if (model) {
            scene->AddModel(name, *model);
        } else {
            utility::LogWarning(
                    "No valid geometry specified to O3DVisualizer. Only "
                    "supported geometries are Geometry3D and TGeometry "
                    "PointClouds and TriangleMeshes.");
        }

        if (no_shadows) {
            scene->GetScene()->GeometryShadows(name, false, false);
        }
        UpdateGeometryVisibility(objects_.back());

        // Bounds have changed, so update the selection point size, since they
        // depend on the bounds.
        SetPointSize(ui_state_.point_size);

        scene_->ForceRedraw();
    }

    void UpdateGeometry(const std::string &name,
                        std::shared_ptr<t::geometry::Geometry> tgeom,
                        uint32_t update_flags) {
        auto t_cloud =
                std::dynamic_pointer_cast<t::geometry::PointCloud>(tgeom);
        if (!t_cloud) {
            utility::LogWarning(
                    "Only TGeometry PointClouds can currently be updated using "
                    "UpdateGeometry. Try removing the geometry that needs to "
                    "be updated then adding the update geometry.");
            return;
        }
        scene_->GetScene()->GetScene()->UpdateGeometry(name, *t_cloud,
                                                       update_flags);
        scene_->ForceRedraw();
    }

    void RemoveGeometry(const std::string &name) {
        std::string group;
        for (size_t i = 0; i < objects_.size(); ++i) {
            if (objects_[i].name == name) {
                group = objects_[i].group;
                settings.object2itemid.erase(objects_[i].name);
                objects_.erase(objects_.begin() + i);
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
        scene_->GetScene()->RemoveGeometry(name);

        // Bounds have changed, so update the selection point size, since they
        // depend on the bounds.
        SetPointSize(ui_state_.point_size);

        scene_->ForceRedraw();
    }

    void ShowGeometry(const std::string &name, bool show) {
        for (auto &o : objects_) {
            if (o.name == name) {
                if (show != o.is_visible) {
                    o.is_visible = show;

                    auto id = settings.object2itemid[o.name];
                    auto cell = settings.geometries->GetItem(id);
                    auto obj_cell =
                            std::dynamic_pointer_cast<DrawObjectTreeCell>(cell);
                    if (obj_cell) {
                        obj_cell->GetCheckbox()->SetChecked(show);
                    }

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

    MaterialRecord GetGeometryMaterial(const std::string &name) {
        for (auto &o : objects_) {
            if (o.name == name) {
                return o.material;
            }
        }
        return MaterialRecord();
    }

    void ModifyGeometryMaterial(const std::string &name,
                                const MaterialRecord *material) {
        for (auto &o : objects_) {
            if (o.name == name) {
                o.material = *material;
                scene_->GetScene()->ModifyGeometryMaterial(name, *material);
                break;
            }
        }
    }

    void CreateInspectionModeMaterial(MaterialRecord &inspect_mat,
                                      bool pcd = false) {
        if (inspect_mat.shader == "defaultLit" ||
            inspect_mat.shader == "defaultLitTransparency" ||
            inspect_mat.shader == "defaultListSSR") {
            inspect_mat.shader = "defaultLit";
            // Set parameters for 'simple' rendering
            inspect_mat.base_color = {1.f, 1.f, 1.f, 1.f};
            inspect_mat.base_metallic = 0.f;
            if (pcd) {
                inspect_mat.base_roughness = 0.1f;
                inspect_mat.base_reflectance = 0.6f;
            } else {
                inspect_mat.base_roughness = 0.5f;
                inspect_mat.base_reflectance = 0.8f;
            }
            inspect_mat.base_clearcoat = 0.f;
            inspect_mat.base_anisotropy = 0.f;
            inspect_mat.albedo_img.reset();
            inspect_mat.normal_img.reset();
            inspect_mat.ao_img.reset();
            inspect_mat.metallic_img.reset();
            inspect_mat.roughness_img.reset();
            inspect_mat.reflectance_img.reset();
            // inspect_mat.sRGB_color = false;
            // inspect_mat.sRGB_vertex_color = true;
        }
    }

    std::shared_ptr<geometry::TriangleMesh> DuplicateGeometryForInspection(
            std::shared_ptr<geometry::TriangleMesh> tmesh) {
        auto new_mesh = std::make_shared<geometry::TriangleMesh>(
                tmesh->vertices_, tmesh->triangles_);
        if (!tmesh->HasTriangleNormals()) {
            new_mesh->ComputeTriangleNormals();
        } else {
            new_mesh->triangle_normals_ = tmesh->triangle_normals_;
        }
        new_mesh->vertex_colors_ = tmesh->vertex_colors_;

        return new_mesh;
    }

    void UpdateGeometryForInspectionMode(bool enable) {
        if (enable) {
            // Copy the objects...
            for (auto &o : objects_) {
                scene_->GetScene()->ShowGeometry(o.name, false);
                inspection_objects_.push_back(o);
                inspection_objects_.back().name = o.name + "_inspect";
            }
            // Add inspection objects to scene
            for (auto &o : inspection_objects_) {
                if (o.geometry) {
                    CreateInspectionModeMaterial(
                            o.material,
                            o.geometry->GetGeometryType() ==
                                    geometry::Geometry::GeometryType::
                                            PointCloud);
                    // Need to compute triangle normals for triangle meshes
                    if (auto tmesh = std::dynamic_pointer_cast<
                                geometry::TriangleMesh>(o.geometry)) {
                        o.geometry = DuplicateGeometryForInspection(tmesh);
                    }
                    scene_->GetScene()->AddGeometry(o.name, o.geometry.get(),
                                                    o.material);
                } else if (o.tgeometry) {
                    CreateInspectionModeMaterial(
                            o.material,
                            o.tgeometry->GetGeometryType() ==
                                    t::geometry::Geometry::GeometryType::
                                            PointCloud);
                    scene_->GetScene()->AddGeometry(o.name, o.tgeometry.get(),
                                                    o.material);
                } else if (o.model) {
                    for (auto &mat : o.model->materials_) {
                        CreateInspectionModeMaterial(mat, false);
                    }
                    for (auto &mi : o.model->meshes_) {
                        auto new_mesh = DuplicateGeometryForInspection(mi.mesh);
                        mi.mesh = new_mesh;
                    }
                    scene_->GetScene()->AddModel(o.name, *o.model);
                }
            }
        } else {
            // Remove inspection geometries
            for (auto &o : inspection_objects_) {
                scene_->GetScene()->RemoveGeometry(o.name);
            }
            inspection_objects_.clear();
            // Show original geometries
            for (auto &o : objects_) {
                scene_->GetScene()->ShowGeometry(o.name, true);
            }
        }
    }

    void UpdateGeometryForWireframeMode(bool enable) {
        if (enable) {
            // Material for wireframe
            MaterialRecord mat;
            mat.shader = "unlitLine";
            mat.line_width = 2.f;
            mat.base_color = {0.f, 0.3f, 1.f, 1.f};
            mat.emissive_color = {10000.f, 10000.f, 10000.f, 1.f};
            // Create line sets for eligible geometry
            for (auto &o : objects_) {
                if (o.geometry &&
                    o.geometry->GetGeometryType() ==
                            geometry::Geometry::GeometryType::TriangleMesh) {
                    // Hide original geometry
                    scene_->GetScene()->ShowGeometry(o.name, false);
                    auto tmesh =
                            std::dynamic_pointer_cast<geometry::TriangleMesh>(
                                    o.geometry);
                    auto lines =
                            geometry::LineSet::CreateFromTriangleMesh(*tmesh);
                    DrawObject draw_obj;
                    draw_obj.name = o.name + "_wireframe";
                    draw_obj.geometry = lines;
                    draw_obj.material = mat;
                    wireframe_objects_.push_back(draw_obj);
                    scene_->GetScene()->AddGeometry(draw_obj.name,
                                                    draw_obj.geometry.get(),
                                                    draw_obj.material);
                } else if (o.tgeometry &&
                           o.tgeometry->GetGeometryType() ==
                                   t::geometry::Geometry::GeometryType::
                                           TriangleMesh) {
                    // Hide original geometry
                    scene_->GetScene()->ShowGeometry(o.name, false);
                    auto tmesh = std::dynamic_pointer_cast<
                            t::geometry::TriangleMesh>(o.tgeometry);
                    auto tmesh_legacy = tmesh->ToLegacy();
                    auto lines = geometry::LineSet::CreateFromTriangleMesh(
                            tmesh_legacy);
                    DrawObject draw_obj;
                    draw_obj.name = o.name + "_wireframe";
                    draw_obj.geometry = lines;
                    draw_obj.material = mat;
                    wireframe_objects_.push_back(draw_obj);
                    scene_->GetScene()->AddGeometry(draw_obj.name,
                                                    draw_obj.geometry.get(),
                                                    draw_obj.material);
                } else if (o.model) {
                    // Hide original geometry
                    scene_->GetScene()->ShowGeometry(o.name, false);
                    for (auto &mi : o.model->meshes_) {
                        auto lines = geometry::LineSet::CreateFromTriangleMesh(
                                *mi.mesh);
                        DrawObject draw_obj;
                        draw_obj.name = mi.mesh_name + "_wireframe";
                        draw_obj.geometry = lines;
                        draw_obj.material = mat;
                        wireframe_objects_.push_back(draw_obj);
                        scene_->GetScene()->AddGeometry(draw_obj.name,
                                                        draw_obj.geometry.get(),
                                                        draw_obj.material);
                    }
                }
            }
        } else {
            // Remove inspection geometries
            for (auto &o : wireframe_objects_) {
                scene_->GetScene()->RemoveGeometry(o.name);
            }
            wireframe_objects_.clear();
            // Show original geometries
            for (auto &o : objects_) {
                scene_->GetScene()->ShowGeometry(o.name, true);
            }
        }
    }

    void Add3DLabel(const Eigen::Vector3f &pos, const char *text) {
        scene_->AddLabel(pos, text);
    }

    void Clear3DLabels() { scene_->ClearLabels(); }

    void SetupCamera(float fov,
                     const Eigen::Vector3f &center,
                     const Eigen::Vector3f &eye,
                     const Eigen::Vector3f &up) {
        scene_->LookAt(center, eye, up);
        scene_->ForceRedraw();
    }

    void SetupCamera(const camera::PinholeCameraIntrinsic &intrinsic,
                     const Eigen::Matrix4d &extrinsic) {
        scene_->SetupCamera(intrinsic, extrinsic,
                            scene_->GetScene()->GetBoundingBox());
        scene_->ForceRedraw();
    }

    void SetupCamera(const Eigen::Matrix3d &intrinsic,
                     const Eigen::Matrix4d &extrinsic,
                     int intrinsic_width_px,
                     int intrinsic_height_px) {
        scene_->SetupCamera(intrinsic, extrinsic, intrinsic_width_px,
                            intrinsic_height_px,
                            scene_->GetScene()->GetBoundingBox());
        scene_->ForceRedraw();
    }

    void ResetCameraToDefault() {
        auto scene = scene_->GetScene();
        scene_->SetupCamera(60.0f, scene->GetBoundingBox(), {0.0f, 0.0f, 0.0f});
        scene_->ForceRedraw();
    }

    void SetBackground(const Eigen::Vector4f &bg_color,
                       std::shared_ptr<geometry::Image> bg_image) {
        auto old_default_color = CalcDefaultUnlitColor();
        ui_state_.bg_color = bg_color;
        settings.bg_color->SetValue(bg_color.x(), bg_color.y(), bg_color.z());
        auto scene = scene_->GetScene();
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

        scene_->ForceRedraw();
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
        settings.show_skybox->SetChecked(show);  // in case called manually
        scene_->GetScene()->ShowSkybox(show);
        scene_->ForceRedraw();
    }

    void ShowAxes(bool show) {
        ui_state_.show_axes = show;
        settings.show_axes->SetChecked(show);  // in case called manually
        scene_->GetScene()->ShowAxes(show);
        scene_->ForceRedraw();
    }

    void ShowGround(bool show) {
        ui_state_.show_ground = show;
        settings.show_ground->SetChecked(show);  // in case called manually
        scene_->GetScene()->ShowGroundPlane(show, ui_state_.ground_plane);
        scene_->ForceRedraw();
    }

    void SetGroundPlane(rendering::Scene::GroundPlane plane) {
        ui_state_.ground_plane = plane;
        if (plane == rendering::Scene::GroundPlane::XZ) {
            settings.ground_plane->SetSelectedIndex(0);
        } else if (plane == rendering::Scene::GroundPlane::XY) {
            settings.ground_plane->SetSelectedIndex(1);
        } else {
            settings.ground_plane->SetSelectedIndex(2);
        }
        // Update ground plane if it is currently showing
        if (ui_state_.show_ground) {
            scene_->GetScene()->ShowGroundPlane(ui_state_.show_ground, plane);
            scene_->ForceRedraw();
        }
    }

    void EnableSunFollowsCamera(bool enable) {
        auto low_scene = scene_->GetScene()->GetScene();
        if (enable) {
            scene_->SetOnCameraChanged([this](rendering::Camera *cam) {
                auto low_scene = scene_->GetScene()->GetScene();
                low_scene->SetSunLightDirection(cam->GetForwardVector());
            });
            low_scene->SetSunLightDirection(
                    scene_->GetScene()->GetCamera()->GetForwardVector());
            scene_->SetSunInteractorEnabled(false);
        } else {
            scene_->SetOnCameraChanged(
                    std::function<void(rendering::Camera *)>());
            low_scene->SetSunLightDirection(ui_state_.sun_dir);
            scene_->SetSunInteractorEnabled(true);
        }
    }

    void EnableInspectionRelatedUI(bool enable) {
        settings.show_skybox->SetEnabled(enable);
        settings.lighting->SetEnabled(enable);
        settings.use_ibl->SetEnabled(enable);
        settings.ibl_intensity->SetEnabled(enable);
        settings.ibl_names->SetEnabled(enable);
        settings.use_sun->SetEnabled(enable);
        settings.sun_dir->SetEnabled(enable);
        settings.sun_color->SetEnabled(enable);
        settings.mouse_buttons[SceneWidget::Controls::ROTATE_SUN]->SetEnabled(
                enable);
        settings.sun_follows_camera->SetEnabled(enable);
        settings.wireframe_mode->SetEnabled(enable);
    }

    void EnableBasicMode(bool enable) {
        auto o3dscene = scene_->GetScene();
        auto view = o3dscene->GetView();
        auto low_scene = o3dscene->GetScene();
        if (enable) {
            // Set lighting environment for inspection
            o3dscene->SetBackground({1.f, 1.f, 1.f, 1.f});
            low_scene->ShowSkybox(false);
            low_scene->EnableIndirectLight(false);
            low_scene->EnableSunLight(true);
            low_scene->SetSunLightIntensity(160000.f);
            view->SetShadowing(false, View::ShadowType::kPCF);
            view->SetPostProcessing(false);
            if (!ui_state_.sun_follows_camera) {
                EnableSunFollowsCamera(true);
                settings.sun_follows_camera->SetChecked(true);
                was_using_sun_follows_cam_ = false;
            } else {
                was_using_sun_follows_cam_ = true;
            }
            // Update geometry for inspection
            settings.wireframe_mode->SetChecked(false);
            EnableWireframeMode(false);
            UpdateGeometryForInspectionMode(true);
            EnableInspectionRelatedUI(false);
            // Force screen redraw
            scene_->ForceRedraw();
        } else {
            view->SetPostProcessing(true);
            view->SetShadowing(true, View::ShadowType::kPCF);
            UpdateGeometryForInspectionMode(false);
            EnableInspectionRelatedUI(true);
            if (!was_using_sun_follows_cam_) {
                settings.sun_follows_camera->SetChecked(false);
                EnableSunFollowsCamera(false);
            }
            SetUIState(ui_state_);
        }
    }

    void EnableWireframeMode(bool enable) {
        auto o3dscene = scene_->GetScene();
        auto view = o3dscene->GetView();
        auto low_scene = o3dscene->GetScene();
        UpdateGeometryForWireframeMode(enable);
        if (enable) {
            o3dscene->SetBackground({0.1f, 0.1f, 0.1f, 1.0f});
            low_scene->ShowSkybox(false);
            view->SetWireframe(true);
            scene_->ForceRedraw();
        } else {
            view->SetWireframe(false);
            SetUIState(ui_state_);
        }
    }

    void SetPointSize(int px) {
        ui_state_.point_size = px;
        settings.point_size->SetValue(double(px));

        px = int(ConvertToScaledPixels(px));
        for (auto &o : objects_) {
            // Ignore Models since they can never be point clouds
            if (o.model) continue;

            o.material.point_size = float(px);
            OverrideMaterial(o.name, o.material, ui_state_.scene_shader);
        }
        for (auto &o : inspection_objects_) {
            o.material.point_size = float(px);
            OverrideMaterial(o.name, o.material, ui_state_.scene_shader);
        }

        auto bbox_extend = scene_->GetScene()->GetBoundingBox().GetExtent();
        auto psize = double(std::max(5, px)) * 0.000666 *
                     std::max(bbox_extend.x(),
                              std::max(bbox_extend.y(), bbox_extend.z()));
        selections_->SetPointSize(psize);

        scene_->SetPickablePointSize(px);
        scene_->ForceRedraw();
    }

    void SetLineWidth(int px) {
        ui_state_.line_width = px;

        px = int(ConvertToScaledPixels(px));
        for (auto &o : objects_) {
            // Ignore Models since they can never be point clouds
            if (o.model) continue;

            o.material.line_width = float(px);
            OverrideMaterial(o.name, o.material, ui_state_.scene_shader);
        }
        for (auto &o : inspection_objects_) {
            o.material.line_width = float(px);
            OverrideMaterial(o.name, o.material, ui_state_.scene_shader);
        }
        scene_->ForceRedraw();
    }

    void SetShader(O3DVisualizer::Shader shader) {
        // Don't force material override if no change
        if (ui_state_.scene_shader == shader) return;

        ui_state_.scene_shader = shader;
        for (auto &o : objects_) {
            OverrideMaterial(o.name, o.material, shader);
        }
        for (auto &o : inspection_objects_) {
            OverrideMaterial(o.name, o.material, shader);
        }
        scene_->ForceRedraw();
    }

    void OverrideMaterial(const std::string &name,
                          const MaterialRecord &original_material,
                          O3DVisualizer::Shader shader) {
        bool is_lines = (original_material.shader == "unlitLine");
        bool is_gradient = (original_material.shader == "unlitGradient");
        auto scene = scene_->GetScene();
        // Lines are already unlit, so keep using the original shader when in
        // unlit mode so that we can keep the wide lines.
        if (shader == Shader::STANDARD || is_gradient ||
            (shader == Shader::UNLIT && is_lines)) {
            scene->ModifyGeometryMaterial(name, original_material);
        } else {
            MaterialRecord m = original_material;
            m.shader = GetShaderString(shader);
            scene->ModifyGeometryMaterial(name, m);
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
        std::string ibl_path;
        if (path == "") {
            // Set IBL back to default
            ibl_path =
                    std::string(Application::GetInstance().GetResourcePath()) +
                    std::string("/") + std::string(kDefaultIBL);
        } else if (utility::filesystem::FileExists(path + "_ibl.ktx")) {
            // Set IBL to named IBL - probably came from selecting in UI
            ibl_path = path;
        } else if (utility::filesystem::FileExists(path)) {
            // User specified full path to IBL file
            if (path.find("_ibl.ktx") == path.size() - 8) {
                ibl_path = path.substr(0, path.size() - 8);
            } else {
                utility::LogWarning(
                        "Could not load IBL path. Filename must be of the form "
                        "'name_ibl.ktx' and be paired with 'name_skybox.ktx'");
                return;
            }
        } else {
            // User specified the IBL 'stem' without the full path
            ibl_path =
                    std::string(Application::GetInstance().GetResourcePath()) +
                    "/" + path;
            if (!utility::filesystem::FileExists(ibl_path + "_ibl.ktx")) {
                return;
            }
        }
        ui_state_.ibl_path = ibl_path;
        scene_->GetScene()->GetScene()->SetIndirectLight(ibl_path);
        scene_->ForceRedraw();
    }

    void SetIBLIntensity(float intensity) {
        ui_state_.ibl_intensity = intensity;
        SetUIState(ui_state_);
    }

    void SetLightingProfile(const LightingProfile &profile) {
        Eigen::Vector3f sun_dir = {0.577f, -0.577f, -0.577f};
        auto scene = scene_->GetScene();
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
        this->settings.lighting->SetSelectedValue(profile.name.c_str());
    }

    void SetMouseMode(SceneWidget::Controls mode) {
        if (selections_->IsActive()) {
            selections_->MakeInactive();
        }

        scene_->SetViewControls(mode);
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

    void SetPanelOpen(const std::string &name, bool open) {
        if (name == settings.mouse_panel->GetText()) {
            settings.mouse_panel->SetIsOpen(open);
        } else if (name == settings.scene_panel->GetText()) {
            settings.scene_panel->SetIsOpen(open);
        } else if (name == settings.light_panel->GetText()) {
            settings.light_panel->SetIsOpen(open);
        } else if (name == settings.geometries_panel->GetText()) {
            settings.geometries_panel->SetIsOpen(open);
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
                    this->scene_->ForceRedraw();
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
        bool is_new_lighting =
                (ibl_path_changed || new_state.use_ibl != ui_state_.use_ibl ||
                 new_state.use_sun != ui_state_.use_sun ||
                 new_state.ibl_intensity != ui_state_.ibl_intensity ||
                 new_state.sun_intensity != ui_state_.sun_intensity ||
                 new_state.sun_dir != ui_state_.sun_dir ||
                 new_state.sun_color != ui_state_.sun_color);
        bool in_basic_mode = settings.basic_mode->IsChecked();

        if (&new_state != &ui_state_) {
            ui_state_ = new_state;
        }

        if (ibl_path_changed) {
            SetIBL(ui_state_.ibl_path);
        }

        ShowSettings(ui_state_.show_settings, false);
        SetShader(ui_state_.scene_shader);
        SetBackground(ui_state_.bg_color, nullptr);
        if (!in_basic_mode) ShowSkybox(ui_state_.show_skybox);
        ShowAxes(ui_state_.show_axes);
        ShowGround(ui_state_.show_ground);

        if (point_size_changed) {
            SetPointSize(ui_state_.point_size);
        }
        if (line_width_changed) {
            SetLineWidth(ui_state_.line_width);
        }

        settings.use_ibl->SetChecked(ui_state_.use_ibl);
        settings.use_sun->SetChecked(ui_state_.use_sun);
        settings.ibl_intensity->SetValue(ui_state_.ibl_intensity);
        settings.sun_intensity->SetValue(ui_state_.sun_intensity);
        settings.sun_dir->SetValue(ui_state_.sun_dir);
        settings.sun_color->SetValue(ui_state_.sun_color);
        // Re-assign intensity in case it was out of range.
        ui_state_.ibl_intensity = settings.ibl_intensity->GetIntValue();
        ui_state_.sun_intensity = settings.sun_intensity->GetIntValue();

        if (ui_state_.sun_follows_camera) {
            settings.sun_dir->SetEnabled(false);
            settings.mouse_buttons[SceneWidget::Controls::ROTATE_SUN]
                    ->SetEnabled(false);
        } else {
            settings.sun_dir->SetEnabled(true);
            settings.mouse_buttons[SceneWidget::Controls::ROTATE_SUN]
                    ->SetEnabled(true);
        }

        if (is_new_lighting) {
            settings.lighting->SetSelectedValue(kCustomName);
        }

        auto *raw_scene = scene_->GetScene()->GetScene();
        if (!in_basic_mode) {
            raw_scene->EnableIndirectLight(ui_state_.use_ibl);
            raw_scene->SetIndirectLightIntensity(
                    float(ui_state_.ibl_intensity));
            raw_scene->SetSunLightColor(ui_state_.sun_color);
            if (!ui_state_.sun_follows_camera) {
                raw_scene->SetSunLightDirection(ui_state_.sun_dir);
            }
        }
        raw_scene->EnableSunLight(ui_state_.use_sun);
        raw_scene->SetSunLightIntensity(ui_state_.sun_intensity);

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

        scene_->ForceRedraw();
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
            auto cell = settings.geometries->GetItem(group_it->second);
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
        TreeView::ItemId parent = settings.geometries->GetRootItem();
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
                parent = settings.geometries->AddItem(parent, cell);
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
        auto id = settings.geometries->AddItem(parent, cell);
        settings.object2itemid[o.name] = id;
    }

    void UpdateObjectTree() {
#if GROUPS_USE_TREE
        settings.group2itemid.clear();
#endif  // GROUPS_USE_TREE
        settings.object2itemid.clear();
        settings.geometries->Clear();

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
        scene_->GetScene()->ShowGeometry(o.name, IsGeometryVisible(o));
        scene_->ForceRedraw();
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
        scene_->ForceRedraw();  // redraw with new selection highlighted
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
        // Count number of meshes stored in TriangleMeshModels
        size_t model_mesh_count = 0;
        size_t model_count = 0;
        for (auto &o : objects_) {
            if (!IsGeometryVisible(o)) {
                continue;
            }
            if (o.model.get()) {
                model_count++;
                model_mesh_count += o.model.get()->meshes_.size();
            }
        }
        pickable.reserve(objects_.size() + model_mesh_count - model_count);
        for (auto &o : objects_) {
            if (!IsGeometryVisible(o)) {
                continue;
            }
            if (o.model.get()) {
                for (auto &g : o.model->meshes_) {
                    pickable.emplace_back(g.mesh_name, g.mesh.get(),
                                          o.tgeometry.get());
                }
            } else {
                pickable.emplace_back(o.name, o.geometry.get(),
                                      o.tgeometry.get());
            }
        }
        selections_->SetSelectableGeometry(pickable);
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
        scene_->EnableSceneCaching(false);
        scene_->GetScene()->GetScene()->RenderToImage(
                [this, path](std::shared_ptr<geometry::Image> image) mutable {
                    if (!io::WriteImage(path, *image)) {
                        this->window_->ShowMessageBox(
                                "Error",
                                (std::string("Could not write image to ") +
                                 path + ".")
                                        .c_str());
                    }
                    scene_->EnableSceneCaching(true);
                });
    }

    void OnAbout() {
        auto &theme = window_->GetTheme();
        auto dlg = std::make_shared<gui::Dialog>("About");

        auto title = std::make_shared<gui::Label>(
                (std::string("Open3D ") + OPEN3D_VERSION).c_str());
        auto text = std::make_shared<gui::Label>(
                "The MIT License (MIT)\n"
                "Copyright (c) 2018-2023 www.open3d.org\n\n"

                "Permission is hereby granted, free of charge, to any person "
                "obtaining a copy of this software and associated "
                "documentation files (the \"Software\"), to deal in the "
                "Software without restriction, including without limitation "
                "the rights to use, copy, modify, merge, publish, distribute, "
                "sublicense, and/or sell copies of the Software, and to "
                "permit persons to whom the Software is furnished to do so, "
                "subject to the following conditions:\n\n"

                "The above copyright notice and this permission notice shall "
                "be included in all copies or substantial portions of the "
                "Software.\n\n"

                "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY "
                "KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE "
                "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
                "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR "
                "COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER "
                "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR "
                "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE "
                "SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.");
        auto ok = std::make_shared<gui::Button>("OK");
        ok->SetOnClicked([this]() { this->window_->CloseDialog(); });

        gui::Margins margins(theme.font_size);
        auto layout = std::make_shared<gui::Vert>(0, margins);
        layout->AddChild(gui::Horiz::MakeCentered(title));
        layout->AddFixed(theme.font_size);
        auto v = std::make_shared<gui::ScrollableVert>(0);
        v->AddChild(text);
        layout->AddChild(v);
        layout->AddFixed(theme.font_size);
        layout->AddChild(gui::Horiz::MakeCentered(ok));
        dlg->AddChild(layout);

        window_->ShowDialog(dlg);
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

    // Create a message processor for incoming messages.
    auto on_geometry = [this](std::shared_ptr<geometry::Geometry3D> geom,
                              const std::string &path, int time,
                              const std::string &layer) {
        impl_->AddGeometry(path, geom, nullptr, nullptr, nullptr, layer, time,
                           true);
        if (impl_->objects_.size() == 1) {
            impl_->ResetCameraToDefault();
        }
    };
    impl_->message_processor_ =
            std::make_shared<MessageProcessor>(this, on_geometry);

    // Create the app menu. We will take over the existing menubar (if any)
    // since a) we need to cache a pointer, and b) we should be the only
    // window, since the whole point of this class is to have an easy way to
    // visualize something with a blocking call to draw().
    auto menu = std::make_shared<Menu>();
#if defined(__APPLE__)
    // The first menu item to be added on macOS becomes the application
    // menu (no matter its name)
    auto app_menu = std::make_shared<Menu>();
    app_menu->AddItem("About", MENU_ABOUT);
    menu->AddMenu("Open3D", app_menu);
#endif  // __APPLE__

    if (Application::GetInstance().UsingNativeWindows()) {
        auto file_menu = std::make_shared<Menu>();
        file_menu->AddItem("Export Current Image...", MENU_EXPORT_RGB);
        file_menu->AddSeparator();
        file_menu->AddItem("Close Window", MENU_CLOSE, KeyName::KEY_W);
        menu->AddMenu("File", file_menu);
    }

    auto actions_menu = std::make_shared<Menu>();
    actions_menu->AddItem("Show Settings", MENU_SETTINGS);
    actions_menu->SetChecked(MENU_SETTINGS, false);
    menu->AddMenu("Actions", actions_menu);
    impl_->settings.actions_menu = actions_menu.get();

#if !defined(__APPLE__)
    auto help_menu = std::make_shared<Menu>();
    help_menu->AddItem("About", MENU_ABOUT);
    menu->AddMenu("Help", help_menu);
#endif  // !__APPLE__

    Application::GetInstance().SetMenubar(menu);

    SetOnMenuItemActivated(MENU_ABOUT, [this]() { this->impl_->OnAbout(); });
    SetOnMenuItemActivated(MENU_EXPORT_RGB,
                           [this]() { this->impl_->OnExportRGB(); });
    SetOnMenuItemActivated(MENU_CLOSE, [this]() { this->impl_->OnClose(); });
    SetOnMenuItemActivated(MENU_SETTINGS,
                           [this]() { this->impl_->OnToggleSettings(); });

    impl_->ShowSettings(false, false);
}

O3DVisualizer::~O3DVisualizer() {}

Open3DScene *O3DVisualizer::GetScene() const {
    return impl_->scene_->GetScene().get();
}

void O3DVisualizer::StartRPCInterface(const std::string &address, int timeout) {
    impl_->receiver_ = std::make_shared<io::rpc::ZMQReceiver>(address, timeout);
    impl_->receiver_->SetMessageProcessor(impl_->message_processor_);

    try {
        utility::LogInfo("Starting to listen on {}", address);
        impl_->receiver_->Start();
    } catch (std::exception &e) {
        utility::LogWarning("Failed to start RPC interface: {}", e.what());
    }
}

void O3DVisualizer::StopRPCInterface() {
    if (impl_->receiver_) {
        utility::LogInfo("Stopping RPC interface");
    }
    impl_->receiver_.reset();
}

void O3DVisualizer::AddAction(const std::string &name,
                              std::function<void(O3DVisualizer &)> callback) {
    // Add button to the "Custom Actions" segment in the UI
    SmallButton *button = new SmallButton(name.c_str());
    button->SetOnClicked([this, callback]() { callback(*this); });
    impl_->settings.actions->AddChild(GiveOwnership(button));

    SetNeedsLayout();
    impl_->settings.actions_panel->SetVisible(true);
    impl_->settings.actions_panel->SetIsOpen(true);

    if (impl_->can_auto_show_settings_ &&
        impl_->settings.actions->size() == 1) {
        impl_->ShowSettings(true);
    }

    // Add menu item
    if (impl_->settings.menuid2action.empty()) {
        impl_->settings.actions_menu->AddSeparator();
    }
    int id = MENU_ACTIONS_BASE + int(impl_->settings.menuid2action.size());
    impl_->settings.actions_menu->AddItem(name.c_str(), id);
    impl_->settings.menuid2action[id] = callback;
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
        const rendering::MaterialRecord *material /*=nullptr*/,
        const std::string &group /*= ""*/,
        double time /*= 0.0*/,
        bool is_visible /*= true*/) {
    impl_->AddGeometry(name, geom, nullptr, nullptr, material, group, time,
                       is_visible);
}

void O3DVisualizer::AddGeometry(
        const std::string &name,
        std::shared_ptr<t::geometry::Geometry> tgeom,
        const rendering::MaterialRecord *material /*=nullptr*/,
        const std::string &group /*= ""*/,
        double time /*= 0.0*/,
        bool is_visible /*= true*/) {
    impl_->AddGeometry(name, nullptr, tgeom, nullptr, material, group, time,
                       is_visible);
}

void O3DVisualizer::AddGeometry(
        const std::string &name,
        std::shared_ptr<rendering::TriangleMeshModel> model,
        const rendering::MaterialRecord *material /*=nullptr*/,
        const std::string &group /*= ""*/,
        double time /*= 0.0*/,
        bool is_visible /*= true*/) {
    impl_->AddGeometry(name, nullptr, nullptr, model, material, group, time,
                       is_visible);
}

void O3DVisualizer::Add3DLabel(const Eigen::Vector3f &pos, const char *text) {
    impl_->Add3DLabel(pos, text);
}

void O3DVisualizer::Clear3DLabels() { impl_->Clear3DLabels(); }

void O3DVisualizer::UpdateGeometry(const std::string &name,
                                   std::shared_ptr<t::geometry::Geometry> tgeom,
                                   uint32_t update_flags) {
    impl_->UpdateGeometry(name, tgeom, update_flags);
}

void O3DVisualizer::RemoveGeometry(const std::string &name) {
    return impl_->RemoveGeometry(name);
}

void O3DVisualizer::ShowGeometry(const std::string &name, bool show) {
    return impl_->ShowGeometry(name, show);
}

O3DVisualizer::DrawObject O3DVisualizer::GetGeometry(
        const std::string &name) const {
    return impl_->GetGeometry(name);
}

MaterialRecord O3DVisualizer::GetGeometryMaterial(
        const std::string &name) const {
    return impl_->GetGeometryMaterial(name);
}

void O3DVisualizer::ModifyGeometryMaterial(
        const std::string &name, const rendering::MaterialRecord *material) {
    impl_->ModifyGeometryMaterial(name, material);
}

void O3DVisualizer::ShowSettings(bool show) { impl_->ShowSettings(show); }

void O3DVisualizer::ShowSkybox(bool show) { impl_->ShowSkybox(show); }

void O3DVisualizer::SetIBL(const std::string &path) { impl_->SetIBL(path); }

void O3DVisualizer::SetIBLIntensity(float intensity) {
    impl_->SetIBLIntensity(intensity);
}

void O3DVisualizer::ShowAxes(bool show) { impl_->ShowAxes(show); }

void O3DVisualizer::ShowGround(bool show) { impl_->ShowGround(show); }

void O3DVisualizer::SetGroundPlane(rendering::Scene::GroundPlane plane) {
    impl_->SetGroundPlane(plane);
}

void O3DVisualizer::EnableBasicMode(bool enable) {
    impl_->EnableBasicMode(enable);
}

void O3DVisualizer::EnableWireframeMode(bool enable) {
    impl_->EnableWireframeMode(enable);
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

void O3DVisualizer::SetPanelOpen(const std::string &name, bool open) {
    impl_->SetPanelOpen(name, open);
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
    int settings_width = 16 * context.theme.font_size;
#if !GROUPS_USE_TREE
    if (impl_->added_groups_.size() >= 2) {
        settings_width += 5 * context.theme.font_size;
    }
#endif  // !GROUPS_USE_TREE
    if (impl_->min_time_ != impl_->max_time_) {
        settings_width += 3 * context.theme.font_size;
    }

    auto f = GetContentRect();
    impl_->settings.actions->SetWidth(settings_width -
                                      int(std::round(1.5 * em)));
    if (impl_->settings.panel->IsVisible()) {
        impl_->scene_->SetFrame(
                Rect(f.x, f.y, f.width - settings_width, f.height));
        impl_->settings.panel->SetFrame(Rect(f.GetRight() - settings_width, f.y,
                                             settings_width, f.height));
    } else {
        impl_->scene_->SetFrame(f);
    }

    Super::Layout(context);
}

}  // namespace visualizer
}  // namespace visualization
}  // namespace open3d
