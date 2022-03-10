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

#include "open3d/visualization/gui/Window.h"

#include <imgui.h>
#include <imgui_internal.h>  // so we can examine the current context

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Dialog.h"
#include "open3d/visualization/gui/ImguiFilamentBridge.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/Menu.h"
#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"
#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/gui/WindowSystem.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

#ifdef BUILD_WEBRTC
#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"
#endif

// ----------------------------------------------------------------------------
namespace open3d {
namespace visualization {
namespace gui {

namespace {

static constexpr int CENTERED_X = -10000;
static constexpr int CENTERED_Y = -10000;
static constexpr int AUTOSIZE_WIDTH = 0;
static constexpr int AUTOSIZE_HEIGHT = 0;

// Assumes the correct ImGuiContext is current
void UpdateImGuiForScaling(float new_scaling) {
    ImGuiStyle& style = ImGui::GetStyle();
    // FrameBorderSize is not adjusted (we want minimal borders)
    style.FrameRounding *= new_scaling;
}

void ChangeAllRenderQuality(
        SceneWidget::Quality quality,
        const std::vector<std::shared_ptr<Widget>>& children) {
    for (auto child : children) {
        auto sw = std::dynamic_pointer_cast<SceneWidget>(child);
        if (sw) {
            sw->SetRenderQuality(quality);
        } else {
            if (child->GetChildren().size() > 0) {
                ChangeAllRenderQuality(quality, child->GetChildren());
            }
        }
    }
}

struct ImguiWindowContext : public FontContext {
    const Theme* theme = nullptr;
    std::unique_ptr<ImguiFilamentBridge> imgui_bridge;
    ImGuiContext* context = nullptr;
    std::vector<ImFont*> fonts;  // references, not owned by us
    float scaling = 1.0;

    void* GetFont(FontId font_id) { return this->fonts[font_id]; }

    void CreateFonts() {
        // ImGUI puts all fonts into one big texture atlas. However, there are
        // separate ImFont* pointers for each conceptual font. This means that
        // while we can have many fonts, all the fonts that we are ever going
        // to use must be loaded up front, which makes a large font selection
        // inconsistent with small memory footprint. Also, we might bump into
        // OpenGL texture size limitations.
        auto& font_descs = Application::GetInstance().GetFontDescriptions();
        this->fonts.reserve(font_descs.size());
        for (auto& fd : font_descs) {
            this->fonts.push_back(AddFont(fd));
        }

        ImGuiIO& io = ImGui::GetIO();
        unsigned char* pixels;
        int textureW, textureH, bytesPerPx;
        io.Fonts->GetTexDataAsAlpha8(&pixels, &textureW, &textureH,
                                     &bytesPerPx);
        // Some fonts seem to result in 0x0 textures (maybe if the font does
        // not contain any of the code points?), which cause Filament to
        // panic. Handle this gracefully.
        if (textureW == 0 || textureH == 0) {
            utility::LogWarning(
                    "Got zero-byte font texture; ignoring custom fonts");
            io.Fonts->Clear();
            this->fonts[0] =
                    io.Fonts->AddFontFromFileTTF(this->theme->font_path.c_str(),
                                                 float(this->theme->font_size));
            for (unsigned int i = 1; i < font_descs.size(); ++i) {
                this->fonts[i] = this->fonts[0];
            }
            io.Fonts->GetTexDataAsAlpha8(&pixels, &textureW, &textureH,
                                         &bytesPerPx);
        }
        this->imgui_bridge->CreateAtlasTextureAlpha8(pixels, textureW, textureH,
                                                     bytesPerPx);
        ImGui::SetCurrentFont(this->fonts[Application::DEFAULT_FONT_ID]);
    }

    ImFont* AddFont(const FontDescription& fd) {
        // We can assume that everything in the FontDescription is usable, since
        // Application::SetFont() should have ensured that it is usable.
        ImFont* imfont = nullptr;

        ImGuiIO& io = ImGui::GetIO();
        float point_size;
        if (fd.point_size_ <= 0) {
            point_size = float(this->theme->font_size);
        } else {
            point_size = this->scaling * float(fd.point_size_);
        }
        // The first range should be "en" from
        // FontDescription::FontDescription()
        if (fd.ranges_.size() == 1) {
            imfont = io.Fonts->AddFontFromFileTTF(fd.ranges_[0].path.c_str(),
                                                  point_size);
        } else {
            imfont = io.Fonts->AddFontFromFileTTF(
                    fd.ranges_[0].path.c_str(), point_size, NULL,
                    io.Fonts->GetGlyphRangesDefault());
        }

        ImFontConfig config;
        config.MergeMode = true;
        for (auto& r : fd.ranges_) {
            if (!r.lang.empty()) {
                const ImWchar* range;
                if (r.lang == "en") {
                    continue;  // added above, don't add cyrillic too
                } else if (r.lang == "ja") {
                    range = io.Fonts->GetGlyphRangesJapanese();
                } else if (r.lang == "ko") {
                    range = io.Fonts->GetGlyphRangesKorean();
                } else if (r.lang == "th") {
                    range = io.Fonts->GetGlyphRangesThai();
                } else if (r.lang == "vi") {
                    range = io.Fonts->GetGlyphRangesVietnamese();
                } else if (r.lang == "zh") {
                    range = io.Fonts->GetGlyphRangesChineseSimplifiedCommon();
                } else if (r.lang == "zh_all") {
                    range = io.Fonts->GetGlyphRangesChineseFull();
                } else {  // so many languages use Cyrillic it can be the
                          // default
                    range = io.Fonts->GetGlyphRangesCyrillic();
                }
                imfont = io.Fonts->AddFontFromFileTTF(
                        r.path.c_str(), point_size, &config, range);
            } else if (!r.code_points.empty()) {
                // TODO: the ImGui docs say that this must exist until
                // CreateAtlastTextureAlpha8().
                ImVector<ImWchar> range;
                ImFontGlyphRangesBuilder builder;
                for (auto c : r.code_points) {
                    builder.AddChar(c);
                }
                builder.BuildRanges(&range);
                imfont = io.Fonts->AddFontFromFileTTF(
                        r.path.c_str(), point_size, &config, range.Data);
            }
        }

        return imfont;
    }
};

}  // namespace

const int Window::FLAG_HIDDEN = (1 << 0);
const int Window::FLAG_TOPMOST = (1 << 1);

struct Window::Impl {
    Impl() {}
    ~Impl() {}

    WindowSystem::OSWindow window_ = nullptr;
    std::string title_;  // there is no glfwGetWindowTitle()...
    bool draw_menu_ = true;
    std::unordered_map<Menu::ItemId, std::function<void()>> menu_callbacks_;
    std::function<bool(void)> on_tick_event_;
    std::function<bool(void)> on_close_;
    std::function<bool(const KeyEvent&)> on_key_event_;
    // We need these for mouse moves and wheel events.
    // The only source of ground truth is button events, so the rest of
    // the time we monitor key up/down events.
    int mouse_mods_ = 0;  // ORed KeyModifiers
    double last_render_time_ = 0.0;
    double last_button_down_time_ = 0.0;  // we have to compute double-click
    MouseButton last_button_down_ = MouseButton::NONE;

    Theme theme_;  // so that the font size can be different based on scaling
    visualization::rendering::FilamentRenderer* renderer_;
    ImguiWindowContext imgui_;
    std::vector<std::shared_ptr<Widget>> children_;

    // Active dialog is owned here. It is not put in the children because
    // we are going to add it and take it out during draw (since that's
    // how an immediate mode GUI works) and that involves changing the
    // children while iterating over it. Also, conceptually it is not a
    // child, it is a child window, and needs to be on top, which we cannot
    // guarantee if it is a child widget.
    std::shared_ptr<Dialog> active_dialog_;

    std::queue<std::function<void()>> deferred_until_before_draw_;
    std::queue<std::function<void()>> deferred_until_draw_;
    Widget* mouse_grabber_widget_ = nullptr;  // only if not ImGUI widget
    Widget* focus_widget_ =
            nullptr;  // only used if ImGUI isn't taking keystrokes
    bool wants_auto_size_ = false;
    bool wants_auto_center_ = false;
    bool needs_layout_ = true;
    bool needs_redraw_ = true;  // set by PostRedraw to defer if already drawing
    bool is_resizing_ = false;
    bool is_drawing_ = false;
};

Window::Window(const std::string& title, int flags /*= 0*/)
    : Window(title, CENTERED_X, CENTERED_Y, AUTOSIZE_WIDTH, AUTOSIZE_HEIGHT) {}

Window::Window(const std::string& title,
               int width,
               int height,
               int flags /*= 0*/)
    : Window(title, CENTERED_X, CENTERED_Y, width, height) {}

Window::Window(const std::string& title,
               int x,
               int y,
               int width,
               int height,
               int flags /*= 0*/)
    : impl_(new Window::Impl()) {
    // Make sure that the Application instance is initialized before creating
    // the window. It is easy to call, e.g. O3DVisualizer() and forgetting to
    // initialize the application. This will cause a crash because the window
    // system will not exist, nor will the resource directory be located, and
    // so the renderer will not load properly and give cryptic messages.
    Application::GetInstance().VerifyIsInitialized();

    impl_->wants_auto_center_ = (x == CENTERED_X || y == CENTERED_Y);
    impl_->wants_auto_size_ =
            (width == AUTOSIZE_WIDTH || height == AUTOSIZE_HEIGHT);

    bool visible = (!(flags & FLAG_HIDDEN) &&
                    (impl_->wants_auto_size_ || impl_->wants_auto_center_));
    int ws_flags = 0;
    if (!visible) {
        ws_flags |= WindowSystem::FLAG_HIDDEN;
    }
    if (flags & FLAG_TOPMOST) {
        ws_flags |= WindowSystem::FLAG_TOPMOST;
    }

    int initial_width = std::max(10, width);
    int initial_height = std::max(10, height);
    auto& ws = Application::GetInstance().GetWindowSystem();
    impl_->window_ = ws.CreateOSWindow(this, initial_width, initial_height,
                                       title.c_str(), ws_flags);
    impl_->title_ = title;

    if (x != CENTERED_X || y != CENTERED_Y) {
        ws.SetWindowPos(impl_->window_, x, y);
    }

    auto& theme = impl_->theme_;  // shorter alias
    impl_->imgui_.context = ImGui::CreateContext();
    auto oldContext = MakeDrawContextCurrent();

    // ImGUI creates a bitmap atlas from a font, so we need to have the correct
    // size when we create it, because we can't change the bitmap without
    // reloading the whole thing (expensive).
    // Note that GetScaling() gets the pixel scaling. On macOS, coordinates are
    // specified in points, not device pixels. The conversion to device pixels
    // is the scaling factor. On Linux, there is no scaling of pixels (just
    // like in Open3D's GUI library), and glfwGetWindowContentScale() returns
    // the appropriate scale factor for text and icons and such.
    float scaling = ws.GetUIScaleFactor(impl_->window_);
    impl_->imgui_.scaling = scaling;
    impl_->theme_ = Application::GetInstance().GetTheme();
    impl_->theme_.font_size =
            int(std::round(impl_->theme_.font_size * scaling));
    impl_->theme_.default_margin =
            int(std::round(impl_->theme_.default_margin * scaling));
    impl_->theme_.default_layout_spacing =
            int(std::round(impl_->theme_.default_layout_spacing * scaling));

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(0, 0);
    style.WindowRounding = 0;
    style.WindowBorderSize = 0;
    style.FrameBorderSize = float(theme.border_width);
    style.FrameRounding = float(theme.border_radius);
    style.ChildRounding = float(theme.border_radius);
    style.Colors[ImGuiCol_WindowBg] = colorToImgui(theme.background_color);
    style.Colors[ImGuiCol_ChildBg] = colorToImgui(theme.background_color);
    style.Colors[ImGuiCol_Text] = colorToImgui(theme.text_color);
    style.Colors[ImGuiCol_Border] = colorToImgui(theme.border_color);
    style.Colors[ImGuiCol_Button] = colorToImgui(theme.button_color);
    style.Colors[ImGuiCol_ButtonHovered] =
            colorToImgui(theme.button_hover_color);
    style.Colors[ImGuiCol_ButtonActive] =
            colorToImgui(theme.button_active_color);
    style.Colors[ImGuiCol_CheckMark] = colorToImgui(theme.checkbox_check_color);
    style.Colors[ImGuiCol_FrameBg] =
            colorToImgui(theme.combobox_background_color);
    style.Colors[ImGuiCol_FrameBgHovered] =
            colorToImgui(theme.combobox_hover_color);
    style.Colors[ImGuiCol_FrameBgActive] =
            style.Colors[ImGuiCol_FrameBgHovered];
    style.Colors[ImGuiCol_SliderGrab] = colorToImgui(theme.slider_grab_color);
    style.Colors[ImGuiCol_SliderGrabActive] =
            colorToImgui(theme.slider_grab_color);
    style.Colors[ImGuiCol_Tab] = colorToImgui(theme.tab_inactive_color);
    style.Colors[ImGuiCol_TabHovered] = colorToImgui(theme.tab_hover_color);
    style.Colors[ImGuiCol_TabActive] = colorToImgui(theme.tab_active_color);

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;

    // ImGUI's io.KeysDown is indexed by our scan codes, and we fill out
    // io.KeyMap to map from our code to ImGui's code.
    io.KeyMap[ImGuiKey_Tab] = KEY_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow] = KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = KEY_PAGEUP;
    io.KeyMap[ImGuiKey_PageDown] = KEY_PAGEDOWN;
    io.KeyMap[ImGuiKey_Home] = KEY_HOME;
    io.KeyMap[ImGuiKey_End] = KEY_END;
    io.KeyMap[ImGuiKey_Insert] = KEY_INSERT;
    io.KeyMap[ImGuiKey_Delete] = KEY_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = KEY_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = ' ';
    io.KeyMap[ImGuiKey_Enter] = KEY_ENTER;
    io.KeyMap[ImGuiKey_Escape] = KEY_ESCAPE;
    io.KeyMap[ImGuiKey_A] = 'a';
    io.KeyMap[ImGuiKey_C] = 'c';
    io.KeyMap[ImGuiKey_V] = 'v';
    io.KeyMap[ImGuiKey_X] = 'x';
    io.KeyMap[ImGuiKey_Y] = 'y';
    io.KeyMap[ImGuiKey_Z] = 'z';
    /*    io.SetClipboardTextFn = [this](void*, const char* text) {
            glfwSetClipboardString(this->impl_->window, text);
        };
        io.GetClipboardTextFn = [this](void*) -> const char* {
            return glfwGetClipboardString(this->impl_->window);
        }; */
    io.ClipboardUserData = nullptr;

    // Restore the context, in case we are creating a window during a draw.
    // (This is quite likely, since ImGUI only handles things like button
    // presses during draw. A file open dialog is likely to create a window
    // after pressing "Open".)
    RestoreDrawContext(oldContext);

    CreateRenderer();
}

void Window::CreateRenderer() {
    // This is a delayed part of the constructor. See comment at end of ctor.
    auto old_context = MakeDrawContextCurrent();

    // On single-threaded platforms, Filament's OpenGL context must be current,
    // not GLFW's context, so create the renderer after the window.
    impl_->renderer_ =
            Application::GetInstance().GetWindowSystem().CreateRenderer(
                    impl_->window_);
    impl_->renderer_->SetClearColor({1.0f, 1.0f, 1.0f, 1.0f});

    impl_->imgui_.imgui_bridge =
            std::make_unique<ImguiFilamentBridge>(impl_->renderer_, GetSize());
    impl_->imgui_.theme = &impl_->theme_;
    impl_->imgui_.CreateFonts();

    RestoreDrawContext(old_context);
}

Window::~Window() {
    impl_->active_dialog_.reset();
    impl_->children_.clear();  // needs to happen before deleting renderer
    ImGui::SetCurrentContext(impl_->imgui_.context);
    ImGui::DestroyContext();
    delete impl_->renderer_;
    DestroyWindow();
}

void Window::DestroyWindow() {
    Application::GetInstance().GetWindowSystem().DestroyWindow(impl_->window_);
    // Ensure DestroyWindow() can be called multiple times, which will
    // happen if you call DestroyWindow() before the destructor.
    impl_->window_ = nullptr;
}

int Window::GetMouseMods() const { return impl_->mouse_mods_; }

std::string Window::GetWebRTCUID() const {
#ifdef BUILD_WEBRTC
    if (auto* webrtc_ws = dynamic_cast<webrtc_server::WebRTCWindowSystem*>(
                &Application::GetInstance().GetWindowSystem())) {
        return webrtc_ws->GetWindowUID(impl_->window_);
    } else {
        return "window_undefined";
    }
#else
    return "window_undefined";
#endif
}

const std::vector<std::shared_ptr<Widget>>& Window::GetChildren() const {
    return impl_->children_;
}

void* Window::MakeDrawContextCurrent() const {
    auto old_context = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(impl_->imgui_.context);
    return old_context;
}

void Window::RestoreDrawContext(void* oldContext) const {
    ImGui::SetCurrentContext((ImGuiContext*)oldContext);
}

const Theme& Window::GetTheme() const { return impl_->theme_; }

visualization::rendering::Renderer& Window::GetRenderer() const {
    return *impl_->renderer_;
}

Rect Window::GetOSFrame() const {
    auto& ws = Application::GetInstance().GetWindowSystem();
    auto pos = ws.GetWindowPos(impl_->window_);
    auto size = ws.GetWindowSize(impl_->window_);
    return Rect(pos.x, pos.y, size.width, size.height);
}

void Window::SetOSFrame(const Rect& r) {
    auto& ws = Application::GetInstance().GetWindowSystem();
    ws.SetWindowPos(impl_->window_, r.x, r.y);
    ws.SetWindowSize(impl_->window_, r.width, r.height);
}

const char* Window::GetTitle() const { return impl_->title_.c_str(); }

void Window::SetTitle(const char* title) {
    impl_->title_ = title;
    return Application::GetInstance().GetWindowSystem().SetWindowTitle(
            impl_->window_, title);
}

// Note: can only be called if the ImGUI context is current (that is,
//       after MakeDrawContextCurrent() has been called), otherwise
//       ImGUI won't be able to access the font.
Size Window::CalcPreferredSize() {
    // If we don't have any children--unlikely, but might happen when you're
    // experimenting and just create an empty window to see if you understand
    // how to config the library--return a non-zero size, since a size of (0, 0)
    // will end up with a crash.
    if (impl_->children_.empty()) {
        return Size(int(std::round(640.0f * impl_->imgui_.scaling)),
                    int(std::round(480.0f * impl_->imgui_.scaling)));
    }

    Rect bbox(0, 0, 0, 0);
    for (auto& child : impl_->children_) {
        auto pref = child->CalcPreferredSize(GetLayoutContext(),
                                             Widget::Constraints());
        Rect r(child->GetFrame().x, child->GetFrame().y, pref.width,
               pref.height);
        bbox = bbox.UnionedWith(r);
    }

    // Note: we are doing (bbox.GetRight() - 0) NOT (bbox.GetRight() - bbox.x)
    //       (and likewise for height) because the origin of the window is
    //       (0, 0) and anything up/left is clipped.
    return Size(bbox.GetRight(), bbox.GetBottom());
}

void Window::SizeToFit() {
    // CalcPreferredSize() can only be called while the ImGUI context
    // is current, but we are probably calling this while setting up the
    // window.
    auto auto_size = [this]() { SetSize(CalcPreferredSize()); };
    impl_->deferred_until_draw_.push(auto_size);
}

void Window::SetSize(const Size& size) {
    // Make sure we do the resize outside of a draw, to avoid unsightly
    // errors if we happen to do this in the middle of a draw.
    auto resize = [this, size /*copy*/]() {
        auto scaling = this->impl_->imgui_.scaling;
        int width = int(std::round(float(size.width) / scaling));
        int height = int(std::round(float(size.height) / scaling));
        Application::GetInstance().GetWindowSystem().SetWindowSize(
                impl_->window_, width, height);
    };
    impl_->deferred_until_before_draw_.push(resize);
}

Size Window::GetSize() const {
    return Application::GetInstance().GetWindowSystem().GetWindowSizePixels(
            impl_->window_);
}

Rect Window::GetContentRect() const {
    auto size = GetSize();
    int menu_height = 0;
    MakeDrawContextCurrent();
    auto menubar = Application::GetInstance().GetMenubar();
    if (menubar && impl_->draw_menu_) {
        menu_height = menubar->CalcHeight(GetTheme());
    }

    return Rect(0, menu_height, size.width, size.height - menu_height);
}

float Window::GetScaling() const {
    return Application::GetInstance().GetWindowSystem().GetWindowScaleFactor(
            impl_->window_);
}

Point Window::GlobalToWindowCoord(int global_x, int global_y) {
    auto pos = Application::GetInstance().GetWindowSystem().GetWindowPos(
            impl_->window_);
    return Point(global_y - pos.x, global_y - pos.y);
}

bool Window::IsVisible() const {
    return Application::GetInstance().GetWindowSystem().GetWindowIsVisible(
            impl_->window_);
}

void Window::Show(bool vis /*= true*/) {
    Application::GetInstance().GetWindowSystem().ShowWindow(impl_->window_,
                                                            vis);
}

void Window::Close() {
    if (impl_->on_close_) {
        bool should_close = impl_->on_close_();
        if (!should_close) {
            Application::GetInstance().GetWindowSystem().CancelUserClose(
                    impl_->window_);
            return;
        }
    }
    Application::GetInstance().RemoveWindow(this);
}

void Window::SetNeedsLayout() { impl_->needs_layout_ = true; }

void Window::PostRedraw() {
    // Windows cannot actually post an expose event, and the actual mechanism
    // requires that PostNativeExposeEvent() not be called while drawing
    // (see the implementation for details).
    if (impl_->is_drawing_) {
        impl_->needs_redraw_ = true;
    } else {
        Application::GetInstance().GetWindowSystem().PostRedrawEvent(
                impl_->window_);
    }
}

void Window::RaiseToTop() const {
    Application::GetInstance().GetWindowSystem().RaiseWindowToTop(
            impl_->window_);
}

bool Window::IsActiveWindow() const {
    return Application::GetInstance().GetWindowSystem().IsActiveWindow(
            impl_->window_);
}

void Window::SetFocusWidget(Widget* w) { impl_->focus_widget_ = w; }

void Window::AddChild(std::shared_ptr<Widget> w) {
    impl_->children_.push_back(w);
    impl_->needs_layout_ = true;
}

void Window::SetOnMenuItemActivated(Menu::ItemId item_id,
                                    std::function<void()> callback) {
    impl_->menu_callbacks_[item_id] = callback;
}

void Window::SetOnTickEvent(std::function<bool()> callback) {
    impl_->on_tick_event_ = callback;
}

void Window::SetOnClose(std::function<bool()> callback) {
    impl_->on_close_ = callback;
}

void Window::SetOnKeyEvent(std::function<bool(const KeyEvent&)> callback) {
    impl_->on_key_event_ = callback;
}

void Window::ShowDialog(std::shared_ptr<Dialog> dlg) {
    if (impl_->active_dialog_) {
        CloseDialog();
    }
    impl_->active_dialog_ = dlg;
    dlg->OnWillShow();

    auto deferred_layout = [this, dlg]() {
        auto context = GetLayoutContext();
        auto content_rect = GetContentRect();
        auto pref = dlg->CalcPreferredSize(context, Widget::Constraints());
        int w = dlg->GetFrame().width;
        int h = dlg->GetFrame().height;
        if (w == 0) {
            w = pref.width;
        }
        if (h == 0) {
            h = pref.height;
        }
        w = std::min(w, int(std::round(0.8 * content_rect.width)));
        h = std::min(h, int(std::round(0.8 * content_rect.height)));
        dlg->SetFrame(gui::Rect((content_rect.width - w) / 2,
                                (content_rect.height - h) / 2, w, h));
        dlg->Layout(context);
    };

    impl_->deferred_until_draw_.push(deferred_layout);
}

void Window::CloseDialog() {
    if (impl_->focus_widget_ == impl_->active_dialog_.get()) {
        SetFocusWidget(nullptr);
    }
    impl_->active_dialog_.reset();

    // The dialog might not be closing from within a draw call, such as when
    // a native file dialog closes, so we need to post a redraw, just in case.
    // If it is from within a draw call, then any redraw request from that will
    // get merged in with this one by the OS.
    PostRedraw();
}

void Window::ShowMessageBox(const char* title, const char* message) {
    auto em = GetTheme().font_size;
    auto margins = Margins(GetTheme().default_margin);
    auto dlg = std::make_shared<Dialog>(title);
    auto layout = std::make_shared<Vert>(em, margins);
    layout->AddChild(std::make_shared<Label>(message));
    auto ok = std::make_shared<Button>("Ok");
    ok->SetOnClicked([this]() { this->CloseDialog(); });
    layout->AddChild(Horiz::MakeCentered(ok));
    dlg->AddChild(layout);
    ShowDialog(dlg);
}

void Window::ShowMenu(bool show) {
    impl_->draw_menu_ = show;
    SetNeedsLayout();
}

LayoutContext Window::GetLayoutContext() { return {GetTheme(), impl_->imgui_}; }

void Window::Layout(const LayoutContext& context) {
    if (impl_->children_.size() == 1) {
        auto r = GetContentRect();
        impl_->children_[0]->SetFrame(r);
        impl_->children_[0]->Layout(context);
    } else {
        for (auto& child : impl_->children_) {
            child->Layout(context);
        }
    }
}

void Window::OnMenuItemSelected(Menu::ItemId item_id) {
    auto callback = impl_->menu_callbacks_.find(item_id);
    if (callback != impl_->menu_callbacks_.end()) {
        callback->second();
        PostRedraw();  // might not be in a draw if from native menu
    }
}

WindowSystem::OSWindow Window::GetOSWindow() const { return impl_->window_; }

namespace {
enum Mode { NORMAL, DIALOG, NO_INPUT };

Widget::DrawResult DrawChild(DrawContext& dc,
                             const char* name,
                             std::shared_ptr<Widget> child,
                             Mode mode) {
    // Note: ImGUI's concept of a "window" is really a moveable child of the
    //       OS window. We want a child to act like a child of the OS window,
    //       like native UI toolkits, Qt, etc. So the top-level widgets of
    //       a window are drawn using ImGui windows whose frame is specified
    //       and which have no title bar, resizability, etc.

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse;
    // Q: When we want no input, why not use ImGui::BeginPopupModal(),
    //    which takes care of blocking input for us, since a modal popup
    //    is the most likely use case for wanting no input?
    // A: It animates an overlay, which would require us to constantly
    //    redraw, otherwise it only animates when the mouse moves. But
    //    we don't need constant animation for anything else, so that would
    //    be a waste of CPU and battery (and really annoys people like me).
    if (mode == NO_INPUT) {
        flags |= ImGuiWindowFlags_NoInputs;
    }
    auto frame = child->GetFrame();
    bool bg_color_not_default = !child->IsDefaultBackgroundColor();
    auto is_3d = (std::dynamic_pointer_cast<SceneWidget>(child) != nullptr);
    if (!is_3d) {
        dc.uiOffsetX = frame.x;
        dc.uiOffsetY = frame.y;
        ImGui::SetNextWindowPos(ImVec2(float(frame.x), float(frame.y)));
        ImGui::SetNextWindowSize(
                ImVec2(float(frame.width), float(frame.height)));
        if (bg_color_not_default) {
            auto& bgColor = child->GetBackgroundColor();
            ImGui::PushStyleColor(ImGuiCol_WindowBg, colorToImgui(bgColor));
        }
        ImGui::Begin(name, nullptr, flags);
    } else {
        dc.uiOffsetX = 0;
        dc.uiOffsetY = 0;
    }

    Widget::DrawResult result;
    result = child->Draw(dc);

    if (!is_3d) {
        ImGui::End();
        if (bg_color_not_default) {
            ImGui::PopStyleColor();
        }
    }

    return result;
}
}  // namespace

Widget::DrawResult Window::DrawOnce(bool is_layout_pass) {
    // These are here to provide fast unique window names. (Hence using
    // char* instead of a std::string, just in case c_str() recreates
    // the buffer on some platform and unwittingly makes
    // ImGui::DrawChild(dc, name.c_str(), ...) slow.
    // If you find yourself needing more than a handful of top-level
    // children, you should probably be using a layout of some sort
    // (gui::Vert, gui::Horiz, gui::VGrid, etc. See Layout.h).
    static const std::vector<const char*> win_names = {
            "win1",  "win2",  "win3",  "win4",  "win5",  "win6",  "win7",
            "win8",  "win9",  "win10", "win11", "win12", "win13", "win14",
            "win15", "win16", "win17", "win18", "win19", "win20"};

    bool needs_layout = false;
    bool needs_redraw = false;

    // ImGUI uses the dt parameter to calculate double-clicks, so it
    // needs to be reasonably accurate.
    double now = Application::GetInstance().Now();
    double dt_sec = now - impl_->last_render_time_;
    impl_->last_render_time_ = now;

    // Run the deferred callbacks that need to happen outside a draw
    while (!impl_->deferred_until_before_draw_.empty()) {
        impl_->deferred_until_before_draw_.front()();
        impl_->deferred_until_before_draw_.pop();
    }

    // Set current context
    MakeDrawContextCurrent();  // make sure our ImGUI context is active
    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = float(dt_sec);

    // Set mouse information
    io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
    auto& ws = Application::GetInstance().GetWindowSystem();
    if (IsActiveWindow()) {
        auto mouse_pos = ws.GetMousePosInWindow(impl_->window_);
        io.MousePos = ImVec2(float(mouse_pos.x), float(mouse_pos.y));
    }
    auto buttons = ws.GetMouseButtons(impl_->window_);
    io.MouseDown[0] = (buttons & int(MouseButton::LEFT));
    io.MouseDown[1] = (buttons & int(MouseButton::RIGHT));
    io.MouseDown[2] = (buttons & int(MouseButton::MIDDLE));

    // Set key information
    io.KeyShift = (impl_->mouse_mods_ & int(KeyModifier::SHIFT));
    io.KeyAlt = (impl_->mouse_mods_ & int(KeyModifier::ALT));
    io.KeyCtrl = (impl_->mouse_mods_ & int(KeyModifier::CTRL));
    io.KeySuper = (impl_->mouse_mods_ & int(KeyModifier::META));

    // Begin an ImGUI frame. We should NOT begin a filament frame here:
    // a) ImGUI always needs to "draw", because event processing happens
    //    during draw for immediate mode GUIs, but if this is a layout
    //    pass (as ImGUI can take up two draws to layout widgets and text)
    //    we aren't actually going to render it.
    // b) Filament pumps events during a beginFrame(), which can cause
    //    a key up event to process and erase the key down state from
    //    the ImGuiIO structure before we get a chance to draw/process it.
    ImGui::NewFrame();
    ImGui::PushFont(
            (ImFont*)impl_->imgui_.GetFont(Application::DEFAULT_FONT_ID));

    // Run the deferred callbacks that need to happen inside a draw
    // In particular, text sizing with ImGUI seems to require being
    // in a frame, otherwise there isn't an GL texture info and we crash.
    while (!impl_->deferred_until_draw_.empty()) {
        impl_->deferred_until_draw_.front()();
        impl_->deferred_until_draw_.pop();
    }

    // Layout if necessary.  This must happen within ImGui setup so that widgets
    // can query font information.
    auto& theme = impl_->theme_;
    if (impl_->needs_layout_) {
        Layout(GetLayoutContext());
        impl_->needs_layout_ = false;
    }

    auto size = GetSize();
    int em = theme.font_size;  // em = font size in digital type (see Wikipedia)
    DrawContext dc{theme,
                   *impl_->renderer_,
                   impl_->imgui_,
                   0,
                   0,
                   size.width,
                   size.height,
                   em,
                   float(dt_sec)};

    // Draw all the widgets. These will get recorded by ImGui.
    size_t win_idx = 0;
    Mode draw_mode = (impl_->active_dialog_ ? NO_INPUT : NORMAL);
    for (auto& child : this->impl_->children_) {
        if (!child->IsVisible()) {
            continue;
        }
        if (win_idx >= win_names.size()) {
            win_idx = win_names.size() - 1;
            utility::LogWarning(
                    "Using too many top-level child widgets; use a layout "
                    "instead.");
        }
        auto result = DrawChild(dc, win_names[win_idx++], child, draw_mode);
        if (result != Widget::DrawResult::NONE) {
            needs_redraw = true;
        }
        if (result == Widget::DrawResult::RELAYOUT) {
            needs_layout = true;
        }
    }

    // Draw menubar after the children so it is always on top (although it
    // shouldn't matter, as there shouldn't be anything under it)
    auto menubar = Application::GetInstance().GetMenubar();
    if (menubar && impl_->draw_menu_) {
        auto id = menubar->DrawMenuBar(dc, !impl_->active_dialog_);
        if (id != Menu::NO_ITEM) {
            OnMenuItemSelected(id);
            needs_redraw = true;
        }
    }

    // Draw any active dialog
    if (impl_->active_dialog_) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize,
                            float(theme.dialog_border_width));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,
                            float(theme.dialog_border_radius));
        if (DrawChild(dc, "dialog", impl_->active_dialog_, DIALOG) !=
            Widget::DrawResult::NONE) {
            needs_redraw = true;
        }
        ImGui::PopStyleVar(2);
    }

    // Finish frame and generate the commands
    ImGui::PopFont();
    ImGui::EndFrame();
    ImGui::Render();  // creates the draw data (i.e. Render()s to data)

    // Draw the ImGui commands
    impl_->imgui_.imgui_bridge->Update(ImGui::GetDrawData());

    // Draw. Since ImGUI is an immediate mode gui, it does layout during
    // draw, and if we are drawing for layout purposes, don't actually
    // draw, because we are just going to draw again after this returns.
    if (!is_layout_pass) {
        impl_->renderer_->BeginFrame();
        impl_->renderer_->Draw();
        impl_->renderer_->EndFrame();
    }

    if (needs_layout) {
        return Widget::DrawResult::RELAYOUT;
    } else if (needs_redraw) {
        return Widget::DrawResult::REDRAW;
    } else {
        return Widget::DrawResult::NONE;
    }
}

void Window::OnDraw() {
    impl_->is_drawing_ = true;
    bool needed_layout = impl_->needs_layout_;

    auto result = DrawOnce(needed_layout);
    if (result == Widget::DrawResult::RELAYOUT) {
        impl_->needs_layout_ = true;
    }

    // ImGUI can take two frames to do its layout, so if we did a layout
    // redraw a second time. This helps prevent a brief red flash when the
    // window first appears, as well as corrupted images if the
    // window initially appears underneath the mouse.
    if (needed_layout || impl_->needs_layout_) {
        DrawOnce(false);
    }

    impl_->is_drawing_ = false;
    if (impl_->needs_redraw_) {
        result = Widget::DrawResult::REDRAW;
        impl_->needs_redraw_ = false;
    }

    if (result == Widget::DrawResult::REDRAW) {
        // Can't just draw here, because Filament sometimes fences within
        // a draw, and then you can get two draws happening at the same
        // time, which ends up with a crash.
        PostRedraw();
    }
}

void Window::OnResize() {
    impl_->needs_layout_ = true;

    Application::GetInstance().GetWindowSystem().ResizeRenderer(
            impl_->window_, impl_->renderer_);

    impl_->imgui_.imgui_bridge->OnWindowResized(*this);

    auto size = GetSize();
    auto scaling = GetScaling();

    auto old_context = MakeDrawContextCurrent();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(size.width), float(size.height));
    if (impl_->imgui_.scaling != scaling) {
        UpdateImGuiForScaling(1.0f / impl_->imgui_.scaling);  // undo previous
        UpdateImGuiForScaling(scaling);
        impl_->imgui_.scaling = scaling;
    }
    io.DisplayFramebufferScale.x = 1.0f;
    io.DisplayFramebufferScale.y = 1.0f;

    if (impl_->wants_auto_size_ || impl_->wants_auto_center_) {
        auto& ws = Application::GetInstance().GetWindowSystem();
        auto screen_size = ws.GetScreenSize(impl_->window_);
        int w = GetOSFrame().width;
        int h = GetOSFrame().height;

        if (impl_->wants_auto_size_) {
            ImGui::NewFrame();
            ImGui::PushFont((ImFont*)impl_->imgui_.GetFont(
                    Application::DEFAULT_FONT_ID));
            auto pref = CalcPreferredSize();
            ImGui::PopFont();
            ImGui::EndFrame();

            w = std::min(screen_size.width,
                         int(std::round(pref.width / impl_->imgui_.scaling)));
            // screen_height is the screen height, not the usable screen height.
            // If we cannot call glfwGetMonitorWorkarea(), then we need to guess
            // at the size. The window titlebar is about 2 * em, and then there
            // is often a global menubar (Linux/GNOME, macOS) or a toolbar
            // (Windows). A toolbar is somewhere around 2 - 3 ems.
            int unusable_height = 4 * impl_->theme_.font_size;
            h = std::min(screen_size.height - unusable_height,
                         int(std::round(pref.height / impl_->imgui_.scaling)));
            ws.SetWindowSize(impl_->window_, w, h);
        }

        if (impl_->wants_auto_center_) {
            int x = (screen_size.width - w) / 2;
            int y = (screen_size.height - h) / 2;
            ws.SetWindowPos(impl_->window_, x, y);
        }

        impl_->wants_auto_size_ = false;
        impl_->wants_auto_center_ = false;

        OnResize();
    }

    // Resizing looks bad if drawing takes a long time, so turn off MSAA
    // while we resize. On macOS this is critical, because the GL driver does
    // not release the memory for all the buffers of the new sizes right away
    // so it eats up GBs of memory rapidly and then resizing looks awful and
    // eventually stops working correctly. Unfortunately, there isn't a good
    // way to tell when we've stopped resizing, so we use the mouse movement.
    // (We get no mouse events while resizing, so any mouse even must mean we
    // are no longer resizing.)
    if (!impl_->is_resizing_) {
        impl_->is_resizing_ = true;
        ChangeAllRenderQuality(SceneWidget::Quality::FAST, impl_->children_);
    }

    RestoreDrawContext(old_context);
    PostRedraw();
}

void Window::OnMouseEvent(const MouseEvent& e) {
    MakeDrawContextCurrent();

    // We don't have a good way of determining when resizing ends; the most
    // likely action after resizing a window is to move the mouse.
    if (impl_->is_resizing_) {
        impl_->is_resizing_ = false;
        ChangeAllRenderQuality(SceneWidget::Quality::BEST, impl_->children_);
    }

    impl_->mouse_mods_ = e.modifiers;

    switch (e.type) {
        case MouseEvent::MOVE:
        case MouseEvent::BUTTON_DOWN:
        case MouseEvent::DRAG:
        case MouseEvent::BUTTON_UP:
            break;
        case MouseEvent::WHEEL: {
            ImGuiIO& io = ImGui::GetIO();
            float dx = 0.0, dy = 0.0;
            if (e.wheel.dx != 0) {
                dx = e.wheel.dx / std::abs(e.wheel.dx);  // get sign
            }
            if (e.wheel.dy != 0) {
                dy = e.wheel.dy / std::abs(e.wheel.dy);  // get sign
            }
            // Note: ImGUI's documentation says that 1 unit of wheel movement
            //       is about 5 lines of text scrolling.
            if (e.wheel.isTrackpad) {
                io.MouseWheelH += dx * 0.25f;
                io.MouseWheel += dy * 0.25f;
            } else {
                io.MouseWheelH += dx;
                io.MouseWheel += dy;
            }
            break;
        }
    }

    if (impl_->mouse_grabber_widget_) {
        impl_->mouse_grabber_widget_->Mouse(e);
        if (e.type == MouseEvent::BUTTON_UP) {
            impl_->mouse_grabber_widget_ = nullptr;
        }
        PostRedraw();
        return;
    }

    // Some ImGUI widgets have popup windows, in particular, the color
    // picker, which creates a popup window when you click on the color
    // patch. Since these aren't gui::Widgets, we don't know about them,
    // and will deliver mouse events to something below them. So find any
    // that would use the mouse, and if it isn't a toplevel child, then
    // eat the event for it.
    if (e.type == MouseEvent::BUTTON_DOWN || e.type == MouseEvent::BUTTON_UP) {
        ImGuiContext* context = ImGui::GetCurrentContext();
        for (auto* w : context->Windows) {
            if (w->Flags & ImGuiWindowFlags_Popup &&
                ImGui::IsPopupOpen(w->PopupId)) {
                Rect r(int(w->Pos.x), int(w->Pos.y), int(w->Size.x),
                       int(w->Size.y));
                if (r.Contains(e.x, e.y)) {
                    bool weKnowThis = false;
                    for (auto child : impl_->children_) {
                        if (child->GetFrame() == r) {
                            weKnowThis = true;
                            break;
                        }
                    }
                    if (!weKnowThis) {
                        // This is not a rect that is one of our children,
                        // must be an ImGUI internal popup. Eat event.
                        PostRedraw();
                        return;
                    }
                }
            }
        }
    }

    // Iterate backwards so that we send mouse events from the top down.
    auto HandleMouseForChild = [this](const MouseEvent& e,
                                      std::shared_ptr<Widget> child) -> bool {
        if (child->GetFrame().Contains(e.x, e.y) && child->IsVisible()) {
            if (e.type == MouseEvent::BUTTON_DOWN) {
                SetFocusWidget(child.get());
            }
            auto result = child->Mouse(e);
            if (e.type == MouseEvent::BUTTON_DOWN) {
                if (result == Widget::EventResult::CONSUMED) {
                    impl_->mouse_grabber_widget_ = child.get();
                }
            } else if (e.type == MouseEvent::BUTTON_UP) {
                impl_->mouse_grabber_widget_ = nullptr;
            }
            return true;
        }
        return false;
    };
    if (impl_->active_dialog_) {
        HandleMouseForChild(e, impl_->active_dialog_);
    } else {
        // Mouse move and wheel always get delivered.
        // Button up and down get delivered if they weren't in an ImGUI popup.
        // Drag should only be delivered if the grabber widget exists;
        // if it is null, then the mouse is being dragged over an ImGUI popup.
        if (e.type != MouseEvent::DRAG || impl_->mouse_grabber_widget_) {
            std::vector<std::shared_ptr<Widget>>& children = impl_->children_;
            for (auto it = children.rbegin(); it != children.rend(); ++it) {
                if (HandleMouseForChild(e, *it)) {
                    break;
                }
            }
        }
    }

    PostRedraw();
}

void Window::OnKeyEvent(const KeyEvent& e) {
    auto this_mod = 0;
    if (e.key == KEY_LSHIFT || e.key == KEY_RSHIFT) {
        this_mod = int(KeyModifier::SHIFT);
    } else if (e.key == KEY_LCTRL || e.key == KEY_RCTRL) {
        this_mod = int(KeyModifier::CTRL);
    } else if (e.key == KEY_ALT) {
        this_mod = int(KeyModifier::ALT);
    } else if (e.key == KEY_META) {
        this_mod = int(KeyModifier::META);
    } else if (e.key == KEY_ESCAPE) {
        Close();
    }

    if (e.type == KeyEvent::UP) {
        impl_->mouse_mods_ &= ~this_mod;
    } else {
        impl_->mouse_mods_ |= this_mod;
    }

    auto old_context = MakeDrawContextCurrent();
    ImGuiIO& io = ImGui::GetIO();
    if (e.key < IM_ARRAYSIZE(io.KeysDown)) {
        io.KeysDown[e.key] = (e.type == KeyEvent::DOWN);
    }

    // If an ImGUI widget is not getting keystrokes, we can send them to
    // non-ImGUI widgets
    if (ImGui::GetCurrentContext()->ActiveId == 0) {
        // dispatch key event to focused widget if not intercepted
        if (!impl_->on_key_event_ || !impl_->on_key_event_(e)) {
            if (impl_->focus_widget_) {
                impl_->focus_widget_->Key(e);
            }
        }
    }

    RestoreDrawContext(old_context);
    PostRedraw();
}

void Window::OnTextInput(const TextInputEvent& e) {
    auto old_context = MakeDrawContextCurrent();
    ImGuiIO& io = ImGui::GetIO();
    io.AddInputCharactersUTF8(e.utf8);
    RestoreDrawContext(old_context);

    PostRedraw();
}

void Window::OnTickEvent(const TickEvent& e) {
    auto old_context = MakeDrawContextCurrent();
    bool redraw = false;

    if (impl_->on_tick_event_) {
        redraw = impl_->on_tick_event_();
    }

    for (auto child : impl_->children_) {
        if (child->Tick(e) == Widget::DrawResult::REDRAW) {
            redraw = true;
        }
    }
    RestoreDrawContext(old_context);

    if (redraw) {
        PostRedraw();
    }
}

void Window::OnDragDropped(const char* path) {}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
