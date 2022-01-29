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

#include "pybind/visualization/gui/gui.h"

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/geometry/Image.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Checkbox.h"
#include "open3d/visualization/gui/Color.h"
#include "open3d/visualization/gui/ColorEdit.h"
#include "open3d/visualization/gui/Combobox.h"
#include "open3d/visualization/gui/Dialog.h"
#include "open3d/visualization/gui/FileDialog.h"
#include "open3d/visualization/gui/Gui.h"
#include "open3d/visualization/gui/ImageWidget.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Label3D.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/ListView.h"
#include "open3d/visualization/gui/NumberEdit.h"
#include "open3d/visualization/gui/ProgressBar.h"
#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/gui/Slider.h"
#include "open3d/visualization/gui/StackedWidget.h"
#include "open3d/visualization/gui/TabControl.h"
#include "open3d/visualization/gui/TextEdit.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/ToggleSwitch.h"
#include "open3d/visualization/gui/TreeView.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/gui/WidgetProxy.h"
#include "open3d/visualization/gui/WidgetStack.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderToBuffer.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/visualization/visualization.h"
#include "pybind11/functional.h"

namespace open3d {
namespace visualization {
namespace gui {

class PythonUnlocker : public Application::EnvUnlocker {
public:
    PythonUnlocker() { unlocker_ = nullptr; }
    ~PythonUnlocker() {
        if (unlocker_) {  // paranoia; this shouldn't happen
            delete unlocker_;
        }
    }

    void unlock() { unlocker_ = new py::gil_scoped_release(); }
    void relock() {
        delete unlocker_;
        unlocker_ = nullptr;
    }

private:
    py::gil_scoped_release *unlocker_;
};

class PyWindow : public Window {
    using Super = Window;

public:
    explicit PyWindow(const std::string &title, int flags = 0)
        : Super(title, flags) {}
    PyWindow(const std::string &title, int width, int height, int flags = 0)
        : Super(title, width, height, flags) {}
    PyWindow(const std::string &title,
             int x,
             int y,
             int width,
             int height,
             int flags = 0)
        : Super(title, x, y, width, height, flags) {}

    std::function<void(const LayoutContext &)> on_layout_;

protected:
    void Layout(const LayoutContext &context) {
        if (on_layout_) {
            // the Python callback sizes the children
            on_layout_(context);
            // and then we need to layout the children
            for (auto child : GetChildren()) {
                child->Layout(context);
            }
        } else {
            Super::Layout(context);
        }
    }
};

// atexit: Filament crashes if the engine was not destroyed before exit().
// As far as I can tell, the bluegl mutex, which is a static variable,
// gets destroyed before the render thread gets around to calling
// bluegl::unbind(), thus crashing. So, we need to make sure Filament gets
// cleaned up before C++ starts cleaning up static variables. But we don't want
// to clean up this way unless something catastrophic happens (e.g. the Python
// interpreter is exiting due to a fatal exception). Some cases we need to
// consider:
//  1) exception before calling Application.instance.run()
//  2) exception during Application.instance.run(), namely within a UI callback
//  3) exception after Application.instance.run() successfully finishes
// If Python is exiting normally, then Application::Run() should have already
// cleaned up Filament. So if we still need to clean up Filament at exit(),
// we must be panicking. It is a little difficult to check this, though, but
// Application::OnTerminate() should work even if we've already cleaned up,
// it will just end up being a no-op.
bool g_installed_atexit = false;
void cleanup_filament_atexit() { Application::GetInstance().OnTerminate(); }

void install_cleanup_atexit() {
    if (!g_installed_atexit) {
        atexit(cleanup_filament_atexit);
    }
}

void InitializeForPython(std::string resource_path /*= ""*/) {
    if (resource_path.empty()) {
        // We need to find the resources directory. Fortunately,
        // Python knows where the module lives (open3d.__file__
        // is the path to
        // __init__.py), so we can use that to find the
        // resources included in the wheel.
        py::object o3d = py::module::import("open3d");
        auto o3d_init_path = o3d.attr("__file__").cast<std::string>();
        auto module_path =
                utility::filesystem::GetFileParentDirectory(o3d_init_path);
        resource_path = module_path + "/resources";
    }
    Application::GetInstance().Initialize(resource_path.c_str());
    install_cleanup_atexit();
}

std::shared_ptr<geometry::Image> RenderToImageWithoutWindow(
        rendering::Open3DScene *scene, int width, int height) {
    return Application::GetInstance().RenderToImage(
            scene->GetRenderer(), scene->GetView(), scene->GetScene(), width,
            height);
}

std::shared_ptr<geometry::Image> RenderToDepthImageWithoutWindow(
        rendering::Open3DScene *scene,
        int width,
        int height,
        bool z_in_view_space /* = false */) {
    return Application::GetInstance().RenderToDepthImage(
            scene->GetRenderer(), scene->GetView(), scene->GetScene(), width,
            height, z_in_view_space);
}

enum class EventCallbackResult { IGNORED = 0, HANDLED, CONSUMED };

void pybind_gui_classes(py::module &m) {
    // ---- FontStyle ----
    py::enum_<FontStyle> font_style(m, "FontStyle", "Font style");
    font_style.value("NORMAL", FontStyle::NORMAL)
            .value("BOLD", FontStyle::BOLD)
            .value("ITALIC", FontStyle::ITALIC)
            .value("BOLD_ITALIC", FontStyle::BOLD_ITALIC);

    // ---- FontDescription ----
    py::class_<FontDescription> fd(m, "FontDescription",
                                   "Class to describe a custom font");
    fd.def_readonly_static("SANS_SERIF", &FontDescription::SANS_SERIF,
                           "Name of the default sans-serif font that comes "
                           "with Open3D")
            .def_readonly_static(
                    "MONOSPACE", &FontDescription::MONOSPACE,
                    "Name of the default monospace font that comes "
                    "with Open3D")
            .def(py::init<const char *, FontStyle, int>(),
                 "typeface"_a = FontDescription::SANS_SERIF,
                 "style"_a = FontStyle::NORMAL, "point_size"_a = 0,
                 "Creates a FontDescription. 'typeface' is a path to a "
                 "TrueType (.ttf), TrueType Collection (.ttc), or "
                 "OpenType (.otf) file, or it is the name of the font, "
                 "in which case the system font paths will be searched "
                 "to find the font file. This typeface will be used for "
                 "roman characters (Extended Latin, that is, European "
                 "languages")
            .def("add_typeface_for_language",
                 &FontDescription::AddTypefaceForLanguage,
                 "Adds code points outside Extended Latin from the specified "
                 "typeface. Supported languages are:\n"
                 "   'ja' (Japanese)\n"
                 "   'ko' (Korean)\n"
                 "   'th' (Thai)\n"
                 "   'vi' (Vietnamese)\n"
                 "   'zh' (Chinese, 2500 most common characters, 50 MB per "
                 "window)\n"
                 "   'zh_all' (Chinese, all characters, ~200 MB per window)\n"
                 "All other languages will be assumed to be Cyrillic. "
                 "Note that generally fonts do not have CJK glyphs unless they "
                 "are specifically a CJK font, although operating systems "
                 "generally use a CJK font for you. We do not have the "
                 "information necessary to do this, so you will need to "
                 "provide a font that has the glyphs you need. In particular, "
                 "common fonts like 'Arial', 'Helvetica', and SANS_SERIF do "
                 "not contain CJK glyphs.")
            .def("add_typeface_for_code_points",
                 &FontDescription::AddTypefaceForCodePoints,
                 "Adds specific code points from the typeface. This is useful "
                 "for selectively adding glyphs, for example, from an icon "
                 "font.");

    // ---- Application ----
    py::class_<Application> application(m, "Application",
                                        "Global application singleton. This "
                                        "owns the menubar, windows, and event "
                                        "loop");
    application
            .def_readonly_static("DEFAULT_FONT_ID",
                                 &Application::DEFAULT_FONT_ID)
            .def("__repr__",
                 [](const Application &app) {
                     return std::string("Application singleton instance");
                 })
            .def_property_readonly_static(
                    "instance",
                    // Seems like we ought to be able to specify
                    // &Application::GetInstance but that gives runtime errors
                    // about number of arguments. It seems that property calls
                    // are made from an object, so that object needs to be in
                    // the function signature.
                    [](py::object) -> Application & {
                        return Application::GetInstance();
                    },
                    py::return_value_policy::reference,
                    "Gets the Application singleton (read-only)")
            .def(
                    "initialize",
                    [](Application &instance) { InitializeForPython(); },
                    "Initializes the application, using the resources included "
                    "in the wheel. One of the `initialize` functions _must_ be "
                    "called prior to using anything in the gui module")
            .def(
                    "initialize",
                    [](Application &instance, const char *resource_dir) {
                        InitializeForPython(resource_dir);
                    },
                    "Initializes the application with location of the "
                    "resources provided by the caller. One of the `initialize` "
                    "functions _must_ be called prior to using anything in the "
                    "gui module")
            .def("add_font", &Application::AddFont,
                 "Adds a font. Must be called after initialize() and before "
                 "a window is created. Returns the font id, which can be used "
                 "to change the font in widgets such as Label which support "
                 "custom fonts.")
            .def("set_font", &Application::SetFont,
                 "Changes the contents of an existing font, for instance, the "
                 "default font.")
            .def(
                    "create_window",
                    [](Application &instance, const std::string &title,
                       int width, int height, int x, int y, int flags) {
                        std::shared_ptr<PyWindow> w;
                        if (x < 0 && y < 0 && width < 0 && height < 0) {
                            w.reset(new PyWindow(title, flags));
                        } else if (x < 0 && y < 0) {
                            w.reset(new PyWindow(title, width, height, flags));
                        } else {
                            w.reset(new PyWindow(title, x, y, width, height,
                                                 flags));
                        }
                        instance.AddWindow(w);
                        return w.get();
                    },
                    "title"_a = std::string(), "width"_a = -1, "height"_a = -1,
                    "x"_a = -1, "y"_a = -1, "flags"_a = 0,
                    "Creates a window and adds it to the application. "
                    "To programmatically destroy the window do window.close()."
                    "Usage: create_window(title, width, height, x, y, flags). "
                    "x, y, and flags are optional.")
            // We need to tell RunOneTick() not to cleanup. Run() and
            // RunOneTick() assume a C++ desktop application approach of
            // init -> run -> cleanup. More problematic for us is that Filament
            // crashes if we don't cleanup. Also, we don't want to force Python
            // script writers to remember to cleanup; this is Python, not C++
            // where you expect to need to thinking about cleanup up after
            // yourself. So as a Python-script-app, the cleanup happens atexit.
            // (Init is still required of the script writer.) This means that
            // run() should NOT cleanup, as it might be called several times
            // to run a UI to visualize the result of a computation.
            .def(
                    "run",
                    [](Application &instance) {
                        PythonUnlocker unlocker;
                        while (instance.RunOneTick(unlocker, false)) {
                            // Enable Ctrl-C to kill Python
                            if (PyErr_CheckSignals() != 0) {
                                throw py::error_already_set();
                            }
                        }
                    },
                    "Runs the event loop. After this finishes, all windows and "
                    "widgets should be considered uninitialized, even if they "
                    "are still held by Python variables. Using them is unsafe, "
                    "even if run() is called again.")
            .def(
                    "run_one_tick",
                    [](Application &instance) {
                        PythonUnlocker unlocker;
                        auto result = instance.RunOneTick(unlocker, false);
                        // Enable Ctrl-C to kill Python
                        if (PyErr_CheckSignals() != 0) {
                            throw py::error_already_set();
                        }
                        return result;
                    },
                    "Runs the event loop once, returns True if the app is "
                    "still running, or False if all the windows have closed "
                    "or quit() has been called.")
            .def(
                    "render_to_image",
                    [](Application &instance, rendering::Open3DScene *scene,
                       int width, int height) {
                        return RenderToImageWithoutWindow(scene, width, height);
                    },
                    "Renders a scene to an image and returns the image. If you "
                    "are rendering without a visible window you should use "
                    "open3d.visualization.rendering.RenderToImage instead")
            .def(
                    "quit", [](Application &instance) { instance.Quit(); },
                    "Closes all the windows, exiting as a result")
            .def(
                    "add_window",
                    // Q: Why not just use &Application::AddWindow here?
                    // A: Because then AddWindow gets passed a shared_ptr with
                    //    a use_count of 0 (but with the correct value for
                    //    .get()), so it never gets freed, and then Filament
                    //    doesn't clean up correctly. TakeOwnership() will
                    //    create the shared_ptr properly.
                    [](Application &instance, UnownedPointer<Window> window) {
                        instance.AddWindow(TakeOwnership(window));
                    },
                    "Adds a window to the application. This is only necessary "
                    "when creating an object that is a Window directly, rather "
                    "than with create_window")
            .def("run_in_thread", &Application::RunInThread,
                 "Runs function in a separate thread. Do not call GUI "
                 "functions on this thread, call post_to_main_thread() if "
                 "this thread needs to change the GUI.")
            .def("post_to_main_thread", &Application::PostToMainThread,
                 py::call_guard<py::gil_scoped_release>(),
                 "Runs the provided function on the main thread. This can "
                 "be used to execute UI-related code at a safe point in "
                 "time. If the UI changes, you will need to manually "
                 "request a redraw of the window with w.post_redraw()")
            .def_property("menubar", &Application::GetMenubar,
                          &Application::SetMenubar,
                          "The Menu for the application (initially None)")
            .def_property_readonly("now", &Application::Now,
                                   "Returns current time in seconds")
            // Note: we cannot export AddWindow and RemoveWindow
            .def_property_readonly("resource_path",
                                   &Application::GetResourcePath,
                                   "Returns a string with the path to the "
                                   "resources directory");

    // ---- LayoutContext ----
    py::class_<LayoutContext> lc(
            m, "LayoutContext",
            "Context passed to Window's on_layout callback");
    //    lc.def_readonly("theme", &LayoutContext::theme);
    // Pybind can't return a reference (since Theme is a unique_ptr), so
    // return a copy instead.
    lc.def_property_readonly("theme",
                             [](const LayoutContext &context) -> Theme {
                                 return context.theme;
                             });

    // ---- Window ----
    // Pybind appears to need to know about the base class. It doesn't have
    // to be named the same as the C++ class, though. The holder object cannot
    // be a shared_ptr or we can crash (see comment for UnownedPointer).
    py::class_<Window, UnownedPointer<Window>> window_base(
            m, "WindowBase", "Application window");
    py::class_<PyWindow, UnownedPointer<PyWindow>, Window> window(
            m, "Window",
            "Application window. Create with "
            "Application.instance.create_window().");
    window.def("__repr__",
               [](const PyWindow &w) { return "Application window"; })
            .def(
                    "add_child",
                    [](PyWindow &w, UnownedPointer<Widget> widget) {
                        w.AddChild(TakeOwnership<Widget>(widget));
                    },
                    "Adds a widget to the window")
            .def_property("os_frame", &PyWindow::GetOSFrame,
                          &PyWindow::SetOSFrame,
                          "Window rect in OS coords, not device pixels")
            .def_property("title", &PyWindow::GetTitle, &PyWindow::SetTitle,
                          "Returns the title of the window")
            .def("size_to_fit", &PyWindow::SizeToFit,
                 "Sets the width and height of window to its preferred size")
            .def_property("size", &PyWindow::GetSize, &PyWindow::SetSize,
                          "The size of the window in device pixels, including "
                          "menubar (except on macOS)")
            .def_property_readonly(
                    "content_rect", &PyWindow::GetContentRect,
                    "Returns the frame in device pixels, relative "
                    " to the window, which is available for widgets "
                    "(read-only)")
            .def_property_readonly(
                    "scaling", &PyWindow::GetScaling,
                    "Returns the scaling factor between OS pixels "
                    "and device pixels (read-only)")
            .def_property_readonly("is_visible", &PyWindow::IsVisible,
                                   "True if window is visible (read-only)")
            .def("show", &PyWindow::Show, "Shows or hides the window")
            .def("close", &PyWindow::Close,
                 "Closes the window and destroys it, unless an on_close "
                 "callback cancels the close.")
            .def("set_needs_layout", &PyWindow::SetNeedsLayout,
                 "Flags window to re-layout")
            .def("post_redraw", &PyWindow::PostRedraw,
                 "Sends a redraw message to the OS message queue")
            .def_property_readonly("is_active_window",
                                   &PyWindow::IsActiveWindow,
                                   "True if the window is currently the active "
                                   "window (read-only)")
            .def("set_focus_widget", &PyWindow::SetFocusWidget,
                 "Makes specified widget have text focus")
            .def("set_on_menu_item_activated",
                 &PyWindow::SetOnMenuItemActivated,
                 "Sets callback function for menu item:  callback()")
            .def("set_on_tick_event", &PyWindow::SetOnTickEvent,
                 "Sets callback for tick event. Callback takes no arguments "
                 "and must return True if a redraw is needed (that is, if "
                 "any widget has changed in any fashion) or False if nothing "
                 "has changed")
            .def("set_on_close", &PyWindow::SetOnClose,
                 "Sets a callback that will be called when the window is "
                 "closed. The callback is given no arguments and should return "
                 "True to continue closing the window or False to cancel the "
                 "close")
            .def(
                    "set_on_layout",
                    [](PyWindow *w,
                       std::function<void(const LayoutContext &)> f) {
                        w->on_layout_ = f;
                    },
                    "Sets a callback function that manually sets the frames of "
                    "children of the window. Callback function will be called "
                    "with one argument: gui.LayoutContext")
            .def_property_readonly("theme", &PyWindow::GetTheme,
                                   "Get's window's theme info")
            .def(
                    "show_dialog",
                    [](PyWindow &w, UnownedPointer<Dialog> dlg) {
                        w.ShowDialog(TakeOwnership<Dialog>(dlg));
                    },
                    "Displays the dialog")
            .def("close_dialog", &PyWindow::CloseDialog,
                 "Closes the current dialog")
            .def("show_message_box", &PyWindow::ShowMessageBox,
                 "Displays a simple dialog with a title and message and okay "
                 "button")
            .def("show_menu", &PyWindow::ShowMenu,
                 "show_menu(show): shows or hides the menu in the window, "
                 "except on macOS since the menubar is not in the window "
                 "and all applications must have a menubar.")
            .def_property_readonly(
                    "renderer", &PyWindow::GetRenderer,
                    "Gets the rendering.Renderer object for the Window");

    // ---- Menu ----
    py::class_<Menu, UnownedPointer<Menu>> menu(m, "Menu",
                                                "A menu, possibly a menu tree");
    menu.def(py::init<>())
            .def(
                    "add_item",
                    [](UnownedPointer<Menu> menu, const char *text,
                       int item_id) { menu->AddItem(text, item_id); },
                    "Adds a menu item with id to the menu")
            .def(
                    "add_menu",
                    [](UnownedPointer<Menu> menu, const char *text,
                       UnownedPointer<Menu> submenu) {
                        menu->AddMenu(text, TakeOwnership<Menu>(submenu));
                    },
                    "Adds a submenu to the menu")
            .def("add_separator", &Menu::AddSeparator,
                 "Adds a separator to the menu")
            .def(
                    "set_enabled",
                    [](UnownedPointer<Menu> menu, int item_id, bool enabled) {
                        menu->SetEnabled(item_id, enabled);
                    },
                    "Sets menu item enabled or disabled")
            .def(
                    "is_checked",
                    [](UnownedPointer<Menu> menu, int item_id) -> bool {
                        return menu->IsChecked(item_id);
                    },
                    "Returns True if menu item is checked")
            .def(
                    "set_checked",
                    [](UnownedPointer<Menu> menu, int item_id, bool checked) {
                        menu->SetChecked(item_id, checked);
                    },
                    "Sets menu item (un)checked");

    // ---- Color ----
    py::class_<Color> color(m, "Color", "Stores color for gui classes");
    color.def(py::init([](float r, float g, float b, float a) {
                  return new Color(r, g, b, a);
              }),
              "r"_a = 1.0, "g"_a = 1.0, "b"_a = 1.0, "a"_a = 1.0)
            .def_property_readonly(
                    "red", &Color::GetRed,
                    "Returns red channel in the range [0.0, 1.0] "
                    "(read-only)")
            .def_property_readonly(
                    "green", &Color::GetGreen,
                    "Returns green channel in the range [0.0, 1.0] "
                    "(read-only)")
            .def_property_readonly(
                    "blue", &Color::GetBlue,
                    "Returns blue channel in the range [0.0, 1.0] "
                    "(read-only)")
            .def_property_readonly(
                    "alpha", &Color::GetAlpha,
                    "Returns alpha channel in the range [0.0, 1.0] "
                    "(read-only)")
            .def("set_color", &Color::SetColor,
                 "Sets red, green, blue, and alpha channels, (range: [0.0, "
                 "1.0])",
                 "r"_a, "g"_a, "b"_a, "a"_a = 1.0);

    // ---- Theme ----
    // Note: no constructor because themes are created by Open3D
    py::class_<Theme> theme(m, "Theme",
                            "Theme parameters such as colors used for drawing "
                            "widgets (read-only)");
    theme.def_readonly("font_size", &Theme::font_size,
                       "Font size (which is also the conventional size of the "
                       "em unit) (read-only)")
            .def_readonly("default_margin", &Theme::default_margin,
                          "Good default value for margins, useful for layouts "
                          "(read-only)")
            .def_readonly("default_layout_spacing",
                          &Theme::default_layout_spacing,
                          "Good value for the spacing parameter in layouts "
                          "(read-only)");

    // ---- Rect ----
    py::class_<Rect> rect(m, "Rect", "Represents a widget frame");
    rect.def(py::init<>())
            .def(py::init<int, int, int, int>())
            .def(py::init([](float x, float y, float w, float h) {
                return Rect(int(std::round(x)), int(std::round(y)),
                            int(std::round(w)), int(std::round(h)));
            }))
            .def("__repr__",
                 [](const Rect &r) {
                     std::stringstream s;
                     s << "Rect (" << r.x << ", " << r.y << "), " << r.width
                       << " x " << r.height;
                     return s.str();
                 })
            .def_readwrite("x", &Rect::x)
            .def_readwrite("y", &Rect::y)
            .def_readwrite("width", &Rect::width)
            .def_readwrite("height", &Rect::height)
            .def("get_left", &Rect::GetLeft)
            .def("get_right", &Rect::GetRight)
            .def("get_top", &Rect::GetTop)
            .def("get_bottom", &Rect::GetBottom);

    // ---- Size ----
    py::class_<Size> size(m, "Size", "Size object");
    size.def(py::init<>())
            .def(py::init<int, int>())
            .def(py::init([](float w, float h) {
                return Size(int(std::round(w)), int(std::round(h)));
            }))
            .def("__repr__",
                 [](const Size &sz) {
                     std::stringstream s;
                     s << "Size (" << sz.width << ", " << sz.height << ")";
                     return s.str();
                 })
            .def_readwrite("width", &Size::width)
            .def_readwrite("height", &Size::height);

    // ---- Widget ----
    // The holder for Widget and all derived classes is UnownedPointer because
    // a Widget may own Filament resources, so we cannot have Python holding
    // on to a shared_ptr after we cleanup Filament. The object is initially
    // "leaked" (as in, Python will not clean it up), but it will be claimed
    // by the object it is added to. There are two consequences to this:
    //  1) adding an object to multiple objects will cause multiple shared_ptrs
    //     to think they own it, leading to a double-free and crash, and
    //  2) if the object is never added, the memory will be leaked.
    py::class_<Widget, UnownedPointer<Widget>> widget(m, "Widget",
                                                      "Base widget class");
    py::enum_<EventCallbackResult> widget_event_callback_result(
            widget, "EventCallbackResult", "Returned by event handlers",
            py::arithmetic());
    widget_event_callback_result
            .value("IGNORED", EventCallbackResult::IGNORED,
                   "Event handler ignored the event, widget will "
                   "handle event normally")
            .value("HANDLED", EventCallbackResult::HANDLED,
                   "Event handler handled the event, but widget "
                   "will still handle the event normally. This is "
                   "useful when you are augmenting base "
                   "functionality")
            .value("CONSUMED", EventCallbackResult::CONSUMED,
                   "Event handler consumed the event, event "
                   "handling stops, widget will not handle the "
                   "event. This is useful when you are replacing "
                   "functionality")
            .export_values();

    py::class_<Widget::Constraints> constraints(
            widget, "Constraints",
            "Constraints object for Widget.calc_preferred_size()");
    constraints.def(py::init<>())
            .def_readwrite("width", &Widget::Constraints::width)
            .def_readwrite("height", &Widget::Constraints::height);

    widget.def(py::init<>())
            .def("__repr__",
                 [](const Widget &w) {
                     std::stringstream s;
                     s << "Widget (" << w.GetFrame().x << ", " << w.GetFrame().y
                       << "), " << w.GetFrame().width << " x "
                       << w.GetFrame().height;
                     return s.str();
                 })
            .def(
                    "add_child",
                    [](Widget &w, UnownedPointer<Widget> child) {
                        w.AddChild(TakeOwnership<Widget>(child));
                    },
                    "Adds a child widget")
            .def("get_children", &Widget::GetChildren,
                 "Returns the array of children. Do not modify.")
            .def_property("frame", &Widget::GetFrame, &Widget::SetFrame,
                          "The widget's frame. Setting this value will be "
                          "overridden if the frame is within a layout.")
            .def_property("visible", &Widget::IsVisible, &Widget::SetVisible,
                          "True if widget is visible, False otherwise")
            .def_property("enabled", &Widget::IsEnabled, &Widget::SetEnabled,
                          "True if widget is enabled, False if disabled")
            .def_property("background_color", &Widget::GetBackgroundColor,
                          &Widget::SetBackgroundColor,
                          "Background color of the widget")
            .def_property("tooltip", &Widget::GetTooltip, &Widget::SetTooltip,
                          "Widget's tooltip that is displayed on mouseover")
            .def("calc_preferred_size", &Widget::CalcPreferredSize,
                 "Returns the preferred size of the widget. This is intended "
                 "to be called only during layout, although it will also work "
                 "during drawing. Calling it at other times will not work, as "
                 "it requires some internal setup in order to function "
                 "properly");

    // ---- WidgetProxy ----
    py::class_<WidgetProxy, UnownedPointer<WidgetProxy>, Widget> widgetProxy(
            m, "WidgetProxy",
            "Widget container to delegate any widget dynamically."
            " Widget can not be managed dynamically. Although it is allowed"
            " to add more child widgets, it's impossible to replace some child"
            " with new on or remove children. WidgetProxy is designed to solve"
            " this problem."
            " When WidgetProxy is created, it's invisible and disabled, so it"
            " won't be drawn or layout, seeming like it does not exist. When"
            " a widget is set by  set_widget, all  Widget's APIs will be"
            " conducted to that child widget. It looks like WidgetProxy is"
            " that widget."
            " At any time, a new widget could be set, to replace the old one."
            " and the old widget will be destroyed."
            " Due to the content changing after a new widget is set or cleared,"
            " a relayout of Window might be called after set_widget."
            " The delegated widget could be retrieved by  get_widget in case"
            "  you need to access it directly, like get check status of a"
            " CheckBox. API other than  set_widget and get_widget has"
            " completely same functions as Widget.");
    widgetProxy.def(py::init<>(), "Creates a widget proxy")
            .def("__repr__",
                 [](const WidgetProxy &c) {
                     std::stringstream s;
                     s << "Proxy (" << c.GetFrame().x << ", " << c.GetFrame().y
                       << "), " << c.GetFrame().width << " x "
                       << c.GetFrame().height;
                     return s.str();
                 })
            .def(
                    "set_widget",
                    [](WidgetProxy &w, UnownedPointer<Widget> proxy) {
                        w.SetWidget(TakeOwnership<Widget>(proxy));
                    },
                    "set a new widget to be delegated by this one."
                    " After set_widget, the previously delegated widget ,"
                    " will be abandon all calls to Widget's API will be "
                    " conducted to widget. Before any set_widget call, "
                    " this widget is invisible and disabled, seems it "
                    " does not exist because it won't be drawn or in a "
                    "layout.")
            .def("get_widget", &WidgetProxy::GetWidget,
                 "Retrieve current delegated widget."
                 "return instance of current delegated widget set by "
                 "set_widget. An empty pointer will be returned "
                 "if there is none.");

    // ---- WidgetStack ----
    py::class_<WidgetStack, UnownedPointer<WidgetStack>, WidgetProxy>
            widgetStack(m, "WidgetStack",
                        "A widget stack saves all widgets pushed into by "
                        "push_widget and always shows the top one. The "
                        "WidgetStack is a subclass of WidgetProxy, in other"
                        "words, the topmost widget will delegate itself to "
                        "WidgetStack. pop_widget will remove the topmost "
                        "widget and callback set by set_on_top taking the "
                        "new topmost widget will be called. The WidgetStack "
                        "disappears in GUI if there is no widget in stack.");
    widgetStack
            .def(py::init<>(),
                 "Creates a widget stack. The widget stack without any"
                 "widget will not be shown in GUI until set_widget is"
                 "called to push a widget.")
            .def("__repr__",
                 [](const WidgetStack &c) {
                     std::stringstream s;
                     s << "Stack (" << c.GetFrame().x << ", " << c.GetFrame().y
                       << "), " << c.GetFrame().width << " x "
                       << c.GetFrame().height;
                     return s.str();
                 })
            .def("push_widget", &WidgetStack::PushWidget,
                 "push a new widget onto the WidgetStack's stack, hiding "
                 "whatever widget was there before and making the new widget "
                 "visible.")
            .def("pop_widget", &WidgetStack::PopWidget,
                 "pop the topmost widget in the stack. The new topmost widget"
                 "of stack will be the widget on the show in GUI.")
            .def("set_on_top", &WidgetStack::SetOnTop,
                 "Callable[[widget] -> None], called while a widget "
                 "becomes the topmost of stack after some widget is popped"
                 "out. It won't be called if a widget is pushed into stack"
                 "by set_widget.");
    // ---- Button ----
    py::class_<Button, UnownedPointer<Button>, Widget> button(m, "Button",
                                                              "Button");
    button.def(py::init<const char *>(), "Creates a button with the given text")
            .def("__repr__",
                 [](const Button &b) {
                     std::stringstream s;
                     s << "Button (" << b.GetFrame().x << ", " << b.GetFrame().y
                       << "), " << b.GetFrame().width << " x "
                       << b.GetFrame().height;
                     return s.str();
                 })
            .def_property("text", &Button::GetText, &Button::SetText,
                          "Gets/sets the button text.")
            .def_property(
                    "toggleable", &Button::GetIsToggleable,
                    &Button::SetToggleable,
                    "True if button is toggleable, False if a push button")
            .def_property(
                    "is_on", &Button::GetIsOn, &Button::SetOn,
                    "True if the button is toggleable and in the on state")
            // It is not possible to overload properties. But we want users
            // to be able to say "o.padding = 1.4" or "o.padding = 1",
            // and float and int are different types. Fortunately, we want
            // a float, which is easily castable from int. So we can pass
            // a py::object and cast it ourselves.
            .def_property(
                    "horizontal_padding_em", &Button::GetHorizontalPaddingEm,
                    [](UnownedPointer<Button> b, const py::object &em) {
                        auto vert = b->GetVerticalPaddingEm();
                        try {
                            b->SetPaddingEm(em.cast<float>(), vert);
                        } catch (const py::cast_error &) {
                            py::print(
                                    "open3d.visualization.gui.Button."
                                    "horizontal_padding_em can only be "
                                    "assigned a numeric type");
                        }
                    },
                    "Horizontal padding in em units")
            .def_property(
                    "vertical_padding_em", &Button::GetVerticalPaddingEm,
                    [](UnownedPointer<Button> b, const py::object &em) {
                        auto horiz = b->GetHorizontalPaddingEm();
                        try {
                            b->SetPaddingEm(horiz, em.cast<float>());
                        } catch (const py::cast_error &) {
                            py::print(
                                    "open3d.visualization.gui.Button."
                                    "vertical_padding_em can only be "
                                    "assigned a numeric type");
                        }
                    },
                    "Vertical padding in em units")
            .def("set_on_clicked", &Button::SetOnClicked,
                 "Calls passed function when button is pressed");

    // ---- Checkbox ----
    py::class_<Checkbox, UnownedPointer<Checkbox>, Widget> checkbox(
            m, "Checkbox", "Checkbox");
    checkbox.def(py::init<const char *>(),
                 "Creates a checkbox with the given text")
            .def("__repr__",
                 [](const Checkbox &c) {
                     std::stringstream s;
                     s << "Checkbox (" << c.GetFrame().x << ", "
                       << c.GetFrame().y << "), " << c.GetFrame().width << " x "
                       << c.GetFrame().height;
                     return s.str();
                 })
            .def_property("checked", &Checkbox::IsChecked,
                          &Checkbox::SetChecked,
                          "True if checked, False otherwise")
            .def("set_on_checked", &Checkbox::SetOnChecked,
                 "Calls passed function when checkbox changes state");

    // ---- ColorEdit ----
    py::class_<ColorEdit, UnownedPointer<ColorEdit>, Widget> coloredit(
            m, "ColorEdit", "Color picker");
    coloredit.def(py::init<>())
            .def("__repr__",
                 [](const ColorEdit &c) {
                     auto &color = c.GetValue();
                     std::stringstream s;
                     s << "ColorEdit [" << color.GetRed() << ", "
                       << color.GetGreen() << ", " << color.GetBlue() << ", "
                       << color.GetAlpha() << "] (" << c.GetFrame().x << ", "
                       << c.GetFrame().y << "), " << c.GetFrame().width << " x "
                       << c.GetFrame().height;
                     return s.str();
                 })
            .def_property(
                    "color_value", &ColorEdit::GetValue,
                    (void (ColorEdit::*)(const Color &)) & ColorEdit::SetValue,
                    "Color value (gui.Color)")
            .def("set_on_value_changed", &ColorEdit::SetOnValueChanged,
                 "Calls f(Color) when color changes by user input");

    // ---- Combobox ----
    py::class_<Combobox, UnownedPointer<Combobox>, Widget> combobox(
            m, "Combobox", "Exclusive selection from a pull-down menu");
    combobox.def(py::init<>(),
                 "Creates an empty combobox. Use add_item() to add items")
            .def("clear_items", &Combobox::ClearItems, "Removes all items")
            .def("add_item", &Combobox::AddItem, "Adds an item to the end")
            .def("change_item",
                 (void (Combobox::*)(int, const char *)) & Combobox::ChangeItem,
                 "Changes the text of the item at index: "
                 "change_item(index, newtext)")
            .def("change_item",
                 (void (Combobox::*)(const char *, const char *)) &
                         Combobox::ChangeItem,
                 "Changes the text of the matching item: "
                 "change_item(text, newtext)")
            .def("remove_item",
                 (void (Combobox::*)(const char *)) & Combobox::RemoveItem,
                 "Removes the first item of the given text")
            .def("remove_item",
                 (void (Combobox::*)(int)) & Combobox::RemoveItem,
                 "Removes the item at the index")
            .def_property_readonly("number_of_items",
                                   &Combobox::GetNumberOfItems,
                                   "The number of items (read-only)")
            .def("get_item", &Combobox::GetItem,
                 "Returns the item at the given index")
            .def_property("selected_index", &Combobox::GetSelectedIndex,
                          &Combobox::SetSelectedIndex,
                          "The index of the currently selected item")
            .def_property("selected_text", &Combobox::GetSelectedValue,
                          &Combobox::SetSelectedValue,
                          "The index of the currently selected item")
            .def("set_on_selection_changed", &Combobox::SetOnValueChanged,
                 "Calls f(str, int) when user selects item from combobox. "
                 "Arguments are the selected text and selected index, "
                 "respectively");

    // ---- ImageWidget ----
    py::class_<UIImage, UnownedPointer<UIImage>> uiimage(
            m, "UIImage", "A bitmap suitable for displaying with ImageWidget");

    py::enum_<UIImage::Scaling> uiimage_scaling(uiimage, "Scaling",
                                                py::arithmetic());
    uiimage_scaling.value("NONE", UIImage::Scaling::NONE)
            .value("ANY", UIImage::Scaling::ANY)
            .value("ASPECT", UIImage::Scaling::ASPECT);

    uiimage.def(py::init<>([](const char *path) { return new UIImage(path); }),
                "Creates a UIImage from the image at the specified path")
            .def(py::init<>([](std::shared_ptr<geometry::Image> image) {
                     return new UIImage(image);
                 }),
                 "Creates a UIImage from the provided image")
            .def("__repr__", [](const UIImage &il) { return "UIImage"; })
            .def_property("scaling", &UIImage::GetScaling, &UIImage::SetScaling,
                          "Sets how the image is scaled:\n"
                          "gui.UIImage.Scaling.NONE: no scaling\n"
                          "gui.UIImage.Scaling.ANY: scaled to fit\n"
                          "gui.UIImage.Scaling.ASPECT: scaled to fit but "
                          "keeping the image's aspect ratio");

    py::class_<ImageWidget, UnownedPointer<ImageWidget>, Widget> imagewidget(
            m, "ImageWidget", "Displays a bitmap");
    imagewidget
            .def(py::init<>([]() { return new ImageWidget(); }),
                 "Creates an ImageWidget with no image")
            .def(py::init<>([](const char *path) {
                     return new ImageWidget(path);
                 }),
                 "Creates an ImageWidget from the image at the specified path")
            .def(py::init<>([](std::shared_ptr<geometry::Image> image) {
                     return new ImageWidget(image);
                 }),
                 "Creates an ImageWidget from the provided image")
            .def(py::init<>([](std::shared_ptr<t::geometry::Image> image) {
                     return new ImageWidget(image);
                 }),
                 "Creates an ImageWidget from the provided tgeometry image")
            .def("__repr__",
                 [](const ImageWidget &il) {
                     std::stringstream s;
                     s << "ImageWidget (" << il.GetFrame().x << ", "
                       << il.GetFrame().y << "), " << il.GetFrame().width
                       << " x " << il.GetFrame().height;
                     return s.str();
                 })
            .def("update_image",
                 py::overload_cast<std::shared_ptr<geometry::Image>>(
                         &ImageWidget::UpdateImage),
                 "Mostly a convenience function for ui_image.update_image(). "
                 "If 'image' is the same size as the current image, will "
                 "update the texture with the contents of 'image'. This is "
                 "the fastest path for setting an image, and is recommended "
                 "if you are displaying video. If 'image' is a different size, "
                 "it will allocate a new texture, which is essentially the "
                 "same as creating a new UIImage and calling SetUIImage(). "
                 "This is the slow path, and may eventually exhaust internal "
                 "texture resources.")
            .def("update_image",
                 py::overload_cast<std::shared_ptr<t::geometry::Image>>(
                         &ImageWidget::UpdateImage),
                 "Mostly a convenience function for ui_image.update_image(). "
                 "If 'image' is the same size as the current image, will "
                 "update the texture with the contents of 'image'. This is "
                 "the fastest path for setting an image, and is recommended "
                 "if you are displaying video. If 'image' is a different size, "
                 "it will allocate a new texture, which is essentially the "
                 "same as creating a new UIImage and calling SetUIImage(). "
                 "This is the slow path, and may eventually exhaust internal "
                 "texture resources.")
            .def_property("ui_image", &ImageWidget::GetUIImage,
                          &ImageWidget::SetUIImage,
                          "Replaces the texture with a new texture. This is "
                          "not a fast path, and is not recommended for video "
                          "as you will exhaust internal texture resources.");

    // ---- Label ----
    py::class_<Label, UnownedPointer<Label>, Widget> label(m, "Label",
                                                           "Displays text");
    label.def(py::init([](const char *title = "") { return new Label(title); }),
              "Creates a Label with the given text")
            .def("__repr__",
                 [](const Label &lbl) {
                     std::stringstream s;
                     s << "Label [" << lbl.GetText() << "] ("
                       << lbl.GetFrame().x << ", " << lbl.GetFrame().y << "), "
                       << lbl.GetFrame().width << " x "
                       << lbl.GetFrame().height;
                     return s.str();
                 })
            .def_property("text", &Label::GetText, &Label::SetText,
                          "The text of the label. Newlines will be treated as "
                          "line breaks")
            .def_property("text_color", &Label::GetTextColor,
                          &Label::SetTextColor,
                          "The color of the text (gui.Color)")
            .def_property("font_id", &Label::GetFontId, &Label::SetFontId,
                          "Set the font using the FontId returned from "
                          "Application.add_font()");

    // ---- Label3D ----
    py::class_<Label3D, UnownedPointer<Label3D>> label3d(
            m, "Label3D", "Displays text in a 3D scene");
    label3d.def(py::init([](const char *text = "",
                            const Eigen::Vector3f &pos = {0.f, 0.f, 0.f}) {
                    return new Label3D(pos, text);
                }),
                "Create a 3D Label with given text and position")
            .def_property("text", &Label3D::GetText, &Label3D::SetText,
                          "The text to display with this label.")
            .def_property("position", &Label3D::GetPosition,
                          &Label3D::SetPosition,
                          "The position of the text in 3D coordinates")
            .def_property("color", &Label3D::GetTextColor,
                          &Label3D::SetTextColor,
                          "The color of the text (gui.Color)")
            .def_property(
                    "scale", &Label3D::GetTextScale, &Label3D::SetTextScale,
                    "The scale of the 3D label. When set to 1.0 (the default) "
                    "text will be rendered at its native font size. Larger and "
                    "smaller values of scale will enlarge or shrink the "
                    "rendered text. Note: large values of scale may result in "
                    "blurry text as the underlying font is not resized.");

    // ---- ListView ----
    py::class_<ListView, UnownedPointer<ListView>, Widget> listview(
            m, "ListView", "Displays a list of text");
    listview.def(py::init<>(), "Creates an empty list")
            .def("__repr__",
                 [](const ListView &lv) {
                     std::stringstream s;
                     s << "Label (" << lv.GetFrame().x << ", "
                       << lv.GetFrame().y << "), " << lv.GetFrame().width
                       << " x " << lv.GetFrame().height;
                     return s.str();
                 })
            .def("set_items", &ListView::SetItems,
                 "Sets the list to display the list of items provided")
            .def("set_max_visible_items", &ListView::SetMaxVisibleItems,
                 "Limit the max visible items shown to user. "
                 "Set to negative number will make list extends vertically "
                 "as much as possible, otherwise the list will at least show "
                 "3 items and at most show num items.")
            .def_property("selected_index", &ListView::GetSelectedIndex,
                          &ListView::SetSelectedIndex,
                          "The index of the currently selected item")
            .def_property_readonly("selected_value",
                                   &ListView::GetSelectedValue,
                                   "The text of the currently selected item")
            .def("set_on_selection_changed", &ListView::SetOnValueChanged,
                 "Calls f(new_val, is_double_click) when user changes "
                 "selection");

    // ---- NumberEdit ----
    py::class_<NumberEdit, UnownedPointer<NumberEdit>, Widget> numedit(
            m, "NumberEdit", "Allows the user to enter a number.");
    py::enum_<NumberEdit::Type> numedit_type(numedit, "Type", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    numedit_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for NumberEdit types.";
            }),
            py::none(), py::none(), "");
    numedit_type.value("INT", NumberEdit::Type::INT)
            .value("DOUBLE", NumberEdit::Type::DOUBLE)
            .export_values();

    numedit.def(py::init<NumberEdit::Type>(),
                "Creates a NumberEdit that is either integers (INT) or "
                "floating point (DOUBLE). The initial value is 0 and the "
                "limits are +/- max integer (roughly).")
            .def("__repr__",
                 [](const NumberEdit &ne) {
                     auto val = ne.GetDoubleValue();
                     std::stringstream s;
                     s << "NumberEdit [" << val << "] (" << ne.GetFrame().x
                       << ", " << ne.GetFrame().y << "), "
                       << ne.GetFrame().width << " x " << ne.GetFrame().height;
                     return s.str();
                 })
            .def_property(
                    "int_value", &NumberEdit::GetIntValue,
                    [](UnownedPointer<NumberEdit> ne, int val) {
                        ne->SetValue(double(val));
                    },
                    "Current value (int)")
            .def_property("double_value", &NumberEdit::GetDoubleValue,
                          &NumberEdit::SetValue, "Current value (double)")
            .def("set_value", &NumberEdit::SetValue, "Sets value")
            .def_property("decimal_precision", &NumberEdit::GetDecimalPrecision,
                          &NumberEdit::SetDecimalPrecision,
                          "Number of fractional digits shown")
            .def_property_readonly("minimum_value",
                                   &NumberEdit::GetMinimumValue,
                                   "The minimum value number can contain "
                                   "(read-only, use set_limits() to set)")
            .def_property_readonly("maximum_value",
                                   &NumberEdit::GetMaximumValue,
                                   "The maximum value number can contain "
                                   "(read-only, use set_limits() to set)")
            .def("set_limits", &NumberEdit::SetLimits,
                 "Sets the minimum and maximum values for the number")
            .def("set_on_value_changed", &NumberEdit::SetOnValueChanged,
                 "Sets f(new_value) which is called with a Float when user "
                 "changes widget's value")
            .def("set_preferred_width", &NumberEdit::SetPreferredWidth,
                 "Sets the preferred width of the NumberEdit")
            .def(
                    "set_preferred_width",
                    [](NumberEdit &ne, double width) {
                        ne.NumberEdit::SetPreferredWidth(int(width));
                    },
                    "Sets the preferred width of the NumberEdit");

    // ---- ProgressBar----
    py::class_<ProgressBar, UnownedPointer<ProgressBar>, Widget> progress(
            m, "ProgressBar", "Displays a progress bar");
    progress.def(py::init<>())
            .def("__repr__",
                 [](const ProgressBar &pb) {
                     std::stringstream s;
                     s << "ProgressBar [" << pb.GetValue() << "] ("
                       << pb.GetFrame().x << ", " << pb.GetFrame().y << "), "
                       << pb.GetFrame().width << " x " << pb.GetFrame().height;
                     return s.str();
                 })
            .def_property(
                    "value", &ProgressBar::GetValue, &ProgressBar::SetValue,
                    "The value of the progress bar, ranges from 0.0 to 1.0");

    // ---- SceneWidget ----
    class PySceneWidget : public SceneWidget {
        using Super = SceneWidget;

    public:
        void SetOnMouse(std::function<int(const MouseEvent &)> f) {
            on_mouse_ = f;
        }
        void SetOnKey(std::function<int(const KeyEvent &)> f) { on_key_ = f; }

        Widget::EventResult Mouse(const MouseEvent &e) override {
            if (on_mouse_) {
                switch (EventCallbackResult(on_mouse_(e))) {
                    case EventCallbackResult::CONSUMED:
                        return Widget::EventResult::CONSUMED;
                    case EventCallbackResult::HANDLED: {
                        auto result = Super::Mouse(e);
                        if (result == Widget::EventResult::IGNORED) {
                            result = Widget::EventResult::CONSUMED;
                        }
                        return result;
                    }
                    case EventCallbackResult::IGNORED:
                    default:
                        return Super::Mouse(e);
                }
            } else {
                return Super::Mouse(e);
            }
        }

        Widget::EventResult Key(const KeyEvent &e) override {
            if (on_key_) {
                switch (EventCallbackResult(on_key_(e))) {
                    case EventCallbackResult::CONSUMED:
                        return Widget::EventResult::CONSUMED;
                    case EventCallbackResult::HANDLED: {
                        auto result = Super::Key(e);
                        if (result == Widget::EventResult::IGNORED) {
                            result = Widget::EventResult::CONSUMED;
                        }
                        return result;
                    }
                    case EventCallbackResult::IGNORED:
                    default:
                        return Super::Key(e);
                }
            } else {
                return Super::Key(e);
            }
        }

    private:
        std::function<int(const MouseEvent &)> on_mouse_;
        std::function<int(const KeyEvent &)> on_key_;
    };

    py::class_<PySceneWidget, UnownedPointer<PySceneWidget>, Widget> scene(
            m, "SceneWidget", "Displays 3D content");
    py::enum_<SceneWidget::Controls> scene_ctrl(scene, "Controls",
                                                py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    scene_ctrl.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class describing mouse interaction.";
            }),
            py::none(), py::none(), "");
    scene_ctrl.value("ROTATE_CAMERA", SceneWidget::Controls::ROTATE_CAMERA)
            .value("ROTATE_CAMERA_SPHERE",
                   SceneWidget::Controls::ROTATE_CAMERA_SPHERE)
            .value("FLY", SceneWidget::Controls::FLY)
            .value("ROTATE_SUN", SceneWidget::Controls::ROTATE_SUN)
            .value("ROTATE_IBL", SceneWidget::Controls::ROTATE_IBL)
            .value("ROTATE_MODEL", SceneWidget::Controls::ROTATE_MODEL)
            .value("PICK_POINTS", SceneWidget::Controls::PICK_POINTS)
            .export_values();

    scene.def(py::init<>(),
              "Creates an empty SceneWidget. Assign a Scene with the 'scene' "
              "property")
            .def_property(
                    "scene", &PySceneWidget::GetScene, &SceneWidget::SetScene,
                    "The rendering.Open3DScene that the SceneWidget renders")
            .def_property("center_of_rotation",
                          &SceneWidget::GetCenterOfRotation,
                          &SceneWidget::SetCenterOfRotation,
                          "Current center of rotation (for ROTATE_CAMERA and "
                          "ROTATE_CAMERA_SPHERE)")
            .def("enable_scene_caching", &PySceneWidget::EnableSceneCaching,
                 "Enable/Disable caching of scene content when the view or "
                 "model is not changing. Scene caching can help improve UI "
                 "responsiveness for large models and point clouds")
            .def("force_redraw", &PySceneWidget::ForceRedraw,
                 "Ensures scene redraws even when scene caching is enabled.")
            .def("set_view_controls", &PySceneWidget::SetViewControls,
                 "Sets mouse interaction, e.g. ROTATE_OBJ")
            .def("setup_camera",
                 py::overload_cast<float,
                                   const geometry::AxisAlignedBoundingBox &,
                                   const Eigen::Vector3f &>(
                         &PySceneWidget::SetupCamera),
                 "Configure the camera: setup_camera(field_of_view, "
                 "model_bounds, center_of_rotation)")
            .def("setup_camera",
                 py::overload_cast<const camera::PinholeCameraIntrinsic &,
                                   const Eigen::Matrix4d &,
                                   const geometry::AxisAlignedBoundingBox &>(
                         &PySceneWidget::SetupCamera),
                 "setup_camera(intrinsics, extrinsic_matrix, model_bounds): "
                 "sets the camera view")
            .def("setup_camera",
                 py::overload_cast<const Eigen::Matrix3d &,
                                   const Eigen::Matrix4d &, int, int,
                                   const geometry::AxisAlignedBoundingBox &>(
                         &PySceneWidget::SetupCamera),
                 "setup_camera(intrinsic_matrix, extrinsic_matrix, "
                 "intrinsic_width_px, intrinsic_height_px, model_bounds): "
                 "sets the camera view")
            .def("look_at", &PySceneWidget::LookAt,
                 "look_at(center, eye, up): sets the "
                 "camera view so that the camera is located at 'eye', pointing "
                 "towards 'center', and oriented so that the up vector is 'up'")
            .def("set_on_mouse", &PySceneWidget::SetOnMouse,
                 "Sets a callback for mouse events. This callback is passed "
                 "a MouseEvent object. The callback must return "
                 "EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, "
                 "or EventCallackResult.CONSUMED.")
            .def("set_on_key", &PySceneWidget::SetOnKey,
                 "Sets a callback for key events. This callback is passed "
                 "a KeyEvent object. The callback must return "
                 "EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, "
                 "or EventCallackResult.CONSUMED.")
            .def("set_on_sun_direction_changed",
                 &PySceneWidget::SetOnSunDirectionChanged,
                 "Callback when user changes sun direction (only called in "
                 "ROTATE_SUN control mode). Called with one argument, the "
                 "[i, j, k] vector of the new sun direction")
            .def("add_3d_label", &PySceneWidget::AddLabel,
                 "Add a 3D text label to the scene. The label will be anchored "
                 "at the specified 3D point.")
            .def("remove_3d_label", &PySceneWidget::RemoveLabel,
                 "Removes the 3D text label from the scene");

    // ---- Slider ----
    py::class_<Slider, UnownedPointer<Slider>, Widget> slider(
            m, "Slider", "A slider widget for visually selecting numbers");
    py::enum_<Slider::Type> slider_type(slider, "Type", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    slider_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Slider types.";
            }),
            py::none(), py::none(), "");
    slider_type.value("INT", Slider::Type::INT)
            .value("DOUBLE", Slider::Type::DOUBLE)
            .export_values();

    slider.def(py::init<Slider::Type>(),
               "Creates a NumberEdit that is either integers (INT) or "
               "floating point (DOUBLE). The initial value is 0 and the limits "
               "are +/- infinity.")
            .def("__repr__",
                 [](const Slider &sl) {
                     auto val = sl.GetDoubleValue();
                     std::stringstream s;
                     s << "TextEdit [" << val << "] (" << sl.GetFrame().x
                       << ", " << sl.GetFrame().y << "), "
                       << sl.GetFrame().width << " x " << sl.GetFrame().height;
                     return s.str();
                 })
            .def_property(
                    "int_value", &Slider::GetIntValue,
                    [](UnownedPointer<Slider> ne, int val) {
                        ne->SetValue(double(val));
                    },
                    "Slider value (int)")
            .def_property("double_value", &Slider::GetDoubleValue,
                          &Slider::SetValue, "Slider value (double)")
            .def_property_readonly("get_minimum_value",
                                   &Slider::GetMinimumValue,
                                   "The minimum value number can contain "
                                   "(read-only, use set_limits() to set)")
            .def_property_readonly("get_maximum_value",
                                   &Slider::GetMaximumValue,
                                   "The maximum value number can contain "
                                   "(read-only, use set_limits() to set)")
            .def("set_limits", &Slider::SetLimits,
                 "Sets the minimum and maximum values for the slider")
            .def("set_on_value_changed", &Slider::SetOnValueChanged,
                 "Sets f(new_value) which is called with a Float when user "
                 "changes widget's value");

    // ---- StackedWidget ----
    py::class_<StackedWidget, UnownedPointer<StackedWidget>, Widget> stacked(
            m, "StackedWidget", "Like a TabControl but without the tabs");
    stacked.def(py::init<>())
            .def_property("selected_index", &StackedWidget::GetSelectedIndex,
                          &StackedWidget::SetSelectedIndex,
                          "Selects the index of the child to display");

    // ---- TabControl ----
    py::class_<TabControl, UnownedPointer<TabControl>, Widget> tabctrl(
            m, "TabControl", "Tab control");
    tabctrl.def(py::init<>())
            .def(
                    "add_tab",
                    [](TabControl &tabs, const char *name,
                       UnownedPointer<Widget> panel) {
                        tabs.AddTab(name, TakeOwnership<Widget>(panel));
                    },
                    "Adds a tab. The first parameter is the title of the tab, "
                    "and the second parameter is a widget--normally this is a "
                    "layout.")
            .def_property("selected_tab_index",
                          &TabControl::GetSelectedTabIndex,
                          &TabControl::SetSelectedTabIndex,
                          "The index of the currently selected item")
            .def("set_on_selected_tab_changed",
                 &TabControl::SetOnSelectedTabChanged,
                 "Calls the provided callback function with the index of the "
                 "currently selected tab whenever the user clicks on a "
                 "different tab");

    // ---- TextEdit ----
    py::class_<TextEdit, UnownedPointer<TextEdit>, Widget> textedit(
            m, "TextEdit", "Allows the user to enter or modify text");
    textedit.def(py::init<>(),
                 "Creates a TextEdit widget with an initial value of an empty "
                 "string.")
            .def("__repr__",
                 [](const TextEdit &te) {
                     auto val = te.GetText();
                     std::stringstream s;
                     s << "TextEdit [" << val << "] (" << te.GetFrame().x
                       << ", " << te.GetFrame().y << "), "
                       << te.GetFrame().width << " x " << te.GetFrame().height;
                     return s.str();
                 })
            .def_property("text_value", &TextEdit::GetText, &TextEdit::SetText,
                          "The value of text")
            .def_property(
                    "placeholder_text", &TextEdit::GetPlaceholderText,
                    &TextEdit::SetPlaceholderText,
                    "The placeholder text displayed when text value is empty")
            .def("set_on_text_changed", &TextEdit::SetOnTextChanged,
                 "Sets f(new_text) which is called whenever the the user makes "
                 "a change to the text")
            .def("set_on_value_changed", &TextEdit::SetOnValueChanged,
                 "Sets f(new_text) which is called with the new text when the "
                 "user completes text editing");

    // ---- ToggleSwitch ----
    py::class_<ToggleSwitch, UnownedPointer<ToggleSwitch>, Widget> toggle(
            m, "ToggleSwitch", "ToggleSwitch");
    toggle.def(py::init<const char *>(),
               "Creates a toggle switch with the given text")
            .def("__repr__",
                 [](const ToggleSwitch &ts) {
                     std::stringstream s;
                     s << "ToggleSwitch (" << ts.GetFrame().x << ", "
                       << ts.GetFrame().y << "), " << ts.GetFrame().width
                       << " x " << ts.GetFrame().height;
                     return s.str();
                 })
            .def_property("is_on", &ToggleSwitch::GetIsOn, &ToggleSwitch::SetOn,
                          "True if is one, False otherwise")
            .def("set_on_clicked", &ToggleSwitch::SetOnClicked,
                 "Sets f(is_on) which is called when the switch changes "
                 "state.");

    // ---- TreeView ----
    py::class_<TreeView, UnownedPointer<TreeView>, Widget> treeview(
            m, "TreeView", "Hierarchical list");
    treeview.def(py::init<>(), "Creates an empty TreeView widget")
            .def("__repr__",
                 [](const TreeView &tv) {
                     std::stringstream s;
                     s << "TreeView (" << tv.GetFrame().x << ", "
                       << tv.GetFrame().y << "), " << tv.GetFrame().width
                       << " x " << tv.GetFrame().height;
                     return s.str();
                 })
            .def("get_root_item", &TreeView::GetRootItem,
                 "Returns the root item. This item is invisible, so its child "
                 "are the top-level items")
            .def(
                    "add_item",
                    [](TreeView &tree, TreeView::ItemId parent_id,
                       UnownedPointer<Widget> item) {
                        return tree.AddItem(parent_id,
                                            TakeOwnership<Widget>(item));
                    },
                    "Adds a child item to the parent. add_item(parent, widget)")
            .def("add_text_item", &TreeView::AddTextItem,
                 "Adds a child item to the parent. add_text_item(parent, text)")
            .def("remove_item", &TreeView::RemoveItem,
                 "Removes an item and all its children (if any)")
            .def("clear", &TreeView::Clear, "Removes all items")
            .def_property(
                    "can_select_items_with_children",
                    &TreeView::GetCanSelectItemsWithChildren,
                    &TreeView::SetCanSelectItemsWithChildren,
                    "If set to False, clicking anywhere on an item with "
                    "will toggle the item open or closed; the item cannot be "
                    "selected. If set to True, items with children can be "
                    "selected, and to toggle open/closed requires clicking "
                    "the arrow or double-clicking the item")
            .def_property("selected_item", &TreeView::GetSelectedItemId,
                          &TreeView::SetSelectedItemId,
                          "The currently selected item")
            .def("set_on_selection_changed", &TreeView::SetOnSelectionChanged,
                 "Sets f(new_item_id) which is called when the user "
                 "changes the selection.");

    // ---- TreeView cells ----
    py::class_<CheckableTextTreeCell, UnownedPointer<CheckableTextTreeCell>,
               Widget>
            checkable_cell(m, "CheckableTextTreeCell",
                           "TreeView cell with a checkbox and text");
    checkable_cell
            .def(py::init<>([](const char *text, bool checked,
                               std::function<void(bool)> on_toggled) {
                     return new CheckableTextTreeCell(text, checked,
                                                      on_toggled);
                 }),
                 "Creates a TreeView cell with a checkbox and text. "
                 "CheckableTextTreeCell(text, is_checked, on_toggled): "
                 "on_toggled takes a boolean and returns None")
            .def_property_readonly("checkbox",
                                   &CheckableTextTreeCell::GetCheckbox,
                                   "Returns the checkbox widget "
                                   "(property is read-only)")
            .def_property_readonly("label", &CheckableTextTreeCell::GetLabel,
                                   "Returns the label widget "
                                   "(property is read-only)");

    py::class_<LUTTreeCell, UnownedPointer<LUTTreeCell>, Widget> lut_cell(
            m, "LUTTreeCell",
            "TreeView cell with checkbox, text, and color edit");
    lut_cell.def(py::init<>([](const char *text, bool checked,
                               const Color &color,
                               std::function<void(bool)> on_enabled,
                               std::function<void(const Color &)> on_color) {
                     return new LUTTreeCell(text, checked, color, on_enabled,
                                            on_color);
                 }),
                 "Creates a TreeView cell with a checkbox, text, and "
                 "a color editor. LUTTreeCell(text, is_checked, color, "
                 "on_enabled, on_color): on_enabled is called when the "
                 "checkbox toggles, and takes a boolean and returns None"
                 "; on_color is called when the user changes the color "
                 "and it takes a gui.Color and returns None.")
            .def_property_readonly("checkbox", &LUTTreeCell::GetCheckbox,
                                   "Returns the checkbox widget "
                                   "(property is read-only)")
            .def_property_readonly("label", &LUTTreeCell::GetLabel,
                                   "Returns the label widget "
                                   "(property is read-only)")
            .def_property_readonly("color_edit", &LUTTreeCell::GetColorEdit,
                                   "Returns the ColorEdit widget "
                                   "(property is read-only)");

    py::class_<ColormapTreeCell, UnownedPointer<ColormapTreeCell>, Widget>
            colormap_cell(m, "ColormapTreeCell",
                          "TreeView cell with a number edit and color edit");
    colormap_cell
            .def(py::init<>([](float value, const Color &color,
                               std::function<void(double)> on_value_changed,
                               std::function<void(const Color &)>
                                       on_color_changed) {
                     return new ColormapTreeCell(value, color, on_value_changed,
                                                 on_color_changed);
                 }),
                 "Creates a TreeView cell with a number and a color edit. "
                 "ColormapTreeCell(value, color, on_value_changed, "
                 "on_color_changed): on_value_changed takes a double "
                 "and returns None; on_color_changed takes a "
                 "gui.Color and returns None")
            .def_property_readonly("number_edit",
                                   &ColormapTreeCell::GetNumberEdit,
                                   "Returns the NumberEdit widget "
                                   "(property is read-only)")
            .def_property_readonly("color_edit",
                                   &ColormapTreeCell::GetColorEdit,
                                   "Returns the ColorEdit widget "
                                   "(property is read-only)");

    // ---- VectorEdit ----
    py::class_<VectorEdit, UnownedPointer<VectorEdit>, Widget> vectoredit(
            m, "VectorEdit", "Allows the user to edit a 3-space vector");
    vectoredit.def(py::init<>())
            .def("__repr__",
                 [](const VectorEdit &ve) {
                     auto val = ve.GetValue();
                     std::stringstream s;
                     s << "VectorEdit [" << val.x() << ", " << val.y() << ", "
                       << val.z() << "] (" << ve.GetFrame().x << ", "
                       << ve.GetFrame().y << "), " << ve.GetFrame().width
                       << " x " << ve.GetFrame().height;
                     return s.str();
                 })
            .def_property("vector_value", &VectorEdit::GetValue,
                          &VectorEdit::SetValue, "Returns value [x, y, z]")
            .def("set_on_value_changed", &VectorEdit::SetOnValueChanged,
                 "Sets f([x, y, z]) which is called whenever the user "
                 "changes the value of a component");

    // ---- Margins ----
    py::class_<Margins, UnownedPointer<Margins>> margins(m, "Margins",
                                                         "Margins for layouts");
    margins.def(py::init([](int left, int top, int right, int bottom) {
                    return new Margins(left, top, right, bottom);
                }),
                "left"_a = 0, "top"_a = 0, "right"_a = 0, "bottom"_a = 0,
                "Creates margins. Arguments are left, top, right, bottom. "
                "Use the em-size (window.theme.font_size) rather than pixels "
                "for more consistency across platforms and monitors. Margins "
                "are the spacing from the edge of the widget's frame to its "
                "content area. They act similar to the 'padding' property in "
                "CSS")
            .def(py::init([](float left, float top, float right, float bottom) {
                     return new Margins(
                             int(std::round(left)), int(std::round(top)),
                             int(std::round(right)), int(std::round(bottom)));
                 }),
                 "left"_a = 0.0f, "top"_a = 0.0f, "right"_a = 0.0f,
                 "bottom"_a = 0.0f,
                 "Creates margins. Arguments are left, top, right, bottom. "
                 "Use the em-size (window.theme.font_size) rather than pixels "
                 "for more consistency across platforms and monitors. Margins "
                 "are the spacing from the edge of the widget's frame to its "
                 "content area. They act similar to the 'padding' property in "
                 "CSS")
            .def_readwrite("left", &Margins::left)
            .def_readwrite("top", &Margins::top)
            .def_readwrite("right", &Margins::right)
            .def_readwrite("bottom", &Margins::bottom)
            .def("get_horiz", &Margins::GetHoriz)
            .def("get_vert", &Margins::GetVert);

    // ---- Layout1D ----
    py::class_<Layout1D, UnownedPointer<Layout1D>, Widget> layout1d(
            m, "Layout1D", "Layout base class");
    layout1d
            // TODO: write the proper constructor
            //        .def(py::init([]() { return new Layout1D(Layout1D::VERT,
            //        0, Margins(), {}); }))
            .def("add_fixed", &Layout1D::AddFixed,
                 "Adds a fixed amount of empty space to the layout")
            .def(
                    "add_fixed",
                    [](UnownedPointer<Layout1D> layout, float px) {
                        layout->AddFixed(int(std::round(px)));
                    },
                    "Adds a fixed amount of empty space to the layout")
            .def("add_stretch", &Layout1D::AddStretch,
                 "Adds empty space to the layout that will take up as much "
                 "extra space as there is available in the layout");

    // ---- Vert ----
    py::class_<Vert, UnownedPointer<Vert>, Layout1D> vlayout(m, "Vert",
                                                             "Vertical layout");
    vlayout.def(py::init([](int spacing, const Margins &margins) {
                    return new Vert(spacing, margins);
                }),
                "spacing"_a = 0, "margins"_a = Margins(),
                "Creates a layout that arranges widgets vertically, top to "
                "bottom, making their width equal to the layout's width. First "
                "argument is the spacing between widgets, the second is the "
                "margins. Both default to 0.")
            .def(py::init([](float spacing, const Margins &margins) {
                     return new Vert(int(std::round(spacing)), margins);
                 }),
                 "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "Creates a layout that arranges widgets vertically, top to "
                 "bottom, making their width equal to the layout's width. "
                 "First argument is the spacing between widgets, the second "
                 "is the margins. Both default to 0.")
            .def_property("preferred_width", &Vert::GetPreferredWidth,
                          &Vert::SetPreferredWidth,
                          "Sets the preferred width of the layout");

    // ---- CollapsableVert ----
    py::class_<CollapsableVert, UnownedPointer<CollapsableVert>, Vert>
            collapsable(m, "CollapsableVert",
                        "Vertical layout with title, whose contents are "
                        "collapsable");
    collapsable
            .def(py::init([](const char *text, int spacing,
                             const Margins &margins) {
                     return new CollapsableVert(text, spacing, margins);
                 }),
                 "text"_a, "spacing"_a = 0, "margins"_a = Margins(),
                 "Creates a layout that arranges widgets vertically, top to "
                 "bottom, making their width equal to the layout's width. "
                 "First argument is the heading text, the second is the "
                 "spacing between widgets, and the third is the margins. "
                 "Both the spacing and the margins default to 0.")
            .def(py::init([](const char *text, float spacing,
                             const Margins &margins) {
                     return new CollapsableVert(text, int(std::round(spacing)),
                                                margins);
                 }),
                 "text"_a, "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "Creates a layout that arranges widgets vertically, top to "
                 "bottom, making their width equal to the layout's width. "
                 "First argument is the heading text, the second is the "
                 "spacing between widgets, and the third is the margins. "
                 "Both the spacing and the margins default to 0.")
            .def("set_is_open", &CollapsableVert::SetIsOpen, "is_open"_a,
                 "Sets to collapsed (False) or open (True). Requires a call to "
                 "Window.SetNeedsLayout() afterwards, unless calling before "
                 "window is visible")
            .def("get_is_open", &CollapsableVert::GetIsOpen,
                 "Check if widget is open.")
            .def_property("font_id", &CollapsableVert::GetFontId,
                          &CollapsableVert::SetFontId,
                          "Set the font using the FontId returned from "
                          "Application.add_font()");

    // ---- ScrollableVert ----
    py::class_<ScrollableVert, UnownedPointer<ScrollableVert>, Vert> slayout(
            m, "ScrollableVert", "Scrollable vertical layout");
    slayout.def(py::init([](int spacing, const Margins &margins) {
                    return new ScrollableVert(spacing, margins);
                }),
                "spacing"_a = 0, "margins"_a = Margins(),
                "Creates a layout that arranges widgets vertically, top to "
                "bottom, making their width equal to the layout's width. First "
                "argument is the spacing between widgets, the second is the "
                "margins. Both default to 0.")
            .def(py::init([](float spacing, const Margins &margins) {
                     return new ScrollableVert(int(std::round(spacing)),
                                               margins);
                 }),
                 "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "Creates a layout that arranges widgets vertically, top to "
                 "bottom, making their width equal to the layout's width. "
                 "First argument is the spacing between widgets, the second "
                 "is the margins. Both default to 0.");

    // ---- Horiz ----
    py::class_<Horiz, UnownedPointer<Horiz>, Layout1D> hlayout(
            m, "Horiz", "Horizontal layout");
    hlayout.def(py::init([](int spacing, const Margins &margins) {
                    return new Horiz(spacing, margins);
                }),
                "spacing"_a = 0, "margins"_a = Margins(),
                "Creates a layout that arranges widgets vertically, left to "
                "right, making their height equal to the layout's height "
                "(which will generally be the largest height of the items). "
                "First argument is the spacing between widgets, the second "
                "is the margins. Both default to 0.")
            .def(py::init([](float spacing, const Margins &margins) {
                     return new Horiz(int(std::round(spacing)), margins);
                 }),
                 "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "Creates a layout that arranges widgets vertically, left to "
                 "right, making their height equal to the layout's height "
                 "(which will generally be the largest height of the items). "
                 "First argument is the spacing between widgets, the second "
                 "is the margins. Both default to 0.")
            .def_property("preferred_height", &Horiz::GetPreferredHeight,
                          &Horiz::SetPreferredHeight,
                          "Sets the preferred height of the layout");

    // ---- VGrid ----
    py::class_<VGrid, UnownedPointer<VGrid>, Widget> vgrid(m, "VGrid",
                                                           "Grid layout");
    vgrid.def(py::init([](int n_cols, int spacing, const Margins &margins) {
                  return new VGrid(n_cols, spacing, margins);
              }),
              "cols"_a, "spacing"_a = 0, "margins"_a = Margins(),
              "Creates a layout that orders its children in a grid, left to "
              "right, top to bottom, according to the number of columns. "
              "The first argument is the number of columns, the second is the "
              "spacing between items (both vertically and horizontally), and "
              "third is the margins. Both spacing and margins default to zero.")
            .def(py::init(
                         [](int n_cols, float spacing, const Margins &margins) {
                             return new VGrid(n_cols, int(std::round(spacing)),
                                              margins);
                         }),
                 "cols"_a, "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "Creates a layout that orders its children in a grid, left to "
                 "right, top to bottom, according to the number of columns. "
                 "The first argument is the number of columns, the second is "
                 "the "
                 "spacing between items (both vertically and horizontally), "
                 "and "
                 "third is the margins. Both spacing and margins default to "
                 "zero.")
            .def_property_readonly(
                    "spacing", &VGrid::GetSpacing,
                    "Returns the spacing between rows and columns")
            .def_property_readonly("margins", &VGrid::GetMargins,
                                   "Returns the margins")
            .def_property("preferred_width", &VGrid::GetPreferredWidth,
                          &VGrid::SetPreferredWidth,
                          "Sets the preferred width of the layout");

    // ---- Dialog ----
    py::class_<Dialog, UnownedPointer<Dialog>, Widget> dialog(m, "Dialog",
                                                              "Dialog");
    dialog.def(py::init<const char *>(),
               "Creates a dialog with the given title");

    // ---- FileDialog ----
    py::class_<FileDialog, UnownedPointer<FileDialog>, Dialog> filedlg(
            m, "FileDialog", "File picker dialog");
    py::enum_<FileDialog::Mode> filedlg_mode(filedlg, "Mode", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    filedlg_mode.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for FileDialog modes.";
            }),
            py::none(), py::none(), "");
    filedlg_mode.value("OPEN", FileDialog::Mode::OPEN)
            .value("SAVE", FileDialog::Mode::SAVE)
            .value("OPEN_DIR", FileDialog::Mode::OPEN_DIR)
            .export_values();
    filedlg.def(py::init<FileDialog::Mode, const char *, const Theme &>(),
                "Creates either an open or save file dialog. The first "
                "parameter is either FileDialog.OPEN or FileDialog.SAVE. The "
                "second is the title of the dialog, and the third is the "
                "theme, "
                "which is used internally by the dialog for layout. The theme "
                "should normally be retrieved from window.theme.")
            .def("set_path", &FileDialog::SetPath,
                 "Sets the initial path path of the dialog")
            .def("add_filter", &FileDialog::AddFilter,
                 "Adds a selectable file-type filter: "
                 "add_filter('.stl', 'Stereolithography mesh'")
            .def("set_on_cancel", &FileDialog::SetOnCancel,
                 "Cancel callback; required")
            .def("set_on_done", &FileDialog::SetOnDone,
                 "Done callback; required");
}

void pybind_gui(py::module &m) {
    py::module m_gui = m.def_submodule("gui");
    pybind_gui_events(m_gui);
    pybind_gui_classes(m_gui);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
