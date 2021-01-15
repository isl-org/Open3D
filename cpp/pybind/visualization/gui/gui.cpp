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

#include "pybind/visualization/gui/gui.h"

#include "open3d/geometry/Image.h"
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
#include "open3d/visualization/gui/ImageLabel.h"
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
#include "open3d/visualization/gui/TreeView.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderToBuffer.h"
#include "pybind/docstring.h"
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

    std::function<void(const Theme)> on_layout_;

protected:
    void Layout(const Theme &theme) {
        if (on_layout_) {
            // the Python callback sizes the children
            on_layout_(theme);
            // and then we need to layout the children
            for (auto child : GetChildren()) {
                child->Layout(theme);
            }
        } else {
            Super::Layout(theme);
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
    PythonUnlocker unlocker;
    return Application::GetInstance().RenderToImage(
            unlocker, scene->GetView(), scene->GetScene(), width, height);
}

enum class EventCallbackResult { IGNORED = 0, HANDLED, CONSUMED };

void pybind_gui_classes(py::module &m) {
    // ---- Application ----
    py::class_<Application> application(
            m, "Application",
            "The global application singleton that that owns the the menubar, "
            "windows, and event loop.");
    application
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
                    "called before using the gui module.")
            .def(
                    "initialize",
                    [](Application &instance, const char *resource_dir) {
                        InitializeForPython(resource_dir);
                    },
                    "Initializes the application with location of the "
                    "resources provided by the caller. One of the `initialize` "
                    "functions "
                    "_must_ be called before using any gui module")
            .def("set_font_for_language", &Application::SetFontForLanguage,
                 "set_font_for_language(font, language_code). The font path "
                 "that can be the path to a TrueType or OpenType font, or "
                 "name of the font in which case the font will be located "
                 "from the system directories. The language code must be "
                 "two-letter, lowercase ISO 639-1 codes. Currently, support is "
                 "available for 'en' (English), 'ja' (Japanese), 'ko' "
                 "(Korean), 'th' (Thai), 'vi' (Vietnamese), 'zh' (common "
                 "Chinese characters), 'zh_all' (all Chinese characters; "
                 "this creates a very large bitmap for each window). All "
                 "other codes are assumed to by Cyrillic. Note that 'ja', "
                 "'zh' will create a 50 MB bitmap, and 'zh_all' creates a "
                 "200 MB bitmap")

            .def("set_font_for_code_points", &Application::SetFontForCodePoints,
                 "set_font_for_code_points(font, [unicode_code_points])."
                 "The font can be the path to a TrueType or OpenType font or "
                 "it can be the name of the font, in which case the font "
                 "will be located from the system directories. You will "
                 "get an error if the font contains glyphs "
                 "for the specified components.")
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
                    "This method creates a window and adds it to the "
                    "application. "
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
                    "The method runs the event loop. After this is complete, "
                    "all windows and "
                    "widgets must be considered uninitialized, even if they "
                    "are still held by Python variables. Using them is unsafe, "
                    "even if you call run() again.")
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
                    "The method runs the event loop once, and returns True if "
                    "the app is "
                    "still running, or False if all the windows are closed "
                    "or quit() has been called.")
            .def(
                    "render_to_image",
                    [](Application &instance, rendering::Open3DScene *scene,
                       int width, int height) {
                        return RenderToImageWithoutWindow(scene, width, height);
                    },
                    "The method renders a scene to an image and returns the "
                    "image. If you "
                    "are rendering without a visible window, then you must use"
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
                    "when "
                    "creating object that is a Window directly, rather than "
                    "with "
                    "create_window")
            .def("run_in_thread", &Application::RunInThread,
                 "The function runs function in a separate thread. You must "
                 "not call GUI "
                 "functions on this thread, rather call post_to_main_thread() "
                 "if "
                 "this thread needs to change the GUI.")
            .def("post_to_main_thread", &Application::PostToMainThread,
                 py::call_guard<py::gil_scoped_release>(),
                 "The function runs the provided function on the main thread. "
                 "You can "
                 "use this to execute UI-related code at a safe point in "
                 "time. If the UI changes, then you have to manually "
                 "request a redraw of the window with w.post_redraw().")
            .def_property("menubar", &Application::GetMenubar,
                          &Application::SetMenubar,
                          "The Menu for the application (initially None)")
            .def_property_readonly("now", &Application::Now,
                                   "The current time in seconds.")
            // Note: we cannot export AddWindow and RemoveWindow
            .def_property_readonly("resource_path",
                                   &Application::GetResourcePath,
                                   "The path to the resources directory.");

    // ---- Window ----
    // Pybind appears to need to know about the base class. It doesn't have
    // to be named the same as the C++ class, though. The holder object cannot
    // be a shared_ptr or we can crash (see comment for UnownedPointer).
    py::class_<Window, UnownedPointer<Window>> window_base(
            m, "WindowBase", "Application window");
    py::class_<PyWindow, UnownedPointer<PyWindow>, Window> window(
            m, "Window",
            "The application window. You can create "
            "a window using the Application.instance.create_window() method.");
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
                          "The window's rectangle in OS coordinates (not "
                          "device pixels).")
            .def_property("title", &PyWindow::GetTitle, &PyWindow::SetTitle,
                          "The title of the window.")
            .def("size_to_fit", &PyWindow::SizeToFit,
                 "The method sets the width and height of window to its "
                 "preferred size.")
            .def_property("size", &PyWindow::GetSize, &PyWindow::SetSize,
                          "The size of the window in device pixels, including "
                          "menubar (except on macOS).")
            .def_property_readonly(
                    "content_rect", &PyWindow::GetContentRect,
                    "The frame in device pixels, relative "
                    " to the window, which is available for widgets "
                    "(read-only).")
            .def_property_readonly("scaling", &PyWindow::GetScaling,
                                   "The scaling factor between OS pixels "
                                   "and device pixels (read-only).")
            .def_property_readonly("is_visible", &PyWindow::IsVisible,
                                   "True if window is visible (read-only).")
            .def("show", &PyWindow::Show,
                 "The method shows or hides the window.")
            .def("close", &PyWindow::Close,
                 "The method closes the window and destructs it.")
            .def("set_needs_layout", &PyWindow::SetNeedsLayout,
                 "The method flags a window for re-layout.")
            .def("post_redraw", &PyWindow::PostRedraw,
                 "The method sends a redraw message to the OS message queue.")
            .def_property_readonly(
                    "is_active_window", &PyWindow::IsActiveWindow,
                    "Indicates if the window is currently the active "
                    "window (read-only)")
            .def("set_focus_widget", &PyWindow::SetFocusWidget,
                 "The method sets the text focus to the specified widget.")
            .def("set_on_menu_item_activated",
                 &PyWindow::SetOnMenuItemActivated,
                 "The method sets the callback function for menu item:  "
                 "callback().")
            .def("set_on_tick_event", &PyWindow::SetOnTickEvent,
                 "The method sets callback for a tick event. A callback takes "
                 "no arguments "
                 "and must return True if a redraw is needed (if "
                 "any widget has changed) or False if nothing "
                 "has changed.")
            .def(
                    "set_on_layout",
                    [](PyWindow *w, std::function<void(const Theme &)> f) {
                        w->on_layout_ = f;
                    },
                    "The method sets a callback function that manually sets "
                    "the frames of "
                    "children of the window.")
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
                 "The method displays a simple dialog with a title and message "
                 "and okay "
                 "button.")
            .def_property_readonly("renderer", &PyWindow::GetRenderer,
                                   "The method gets the rendering.Renderer "
                                   "object for the Window.");

    // ---- Menu ----
    py::class_<Menu, UnownedPointer<Menu>> menu(
            m, "Menu",
            "The class lets you manage the menu bar, or menu tree, for the "
            "window.");
    menu.def(py::init<>())
            .def(
                    "add_item",
                    [](UnownedPointer<Menu> menu, const char *text,
                       int item_id) { menu->AddItem(text, item_id); },
                    "The method adds a menu item with id to the menu.")
            .def(
                    "add_menu",
                    [](UnownedPointer<Menu> menu, const char *text,
                       UnownedPointer<Menu> submenu) {
                        menu->AddMenu(text, TakeOwnership<Menu>(submenu));
                    },
                    "The method adds a submenu to a menu item.")
            .def("add_separator", &Menu::AddSeparator,
                 "The method adds a separator to the menu.")
            .def(
                    "set_enabled",
                    [](UnownedPointer<Menu> menu, int item_id, bool enabled) {
                        menu->SetEnabled(item_id, enabled);
                    },
                    "The function sets menu item enabled or disabled.")
            .def(
                    "is_checked",
                    [](UnownedPointer<Menu> menu, int item_id) -> bool {
                        return menu->IsChecked(item_id);
                    },
                    "Indicates if the menu item is checked.")
            .def(
                    "set_checked",
                    [](UnownedPointer<Menu> menu, int item_id, bool checked) {
                        menu->SetChecked(item_id, checked);
                    },
                    "The function checks or unchecks a menu item.");

    // ---- Color ----
    py::class_<Color> color(m, "Color",
                            "The class lets you store colors for GUI classes.");
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
    py::class_<Theme> theme(
            m, "Theme",
            "(Read-only) The Theme parameters such as colors used for drawing "
            "widgets.");
    theme.def_readonly("font_size", &Theme::font_size,
                       "The font size, which is also the "
                       "conventional size of the "
                       "em unit. (read-only)")
            .def_readonly("default_margin", &Theme::default_margin,
                          "The default value for margins, used for "
                          "layouts. (read-only)")
            .def_readonly("default_layout_spacing",
                          &Theme::default_layout_spacing,
                          "A value for the spacing parameter in "
                          "layouts. (read-only)");

    // ---- Rect ----
    py::class_<Rect> rect(m, "Rect",
                          "The class lets you manage the widget frame.");
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
    py::class_<Size> size(m, "Size", "The size of the window.");
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
    py::class_<Widget, UnownedPointer<Widget>> widget(
            m, "Widget", "The base class that is used to create widgets.");
    py::enum_<EventCallbackResult> widget_event_callback_result(
            widget, "EventCallbackResult", "Returned by event handlers",
            py::arithmetic());
    widget_event_callback_result
            .value("IGNORED", EventCallbackResult::IGNORED,
                   "Event handler ignored the event, widget will "
                   "handle event normally.")
            .value("HANDLED", EventCallbackResult::HANDLED,
                   "Event handler handled the event, but widget "
                   "will still handle the event normally. This is "
                   "useful when you are augmenting base "
                   "functionality.")
            .value("CONSUMED", EventCallbackResult::CONSUMED,
                   "Event handler consumed the event, event "
                   "handling stops, widget will not handle the "
                   "event. This is useful when you are replacing "
                   "functionality.")
            .export_values();

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
                 "The method returns the array of children. Do not modify.")
            .def_property("frame", &Widget::GetFrame, &Widget::SetFrame,
                          "The widget's frame. This value is overridden "
                          "if the frame is within a layout.")
            .def_property("visible", &Widget::IsVisible, &Widget::SetVisible,
                          "Indicates if the widget is visible.")
            .def_property("enabled", &Widget::IsEnabled, &Widget::SetEnabled,
                          "Indicates if the widget is enabled.")
            .def("calc_preferred_size", &Widget::CalcPreferredSize,
                 "The function returns the preferred size of a widget. It is "
                 "recommended to use this function only during layout, though, "
                 "it will also "
                 "work during drawing. The function does not work in other "
                 "scenarios "
                 "because it requires some internal setup in order to function "
                 "properly.");

    // ---- Button ----
    py::class_<Button, UnownedPointer<Button>, Widget> button(
            m, "Button", "The buttons that can be added to the window.");
    button.def(py::init<const char *>(),
               "The function creates a button with the given text.")
            .def("__repr__",
                 [](const Button &b) {
                     std::stringstream s;
                     s << "Button (" << b.GetFrame().x << ", " << b.GetFrame().y
                       << "), " << b.GetFrame().width << " x "
                       << b.GetFrame().height;
                     return s.str();
                 })
            .def_property("text", &Button::GetText, &Button::SetText,
                          "Gets or set the button text.")
            .def_property(
                    "toggleable", &Button::GetIsToggleable,
                    &Button::SetToggleable,
                    "Indicates if the button is toggleable or a push button.")
            .def_property("is_on", &Button::GetIsOn, &Button::SetOn,
                          "Indicates if the button is toggleable and in the On "
                          "state.")
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
                    "Horizontal padding in em units.")
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
                 "The method to call when the button is pressed.");

    // ---- Checkbox ----
    py::class_<Checkbox, UnownedPointer<Checkbox>, Widget> checkbox(
            m, "Checkbox", "The class lets you manage a checkbox.");
    checkbox.def(py::init<const char *>(),
                 "The function creates a checkbox with the given text.")
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
                 "The functionality to call when the checkbox is checked.");

    // ---- ColorEdit ----
    py::class_<ColorEdit, UnownedPointer<ColorEdit>, Widget> coloredit(
            m, "ColorEdit", "The class lets you manage a colorpicker.");
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
            m, "Combobox", "The class lets you manage a pull-down menu.");
    combobox.def(py::init<>(),
                 "Creates an empty combobox. You must use add_item() to add "
                 "items.")
            .def("clear_items", &Combobox::ClearItems, "Removes all items.")
            .def("add_item", &Combobox::AddItem, "Adds an item to the end.")
            .def("change_item",
                 (void (Combobox::*)(int, const char *)) & Combobox::ChangeItem,
                 "Changes the text of the item at index: "
                 "change_item(index, newtext).")
            .def("change_item",
                 (void (Combobox::*)(const char *, const char *)) &
                         Combobox::ChangeItem,
                 "Changes the text of the matching item: "
                 "change_item(text, newtext).")
            .def("remove_item",
                 (void (Combobox::*)(const char *)) & Combobox::RemoveItem,
                 "Removes the first item of the given text.")
            .def("remove_item",
                 (void (Combobox::*)(int)) & Combobox::RemoveItem,
                 "Removes the item at the index")
            .def_property_readonly(
                    "number_of_items", &Combobox::GetNumberOfItems,
                    "(Read-Only) The number of items in the list.")
            .def("get_item", &Combobox::GetItem,
                 "Returns the item at the given index")
            .def_property("selected_index", &Combobox::GetSelectedIndex,
                          &Combobox::SetSelectedIndex,
                          "The index of the currently selected item.")
            .def_property("selected_text", &Combobox::GetSelectedValue,
                          &Combobox::SetSelectedValue,
                          "The value of the currently selected item")
            .def("set_on_selection_changed", &Combobox::SetOnValueChanged,
                 "This method calls f(str, int) when user selects item"
                 " from combobox. Arguments are the selected text "
                 " and selected index.");

    // ---- ImageLabel ----
    py::class_<ImageLabel, UnownedPointer<ImageLabel>, Widget> imagelabel(
            m, "ImageLabel",
            "The class displays a bitmap or image on the window.");
    imagelabel
            .def(py::init<>(
                         [](const char *path) { return new ImageLabel(path); }),
                 "The method creates an ImageLabel from the image at the "
                 "specified path.")
            .def("__repr__", [](const ImageLabel &il) {
                std::stringstream s;
                s << "ImageLabel (" << il.GetFrame().x << ", "
                  << il.GetFrame().y << "), " << il.GetFrame().width << " x "
                  << il.GetFrame().height;
                return s.str();
            });
    // TODO: add the other functions and UIImage?

    // ---- Label ----
    py::class_<Label, UnownedPointer<Label>, Widget> label(
            m, "Label", "The class manages labels in the windows.");
    label.def(py::init([](const char *title = "") { return new Label(title); }),
              "The method creates a label with the given text.")
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
                          "The text of the label. All newlines in the text "
                          "will be treated as line breaks")
            .def_property("text_color", &Label::GetTextColor,
                          &Label::SetTextColor,
                          "The color of the text (gui.Color)");

    // ---- Label3D ----
    py::class_<Label3D> label3d(m, "Label3D",
                                "The class displays text in a 3D scene");
    label3d.def_property("text", &Label3D::GetText, &Label3D::SetText,
                         "The text to display with this label.")
            .def_property("position", &Label3D::GetPosition,
                          &Label3D::SetPosition,
                          "The position of the text in 3D coordinates.")
            .def_property("color", &Label3D::GetTextColor,
                          &Label3D::SetTextColor,
                          "The color of the text (gui.Color).");

    // ---- ListView ----
    py::class_<ListView, UnownedPointer<ListView>, Widget> listview(
            m, "ListView", "The class manages a list of text.");
    listview.def(py::init<>(), "The method creates an empty list.")
            .def("__repr__",
                 [](const ListView &lv) {
                     std::stringstream s;
                     s << "Label (" << lv.GetFrame().x << ", "
                       << lv.GetFrame().y << "), " << lv.GetFrame().width
                       << " x " << lv.GetFrame().height;
                     return s.str();
                 })
            .def("set_items", &ListView::SetItems,
                 "This sets the list to display the list of items provided.")
            .def_property("selected_index", &ListView::GetSelectedIndex,
                          &ListView::SetSelectedIndex,
                          "The index of the currently selected item.")
            .def_property_readonly("selected_value",
                                   &ListView::GetSelectedValue,
                                   "The text of the currently selected item")
            .def("set_on_selection_changed", &ListView::SetOnValueChanged,
                 "The method calls f(new_val, is_double_click) when user "
                 "changes "
                 "selection");

    // ---- NumberEdit ----
    py::class_<NumberEdit, UnownedPointer<NumberEdit>, Widget> numedit(
            m, "NumberEdit", "The class allows the user to enter a number.");
    py::enum_<NumberEdit::Type> numedit_type(numedit, "Type", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    numedit_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "The enum class for NumberEdit types.";
            }),
            py::none(), py::none(), "");
    numedit_type.value("INT", NumberEdit::Type::INT)
            .value("DOUBLE", NumberEdit::Type::DOUBLE)
            .export_values();

    numedit.def(py::init<NumberEdit::Type>(),
                "The method creates a NumberEdit that is either integers (INT) "
                "or "
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
                          "The number of fractional digits shown.")
            .def_property_readonly("minimum_value",
                                   &NumberEdit::GetMinimumValue,
                                   "The minimum value number can contain "
                                   "(read-only, use set_limits() to set).")
            .def_property_readonly("maximum_value",
                                   &NumberEdit::GetMaximumValue,
                                   "The maximum value number can contain "
                                   "(read-only, use set_limits() to set).")
            .def("set_limits", &NumberEdit::SetLimits,
                 "This sets the minimum and maximum values for the number.")
            .def("set_on_value_changed", &NumberEdit::SetOnValueChanged,
                 "The sets f(new_value) which is called with a Float when user "
                 "changes widget's value")
            .def("set_preferred_width", &NumberEdit::SetPreferredWidth,
                 "The preferred width of the NumberEdit.")
            .def(
                    "set_preferred_width",
                    [](NumberEdit &ne, double width) {
                        ne.NumberEdit::SetPreferredWidth(int(width));
                    },
                    "The preferred width of the NumberEdit.");

    // ---- ProgressBar----
    py::class_<ProgressBar, UnownedPointer<ProgressBar>, Widget> progress(
            m, "ProgressBar", "The class displays a progress bar.");
    progress.def(py::init<>())
            .def("__repr__",
                 [](const ProgressBar &pb) {
                     std::stringstream s;
                     s << "ProgressBar [" << pb.GetValue() << "] ("
                       << pb.GetFrame().x << ", " << pb.GetFrame().y << "), "
                       << pb.GetFrame().width << " x " << pb.GetFrame().height;
                     return s.str();
                 })
            .def_property("value", &ProgressBar::GetValue,
                          &ProgressBar::SetValue,
                          "The value of the progress bar. It ranges from 0.0 "
                          "to 1.0.");

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
            m, "SceneWidget", "The class lets you manage 3D content.");
    py::enum_<SceneWidget::Controls> scene_ctrl(scene, "Controls",
                                                py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    scene_ctrl.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "The enum class describing mouse interaction."
                       " The following interactions is captured and intepreted "
                       "using other methods:";
            }),
            py::none(), py::none(), "");
    scene_ctrl.value("ROTATE_CAMERA", SceneWidget::Controls::ROTATE_CAMERA)
            .value("FLY", SceneWidget::Controls::FLY)
            .value("ROTATE_SUN", SceneWidget::Controls::ROTATE_SUN)
            .value("ROTATE_IBL", SceneWidget::Controls::ROTATE_IBL)
            .value("ROTATE_MODEL", SceneWidget::Controls::ROTATE_MODEL)
            .value("PICK_POINTS", SceneWidget::Controls::PICK_POINTS)
            .export_values();

    scene.def(py::init<>(),
              "The method creates an empty SceneWidget. You must assign "
              " assign a Scene to the widget using 'scene' property")
            .def_property(
                    "scene", &PySceneWidget::GetScene, &SceneWidget::SetScene,
                    "The rendering.Open3DScene that the SceneWidget renders")
            .def("enable_scene_caching", &PySceneWidget::EnableSceneCaching,
                 "Enable/Disable caching of scene content when the view or "
                 "model is not changing. Scene caching can help improve UI "
                 "responsiveness for large models and point clouds")
            .def("force_redraw", &PySceneWidget::ForceRedraw,
                 "This marks the widget for a forced scene redraw, even when "
                 "scene caching is enabled.")
            .def("set_view_controls", &PySceneWidget::SetViewControls,
                 "This sets the mouse interaction, such as ROTATE_OBJ.")
            .def("setup_camera", &PySceneWidget::SetupCamera,
                 "This configures the camera using the paramters "
                 "field_of_view, "
                 "model_bounds, and "
                 "center_of_rotation)")
            .def("set_on_mouse", &PySceneWidget::SetOnMouse,
                 "This sets a callback for mouse events, and takes "
                 " the scene widget as an argument. The widget then"
                 " calls a function to widget as an argument. The widget then"
                 "call the function and passes the MouseEvent object."
                 "The callback returns an "
                 "EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, "
                 "or EventCallackResult.CONSUMED.")
            .def("set_on_key", &PySceneWidget::SetOnKey,
                 "This sets a callback for key events, and takes "
                 " the scene widget as an argument. The widget then"
                 " calls a function to widget as an argument. The widget then"
                 "call the function and passes the KeyEvent object."
                 "The callback returns an "
                 "EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, "
                 "or EventCallackResult.CONSUMED.")
            .def("set_on_sun_direction_changed",
                 &PySceneWidget::SetOnSunDirectionChanged,
                 "The callback when user changes sun direction, and used only "
                 "in ROTATE_SUN control mode. The callback function accepts"
                 " the [i, j, k] vector of the new sun direction.")
            .def("add_3d_label", &PySceneWidget::AddLabel,
                 "This adds a 3D text label to the scene. The label will be "
                 "anchored "
                 "at the specified 3D point.")
            .def("remove_3d_label", &PySceneWidget::RemoveLabel,
                 "This removes the 3D text label from the scene");

    // ---- Slider ----
    py::class_<Slider, UnownedPointer<Slider>, Widget> slider(
            m, "Slider", "The class lets you add and manage a slider.");
    py::enum_<Slider::Type> slider_type(slider, "Type", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    slider_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "The enum class for Slider types.";
            }),
            py::none(), py::none(), "");
    slider_type.value("INT", Slider::Type::INT)
            .value("DOUBLE", Slider::Type::DOUBLE)
            .export_values();

    slider.def(py::init<Slider::Type>(),
               "This method creates a NumberEdit that is either integers (INT) "
               "or "
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
                                   "(read-only, use set_limits() to set).")
            .def_property_readonly("get_maximum_value",
                                   &Slider::GetMaximumValue,
                                   "The maximum value number can contain "
                                   "(read-only, use set_limits() to set).")
            .def("set_limits", &Slider::SetLimits,
                 "Sets the minimum and maximum values for the slider.")
            .def("set_on_value_changed", &Slider::SetOnValueChanged,
                 "Sets f(new_value) which is called with a Float when user "
                 "changes widget's value.");

    // ---- StackedWidget ----
    py::class_<StackedWidget, UnownedPointer<StackedWidget>, Widget> stacked(
            m, "StackedWidget",
            "The class lets you create a stacked widget"
            " that is similar to a TabControl but without the tabs.");
    stacked.def(py::init<>())
            .def_property(
                    "selected_index", &StackedWidget::GetSelectedIndex,
                    &StackedWidget::SetSelectedIndex,
                    "The method selects the index of the child to display.");

    // ---- TabControl ----
    py::class_<TabControl, UnownedPointer<TabControl>, Widget> tabctrl(
            m, "TabControl", "The Tab control.");
    tabctrl.def(py::init<>())
            .def(
                    "add_tab",
                    [](TabControl &tabs, const char *name,
                       UnownedPointer<Widget> panel) {
                        tabs.AddTab(name, TakeOwnership<Widget>(panel));
                    },
                    "Adds a tab. The first parameter is the title of the tab, "
                    "and "
                    "the second parameter is a widget that is usually a "
                    "layout.")
            .def("set_on_selected_tab_changed",
                 &TabControl::SetOnSelectedTabChanged,
                 "The function calls the provided callback function with the "
                 "index of the "
                 "currently selected tab whenever the user clicks on a "
                 "different tab");

    // ---- TextEdit ----
    py::class_<TextEdit, UnownedPointer<TextEdit>, Widget> textedit(
            m, "TextEdit", "The class lets you enter or modify text.");
    textedit.def(py::init<>(),
                 "The function creates a TextEdit widget with an initial value "
                 "of an empty "
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
                          "The text in the widget.")
            .def_property(
                    "placeholder_text", &TextEdit::GetPlaceholderText,
                    &TextEdit::SetPlaceholderText,
                    "The placeholder text displayed when text value is empty.")
            .def("set_on_text_changed", &TextEdit::SetOnTextChanged,
                 "This method sets f(new_text) which is called whenever the "
                 "the user makes "
                 "a change to the text.")
            .def("set_on_value_changed", &TextEdit::SetOnValueChanged,
                 "This method sets f(new_text) which is called with the new "
                 "text when the "
                 "user completes text editing.");

    // ---- TreeView ----
    py::class_<TreeView, UnownedPointer<TreeView>, Widget> treeview(
            m, "TreeView", "The class is a hierarchical list.");
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
                 "items are the top-level items.")
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
                    "If set to False, clicking on an item does not select it. "
                    "It toggles the item's open or close. If set to True,"
                    "clicking on an item with children selects it. "
                    "To toggle the open closed state, you must click "
                    "the arrow or double-clicking the item.")
            .def_property("selected_item", &TreeView::GetSelectedItemId,
                          &TreeView::SetSelectedItemId,
                          "The currently selected item")
            .def("set_on_selection_changed", &TreeView::SetOnSelectionChanged,
                 "The method sets f(new_item_id) which is called when the user "
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
                 "The method creates a cell in the treeview with a checkbox "
                 "and text. "
                 "CheckableTextTreeCell(text, is_checked, on_toggled): "
                 "on_toggled takes a boolean and returns None.")
            .def_property_readonly("checkbox",
                                   &CheckableTextTreeCell::GetCheckbox,
                                   "Returns the checkbox widget "
                                   "(The property is read-only)")
            .def_property_readonly("label", &CheckableTextTreeCell::GetLabel,
                                   "Returns the label widget "
                                   "(The property is read-only)");

    py::class_<LUTTreeCell, UnownedPointer<LUTTreeCell>, Widget> lut_cell(
            m, "LUTTreeCell",
            "The class creates TreeView cell with checkbox, text, and color "
            "edit.");
    lut_cell.def(py::init<>([](const char *text, bool checked,
                               const Color &color,
                               std::function<void(bool)> on_enabled,
                               std::function<void(const Color &)> on_color) {
                     return new LUTTreeCell(text, checked, color, on_enabled,
                                            on_color);
                 }),
                 "The function creates a TreeView cell with a checkbox, text, "
                 "and a color editor."
                 "LUTTreeCell(text, is_checked, color, "
                 "on_enabled, on_color): on_enabled is called when the "
                 "checkbox toggles, and takes a boolean and returns None"
                 "; on_color is called when the user changes the color "
                 "and it takes a gui.Color and returns None.")
            .def_property_readonly("checkbox", &LUTTreeCell::GetCheckbox,
                                   "Returns the checkbox widget "
                                   "(The property is read-only)")
            .def_property_readonly("label", &LUTTreeCell::GetLabel,
                                   "Returns the label widget "
                                   "(The property is read-only)")
            .def_property_readonly("color_edit", &LUTTreeCell::GetColorEdit,
                                   "Returns the ColorEdit widget "
                                   "(The property is read-only)");

    py::class_<ColormapTreeCell, UnownedPointer<ColormapTreeCell>, Widget>
            colormap_cell(m, "ColormapTreeCell",
                          "The class is a TreeView cell with "
                          "a number edit and color edit.");
    colormap_cell
            .def(py::init<>([](float value, const Color &color,
                               std::function<void(double)> on_value_changed,
                               std::function<void(const Color &)>
                                       on_color_changed) {
                     return new ColormapTreeCell(value, color, on_value_changed,
                                                 on_color_changed);
                 }),
                 "The function creates a TreeView cell with a number and a "
                 "color edit. "
                 "You pass value, color (gui.color), on_value_changed, "
                 "on_color_changed."
                 "The on_value_changed takes a double and returns None."
                 "The on_color_changed takes a gui.Color and returns None.")
            .def_property_readonly("number_edit",
                                   &ColormapTreeCell::GetNumberEdit,
                                   "Returns the NumberEdit widget "
                                   "(The property is read-only)")
            .def_property_readonly("color_edit",
                                   &ColormapTreeCell::GetColorEdit,
                                   "Returns the ColorEdit widget "
                                   "(The property is read-only)");

    // ---- VectorEdit ----
    py::class_<VectorEdit, UnownedPointer<VectorEdit>, Widget> vectoredit(
            m, "VectorEdit", "The class lets you edit a 3-space vector.");
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
                 "changes the value of a component.");

    // ---- Margins ----
    py::class_<Margins, UnownedPointer<Margins>> margins(m, "Margins",
                                                         "Margins for layouts");
    margins.def(py::init([](int left, int top, int right, int bottom) {
                    return new Margins(left, top, right, bottom);
                }),
                "left"_a = 0, "top"_a = 0, "right"_a = 0, "bottom"_a = 0,
                "The function creates margins using integers. Margins are the "
                "spacings from the edge"
                " of the widget's frame to its content area. They are similar "
                "to the "
                "`padding` property in CSS. "
                "You pass left, top, right, bottom. You must use the em-size "
                "(window.theme.font_size) rather than pixels for more "
                "consistency across desktop environments.")
            .def(py::init([](float left, float top, float right, float bottom) {
                     return new Margins(
                             int(std::round(left)), int(std::round(top)),
                             int(std::round(right)), int(std::round(bottom)));
                 }),
                 "left"_a = 0.0f, "top"_a = 0.0f, "right"_a = 0.0f,
                 "bottom"_a = 0.0f,
                 "The function creates margins using floating values. Margins "
                 "are the spacings from the edge"
                 " of the widget's frame to its content area. They are similar "
                 "to the "
                 "`padding` property in CSS. "
                 "You pass left, top, right, bottom. You must use the em-size "
                 "(window.theme.font_size) rather than pixels for more "
                 "consistency across desktop environments.")
            .def_readwrite("left", &Margins::left)
            .def_readwrite("top", &Margins::top)
            .def_readwrite("right", &Margins::right)
            .def_readwrite("bottom", &Margins::bottom)
            .def("get_horiz", &Margins::GetHoriz)
            .def("get_vert", &Margins::GetVert);

    // ---- Layout1D ----
    py::class_<Layout1D, UnownedPointer<Layout1D>, Widget> layout1d(
            m, "Layout1D", "The class lets you manage Layouts.");
    layout1d
            // TODO: write the proper constructor
            //        .def(py::init([]() { return new Layout1D(Layout1D::VERT,
            //        0, Margins(), {}); }))
            .def("add_fixed", &Layout1D::AddFixed,
                 "Adds a fixed amount of empty space to the layout.")
            .def(
                    "add_fixed",
                    [](UnownedPointer<Layout1D> layout, float px) {
                        layout->AddFixed(int(std::round(px)));
                    },
                    "Adds a fixed amount of empty space to the layout.")
            .def("add_stretch", &Layout1D::AddStretch,
                 "Adds empty space to the layout that will take up as much "
                 "extra space as there is available in the layout.");

    // ---- Vert ----
    py::class_<Vert, UnownedPointer<Vert>, Layout1D> vlayout(
            m, "Vert", "The class lets you manage vertical layouts.");
    vlayout.def(py::init([](int spacing, const Margins &margins) {
                    return new Vert(spacing, margins);
                }),
                "spacing"_a = 0, "margins"_a = Margins(),
                "The function creates a layout that arranges widgets "
                "vertically,"
                " making their width equal to the layout's width. You pass the "
                "spacing between widgets and the margins. Both default to 0.")
            .def(py::init([](float spacing, const Margins &margins) {
                     return new Vert(int(std::round(spacing)), margins);
                 }),
                 "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "The function creates a layout that arranges widgets "
                 "vertically, top to "
                 "bottom, making their width equal to the layout's width. "
                 "First argument is the spacing between widgets, the second "
                 "is the margins. Both default to 0.");

    // ---- CollapsableVert ----
    py::class_<CollapsableVert, UnownedPointer<CollapsableVert>, Vert>
            collapsable(m, "CollapsableVert",
                        "The class creates a vertical layout with title and"
                        " collapsable contents.");
    collapsable
            .def(py::init([](const char *text, int spacing,
                             const Margins &margins) {
                     return new CollapsableVert(text, spacing, margins);
                 }),
                 "text"_a, "spacing"_a = 0, "margins"_a = Margins(),
                 "The function creates a layout that arranges widgets "
                 "vertically,"
                 " making their width equal to the layout's width. You pass "
                 "the "
                 "heading text, spacing between widgets, and the margins. "
                 "Both the spacing and the margins default to 0.")
            .def(py::init([](const char *text, float spacing,
                             const Margins &margins) {
                     return new CollapsableVert(text, int(std::round(spacing)),
                                                margins);
                 }),
                 "text"_a, "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "The function creates a layout that arranges widgets "
                 "vertically,"
                 " making their width equal to the layout's width. You pass "
                 "the "
                 "heading text, spacing between widgets, and the margins. "
                 "Both the spacing and the margins default to 0.")
            .def("set_is_open", &CollapsableVert::SetIsOpen,
                 "Sets to collapsed (False) or open (True). You must call "
                 "Window.SetNeedsLayout() after the function, unless you "
                 "this before the window is visible.");

    // ---- Horiz ----
    py::class_<Horiz, UnownedPointer<Horiz>, Layout1D> hlayout(
            m, "Horiz",
            "The class lets you create and manage a horizontal layout.");
    hlayout.def(py::init([](int spacing, const Margins &margins) {
                    return new Horiz(spacing, margins);
                }),
                "spacing"_a = 0, "margins"_a = Margins(),
                "The function creates a layout that arranges widgets "
                "vertically. "
                "This makes their height equal to the layout's height . You "
                "pass the spacing and the margin as numbers. Both default to "
                "0.")
            .def(py::init([](float spacing, const Margins &margins) {
                     return new Horiz(int(std::round(spacing)), margins);
                 }),
                 "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "The function creates a layout that arranges widgets "
                 "vertically. "
                 "This makes their height equal to the layout's height . You "
                 "pass the spacing and the margin as float values. Both "
                 "default to 0.");

    // ---- VGrid ----
    py::class_<VGrid, UnownedPointer<VGrid>, Widget> vgrid(m, "VGrid",
                                                           "Grid layout");
    vgrid.def(py::init([](int n_cols, int spacing, const Margins &margins) {
                  return new VGrid(n_cols, spacing, margins);
              }),
              "cols"_a, "spacing"_a = 0, "margins"_a = Margins(),
              "The function creates a layout that orders its children in a "
              "grid from left to right and top to bottom, according to "
              "the number of columns. You pass the number of columns, the "
              "spacing between items (both vertically and horizontally), and "
              "margins as number values. Both spacing and margins default to "
              "0.")
            .def(py::init(
                         [](int n_cols, float spacing, const Margins &margins) {
                             return new VGrid(n_cols, int(std::round(spacing)),
                                              margins);
                         }),
                 "cols"_a, "spacing"_a = 0.0f, "margins"_a = Margins(),
                 "The function creates a layout that orders its children in a "
                 "grid from left to right and top to bottom, according to "
                 "the number of columns. You pass the number of columns, the "
                 "spacing between items (both vertically and horizontally), "
                 "and "
                 "margins as float values. Both spacing and margins default to "
                 "0.")
            .def_property_readonly(
                    "spacing", &VGrid::GetSpacing,
                    "Returns the spacing between rows and columns")
            .def_property_readonly("margins", &VGrid::GetMargins,
                                   "Returns the margins");

    // ---- Dialog ----
    py::class_<Dialog, UnownedPointer<Dialog>, Widget> dialog(
            m, "Dialog",
            "The lets you create and "
            " manage a dialog box.");
    dialog.def(py::init<const char *>(),
               "The function creates a dialog with the given title.");

    // ---- FileDialog ----
    py::class_<FileDialog, UnownedPointer<FileDialog>, Dialog> filedlg(
            m, "FileDialog",
            "The class let you create and manage a File picker dialog.");
    py::enum_<FileDialog::Mode> filedlg_mode(filedlg, "Mode", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    filedlg_mode.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "The enum class for FileDialog modes.";
            }),
            py::none(), py::none(), "");
    filedlg_mode.value("OPEN", FileDialog::Mode::OPEN)
            .value("SAVE", FileDialog::Mode::SAVE)
            .export_values();
    filedlg.def(py::init<FileDialog::Mode, const char *, const Theme &>(),
                "The function create an open or save file dialog. You pass "
                "FileDialog.OPEN or FileDialog.SAVE, title of the dialog, "
                "and the theme, which is used internally by the dialog "
                "for layout. You normally retrieve the theme from "
                "window.theme.")
            .def("set_path", &FileDialog::SetPath,
                 "Sets the initial path path of the dialog.")
            .def("add_filter", &FileDialog::AddFilter,
                 "Adds a selectable file-type filter: "
                 "add_filter('.stl', 'Stereolithography mesh').")
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
