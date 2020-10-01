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
#include "pybind/docstring.h"
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

void pybind_gui_classes(py::module &m) {
    // ---- Application ----
    py::class_<Application> application(m, "Application",
                                        "Global application singleton. This "
                                        "owns the menubar, windows, and event "
                                        "loop");
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
                    [](Application &instance) {
                        // We need to find the resources directory. Fortunately,
                        // Python knows where the module lives (open3d.__file__
                        // is the path to
                        // __init__.py), so we can use that to find the
                        // resources included in the wheel.
                        py::object o3d = py::module::import("open3d");
                        auto o3d_init_path =
                                o3d.attr("__file__").cast<std::string>();
                        auto module_path =
                                utility::filesystem::GetFileParentDirectory(
                                        o3d_init_path);
                        auto resource_path = module_path + "/resources";
                        instance.Initialize(resource_path.c_str());
                        install_cleanup_atexit();
                    },
                    "Initializes the application, using the resources included "
                    "in the wheel. One of the `initialize` functions _must_ be "
                    "called prior to using anything in the gui module")
            .def(
                    "initialize",
                    [](Application &instance, const char *resource_dir) {
                        instance.Initialize(resource_dir);
                        install_cleanup_atexit();
                    },
                    "Initializes the application with location of the "
                    "resources "
                    "provided by the caller. One of the `initialize` functions "
                    "_must_ be called prior to using anything in the gui "
                    "module")
            .def(
                    "run",
                    [](Application &instance) {
                        PythonUnlocker unlocker;
                        while (instance.RunOneTick(unlocker)) {
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
                    "quit", [](Application &instance) { instance.Quit(); },
                    "Closes all the windows, exiting as a result")
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
            .def("add_window", &Application::AddWindow,
                 "Adds the window to the application")
            .def("remove_window", &Application::AddWindow,
                 "Removes the window from the application, closing it. If "
                 "there are no open windows left the event loop will exit.")
            .def_property_readonly("resource_path",
                                   &Application::GetResourcePath,
                                   "Returns a string with the path to the "
                                   "resources directory");

    // ---- Window ----
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

    // Pybind appears to need to know about the base class. It doesn't have
    // to be named the same as the C++ class, though.
    py::class_<Window, std::shared_ptr<Window>> window_base(
            m, "WindowBase", "Application window");
    py::class_<PyWindow, std::shared_ptr<PyWindow>, Window> window(
            m, "Window", "Application window");
    window.def(py::init([](const std::string &title, int width, int height,
                           int x, int y, int flags) {
                   if (x < 0 && y < 0 && width < 0 && height < 0) {
                       return new PyWindow(title, flags);
                   } else if (x < 0 && y < 0) {
                       return new PyWindow(title, width, height, flags);
                   } else {
                       return new PyWindow(title, x, y, width, height, flags);
                   }
               }),
               "title"_a = std::string(), "width"_a = -1, "height"_a = -1,
               "x"_a = -1, "y"_a = -1, "flags"_a = 0)
            .def("__repr__",
                 [](const PyWindow &w) { return "Application window"; })
            .def("add_child", &PyWindow::AddChild,
                 "Adds a widget to the window")
            .def_property("os_frame", &PyWindow::GetOSFrame,
                          &PyWindow::SetOSFrame,
                          "Window rect in OS coords, not device pixels")
            .def_property("title", &PyWindow::GetTitle, &PyWindow::SetTitle,
                          "Returns the title of the window")
            .def("size_to_fit", &PyWindow::SizeToFit,
                 "Sets the width and height of window to its preferred size")
            .def_property("get_size", &PyWindow::GetSize, &PyWindow::SetSize,
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
            .def("close", &PyWindow::Close, "Closes the window and destroys it")
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
            .def(
                    "set_on_layout",
                    [](PyWindow *w, std::function<void(const Theme &)> f) {
                        w->on_layout_ = f;
                    },
                    "Sets a callback function that manually sets the frames of "
                    "children of the window")
            .def_property_readonly("theme", &PyWindow::GetTheme,
                                   "Get's window's theme info")
            .def("show_dialog", &PyWindow::ShowDialog, "Displays the dialog")
            .def("close_dialog", &PyWindow::CloseDialog,
                 "Closes the current dialog")
            .def("show_message_box", &PyWindow::ShowMessageBox,
                 "Displays a simple dialog with a title and message and okay "
                 "button")
            .def_property_readonly(
                    "renderer", &PyWindow::GetRenderer,
                    "Gets the rendering.Renderer object for the Window");

    // ---- Menu ----
    py::class_<Menu, std::shared_ptr<Menu>> menu(
            m, "Menu", "A menu, possibly a menu tree");
    menu.def(py::init<>())
            .def(
                    "add_item",
                    [](std::shared_ptr<Menu> menu, const char *text,
                       int item_id) { menu->AddItem(text, item_id); },
                    "Adds a menu item with id to the menu")
            .def(
                    "add_menu",
                    [](std::shared_ptr<Menu> menu, const char *text,
                       std::shared_ptr<Menu> submenu) {
                        menu->AddMenu(text, submenu);
                    },
                    "Adds a submenu to the menu")
            .def("add_separator", &Menu::AddSeparator,
                 "Adds a separator to the menu")
            .def(
                    "set_enabled",
                    [](std::shared_ptr<Menu> menu, int item_id, bool enabled) {
                        menu->SetEnabled(item_id, enabled);
                    },
                    "Sets menu item enabled or disabled")
            .def(
                    "is_checked",
                    [](std::shared_ptr<Menu> menu, int item_id) -> bool {
                        return menu->IsChecked(item_id);
                    },
                    "Returns True if menu item is checked")
            .def(
                    "set_checked",
                    [](std::shared_ptr<Menu> menu, int item_id, bool checked) {
                        menu->SetChecked(item_id, checked);
                    },
                    "Sets menu item (un)checked");

    // ---- Color ----
    py::class_<Color, std::shared_ptr<Color>> color(
            m, "Color", "Stores color for gui classes");
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
    py::class_<Theme, std::shared_ptr<Theme>> theme(m, "Theme",
                                                    "Theme parameters such as "
                                                    "colors used for drawing "
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
    py::class_<Widget, std::shared_ptr<Widget>> widget(m, "Widget",
                                                       "Base widget class");
    widget.def(py::init<>())
            .def("__repr__",
                 [](const Widget &w) {
                     std::stringstream s;
                     s << "Widget (" << w.GetFrame().x << ", " << w.GetFrame().y
                       << "), " << w.GetFrame().width << " x "
                       << w.GetFrame().height;
                     return s.str();
                 })
            .def("add_child", &Widget::AddChild, "Adds a child widget")
            .def("get_children", &Widget::GetChildren,
                 "Returns the array of children. Do not modify.")
            .def_property("frame", &Widget::GetFrame, &Widget::SetFrame,
                          "The widget's frame. Setting this value will be "
                          "overridden if the frame is within a layout.")
            .def_property("visible", &Widget::IsVisible, &Widget::SetVisible,
                          "True if widget is visible, False otherwise")
            .def_property("enabled", &Widget::IsEnabled, &Widget::SetEnabled,
                          "True if widget is enabled, False if disabled")
            .def("calc_preferred_size", &Widget::CalcPreferredSize,
                 "Returns the preferred size of the widget. This is intended "
                 "to be called only during layout, although it will also work "
                 "during drawing. Calling it at other times will not work, as "
                 "it requires some internal setup in order to function "
                 "properly");

    // ---- Button ----
    py::class_<Button, std::shared_ptr<Button>, Widget> button(m, "Button",
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
                    [](std::shared_ptr<Button> b, const py::object &em) {
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
                    [](std::shared_ptr<Button> b, const py::object &em) {
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
    py::class_<Checkbox, std::shared_ptr<Checkbox>, Widget> checkbox(
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
    py::class_<ColorEdit, std::shared_ptr<ColorEdit>, Widget> coloredit(
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
    py::class_<Combobox, std::shared_ptr<Combobox>, Widget> combobox(
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

    // ---- ImageLabel ----
    py::class_<ImageLabel, std::shared_ptr<ImageLabel>, Widget> imagelabel(
            m, "ImageLabel", "Displays a bitmap");
    imagelabel
            .def(py::init<>(
                         [](const char *path) { return new ImageLabel(path); }),
                 "Creates an ImageLabel from the image at the specified path")
            .def("__repr__", [](const ImageLabel &il) {
                std::stringstream s;
                s << "ImageLabel (" << il.GetFrame().x << ", "
                  << il.GetFrame().y << "), " << il.GetFrame().width << " x "
                  << il.GetFrame().height;
                return s.str();
            });
    // TODO: add the other functions and UIImage?

    // ---- Label ----
    py::class_<Label, std::shared_ptr<Label>, Widget> label(m, "Label",
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
                          "The color of the text (gui.Color)");

    // ---- ListView ----
    py::class_<ListView, std::shared_ptr<ListView>, Widget> listview(
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
    py::class_<NumberEdit, std::shared_ptr<NumberEdit>, Widget> numedit(
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
                    [](std::shared_ptr<NumberEdit> ne, int val) {
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
                 "changes widget's value");

    // ---- ProgressBar----
    py::class_<ProgressBar, std::shared_ptr<ProgressBar>, Widget> progress(
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
    py::class_<SceneWidget, std::shared_ptr<SceneWidget>, Widget> scene(
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
            .value("FLY", SceneWidget::Controls::FLY)
            .value("ROTATE_SUN", SceneWidget::Controls::ROTATE_SUN)
            .value("ROTATE_IBL", SceneWidget::Controls::ROTATE_IBL)
            .value("ROTATE_MODEL", SceneWidget::Controls::ROTATE_MODEL)
            .export_values();

    scene.def(py::init<>(),
              "Creates an empty SceneWidget. Assign a Scene with the 'scene' "
              "property")
            .def_property(
                    "scene", &SceneWidget::GetScene, &SceneWidget::SetScene,
                    "The rendering.Open3DScene that the SceneWidget renders")
            .def("enable_scene_caching", &SceneWidget::EnableSceneCaching,
                 "Enable/Disable caching of scene content when the view or "
                 "model is not changing. Scene caching can help improve UI "
                 "responsiveness for large models and point clouds")
            .def("force_redraw", &SceneWidget::ForceRedraw,
                 "Ensures scene redraws even when scene caching is enabled.")
            .def("set_view_controls", &SceneWidget::SetViewControls,
                 "Sets mouse interaction, e.g. ROTATE_OBJ")
            .def("setup_camera", &SceneWidget::SetupCamera,
                 "Configure the camera: setup_camera(field_of_view, "
                 "model_bounds, "
                 "center_of_rotation)")
            .def("set_on_sun_direction_changed",
                 &SceneWidget::SetOnSunDirectionChanged,
                 "Callback when user changes sun direction (only called in "
                 "ROTATE_SUN control mode). Called with one argument, the "
                 "[i, j, k] vector of the new sun direction");

    // ---- Slider ----
    py::class_<Slider, std::shared_ptr<Slider>, Widget> slider(
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
                    [](std::shared_ptr<Slider> ne, int val) {
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
    py::class_<StackedWidget, std::shared_ptr<StackedWidget>, Widget> stacked(
            m, "StackedWidget", "Like a TabControl but without the tabs");
    stacked.def(py::init<>())
            .def_property("selected_index", &StackedWidget::GetSelectedIndex,
                          &StackedWidget::SetSelectedIndex,
                          "Selects the index of the child to display");

    // ---- TabControl ----
    py::class_<TabControl, std::shared_ptr<TabControl>, Widget> tabctrl(
            m, "TabControl", "Tab control");
    tabctrl.def(py::init<>())
            .def("add_tab", &TabControl::AddTab,
                 "Adds a tab. The first parameter is the title of the tab, and "
                 "the second parameter is a widget--normally this is a "
                 "layout.")
            .def("set_on_selected_tab_changed",
                 &TabControl::SetOnSelectedTabChanged,
                 "Calls the provided callback function with the index of the "
                 "currently selected tab whenever the user clicks on a "
                 "different tab");

    // ---- TextEdit ----
    py::class_<TextEdit, std::shared_ptr<TextEdit>, Widget> textedit(
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

    // ---- TreeView ----
    py::class_<TreeView, std::shared_ptr<TreeView>, Widget> treeview(
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
                 "are "
                 "the top-level items")
            .def("add_item", &TreeView::AddItem,
                 "Adds a child item to the parent. add_item(parent, widget)")
            .def("add_text_item", &TreeView::AddTextItem,
                 "Adds a child item to the parent. add_item(parent, text)")
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
    py::class_<CheckableTextTreeCell, std::shared_ptr<CheckableTextTreeCell>,
               Widget>
            checkable_cell(m, "CheckableTextTreeCell",
                           "TreeView cell with a checkbox and text");
    checkable_cell
            .def(py::init<>([](const char *text, bool checked,
                               std::function<void(bool)> on_toggled) {
                     return std::make_shared<CheckableTextTreeCell>(
                             text, checked, on_toggled);
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

    py::class_<LUTTreeCell, std::shared_ptr<LUTTreeCell>, Widget> lut_cell(
            m, "LUTTreeCell",
            "TreeView cell with checkbox, text, and color edit");
    lut_cell.def(py::init<>([](const char *text, bool checked,
                               const Color &color,
                               std::function<void(bool)> on_enabled,
                               std::function<void(const Color &)> on_color) {
                     return std::make_shared<LUTTreeCell>(text, checked, color,
                                                          on_enabled, on_color);
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

    py::class_<ColormapTreeCell, std::shared_ptr<ColormapTreeCell>, Widget>
            colormap_cell(m, "ColormapTreeCell",
                          "TreeView cell with a number edit and color edit");
    colormap_cell
            .def(py::init<>([](float value, const Color &color,
                               std::function<void(double)> on_value_changed,
                               std::function<void(const Color &)>
                                       on_color_changed) {
                     return std::make_shared<ColormapTreeCell>(
                             value, color, on_value_changed, on_color_changed);
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
    py::class_<VectorEdit, std::shared_ptr<VectorEdit>, Widget> vectoredit(
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
    py::class_<Margins, std::shared_ptr<Margins>> margins(
            m, "Margins", "Margins for layouts");
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
    py::class_<Layout1D, std::shared_ptr<Layout1D>, Widget> layout1d(
            m, "Layout1D", "Layout base class");
    layout1d
            // TODO: write the proper constructor
            //        .def(py::init([]() { return new Layout1D(Layout1D::VERT,
            //        0, Margins(), {}); }))
            .def("add_fixed", &Layout1D::AddFixed,
                 "Adds a fixed amount of empty space to the layout")
            .def(
                    "add_fixed",
                    [](std::shared_ptr<Layout1D> layout, float px) {
                        layout->AddFixed(int(std::round(px)));
                    },
                    "Adds a fixed amount of empty space to the layout")
            .def("add_stretch", &Layout1D::AddStretch,
                 "Adds empty space to the layout that will take up as much "
                 "extra space as there is available in the layout");

    // ---- Vert ----
    py::class_<Vert, std::shared_ptr<Vert>, Layout1D> vlayout(
            m, "Vert", "Vertical layout");
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
                 "is the margins. Both default to 0.");

    // ---- CollapsableVert ----
    py::class_<CollapsableVert, std::shared_ptr<CollapsableVert>, Vert>
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
            .def("set_is_open", &CollapsableVert::SetIsOpen,
                 "Sets to collapsed (False) or open (True). Requires a call to "
                 "Window.SetNeedsLayout() afterwards, unless calling before "
                 "window is visible");

    // ---- Horiz ----
    py::class_<Horiz, std::shared_ptr<Horiz>, Layout1D> hlayout(
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
                 "is the margins. Both default to 0.");

    // ---- VGrid ----
    py::class_<VGrid, std::shared_ptr<VGrid>, Widget> vgrid(m, "VGrid",
                                                            "Gride layout");
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
                                   "Returns the margins");

    // ---- Dialog ----
    py::class_<Dialog, std::shared_ptr<Dialog>, Widget> dialog(m, "Dialog",
                                                               "Dialog");
    dialog.def(py::init<const char *>(),
               "Creates a dialog with the given title");

    // ---- FileDialog ----
    py::class_<FileDialog, std::shared_ptr<FileDialog>, Dialog> filedlg(
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
    pybind_gui_classes(m_gui);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
