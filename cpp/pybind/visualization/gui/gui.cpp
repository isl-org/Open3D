// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "pybind/docstring.h"
#include "pybind11/functional.h"

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
#include "open3d/visualization/gui/Slider.h"
#include "open3d/visualization/gui/TabControl.h"
#include "open3d/visualization/gui/TextEdit.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/gui/Window.h"

using namespace open3d::visualization::gui;

namespace open3d {

void pybind_gui_classes(py::module &m) {
    // ---- Application ----
    py::class_<Application> application(m, "Application", "Deals with global application tasks");
    application 
        .def("__repr__",
             [](const Application &app) {
                 return std::string("Application singleton instance");
             })
        .def_property_readonly_static("instance",
             // Seems like we ought to be able to specify
             // &Application::GetInstance but that gives runtime errors about
             // number of arguments. It seems that property calls are made from
             // an object, so that object needs to be in the function signature.
             [](py::object) -> Application& {
                 return Application::GetInstance();
             },
             py::return_value_policy::reference,
             "Gets the Application singleton (read-only)")
        .def("initialize",
             [](Application& instance) {
                 // We need to find the resources directory. Fortunately, Python
                 // knows where the module lives (open3d.__file__ is the path to
                 // __init__.py), so we can use that to find the resources
                 // included in the wheel.
                 py::object o3d = py::module::import("open3d");
                 auto o3d_init_path = o3d.attr("__file__").cast<std::string>();
                 auto module_path = utility::filesystem::GetFileParentDirectory(o3d_init_path);
                 auto resource_path = module_path + "/resources";
                 instance.Initialize(resource_path.c_str());
             })
        .def("initialize", [](Application& instance, const char *resource_dir) {
                 instance.Initialize(resource_dir);
             })
        .def("run",
             [](Application& instance) {
                 while (instance.RunOneTick()) {
                     // Enable Ctrl-C to kill Python
                     if (PyErr_CheckSignals() != 0) {
                         throw py::error_already_set();
                     }
                 }
             },
             "Runs the event loop")
        .def("quit",
             [](Application& instance) { instance.Quit(); },
             "Closes all the windows, exiting as a result")
        .def_property("menubar",
                      &Application::GetMenubar, &Application::SetMenubar,
                      "The Menu for the application (initially None)")
        .def("add_window",
             &Application::AddWindow,
             "Adds the window to the application")
        .def("remove_window",
             &Application::AddWindow,
             "Removes the window from the application, closing it. If there "
             "are no open windows left the run loop will exit.")
        ;

    // ---- Window ----
    py::class_<Window, std::shared_ptr<Window>> window(m, "Window", "Application window");
    window
        .def(py::init([](const std::string& title, int width, int height,
                         int x, int y, int flags) {
                 if (x < 0 && y < 0 && width < 0 && height < 0) {
                     return new Window(title, flags);
                 } else if (x < 0 && y < 0) {
                     return new Window(title, width, height, flags);
                 } else {
                     return new Window(title, x, y, width, height, flags);
                 }
                      }),
             "title"_a = std::string(), "width"_a = -1, "height"_a = -1,
             "x"_a = -1, "y"_a = -1, "flags"_a = 0)
        .def("__repr__",
             [](const Window &w) { return "Application window"; })
        .def("add_child", &Window::AddChild,
             "Adds a widget to the window")
        .def_property("os_frame", &Window::GetOSFrame, &Window::SetOSFrame,
                      "Window rect in OS coords, not device pixels")
        .def_property("title", &Window::GetTitle, &Window::SetTitle,
                      "Returns the title of the window")
        .def("size_to_fit", &Window::SizeToFit,
             "Sets the width and height of window to its preferred size")
        .def_property("get_size", &Window::GetSize, &Window::SetSize,
                      "The size of the window in device pixels, including "
                      "menubar (except on macOS)")
        .def_property_readonly("content_rect", &Window::GetContentRect,
                               "Returns the frame in device pixels, relative "
                               " to the window, which is available for widgets "
                               "(read-only)")
        .def_property_readonly("scaling", &Window::GetScaling,
                               "Returns the scaling factor between OS pixels "
                               "and device pixels (read-only)")
        .def_property_readonly("is_visible", &Window::IsVisible,
                               "True if window is visible (read-only)")
        .def("show", &Window::Show, "Shows or hides the window")
        .def("close", &Window::Close, "Closes the window and destroys it")
        .def("set_needs_layout", &Window::SetNeedsLayout,
             "Flags window to re-layout")
        .def("post_redraw", &Window::PostRedraw,
             "Sends a redraw message to the OS message queue")
        .def_property_readonly("is_active_window", &Window::IsActiveWindow,
                               "True if the window is currently the active "
                               "window (read-only)")
        .def("set_focus_widget", &Window::SetFocusWidget,
             "Makes specified widget have text focus")
        .def("set_on_menu_item_activated", &Window::SetOnMenuItemActivated,
             "Sets callback function for menu item:  callback()")
        .def_property_readonly("theme", &Window::GetTheme,
                               "Get's window's theme info")
        .def("show_dialog", &Window::ShowDialog, "Displays the dialog")
        .def("close_dialog", &Window::CloseDialog, "Closes the current dialog")
        .def("show_message_box", &Window::ShowMessageBox,
             "Displays a simple dialog with a title and message and okay button")
        ;

    // ---- Menu ----
    py::class_<Menu, std::shared_ptr<Menu>> menu(m, "Menu", "Menu class");
    menu
        .def(py::init<>())
        .def("add_item", [](std::shared_ptr<Menu> menu, const char* text,
                            int item_id) {
                menu->AddItem(text, item_id);
             },
             "Adds a menu item with id to the menu")
        .def("add_menu", [](std::shared_ptr<Menu> menu, const char* text,
                            std::shared_ptr<Menu> submenu) {
                menu->AddMenu(text, submenu);
             },
             "Adds a submenu to the menu")
        .def("add_separator", &Menu::AddSeparator,
             "Adds a separator to the menu")
        .def("set_enabled", [](std::shared_ptr<Menu> menu, int item_id,
                               bool enabled) {
                 menu->SetEnabled(item_id, enabled);
             },
             "Sets menu item enabled or disabled")
        .def("is_checked", [](std::shared_ptr<Menu> menu, int item_id) -> bool {
                return menu->IsChecked(item_id);
             },
             "Returns True if menu item is checked")
        .def("set_checked", [](std::shared_ptr<Menu> menu, int item_id,
                               bool checked) {
                 menu->SetChecked(item_id, checked);
             },
             "Sets menu item (un)checked")
        ;
    
    // ---- Color ----
    py::class_<Color, std::shared_ptr<Color>> color(m, "Color", "Color class");
    color
        .def(py::init([](float r, float g, float b, float a) {
                          return new Color(r, g, b, a);
                      }),
             "r"_a = 1.0, "g"_a = 1.0, "b"_a = 1.0, "a"_a = 1.0)
        .def_property_readonly("red", &Color::GetRed,
                               "Returns red channel in the range [0.0, 1.0] "
                               "(read-only)")
        .def_property_readonly("green", &Color::GetGreen,
                               "Returns green channel in the range [0.0, 1.0]"
                               "(read-only)")
        .def_property_readonly("blue", &Color::GetBlue,
                               "Returns blue channel in the range [0.0, 1.0]"
                               "(read-only)")
        .def_property_readonly("alpha", &Color::GetAlpha,
                               "Returns alpha channel in the range [0.0, 1.0]"
                               "(read-only)")
        .def("set_color", &Color::SetColor,
             "Sets red, green, blue, and alpha channels, (range: [0.0, 1.0])",
             "r"_a, "g"_a, "b"_a, "a"_a = 1.0)
        ;

    // ---- Theme ----
    // Note: no constructor because themes are created by Open3D
    py::class_<Theme, std::shared_ptr<Theme>> theme(m, "Theme", "Theme class");
    theme
        .def_readonly("font_size", &Theme::font_size,
                      "Font size (which is also the conventional size of the "
                      "em unit) [read-only]")
        .def_readonly("default_margin", &Theme::default_margin,
                      "Good default value for margins, useful for layouts "
                      "[read-only")
        .def_readonly("default_layout_spacing", &Theme::default_layout_spacing,
                      "Good value for the spacing parameter in layouts "
                      "[read-only]")
        ;

    // ---- Rect ----
    py::class_<Rect, std::shared_ptr<Rect>> rect(m, "Rect", "Rect class");
    rect
        .def(py::init<>())
        .def(py::init([](int x, int y, int w, int h) {
                          return new Rect(x, y, w, h); }))
        .def_readwrite("x", &Rect::x)
        .def_readwrite("y", &Rect::y)
        .def_readwrite("width", &Rect::width)
        .def_readwrite("height", &Rect::height)
        .def("get_left", &Rect::GetLeft)
        .def("get_right", &Rect::GetRight)
        .def("get_top", &Rect::GetTop)
        .def("get_bottom", &Rect::GetBottom)
        ;

    // ---- Widget ----
    py::class_<Widget, std::shared_ptr<Widget>> widget(m, "Widget", "Base widget class");
    widget
        .def(py::init<>())
        .def("__repr__", [](const Widget &w) {
                std::stringstream s;
                s << "Widget (" << w.GetFrame().x << ", " << w.GetFrame().y
                  << "), " << w.GetFrame().width << " x " << w.GetFrame().height;
                return s.str().c_str();
             })
        .def("add_child", &Widget::AddChild, "Adds a child widget")
        .def("get_children", &Widget::GetChildren,
             "Returns the child array. Do not modify")
        .def_property("frame", &Widget::GetFrame, &Widget::SetFrame,
                      "The widget's frame. Setting this value will be "
                      "overridden if the frame is within a layout.")
        .def_property("visible", &Widget::IsVisible, &Widget::SetVisible,
                      "True if widget is visible, False otherwise")
        .def_property("enabled", &Widget::IsEnabled, &Widget::SetEnabled,
                      "True if widget is enabled, False if disabled")
        ;

    // ---- Button ----
    py::class_<Button, std::shared_ptr<Button>, Widget> button(m, "Button", "Button class");
    button
        .def(py::init<const char *>())
        .def("__repr__", [](const Button &b) {
                std::stringstream s;
                s << "Button (" << b.GetFrame().x << ", " << b.GetFrame().y
                  << "), " << b.GetFrame().width << " x " << b.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("toggleable",
                      &Button::GetIsToggleable, &Button::SetToggleable,
                      "True if button is toggleable, False if a push button")
        .def_property("is_on", &Button::GetIsOn, &Button::SetOn,
                      "True if the button is toggleable and in the on state")
        .def("set_on_clicked", &Button::SetOnClicked,
             "Calls passed function when button is pressed")
        ;

    // ---- Checkbox ----
    py::class_<Checkbox, std::shared_ptr<Checkbox>, Widget> checkbox(m, "Checkbox", "Checkbox class");
    checkbox
        .def(py::init<const char *>())
        .def("__repr__", [](const Checkbox &c) {
                std::stringstream s;
                s << "Checkbox (" << c.GetFrame().x << ", " << c.GetFrame().y
                  << "), " << c.GetFrame().width << " x " << c.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("checked", &Checkbox::IsChecked, &Checkbox::SetChecked,
                      "True if checked, False otherwise")
        .def("set_on_checked", &Checkbox::SetOnChecked,
             "Calls passed function when checkbox changes state")
        ;

    // ---- ColorEdit ----
    py::class_<ColorEdit, std::shared_ptr<ColorEdit>, Widget> coloredit(m, "ColorEdit", "ColorEdit class");
    coloredit
        .def(py::init<>())
        .def("__repr__", [](const ColorEdit &c) {
                auto &color = c.GetValue();
                std::stringstream s;
                s << "ColorEdit [" << color.GetRed() << ", "
                  << color.GetGreen() << ", " << color.GetBlue() << ", "
                  << color.GetAlpha() << "] (" << c.GetFrame().x << ", "
                  << c.GetFrame().y << "), " << c.GetFrame().width << " x "
                  << c.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("color_value",
                      &ColorEdit::GetValue,
                      (void (ColorEdit::*)(const Color&))&ColorEdit::SetValue,
                      "Color value (gui.Color)")
        .def("set_on_value_changed", &ColorEdit::SetOnValueChanged,
             "Calls f(Color) when color changes by user input")
        ;

    // ---- Combobox ----
    py::class_<Combobox, std::shared_ptr<Combobox>, Widget> combobox(m, "Combobox", "Combobox class");
    combobox
        .def(py::init<>())
        .def("clear_items", &Combobox::ClearItems, "Removes all items")
        .def("add_item",  &Combobox::AddItem, "Adds an item to the end")
        .def("change_item",
             (void (Combobox::*)(int, const char*))&Combobox::ChangeItem,
             "Changes the text of the item at index: "
             "change_item(index, newtext)")
        .def("change_item",
             (void (Combobox::*)(const char*, const char*))&Combobox::ChangeItem,
             "Changes the text of the matching item: "
             "change_item(text, newtext)")
        .def("remove_item",
             (void (Combobox::*)(const char *))&Combobox::RemoveItem,
             "Removes the first item of the given text")
        .def("remove_item",
             (void (Combobox::*)(int))&Combobox::RemoveItem,
             "Removes the item at the index")
//        .def_readonly("number_of_items", &Combobox::GetNumberOfItems,
//             "The number of items (read-only)")
        .def("get_item", &Combobox::GetItem,
             "Returns the item at the given index")
        .def_property("selected_index",
                      &Combobox::GetSelectedIndex,
                      &Combobox::SetSelectedIndex,
                      "The index of the currently selected item")
        .def_property("selected_text",
                      &Combobox::GetSelectedValue,
                      &Combobox::SetSelectedValue,
                      "The index of the currently selected item")
        .def("set_on_selection_changed", &Combobox::SetOnValueChanged,
             "Calls f(str, int) when user selects item from combobox. Arguments "
             "are the selected text and selected index, respectively")
        ;

    // ---- ImageLabel ----
    py::class_<ImageLabel, std::shared_ptr<ImageLabel>, Widget> imagelabel(m, "ImageLabel", "ImageLabel class");
    imagelabel
        .def(py::init<>([](const char *path) { return new ImageLabel(path); }))
        .def("__repr__", [](const ImageLabel &il) {
                std::stringstream s;
                s << "ImageLabel (" << il.GetFrame().x
                  << ", " << il.GetFrame().y << "), " << il.GetFrame().width
                  << " x " << il.GetFrame().height;
                return s.str().c_str();
             })
        ;
    // TODO: add the other functions and UIImage?

    // ---- Label ----
    py::class_<Label, std::shared_ptr<Label>, Widget> label(m, "Label", "Label class");
    label
        .def(py::init([](const char *title = "") {
                          return new Label(title); }))
        .def("__repr__", [](const Label &lbl) {
                std::stringstream s;
                s << "Label [" << lbl.GetText() << "] (" << lbl.GetFrame().x
                  << ", " << lbl.GetFrame().y << "), " << lbl.GetFrame().width
                  << " x " << lbl.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("text", &Label::GetText, &Label::SetText,
                      "The text of the label")
        .def_property("text_color", &Label::GetTextColor, &Label::SetTextColor,
                      "The color of the text (gui.Color)")
        ;

    // ---- ListView ----
    py::class_<ListView, std::shared_ptr<ListView>, Widget> listview(m, "ListView", "ListViewclass");
    listview
        .def(py::init<>())
        .def("__repr__", [](const ListView &lv) {
                std::stringstream s;
                s << "Label (" << lv.GetFrame().x << ", " << lv.GetFrame().y
                  << "), " << lv.GetFrame().width << " x "
                  << lv.GetFrame().height;
                return s.str().c_str();
             })
        .def("set_items", &ListView::SetItems,
             "Sets the list to display the list of items provided")
        .def_property("selected_index",
                      &ListView::GetSelectedIndex, &ListView::SetSelectedIndex,
                      "The index of the currently selected item")
        .def_property_readonly("selected_value", &ListView::GetSelectedValue,
                               "The text of the currently selected item")
        .def("set_on_selection_changed", &ListView::SetOnValueChanged,
             "Calls f(new_val, is_double_click) when user changes selection")
        ;

    // ---- NumberEdit ----
    py::class_<NumberEdit, std::shared_ptr<NumberEdit>, Widget> numedit(m, "NumberEdit", "NumberEdit class");
    py::enum_<NumberEdit::Type> numedit_type(numedit, "Type", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    numedit_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for NumberEdit types.";
            }),
            py::none(), py::none(), "");
    numedit_type
        .value("INT", NumberEdit::Type::INT)
        .value("DOUBLE", NumberEdit::Type::DOUBLE)
        .export_values();

    numedit
        .def(py::init<NumberEdit::Type>())
        .def("__repr__", [](const NumberEdit &ne) {
                auto val = ne.GetDoubleValue();
                std::stringstream s;
                s << "NumberEdit [" << val << "] (" << ne.GetFrame().x
                  << ", " << ne.GetFrame().y << "), " << ne.GetFrame().width
                  << " x " << ne.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("int_value",
                      &NumberEdit::GetIntValue,
                      [](std::shared_ptr<NumberEdit> ne, int val) {
                          ne->SetValue(double(val));
                      },
                      "Current value (int)")
        .def_property("double_value",
                      &NumberEdit::GetDoubleValue, &NumberEdit::SetValue,
                      "Current value (double)")
        .def("set_value", &NumberEdit::SetValue, "Sets value")
        .def_property("decimal_precision",
                      &NumberEdit::GetDecimalPrecision,
                      &NumberEdit::SetDecimalPrecision,
                      "Number of fractional digits shown")
        .def_property_readonly("minimum_value", &NumberEdit::GetMinimumValue,
                               "The minimum value number can contain "
                               "(read-only, use set_limits() to set)")
        .def_property_readonly("maximum_value", &NumberEdit::GetMaximumValue,
                               "The maximum value number can contain "
                               "(read-only, use set_limits() to set)")
        .def("set_limits", &NumberEdit::SetLimits,
             "Sets the minimum and maximum values for the number")
        .def("set_on_value_changed", &NumberEdit::SetOnValueChanged,
             "Sets f(new_value) which is called with a Float when user changes "
             "widget's value")
        ;

    // ---- ProgressBar----
    py::class_<ProgressBar, std::shared_ptr<ProgressBar>, Widget> progress(m, "ProgressBar", "ProgressBar class");
    progress
        .def(py::init<>())
        .def("__repr__", [](const ProgressBar &pb) {
                std::stringstream s;
                s << "ProgressBar [" << pb.GetValue() << "] ("
                  << pb.GetFrame().x << ", " << pb.GetFrame().y << "), "
                  << pb.GetFrame().width << " x " << pb.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("value", &ProgressBar::GetValue, &ProgressBar::SetValue,
                      "The value of the progress bar, ranges from 0.0 to 1.0")
        ;

    // ---- Slider ----
    py::class_<Slider, std::shared_ptr<Slider>, Widget> slider(m, "Slider", "Slider class");
    py::enum_<Slider::Type> slider_type(slider, "Type", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    slider_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Slider types.";
            }),
            py::none(), py::none(), "");
    slider_type
        .value("INT", Slider::Type::INT)
        .value("DOUBLE", Slider::Type::DOUBLE)
        .export_values();

    slider
        .def(py::init<Slider::Type>())
        .def("__repr__", [](const Slider &sl) {
                auto val = sl.GetDoubleValue();
                std::stringstream s;
                s << "TextEdit [" << val << "] (" << sl.GetFrame().x
                  << ", " << sl.GetFrame().y << "), " << sl.GetFrame().width
                  << " x " << sl.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("int_value",
                      &Slider::GetIntValue,
                      [](std::shared_ptr<Slider> ne, int val) {
                          ne->SetValue(double(val));
                      },
                      "Slider value (int)")
        .def_property("double_value",
                      &Slider::GetDoubleValue, &Slider::SetValue,
                      "Slider value (double)")
        .def_property_readonly("get_minimum_value", &Slider::GetMinimumValue,
                               "The minimum value number can contain "
                               "(read-only, use set_limits() to set)")
        .def_property_readonly("get_maximum_value", &Slider::GetMaximumValue,
                               "The maximum value number can contain "
                               "(read-only, use set_limits() to set)")
        .def("set_limits", &Slider::SetLimits,
             "Sets the minimum and maximum values for the slider")
        .def("set_on_value_changed", &Slider::SetOnValueChanged,
             "Sets f(new_value) which is called with a Float when user changes "
             "widget's value")
        ;

    // ---- TabControl ----
    py::class_<TabControl, std::shared_ptr<TabControl>, Widget> tabctrl(m, "TabControl", "TabControl class");
    tabctrl
        .def(py::init<>())
        .def("add_tab", &TabControl::AddTab, "Adds a tab")
        ;

    // ---- TextEdit ----
    py::class_<TextEdit, std::shared_ptr<TextEdit>, Widget> textedit(m, "TextEdit", "TextEdit class");
    textedit
        .def(py::init<>())
        .def("__repr__", [](const TextEdit &te) {
                auto val = te.GetText();
                std::stringstream s;
                s << "TextEdit [" << val << "] (" << te.GetFrame().x
                  << ", " << te.GetFrame().y << "), " << te.GetFrame().width
                  << " x " << te.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("text_value", &TextEdit::GetText, &TextEdit::SetText,
                      "The value of text")
        .def_property("placeholder_text",
                      &TextEdit::GetPlaceholderText,
                      &TextEdit::SetPlaceholderText,
                      "The placeholder text displayed when text value is empty")
        .def("set_on_text_changed", &TextEdit::SetOnTextChanged,
             "Sets f(new_text) which is called whenever the the user makes a "
             "change to the text")
        .def("set_on_value_changed", &TextEdit::SetOnValueChanged,
             "Sets f(new_text) which is called with the new text when the user "
             "completes text editing")
        ;

    // ---- VectorEdit ----
    py::class_<VectorEdit, std::shared_ptr<VectorEdit>, Widget> vectoredit(m, "VectorEdit", "VectorEdit class");
    vectoredit
        .def(py::init<>())
        .def("__repr__", [](const VectorEdit &ve) {
                auto val = ve.GetValue();
                std::stringstream s;
                s << "VectorEdit [" << val.x() << ", " << val.y() << ", "
                  << val.z() << "] (" << ve.GetFrame().x
                  << ", " << ve.GetFrame().y << "), " << ve.GetFrame().width
                  << " x " << ve.GetFrame().height;
                return s.str().c_str();
             })
        .def_property("vector_value",
                      &VectorEdit::GetValue, &VectorEdit::SetValue,
                      "Returns value [x, y, z]")
        .def("set_on_value_changed", &VectorEdit::SetOnValueChanged,
             "Sets f([x, y, z]) which is called whenever the user changes the "
             "value of a component")
        ;

    // ---- Margins ----
    py::class_<Margins, std::shared_ptr<Margins>> margins(m, "Margins", "Margins class");
    margins
        .def(py::init([](int left, int top, int right, int bottom) {
                 return new Margins(left, top, right, bottom);
             }),
            "left"_a = 0, "top"_a = 0, "right"_a = 0, "bottom"_a = 0)
        .def(py::init([](float left, float top, float right, float bottom) {
                 return new Margins(int(std::round(left)),
                                    int(std::round(top)),
                                    int(std::round(right)),
                                    int(std::round(bottom)));
             }),
            "left"_a = 0.0f, "top"_a = 0.0f, "right"_a = 0.0f, "bottom"_a = 0.0f)
        .def_readwrite("left", &Margins::left)
        .def_readwrite("top", &Margins::top)
        .def_readwrite("right", &Margins::right)
        .def_readwrite("bottom", &Margins::bottom)
        .def("get_horiz", &Margins::GetHoriz)
        .def("get_vert", &Margins::GetVert)
        ;
    
    // ---- Layout1D ----
    py::class_<Layout1D, std::shared_ptr<Layout1D>, Widget> layout1d(m, "Layout1D", "Layout1D class");
    layout1d
        // TODO: write the proper constructor
//        .def(py::init([]() { return new Layout1D(Layout1D::VERT, 0, Margins(), {}); }))
        .def("add_fixed", &Layout1D::AddFixed,
             "Adds a fixed amount of empty space to the layout")
        .def("add_stretch", &Layout1D::AddStretch,
             "Adds empty space to the layout that will take up as much extra "
             "space as there is available in the layout")
        ;

    // ---- Vert ----
    py::class_<Vert, std::shared_ptr<Vert>, Layout1D> vlayout(m, "Vert", "Vert class");
    vlayout
        .def(py::init([](int spacing, const Margins& margins) {
                          return new Vert(spacing, margins);
                      }),
            "spacing"_a = 0, "margins"_a = Margins())
        ;

    // ---- CollapsableVert ----
    py::class_<CollapsableVert, std::shared_ptr<CollapsableVert>, Vert> collapsable(m, "CollapsableVert", "CollapsableVert class");
    collapsable
        .def(py::init([](const char *text, int spacing, const Margins& margins) {
                          return new CollapsableVert(text, spacing, margins);
                      }),
            "text"_a, "spacing"_a = 0, "margins"_a = Margins())
        .def("set_is_open", &CollapsableVert::SetIsOpen,
             "Sets to collapsed (False) or open (True). Requires a call to "
             "Window.SetNeedsLayout() afterwards, unless calling before window "
             "is visible")
        ;

    // ---- Horiz ----
    py::class_<Horiz, std::shared_ptr<Horiz>, Layout1D> hlayout(m, "Horiz", "Horiz class");
    hlayout
        .def(py::init([](int spacing, const Margins& margins) {
                          return new Horiz(spacing, margins);
                      }),
            "spacing"_a = 0, "margins"_a = Margins())
        ;

    // ---- VGrid ----
    py::class_<VGrid, std::shared_ptr<VGrid>, Widget> vgrid(m, "VGrid", "VGrid class");
    vgrid 
        .def(py::init([](int n_cols, int spacing, const Margins& margins) {
                          return new VGrid(n_cols, spacing, margins);
                      }),
            "cols"_a, "spacing"_a = 0, "margins"_a = Margins())
        .def_property_readonly("spacing", &VGrid::GetSpacing,
                               "Returns the spacing between rows and columns")
        .def_property_readonly("margins", &VGrid::GetMargins,
                               "Returns the margins")
        ;

    // ---- Dialog ----
    py::class_<Dialog, std::shared_ptr<Dialog>, Widget> dialog(m, "Dialog", "Dialog class");
    dialog
        .def(py::init<const char *>())
        ;

    // ---- FileDialog ----
    py::class_<FileDialog, std::shared_ptr<FileDialog>, Dialog> filedlg(m, "FileDialog", "FileDialog class");
    py::enum_<FileDialog::Mode> filedlg_mode(filedlg, "Mode", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    filedlg_mode.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for FileDialog modes.";
            }),
            py::none(), py::none(), "");
    filedlg_mode
        .value("OPEN", FileDialog::Mode::OPEN)
        .value("SAVE", FileDialog::Mode::SAVE)
        .export_values();
    filedlg
        .def(py::init<FileDialog::Mode, const char*, const Theme&>())
        .def("set_path", &FileDialog::SetPath,
             "Sets path to directory the dialog start in")
        .def("add_filter", &FileDialog::AddFilter,
             "Adds a selectable file-type filter: "
             "add_filter('.stl', 'Stereolithography mesh'")
        .def("set_on_cancel", &FileDialog::SetOnCancel,
             "Cancel callback; required")
        .def("set_on_done", &FileDialog::SetOnDone, "Done callback; required")
        ;
}

void pybind_gui(py::module &m) {
    py::module m_gui = m.def_submodule("gui");
    pybind_gui_classes(m_gui);
}

}  // namespace open3d

