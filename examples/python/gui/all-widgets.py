#!/usr/bin/env python
import open3d.visualization.gui as gui
import os.path

basedir = os.path.dirname(os.path.realpath(__file__))


class ExampleWindow:
    MENU_CHECKABLE = 1
    MENU_DISABLED = 2
    MENU_QUIT = 3

    def __init__(self):
        self.window = gui.Application.instance.create_window("Test", 400, 768)
        # self.window = gui.Application.instance.create_window("Test", 400, 768,
        #                                                        x=50, y=100)
        w = self.window  # for more concise code

        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row.
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

        # Create the menu. The menu is global (because the macOS menu is global),
        # so only create it once.
        if gui.Application.instance.menubar is None:
            menubar = gui.Menu()
            test_menu = gui.Menu()
            test_menu.add_item("An option", ExampleWindow.MENU_CHECKABLE)
            test_menu.set_checked(ExampleWindow.MENU_CHECKABLE, True)
            test_menu.add_item("Unavailable feature",
                               ExampleWindow.MENU_DISABLED)
            test_menu.set_enabled(ExampleWindow.MENU_DISABLED, False)
            test_menu.add_separator()
            test_menu.add_item("Quit", ExampleWindow.MENU_QUIT)
            # On macOS the first menu item is the application menu item and will
            # always be the name of the application (probably "Python"),
            # regardless of what you pass in here. The application menu is
            # typically where About..., Preferences..., and Quit go.
            menubar.add_menu("Test", test_menu)
            gui.Application.instance.menubar = menubar

        # Each window needs to know what to do with the menu items, so we need
        # to tell the window how to handle menu items.
        w.set_on_menu_item_activated(ExampleWindow.MENU_CHECKABLE,
                                     self._on_menu_checkable)
        w.set_on_menu_item_activated(ExampleWindow.MENU_QUIT,
                                     self._on_menu_quit)

        # Create a file-chooser widget. One part will be a text edit widget for
        # the filename and clicking on the button will let the user choose using
        # the file dialog.
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        # (Create the horizontal widget for the row. This will make sure the
        # text editor takes up as much space as it can.)
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Model file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        # add to the top-level (vertical) layout
        layout.add_child(fileedit_layout)

        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use. All layouts take a spacing parameter,
        # which is the spacinging between items in the widget, and a margins
        # parameter, which specifies the spacing of the left, top, right,
        # bottom margins. (This acts like the 'padding' property in CSS.)
        collapse = gui.CollapsableVert("Widgets", 0.33 * em,
                                       gui.Margins(em, 0, 0, 0))
        self._label = gui.Label("Lorem ipsum dolor")
        self._label.text_color = gui.Color(1.0, 0.5, 0.0)
        collapse.add_child(self._label)

        # Create a checkbox. Checking or unchecking would usually be used to set
        # a binary property, but in this case it will show a simple message box,
        # which illustrates how to create simple dialogs.
        cb = gui.Checkbox("Enable some really cool effect")
        cb.set_on_checked(self._on_cb)  # set the callback function
        collapse.add_child(cb)

        # Create a color editor. We will change the color of the orange label
        # above when the color changes.
        color = gui.ColorEdit()
        color.color_value = self._label.text_color
        color.set_on_value_changed(self._on_color)
        collapse.add_child(color)

        # This is a combobox, nothing fancy here, just set a simple function to
        # handle the user selecting an item.
        combo = gui.Combobox()
        combo.add_item("Show point labels")
        combo.add_item("Show point velocity")
        combo.add_item("Show bounding boxes")
        combo.set_on_selection_changed(self._on_combo)
        collapse.add_child(combo)

        # Add a simple image
        logo = gui.ImageLabel(basedir + "/icon-32.png")
        collapse.add_child(logo)

        # Add a list of items
        lv = gui.ListView()
        lv.set_items(["Ground", "Trees", "Buildings" "Cars", "People"])
        lv.selected_index = lv.selected_index + 2  # initially is -1, so now 1
        lv.set_on_selection_changed(self._on_list)
        collapse.add_child(lv)

        # Add a tree view
        tree = gui.TreeView()
        tree.add_text_item(tree.get_root_item(), "Camera")
        geo_id = tree.add_text_item(tree.get_root_item(), "Geometries")
        mesh_id = tree.add_text_item(geo_id, "Mesh")
        tree.add_text_item(mesh_id, "Triangles")
        tree.add_text_item(mesh_id, "Albedo texture")
        tree.add_text_item(mesh_id, "Normal map")
        points_id = tree.add_text_item(geo_id, "Points")
        tree.can_select_items_with_children = True
        tree.set_on_selection_changed(self._on_tree)
        # does not call on_selection_changed: user did not change selection
        tree.selected_item = points_id
        collapse.add_child(tree)

        # Add two number editors, one for integers and one for floating point
        # Number editor can clamp numbers to a range, although this is more
        # useful for integers than for floating point.
        intedit = gui.NumberEdit(gui.NumberEdit.INT)
        intedit.int_value = 0
        intedit.set_limits(1, 19)  # value coerced to 1
        intedit.int_value = intedit.int_value + 2  # value should be 3
        doubleedit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        numlayout = gui.Horiz()
        numlayout.add_child(gui.Label("int"))
        numlayout.add_child(intedit)
        numlayout.add_fixed(em)  # manual spacing (could set it in Horiz() ctor)
        numlayout.add_child(gui.Label("double"))
        numlayout.add_child(doubleedit)
        collapse.add_child(numlayout)

        # Create a progress bar. It ranges from 0.0 to 1.0.
        self._progress = gui.ProgressBar()
        self._progress.value = 0.25  # 25% complete
        self._progress.value = self._progress.value + 0.08  # 0.25 + 0.08 = 33%
        prog_layout = gui.Horiz(em)
        prog_layout.add_child(gui.Label("Progress..."))
        prog_layout.add_child(self._progress)
        collapse.add_child(prog_layout)

        # Create a slider. It acts very similar to NumberEdit except that the
        # user moves a slider and cannot type the number.
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(5, 13)
        slider.set_on_value_changed(self._on_slider)
        collapse.add_child(slider)

        # Create a text editor. The placeholder text (if not empty) will be
        # displayed when there is no text, as concise help, or visible tooltip.
        tedit = gui.TextEdit()
        tedit.placeholder_text = "Edit me some text here"

        # on_text_changed fires whenever the user changes the text (but not if
        # the text_value property is assigned to).
        tedit.set_on_text_changed(self._on_text_changed)

        # on_value_changed fires whenever the user signals that they are finished
        # editing the text, either by pressing return or by clicking outside of
        # the text editor, thus losing text focus.
        tedit.set_on_value_changed(self._on_value_changed)
        collapse.add_child(tedit)

        # Create a widget for showing/editing a 3D vector
        vedit = gui.VectorEdit()
        vedit.vector_value = [1, 2, 3]
        vedit.set_on_value_changed(self._on_vedit)
        collapse.add_child(vedit)

        # Create a VGrid layout. This layout specifies the number of columns
        # (two, in this case), and will place the first child in the first
        # column, the second in the second, the third in the first, the fourth
        # in the second, etc.
        # So:
        #      2 cols             3 cols                  4 cols
        #   |  1  |  2  |   |  1  |  2  |  3  |   |  1  |  2  |  3  |  4  |
        #   |  3  |  4  |   |  4  |  5  |  6  |   |  5  |  6  |  7  |  8  |
        #   |  5  |  6  |   |  7  |  8  |  9  |   |  9  | 10  | 11  | 12  |
        #   |    ...    |   |       ...       |   |         ...           |
        vgrid = gui.VGrid(2)
        vgrid.add_child(gui.Label("Trees"))
        vgrid.add_child(gui.Label("12 items"))
        vgrid.add_child(gui.Label("People"))
        vgrid.add_child(gui.Label("2 (93% certainty)"))
        vgrid.add_child(gui.Label("Cars"))
        vgrid.add_child(gui.Label("5 (87% certainty)"))
        collapse.add_child(vgrid)

        # Create a tab control. This is really a set of N layouts on top of each
        # other, but with only one selected.
        tabs = gui.TabControl()
        tab1 = gui.Vert()
        tab1.add_child(gui.Checkbox("Enable option 1"))
        tab1.add_child(gui.Checkbox("Enable option 2"))
        tab1.add_child(gui.Checkbox("Enable option 3"))
        tabs.add_tab("Options", tab1)
        tab2 = gui.Vert()
        tab2.add_child(gui.Label("No plugins detected"))
        tab2.add_stretch()
        tabs.add_tab("Plugins", tab2)
        collapse.add_child(tabs)

        # Quit button. (Typically this is a menu item)
        button_layout = gui.Horiz()
        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self._on_ok)
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        layout.add_child(collapse)
        layout.add_child(button_layout)

        # We're done, set the window's layout
        w.add_child(layout)

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",
                                 self.window.theme)
        filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self.window.close_dialog()

    def _on_cb(self, is_checked):
        if is_checked:
            text = "Sorry, effects are unimplemented"
        else:
            text = "Good choice"

        self.show_message_dialog("There might be a problem...", text)

    # This function is essentially the same as window.show_message_box(),
    # so for something this simple just use that, but it illustrates making a
    # dialog.
    def show_message_dialog(self, title, message):
        # A Dialog is just a widget, so you make its child a layout just like
        # a Window.
        dlg = gui.Dialog(title)

        # Add the message text
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self._on_dialog_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        # Add the button layout,
        dlg_layout.add_child(button_layout)
        # ... then add the layout as the child of the Dialog
        dlg.add_child(dlg_layout)
        # ... and now we can show the dialog
        self.window.show_dialog(dlg)

    def _on_dialog_ok(self):
        self.window.close_dialog()

    def _on_color(self, new_color):
        self._label.text_color = new_color

    def _on_combo(self, new_val, new_idx):
        print(new_idx, new_val)

    def _on_list(self, new_val, is_dbl_click):
        print(new_val)

    def _on_tree(self, new_item_id):
        print(new_item_id)

    def _on_slider(self, new_val):
        self._progress.value = new_val / 20.0

    def _on_text_changed(self, new_text):
        print("edit:", new_text)

    def _on_value_changed(self, new_text):
        print("value:", new_text)

    def _on_vedit(self, new_val):
        print(new_val)

    def _on_ok(self):
        gui.Application.instance.quit()

    def _on_menu_checkable(self):
        gui.Application.instance.menubar.set_checked(
            ExampleWindow.MENU_CHECKABLE,
            not gui.Application.instance.menubar.is_checked(
                ExampleWindow.MENU_CHECKABLE))

    def _on_menu_quit(self):
        gui.Application.instance.quit()


# This class is essentially the same as window.show_message_box(),
# so for something this simple just use that, but it illustrates making a
# dialog.
class MessageBox:

    def __init__(self, title, message):
        self._window = None

        # A Dialog is just a widget, so you make its child a layout just like
        # a Window.
        dlg = gui.Dialog(title)

        # Add the message text
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self._on_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        # Add the button layout,
        dlg_layout.add_child(button_layout)
        # ... then add the layout as the child of the Dialog
        dlg.add_child(dlg_layout)

    def show(self, window):
        self._window = window

    def _on_ok(self):
        self._window.close_dialog()


def main():
    # We need to initalize the application, which finds the necessary shaders for
    # rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = ExampleWindow()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
