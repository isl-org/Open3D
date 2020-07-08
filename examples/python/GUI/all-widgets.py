#!/usr/bin/env python
import open3d as o3d
gui = o3d.visualization.gui

basedir = "/some/path"
gui.Application.instance.initialize()
w = gui.Window("Test")
#w = gui.Window("Test", 640, 480)
#w = gui.Window("Test", 640, 480, x=50, y=100)
em = w.theme.font_size
layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

MENU_CHECKABLE = 1
MENU_DISABLED = 2
MENU_QUIT = 3
if gui.Application.instance.menubar is None:
    print("[py] adding menubar")
    menubar = gui.Menu()
    test_menu = gui.Menu()
    test_menu.add_item("An option", MENU_CHECKABLE)
    test_menu.set_checked(MENU_CHECKABLE, True)
    test_menu.add_item("Unavailable feature", MENU_DISABLED)
    test_menu.set_enabled(MENU_DISABLED, False)
    test_menu.add_separator()
    test_menu.add_item("Quit", MENU_QUIT)
    menubar.add_menu("Test", test_menu)
    gui.Application.instance.menubar = menubar

    def on_checkable():
        test_menu.set_checked(MENU_CHECKABLE,
                              not test_menu.is_checked(MENU_CHECKABLE))

    def on_quit():
        gui.Application.instance.quit()

    w.set_on_menu_item_activated(MENU_CHECKABLE, on_checkable)
    w.set_on_menu_item_activated(MENU_QUIT, on_quit)

fileedit = gui.TextEdit()
filedlgbutton = gui.Button("...")


def on_filedlg():
    filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", w.theme)
    filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
    filedlg.add_filter("", "All files")

    def filedlg_cancel():
        w.close_dialog()

    def filedlg_done(path):
        fileedit.text_value = path
        w.close_dialog()

    filedlg.set_on_cancel(filedlg_cancel)
    filedlg.set_on_done(filedlg_done)
    w.show_dialog(filedlg)


filedlgbutton.set_on_clicked(on_filedlg)
fileedit_layout = gui.Horiz()
fileedit_layout.add_child(gui.Label("Model file"))
fileedit_layout.add_child(fileedit)
fileedit_layout.add_child(filedlgbutton)
layout.add_child(fileedit_layout)

collapse = gui.CollapsableVert("Widgets", int(0.33 * em),
                               gui.Margins(em, 0, 0, 0))
label = gui.Label("Lorem ipsum dolor")
label.text_color = gui.Color(1.0, 0.5, 0.0)
collapse.add_child(label)

cb = gui.Checkbox("Enable some really cool effects")


def on_cb(is_checked):
    if is_checked:
        text = "Sorry, effects are unimplemented"
    else:
        text = "Good choice"
    dlg = gui.Dialog("There might be a problem...")
    dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
    dlg_layout.add_child(gui.Label(text))
    button_layout = gui.Horiz()
    ok_button = gui.Button("Ok")

    def on_ok():
        w.close_dialog()

    ok_button.set_on_clicked(on_ok)
    button_layout.add_stretch()
    button_layout.add_child(ok_button)
    dlg_layout.add_child(button_layout)
    dlg.add_child(dlg_layout)
    w.show_dialog(dlg)


cb.set_on_checked(on_cb)
collapse.add_child(cb)

color = gui.ColorEdit()
color.color_value = gui.Color(1.0, 0.5, 0.0)


def on_color(new_color):
    label.text_color = new_color


color.set_on_value_changed(on_color)
collapse.add_child(color)

combo = gui.Combobox()
combo.add_item("Show point labels")
combo.add_item("Show point velocity")
combo.add_item("Show bounding boxes")


def on_combo(new_val, new_idx):
    print(new_idx, new_val)


combo.set_on_selection_changed(on_combo)
collapse.add_child(combo)

logo = gui.ImageLabel(basedir + "icon-32@2x.png")
collapse.add_child(logo)

lv = gui.ListView()
lv.set_items(["Ground", "Trees", "Buildings" "Cars", "People"])
lv.selected_index = lv.selected_index + 2  # initial value is -1, so will be 1


def on_list(new_val, is_dbl_click):
    print(new_val)


lv.set_on_selection_changed(on_list)
collapse.add_child(lv)

intedit = gui.NumberEdit(gui.NumberEdit.INT)
intedit.int_value = 0
intedit.set_limits(1, 19)  # value coerced to 1
intedit.int_value = intedit.int_value + 2  # value should be 3
doubleedit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
numlayout = gui.Horiz()
numlayout.add_child(gui.Label("int"))
numlayout.add_child(intedit)
numlayout.add_fixed(em)
numlayout.add_child(gui.Label("double"))
numlayout.add_child(doubleedit)
collapse.add_child(numlayout)

progress = gui.ProgressBar()
progress.value = 0.25
progress.value = progress.value + 0.08
prog_layout = gui.Horiz(em)
prog_layout.add_child(gui.Label("Troning..."))
prog_layout.add_child(progress)
collapse.add_child(prog_layout)

slider = gui.Slider(gui.Slider.INT)
slider.set_limits(5, 13)


def on_slider(new_val):
    progress.value = new_val / 20.0


slider.set_on_value_changed(on_slider)
collapse.add_child(slider)

tedit = gui.TextEdit()
tedit.placeholder_text = "Edit me some text here"


def on_text_changed(new_text):
    print("edit:", new_text)


def on_value_changed(new_text):
    print("value:", new_text)


tedit.set_on_text_changed(on_text_changed)
tedit.set_on_value_changed(on_value_changed)
collapse.add_child(tedit)

vedit = gui.VectorEdit()
vedit.vector_value = [1, 2, 3]


def on_vedit(new_val):
    print(new_val)


vedit.set_on_value_changed(on_vedit)
collapse.add_child(vedit)

vgrid = gui.VGrid(2)
vgrid.add_child(gui.Label("Trees"))
vgrid.add_child(gui.Label("12 items"))
vgrid.add_child(gui.Label("People"))
vgrid.add_child(gui.Label("2 (93% certainty)"))
vgrid.add_child(gui.Label("Cars"))
vgrid.add_child(gui.Label("5 (87% certainty)"))
collapse.add_child(vgrid)
collapse.add_child(vgrid)

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

button_layout = gui.Horiz()
ok_button = gui.Button("Ok")


def on_ok():
    gui.Application.instance.quit()


ok_button.set_on_clicked(on_ok)
button_layout.add_stretch()
button_layout.add_child(ok_button)

layout.add_child(collapse)
layout.add_child(button_layout)

w.add_child(layout)
gui.Application.instance.add_window(w)
gui.Application.instance.run()
