#!/usr/bin/env python3
import open3d as o3d
gui = o3d.visualization.gui

gui.Application.instance.initialize()
w = gui.Window("Open3D", 1024, 768)

scene = gui.SceneWidget(w)

# Settings
em = w.theme.font_size
separation_height = int(round(0.5 * em))
settings_panel = gui.Vert(
    0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

view_ctrls = gui.CollapsableVert("View controls", 0, gui.Margins(em, 0, 0, 0))

arcball_button = gui.Button("Arcball")
fly_button = gui.Button("Fly")
model_button = gui.Button("Model")
sun_button = gui.Button("Sun")
ibl_button = gui.Button("Environment")
view_ctrls.add_child(gui.Label("Mouse controls"))
h = gui.Horiz(0.25 * em)
h.add_stretch()
h.add_child(arcball_button)
h.add_child(fly_button)
h.add_child(model_button)
h.add_stretch()
view_ctrls.add_child(h)
h = gui.Horiz(0.25 * em)
h.add_stretch()
h.add_child(sun_button)
h.add_child(ibl_button)
h.add_stretch()
view_ctrls.add_child(h)

show_skymap = gui.Checkbox("Show skymap")
view_ctrls.add_fixed(separation_height)
view_ctrls.add_child(show_skymap)

bg_color = gui.ColorEdit()
bg_color.color_value = gui.Color(1, 1, 1, 1)


def on_bg_color(new_color):
    scene.set_background_color(new_color)


on_bg_color(bg_color.color_value)
bg_color.set_on_value_changed(on_bg_color)
grid = gui.VGrid(2, 0.25 * em)
grid.add_child(gui.Label("BG Color"))
grid.add_child(bg_color)
view_ctrls.add_child(grid)

show_axes = gui.Checkbox("Show axes")
view_ctrls.add_fixed(separation_height)
view_ctrls.add_child(show_axes)

profiles = gui.Combobox()
profiles.add_item("Bright day with sun at +Y [default]")
profiles.add_item("Bright day with sun at -Y")
profiles.add_item("Bright day with sun at +Z")
profiles.add_item("Less bright day with sun at +Y")
profiles.add_item("Less bright day with sun at -Y")
profiles.add_item("Less bright day with sun at +Z")
profiles.add_item("Cloudy day (no direct sun)")
profiles.add_item("Custom")
view_ctrls.add_fixed(separation_height)
view_ctrls.add_child(gui.Label("Lighting profiles"))
view_ctrls.add_child(profiles)
settings_panel.add_fixed(separation_height)
settings_panel.add_child(view_ctrls)

advanced = gui.CollapsableVert("Advanced lighting", 0, gui.Margins(em, 0, 0, 0))

show_skybox = gui.Checkbox("HDR map")
show_skybox.checked = True
show_sunlight = gui.Checkbox("Sun")
show_sunlight.checked = True
advanced.add_child(gui.Label("Light sources"))
h = gui.Horiz(em)
h.add_child(show_skybox)
h.add_child(show_sunlight)
advanced.add_child(h)

ibl_map = gui.Combobox()
ibl_map.add_item("default")
ibl_intensity = gui.Slider(gui.Slider.INT)
ibl_intensity.set_limits(0, 200000)
ibl_intensity.int_value = 45000
grid = gui.VGrid(2, 0.25 * em)
grid.add_child(gui.Label("HDR map"))
grid.add_child(ibl_map)
grid.add_child(gui.Label("Intensity"))
grid.add_child(ibl_intensity)
advanced.add_fixed(separation_height)
advanced.add_child(gui.Label("Environment"))
advanced.add_child(grid)

sun_intensity = gui.Slider(gui.Slider.INT)
sun_intensity.set_limits(0, 200000)
sun_intensity.int_value = 45000
sun_dir = gui.VectorEdit()
sun_dir.vector_value = [0.577, -0.577, -0.577]
sun_color = gui.ColorEdit()
sun_color.color_value = gui.Color(1, 1, 1, 1)
grid = gui.VGrid(2, 0.25 * em)
grid.add_child(gui.Label("Intensity"))
grid.add_child(sun_intensity)
grid.add_child(gui.Label("Direction"))
grid.add_child(sun_dir)
grid.add_child(gui.Label("Color"))
grid.add_child(sun_color)
advanced.add_fixed(separation_height)
advanced.add_child(gui.Label("Sun (Directional light)"))
advanced.add_child(grid)

settings_panel.add_fixed(separation_height)
settings_panel.add_child(advanced)

material_settings = gui.CollapsableVert("Material settings", 0,
                                        gui.Margins(em, 0, 0, 0))

shader = gui.Combobox()
shader.add_item("Lit")
shader.add_item("Unlit")
shader.add_item("Normals")
shader.add_item("Depth")
material_profile = gui.Combobox()
material_profile.add_item("Clay")
material_profile.add_item("Glazed ceramic")
material_profile.add_item("Metal (rougher)")
material_profile.add_item("Metal (smoother)")
material_profile.add_item("Plastic")
material_profile.add_item("Polished ceramic [default]")
material_color = gui.ColorEdit()
material_color.color_value = gui.Color(1, 1, 1, 1)
point_size = gui.Slider(gui.Slider.INT)
point_size.set_limits(1, 10)
point_size.int_value = 3

grid = gui.VGrid(2, 0.25 * em)
grid.add_child(gui.Label("Type"))
grid.add_child(shader)
grid.add_child(gui.Label("Material"))
grid.add_child(material_profile)
grid.add_child(gui.Label("Color"))
grid.add_child(material_color)
grid.add_child(gui.Label("Point size"))
grid.add_child(point_size)
material_settings.add_child(grid)

settings_panel.add_fixed(separation_height)
settings_panel.add_child(material_settings)

w.add_child(scene)
w.add_child(settings_panel)


def on_layout(theme):
    r = w.content_rect
    scene.frame = r
    width = 17 * theme.font_size
    settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)


w.set_on_layout(on_layout)

# Menu
MENU_OPEN = 1
MENU_EXPORT = 2
MENU_QUIT = 3
MENU_SHOW_SETTINGS = 11
MENU_ABOUT = 21
if gui.Application.instance.menubar is None:
    file_menu = gui.Menu()
    file_menu.add_item("Open...", MENU_OPEN)
    file_menu.add_item("Export Current Image...", MENU_EXPORT)
    file_menu.set_enabled(MENU_EXPORT, False)
    file_menu.add_separator()
    file_menu.add_item("Quit", MENU_QUIT)
    settings_menu = gui.Menu()
    settings_menu.add_item("Lighting & Materials", MENU_SHOW_SETTINGS)
    settings_menu.set_checked(MENU_SHOW_SETTINGS, True)
    help_menu = gui.Menu()
    help_menu.add_item("About", MENU_ABOUT)

    menu = gui.Menu()
    menu.add_menu("File", file_menu)
    menu.add_menu("Settings", settings_menu)
    menu.add_menu("Help", help_menu)
    gui.Application.instance.menubar = menu


def on_open():
    dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", w.theme)

    def on_cancel():
        w.close_dialog()

    def on_done(filename):
        w.close_dialog()
        w.show_message_box("Error", "Loading has not been implemented yet")

    dlg.set_on_cancel(on_cancel)
    dlg.set_on_done(on_done)
    w.show_dialog(dlg)


def on_quit():
    gui.Application.instance.quit()


def on_toggle_settings_panel():
    settings_panel.visible = not settings_panel.visible
    gui.Application.instance.menubar.set_checked(MENU_SHOW_SETTINGS,
                                                 settings_panel.visible)


def on_about():
    em = w.theme.font_size
    dlg = gui.Dialog("About")
    dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
    dlg_layout.add_child(gui.Label("Open3D GUI Example"))
    ok = gui.Button("OK")

    def on_ok():
        w.close_dialog()

    ok.set_on_clicked(on_ok)
    h = gui.Horiz()
    h.add_stretch()
    h.add_child(ok)
    h.add_stretch()
    dlg_layout.add_child(h)
    dlg.add_child(dlg_layout)
    w.show_dialog(dlg)


w.set_on_menu_item_activated(MENU_OPEN, on_open)
w.set_on_menu_item_activated(MENU_QUIT, on_quit)
w.set_on_menu_item_activated(MENU_SHOW_SETTINGS, on_toggle_settings_panel)
w.set_on_menu_item_activated(MENU_ABOUT, on_about)

# Start
gui.Application.instance.add_window(w)
gui.Application.instance.run()
