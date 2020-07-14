#!/usr/bin/env python3
import open3d as o3d
gui = o3d.visualization.gui
import platform

isMacOS = (platform.system() == "Darwin")

# We need to initalize the application, which finds the necessary shaders for
# rendering and prepares the cross-platform window abstraction.
gui.Application.instance.initialize()

# Now we can create the window
w = gui.Window("Open3D", 1024, 768)

# Create the widget for rendering now, so that we can refer to it in callback
# functions.
scene = gui.SceneWidget(w)

# ---- Settings panel ----
# Rather than specifying sizes in pixels, which may vary in size based on the
# monitor, especially on macOS which has 220 dpi monitors, use the em-size.
# This way sizings will be proportional to the font size,
em = w.theme.font_size
separation_height = int(round(0.5 * em))

# Widgets are laid out in layouts: gui.Horiz, gui.Vert, gui.CollapsableVert,
# and gui.VGrid. By nesting the layouts we can achieve complex designs.
# Usually we use a vertical layout as the topmost widget, since widgets tend
# to be organized from top to bottom. Within that, we usually have a series
# of horizontal layouts for each row. All layouts take a spacing parameter,
# which is the spacinging between items in the widget, and a margins parameter,
# which specifies the spacing of the left, top, right, bottom margins. (This
# acts like the 'padding' property in CSS.)
settings_panel = gui.Vert(
    0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

# Create a collapsable vertical widget, which takes up enough vertical space for
# all its children when open, but only enough for text when closed. This is
# useful for property pages, so the user can hide sets of properties they rarely
# use.
view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                 gui.Margins(em, 0, 0, 0))

arcball_button = gui.Button("Arcball")
fly_button = gui.Button("Fly")
model_button = gui.Button("Model")
sun_button = gui.Button("Sun")
ibl_button = gui.Button("Environment")
view_ctrls.add_child(gui.Label("Mouse controls"))
# We want two rows of buttons, so make two horizontal layouts. We also want the
# buttons centered, which we can do be putting a stretch item as the first
# and last item. Stretch items take up as much space as possible, and since there
# are two, they will each take half the extra space, thus centering the buttons.
h = gui.Horiz(0.25 * em)  # row 1
h.add_stretch()
h.add_child(arcball_button)
h.add_child(fly_button)
h.add_child(model_button)
h.add_stretch()
view_ctrls.add_child(h)
h = gui.Horiz(0.25 * em)  # row 2
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
# ----

# Normally our user interface can be children of all one layout (usually a
# vertical layout), which is then the only child of the window. In our case
# we want the scene to take up all the space and the settings panel to go
# above it. We can do this custom layout by providing an on_layout callback.
# The on_layout callback should set the frame (position + size) of every
# child correctly. After the callback is done the window will layout the
# grandchildren.
w.add_child(scene)
w.add_child(settings_panel)


def on_layout(theme):
    r = w.content_rect
    scene.frame = r
    width = 17 * theme.font_size
    settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)


w.set_on_layout(on_layout)

# ---- Menu ----
MENU_OPEN = 1
MENU_EXPORT = 2
MENU_QUIT = 3
MENU_SHOW_SETTINGS = 11
MENU_ABOUT = 21
# The menu is global (because the macOS menu is global), so only create it once
if gui.Application.instance.menubar is None:
    if isMacOS:
        app_menu = gui.Menu()
        app_menu.add_item("About", MENU_ABOUT)
        app_menu.add_separator()
        app_menu.add_item("Quit", MENU_QUIT)
    file_menu = gui.Menu()
    file_menu.add_item("Open...", MENU_OPEN)
    file_menu.add_item("Export Current Image...", MENU_EXPORT)
    file_menu.set_enabled(MENU_EXPORT, False)
    if not isMacOS:
        file_menu.add_separator()
        file_menu.add_item("Quit", MENU_QUIT)
    settings_menu = gui.Menu()
    settings_menu.add_item("Lighting & Materials", MENU_SHOW_SETTINGS)
    settings_menu.set_checked(MENU_SHOW_SETTINGS, True)
    help_menu = gui.Menu()
    help_menu.add_item("About", MENU_ABOUT)

    menu = gui.Menu()
    if isMacOS:
        # On macOS the first menu item will be named for the running application
        # (in our case, probably "Python"), regardless of what we call it.
        # This is the application menu, and it is where the About...,
        # Preferences..., and Quit menu items typically go
        menu.add_menu("Example", app_menu)
        menu.add_menu("File", file_menu)
        menu.add_menu("Settings", settings_menu)
        # Don't include help menu unless it has something more than About...
    else:
        menu.add_menu("File", file_menu)
        menu.add_menu("Settings", settings_menu)
        menu.add_menu("Help", help_menu)
    gui.Application.instance.menubar = menu


# Define the menu callbacks
def on_open():
    dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", w.theme)

    # A file dialog MUST define on_cancel and on_done functions
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
    # Show a simple dialog. Although the Dialog is actually a widget, you can
    # treat it similar to a Window for layout and put all the widgets in a
    # layout which you make the only child of the Dialog.
    em = w.theme.font_size
    dlg = gui.Dialog("About")

    # Add the text
    dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
    dlg_layout.add_child(gui.Label("Open3D GUI Example"))

    # Add the Ok button. We need to define a callback function to handle
    # the click.
    ok = gui.Button("OK")

    def on_ok():
        w.close_dialog()

    ok.set_on_clicked(on_ok)

    # We want the Ok button to be an the right side, so we need to add
    # a stretch item to the layout, otherwise the button will be the size
    # of the entire row. A stretch item takes up as much space as it can,
    # which forces the button to be its minimum size.
    h = gui.Horiz()
    h.add_stretch()
    h.add_child(ok)
    h.add_stretch()
    dlg_layout.add_child(h)

    dlg.add_child(dlg_layout)
    w.show_dialog(dlg)


# The menubar is global, but we need to connect the menu items to the window,
# so that the window can call the appropriate function when the menu item is
# activated.
w.set_on_menu_item_activated(MENU_OPEN, on_open)
w.set_on_menu_item_activated(MENU_QUIT, on_quit)
w.set_on_menu_item_activated(MENU_SHOW_SETTINGS, on_toggle_settings_panel)
w.set_on_menu_item_activated(MENU_ABOUT, on_about)
#----

# Add the window to the applicaiton, which will make it visible
gui.Application.instance.add_window(w)

# Run the event loop. This will not return until the last window is closed.
gui.Application.instance.run()
