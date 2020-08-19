#!/usr/bin/env python
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import pathlib
import random # development testing, remove
import sys
import torch

# -------- temporary until tgeometry::PointCloud gets python bindings ----------
class TPointCloud:
    def __init__(self):
        self._points = None
        self._attributes = {}

    def add_points(self, points):
        self._points = points

    def get_points(self):
        return self._points

    def add_attribute(self, name, a):
        self._attributes[name] = a

    def get_attribute(self, name):
        return self._attributes[name]

    def get_attribute_names(self):
        return self._attributes.keys()
#-------------------------------------------------------------------------------

class MLDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.datasets = []
        self.labels = {}  # should be { value: ("name", [r, g, b]), ... }

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

    _g_gui_is_initialized = False

    def visualize(self, idx_or_list):
        if not MLDataset._g_gui_is_initialized:
            gui.Application.instance.initialize()
            MLDataset._g_gui_is_initialized = True

        vis = MLVisualizer()
        vis.set_labels(self.labels)
        indices = idx_or_list
        if not isinstance(idx_or_list, list):
            indices = [idx_or_list]
        for i in indices:
            self[i].load_data()
            vis.add(self[i].name, self[i].make_point_cloud())
        vis.setup_camera()

        gui.Application.instance.add_window(vis.window)
        gui.Application.instance.run()

class SemanticKITTIDatum:
    def __init__(self, dataset_path, sequence_id, timestep_id):
        self.name = sequence_id + "/" + timestep_id
        self.time = -9.999e10
        self.image = dataset_path + "/sequences/" + sequence_id + "/image_3/" + timestep_id + ".png"
        self._points_path = dataset_path + "/sequences_0.06/" + sequence_id + "/velodyne/" + timestep_id + ".npy"
        self._labels_path = dataset_path + "/sequences_0.06/" + sequence_id + "/labels/" + timestep_id + ".npy"
        self._kdtree_path = dataset_path + "/sequences_0.06/" + sequence_id + "/KDTree/" + timestep_id + ".pkl"
        self.points = None
        self.labels = None
        self.intensity = None # SemanticKITTI doesn't have these ?
        self.kdtree = None

    def load_data(self):
        if self.points is None:
            self.points = np.load(self._points_path)
        if self.labels is None:
            self.labels = np.squeeze(np.load(self._labels_path))
        # It seems like SemanticKITTI doesn't have intensities, so fake for
        # developement.
        self.intensity = [random.random() for _ in self.points]

    def make_point_cloud(self):
        pc = TPointCloud()
        pc.add_points(self.points)
        pc.add_attribute("labels", self.labels)
        pc.add_attribute("intensity", self.intensity)
        return pc

class SemanticKITTI(MLDataset):
    def __init__(self, dataset_path, sequence_id):
        self.labels = {0: ('unlabeled', [0.5, 0.5, 0.5]),
                       1: ('car', [0.5, 0.5, 1.0]),
                       2: ('bicycle', [0.5, 0.0, 0.0]),
                       3: ('motorcycle', [1.0, 0.0, 0.0]),
                       4: ('truck', [0.0, 0.0, 1.0]),
                       5: ('other-vehicle', [0.75, 0.0, 1.0]),
                       6: ('person', [0.0, 1.0, 1.0]),
                       7: ('bicyclist', [0.4, 0.1, 0.0, 0.0]),
                       8: ('motorcyclist', [0.666, 0.0, 0.0]),
                       9: ('road', [1.0, 0.0, 1.0]),
                       10: ('parking', [0.8, 0.0, 1.0]),
                       11: ('sidewalk', [0.8, 0.0, 0.8]),
                       12: ('other-ground', [0.5, 0.0, 0.8]),
                       13: ('building', [1.0, 0.75, 0.0]),
                       14: ('fence', [0.5, 0.40, 0.0]),
                       15: ('vegetation', [0.0, 0.8, 0.0]),
                       16: ('trunk', [0.5, 0.25, 0.0]),
                       17: ('terrain', [0.25, 1.0, 0.0]),
                       18: ('pole', [0.75, 0.1, 0.0]),
                       19: ('traffic-sign', [1.0, 0.2, 0.0])}
        path = dataset_path + "/sequences_0.06/" + sequence_id + "/velodyne"
        files = []
        try:
            (_, _, files) = next(os.walk(path))
        except StopIteration:
            print('Error: bad path "' + path + '"')
            
        timesteps = sorted([pathlib.Path(f).stem for f in files])
        if len(timesteps) == 0:
            print('Error: No data in "' + path + '"')

        self.datasets = [SemanticKITTIDatum(dataset_path, sequence_id, t)
                         for t in timesteps]

        with open(dataset_path + "/sequences/" + sequence_id + "/times.txt") as f:
            idx = 0
            for line in f.readlines():
                self.datasets[idx].time = float(line)
                idx += 1

#-------------------------------------------------------------------------------
class Colormap:
    class Point:
        def __init__(self, value, color):
            assert(value >= 0.0)
            assert(value <= 1.0)

            self.value = value
            self.color = color

    # The value of each Point must be greater than the previous
    # (e.g. [0.0, 0.1, 0.4, 1.0], not [0.0, 0.4, 0.1, 1.0]
    def __init__(self, points):
        self.points = points

    def calc_u(self, value, range_min, range_max):
        u = (value - range_min) / (range_max - range_min)
        return min(1.0, max(0.0, u))

    # TODO: this will be done by the colormap shader
    def calc_color(self, u):
        i = 0
        while i < len(self.points) and u >= self.points[i].value:
            i += 1
        i = max(0, i - 1)
        i = min(i, len(self.points) - 2)

        p0 = self.points[i]
        p1 = self.points[i+1]
        dist = p1.value - p0.value
        # Calc weights between 0 and 1
        w0 = 1.0 - (u - p0.value) / dist
        w1 = (u - p0.value) / dist
        return [w0 * p0.color[0] + w1 * p1.color[0],
                w0 * p0.color[1] + w1 * p1.color[1],
                w0 * p0.color[2] + w1 * p1.color[2]]

Grayscale = Colormap([Colormap.Point(0.0, [0.0, 0.0, 0.0]),
                      Colormap.Point(1.0, [1.0, 1.0, 1.0])])
Rainbow = Colormap([Colormap.Point(0.000, [1.0, 0.0, 0.0]),
                    Colormap.Point(0.125, [1.0, 0.5, 0.0]),
                    Colormap.Point(0.250, [1.0, 1.0, 0.0]),
                    Colormap.Point(0.375, [0.5, 1.0, 0.0]),
                    Colormap.Point(0.500, [0.0, 1.0, 0.0]),
                    Colormap.Point(0.625, [0.0, 1.0, 0.5]),
                    Colormap.Point(0.750, [0.0, 1.0, 1.0]),
                    Colormap.Point(0.875, [0.0, 0.5, 1.0]),
                    Colormap.Point(1.000, [0.0, 0.0, 1.0])])

class MLVisualizer:
    def __init__(self):
        self._data = {}
        self._known_attrs = set()
        self._name2treenode = {}
        self._name2treeid = {}
        self._label2color = {}
        self._shader2updater = {}
        self.window = gui.Window("ML Visualizer", 1024, 768)
        self.window.set_on_layout(self._on_layout)

        em = self.window.theme.font_size

        self._3d = gui.SceneWidget()
        self._3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._3d)

        self._panel = gui.Vert()
        self.window.add_child(self._panel)

        indented_margins = gui.Margins(em, 0, em, 0)

        # View controls
        ctrl = gui.CollapsableVert("Mouse Controls", 0, indented_margins)

        arcball = gui.Button("Arcball")
        arcball.horizontal_padding_em = 0.5
        arcball.vertical_padding_em = 0
        fly = gui.Button("Fly")
        fly.horizontal_padding_em = 0.5
        fly.vertical_padding_em = 0
        reset = gui.Button("Reset")
        reset.horizontal_padding_em = 0.5
        reset.vertical_padding_em = 0
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(arcball)
        h.add_child(fly)
        h.add_fixed(em)
        h.add_child(reset)
        h.add_stretch()
        ctrl.add_child(h)

        ctrl.add_fixed(em)
        self._panel.add_child(ctrl)

        # Dataset
        model = gui.CollapsableVert("Dataset", 0, indented_margins)

        bgcolor = gui.ColorEdit()
        bgcolor.color_value = gui.Color(1, 1, 1)
        self._on_bgcolor_changed(bgcolor.color_value)
        bgcolor.set_on_value_changed(self._on_bgcolor_changed)
        h = gui.Horiz(em)
        h.add_child(gui.Label("BG Color"))
        h.add_child(bgcolor)
        model.add_child(h)

        self._dataset = gui.TreeView()
        model.add_fixed(0.5 * em)
        model.add_child(self._dataset)

        model.add_fixed(em)
        self._panel.add_child(model)

        # Coloring
        properties = gui.CollapsableVert("Properties", 0, indented_margins)

        grid = gui.VGrid(2, 0.25 * em)
        
        # ... data source
        self._datasource_combobox = gui.Combobox()
        self._datasource_combobox.set_on_selection_changed(self._on_datasource_changed)
        grid.add_child(gui.Label("Source data"))
        grid.add_child(self._datasource_combobox)

        # ... shader
        self._shader = gui.Combobox()
        self._shader.add_item("Solid Color")
        self._shader.add_item("Labels")
        self._shader.add_item("Colormap (Rainbow)")
        self._shader.add_item("Colormap (Grayscale)")
        self._shader2updater[0] = self._make_color_data_uniform
        self._shader2updater[1] = self._make_color_data_labels
        self._shader2updater[2] = self._make_colormap_callback(Rainbow)
        self._shader2updater[3] = self._make_colormap_callback(Grayscale)
        self._shader.selected_index = 0
        self._shader.set_on_selection_changed(self._on_shader_changed)
        grid.add_child(gui.Label("Shader"))
        grid.add_child(self._shader)

        properties.add_child(grid)

        # ... shader panels
        self._shader_panels = gui.StackedWidget()

        #     ... sub-panel: single color
        self._color_panel = gui.Vert()
        self._shader_panels.add_child(self._color_panel)
        self._color = gui.ColorEdit()
        self._color.color_value = gui.Color(0.5, 0.5, 0.5)
        self._color.set_on_value_changed(self._on_shader_color_changed)
        h = gui.Horiz()
        h.add_child(gui.Label("Color"))
        h.add_child(self._color)
        self._color_panel.add_child(h)

        #     ... sub-panel: labels
        self._labels_panel = gui.Vert()
        self._shader_panels.add_child(self._labels_panel)
        self._lut = gui.TreeView()
        self._labels_panel.add_child(gui.Label("Labels"))
        self._labels_panel.add_child(self._lut)

        #     ... sub-panel: colormap
        self._colormap_panel = gui.Vert()
        self._shader_panels.add_child(self._colormap_panel)
        self._min_label = gui.Label("")
        self._max_label = gui.Label("")
        grid = gui.VGrid(2)
        grid.add_child(gui.Label("Range (min):"))
        grid.add_child(self._min_label)
        grid.add_child(gui.Label("Range (max):"))
        grid.add_child(self._max_label)
        self._colormap_panel.add_child(grid)
        self._colormap_panel.add_fixed(0.5 * em)
        self._colormap_panel.add_child(gui.Label("Colormap"))
        self._colormap_edit = gui.TreeView()
        self._colormap_panel.add_child(self._colormap_edit)

        properties.add_fixed(em)
        properties.add_child(self._shader_panels)
        self._panel.add_child(properties)

    def set_labels(self, labels):
        self._lut.clear()
        root = self._lut.get_root_item()
        for key in sorted(labels.keys()):
            name, color = labels[key]
            self._label2color[key] = color
            color = gui.Color(color[0], color[1], color[2])
            cell = gui.LUTTreeCell(str(key) + ": " + name, True,
                                   color, None, None)
            self._lut.add_item(root, cell)
        
    def clear():
        self._data = {}
        self._name2treenode = {}
        self._name2treeid = {}
        self._dataset.clear()
        self._lut.clear()

    def add(self, name, cloud):
        self._data[name] = cloud

        names = name.split("/")
        parent = self._dataset.get_root_item()
        for i in range(0, len(names) - 1):
            n = "/".join(names[:i + 1]) + "/"
            if n in self._name2treeid:
                parent = self._name2treeid[n]
            else:
                def on_parent_checked(checked):
                    self.show_geometries_under(n, checked)
                cell = gui.CheckableTextTreeCell(n, True, on_parent_checked)
                parent = self._dataset.add_item(parent, cell)
                self._name2treenode[n] = cell
                self._name2treeid[n] = parent

        def on_checked(checked):
            self._3d.scene.show_geometry(name, checked)

        cell = gui.CheckableTextTreeCell(names[-1], True, on_checked)
        node = self._dataset.add_item(parent, cell)
        self._name2treenode[name] = cell

        for attr_name in cloud.get_attribute_names():
            if attr_name not in self._known_attrs:
                self._datasource_combobox.add_item(attr_name)
                self._known_attrs.add(attr_name)

        self._update_legacy_point_cloud(name, cloud)

    def setup_camera(self):
        bounds = self._3d.scene.bounding_box
        self._3d.setup_camera(60, bounds, bounds.get_center())

    def show_geometries_under(self, name, show):
        prefix = name
        for (n,node) in self._name2treenode.items():
            if n.startswith(prefix):
                self._3d.scene.show_geometry(n, show)
                node.checkbox.checked = show

    def _update_colormap_widget(self, cmap):
        self._colormap_edit.clear()
        root_id = self._colormap_edit.get_root_item()
        for p in cmap.points:
            color = gui.Color(p.color[0], p.color[1], p.color[2])
            cell = gui.ColormapTreeCell(p.value, color, None, None)
            self._colormap_edit.add_item(root_id, cell)

    def _update_legacy(self):
        for n,tcloud in self._data.items():
            self._update_legacy_point_cloud(n, tcloud)
        
    def _update_legacy_point_cloud(self, name, tcloud):
        self._3d.scene.remove_geometry(name)
        pts = tcloud.get_points()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        attr_name = self._datasource_combobox.selected_text
        attr = tcloud.get_attribute(attr_name)
        if attr is not None:
            make_colors = self._shader2updater[self._shader.selected_index]
            if make_colors is not None:
                this = self
                colors = make_colors(this, attr)
                assert(len(colors) == len(attr))
                assert(len(colors) == len(pts))
            else:
                print("[warning] dataset '" + str(name) + "' has no attribute named '" + self._shader.selected_text + "'")
                colors = [[1.0, 0.0, 1.0]] * len(pts)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        material = rendering.Material()
        material.shader = "defaultUnlit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        self._3d.scene.add_geometry(name, cloud, material)

    @staticmethod
    def _make_color_data_uniform(this, attr):
        color = this._color.color_value
        return [[color.red, color.green, color.blue]] * len(attr)

    @staticmethod
    def _make_color_data_labels(this, attr):
        colors = [[1.0, 0.0, 1.0]] * len(attr)
        for i in range(0, len(attr)):
            if attr[i] in this._label2color:
                colors[i] = this._label2color[attr[i]]
        return colors

    @staticmethod
    def _make_color_data_colormap(this, cmap, attr):
        min_val = min(attr)
        max_val = max(attr)
        this._min_label.text = str(min_val)
        this._max_label.text = str(max_val)
        this._update_colormap_widget(cmap)

        return [cmap.calc_color(cmap.calc_u(v, min_val, max_val)) for v in attr]

    def _make_colormap_callback(self, cmap):
        def callback(this, attr):
            return MLVisualizer._make_color_data_colormap(this, cmap, attr)
        return callback

    def _on_layout(self, theme):
        frame = self.window.content_rect
        em = theme.font_size
        panel_width = 20 * em
        panel_rect = gui.Rect(frame.get_right() - panel_width, frame.y,
                              panel_width, frame.height - frame.y)
        self._panel.frame = panel_rect
        self._3d.frame = gui.Rect(frame.x, frame.y,
                                  panel_rect.x - frame.x, frame.height - frame.y)

    def _on_bgcolor_changed(self, new_color):
        self._3d.set_background_color(new_color)

    def _on_datasource_changed(self, attr_name, idx):
        self._update_legacy()

    def _on_shader_changed(self, name, idx):
        # Last items are all colormaps, so just clamp to n_children - 1
        idx = min(idx, len(self._shader_panels.get_children()) - 1)
        self._shader_panels.selected_index = idx
        self._update_legacy()

    def _on_shader_color_changed(self, color):
        self._update_legacy()

#------------------------------ Example usage ----------------------------------
def main():
    dataset_path = None
    sequence_id = None
    idx = 1
    while idx < len(sys.argv):
        if dataset_path is None:
            dataset_path = sys.argv[idx]
        elif sequence_id is None:
            sequence_id = sys.argv[idx]
        else:
            print("Unknown argument '" + sys.argv[idx] + "'")
        idx += 1

    if dataset_path is None or sequence_id is None:
        print("Usage:")
        print("   ", sys.argv[0], "dataset_path sequence_id")
        sys.exit(1)

    dataset = SemanticKITTI(dataset_path, sequence_id)
    dataset.visualize([0, 2, 4, 6])
#    dataset.visualize(0)

if __name__ == "__main__":
    main()
