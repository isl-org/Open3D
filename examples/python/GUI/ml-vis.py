#!/usr/bin/env python
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import pathlib
import sys
import torch

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
            vis.add(self[i])
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
        self.kdtree = None

    def load_data(self):
        if self.points is None:
            self.points = np.load(self._points_path)
        if self.labels is None:
            self.labels = np.squeeze(np.load(self._labels_path))

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

class MLVisualizer:
    def __init__(self):
        self._data = {}
        self._name2treenode = {}
        self._name2treeid = {}
        self._label2color = {}
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

        # Material
        material = gui.CollapsableVert("Material", 0, indented_margins)

        self._shader = gui.Combobox()
        self._shader.add_item("LUT")
        self._shader.add_item("Solid Color")
        self._shader.selected_index = 0
        self._shader.set_on_selection_changed(self._on_shader_changed)
        h = gui.Horiz()
        h.add_child(gui.Label("Shader"))
        h.add_child(self._shader)
        material.add_child(h)

        # ... LUT
        self._lut_shader_panel = gui.Vert()
        self._array_index_combobox = gui.Combobox()
        h = gui.Horiz()
        h.add_child(gui.Label("Array"))
        h.add_child(self._array_index_combobox)
        self._lut_shader_panel.add_child(h)

        self._lut = gui.TreeView()
        self._lut_shader_panel.add_child(gui.Label("Labels"))
        self._lut_shader_panel.add_child(self._lut)

        material.add_child(self._lut_shader_panel)

        # ... solid color
        self._color_shader_panel = gui.Vert()
        self._color_shader_panel.visible = False
        self._color = gui.ColorEdit()
        h = gui.Horiz()
        h.add_child(gui.Label("Color"))
        h.add_child(self._color)
        self._color_shader_panel.add_child(h)
        material.add_child(self._color_shader_panel)

        material.add_fixed(em)
        self._panel.add_child(material)

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

    def add(self, datum):
        self._data[datum.name] = datum

        names = datum.name.split("/")
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
            self._3d.scene.show_geometry(datum.name, checked)

        cell = gui.CheckableTextTreeCell(names[-1], True, on_checked)
        node = self._dataset.add_item(parent, cell)
        self._name2treenode[datum.name] = cell
        if datum.points is not None:
            self._dataset.add_text_item(node, "Raw data (" + str(len(datum.points)) + " pts)")
        if datum.labels is not None:
            self._dataset.add_text_item(node, "Labels")

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(datum.points))
        colors = [[1.0, 0.0, 1.0]] * len(datum.points)
        for i in range(0, len(datum.points)):
            if datum.labels[i] in self._label2color:
                colors[i] = self._label2color[datum.labels[i]]
        cloud.colors = o3d.utility.Vector3dVector(colors)
        material = rendering.Material()
        material.shader = "defaultUnlit"
        material.base_color = [0.5, 0.5, 0.5, 1.0]
        self._3d.scene.add_geometry(datum.name, cloud, material)

    def setup_camera(self):
        bounds = self._3d.scene.bounding_box
        self._3d.setup_camera(60, bounds, bounds.get_center())

    def show_geometries_under(self, name, show):
        prefix = name
        for (n,node) in self._name2treenode.items():
            if n.startswith(prefix):
                self._3d.scene.show_geometry(n, show)
                node.checkbox.checked = show

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

    def _on_shader_changed(self, name, idx):
        if name == "LUT":
            self._lut_shader_panel.visible = True
            self._color_shader_panel.visible = False
        else:
            self._lut_shader_panel.visible = False
            self._color_shader_panel.visible = True

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
