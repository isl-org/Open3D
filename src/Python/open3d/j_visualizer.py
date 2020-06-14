# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import ipywidgets as widgets
from traitlets import Unicode, Float, List, Instance
from IPython.display import display
import open3d as o3
import numpy as np


def geometry_to_json(geometry):
    """Convert Open3D geometry to Json (Dict)"""
    json = dict()
    if isinstance(geometry, o3.geometry.PointCloud):
        json['type'] = 'PointCloud'
        # TODO: do not flatten
        json['points'] = np.asarray(geometry.points,
                                    dtype=np.float32).reshape(-1).tolist()
        json['colors'] = np.asarray(geometry.colors,
                                    dtype=np.float32).reshape(-1).tolist()
    else:
        raise NotImplementedError(
            "Only supporting geometry_to_json for PointCloud")
    return json


@widgets.register
class JVisualizer(widgets.DOMWidget):
    _view_name = Unicode('JVisualizerView').tag(sync=True)
    _view_module = Unicode('open3d').tag(sync=True)
    _view_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)
    _model_name = Unicode('JVisualizerModel').tag(sync=True)
    _model_module = Unicode('open3d').tag(sync=True)
    _model_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)

    # We need to declare class attributes for traitlets to work
    geometry_jsons = List(Instance(dict)).tag(sync=True)

    def __init__(self):
        super(JVisualizer, self).__init__()
        self.geometry_jsons = []
        self.geometries = []

    def __repr__(self):
        return "JVisualizer with %s geometries" % len(self.geometry_jsons)

    def add_geometry(self, geometry):
        # TODO: See if we can use self.send(content=content)
        #       For some reason self.geometry_jsons has to be directly assigned,
        #       so we keep track of self.geometries and self.geometry_jsons.
        self.geometries.append(geometry)
        self.geometry_jsons = [geometry_to_json(g) for g in self.geometries]

    def clear(self):
        self.geometries = []
        self.geometry_jsons = []

    # TODO: consider using this mechanism to send geometry data
    # def send_dog(self):
    #     print("py: sending gwen")
    #     content = {
    #         "type": "dog",
    #         "name": "gwen"
    #     }
    #     self.send(content=content)

    def show(self):
        display(self)
