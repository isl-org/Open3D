# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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
"""Internal information about the Open3D plugin."""

from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from .plugin_data_pb2 import LabelToNames

PLUGIN_NAME = "Open3D"

# The most recent value for the `version` field of the
# `Open3DPluginData` proto. Sync with Open3D version (MAJOR*100 + MINOR)
_VERSION = 14

SUPPORTED_FILEFORMAT_VERSIONS = [14]

GEOMETRY_PROPERTY_DIMS = {
    'vertex_positions': (3,),
    'vertex_normals': (3,),
    'vertex_colors': (3,),
    'vertex_texture_uvs': (2,),
    'triangle_indices': (3,),
    'triangle_normals': (3,),
    'triangle_colors': (3,),
    'triangle_texture_uvs': (3, 2),
    'line_indices': (2,),
    'line_colors': (3,)
}
VERTEX_PROPERTIES = ('vertex_normals', 'vertex_colors', 'vertex_texture_uvs')
TRIANGLE_PROPERTIES = ('triangle_normals', 'triangle_colors',
                       'triangle_texture_uvs')
LINE_PROPERTIES = ('line_colors',)
MATERIAL_SCALAR_PROPERTIES = (
    'point_size',
    'line_width',
    'metallic',
    'roughness',
    'reflectance',
    # 'sheen_roughness',
    'clear_coat',
    'clear_coat_roughness',
    'anisotropy',
    'ambient_occlusion',
    # 'ior',
    'transmission',
    # 'micro_thickness',
    'thickness',
    'absorption_distance',
)
MATERIAL_VECTOR_PROPERTIES = (
    'base_color',
    # 'sheen_color',
    # 'anisotropy_direction',
    'normal',
    # 'bent_normal',
    # 'clear_coat_normal',
    # 'emissive',
    # 'post_lighting_color',
    'absorption_color',
)
MATERIAL_TEXTURE_MAPS = (
    'albedo',  # same as base_color
    # Ambient occlusion, roughness, and metallic maps in a single 3 channel
    # texture. Commonly used in glTF models.
    'ao_rough_metal',
)

SUPPORTED_PROPERTIES = set(
    tuple(GEOMETRY_PROPERTY_DIMS.keys()) + ("material_name",) +
    tuple("material_scalar_" + p for p in MATERIAL_SCALAR_PROPERTIES) +
    tuple("material_vector_" + p for p in MATERIAL_VECTOR_PROPERTIES) +
    tuple("material_texture_map_" + p for p in
          (MATERIAL_SCALAR_PROPERTIES[2:] +  # skip point_size, line_width
           MATERIAL_VECTOR_PROPERTIES[1:] +  # skip base_color
           MATERIAL_TEXTURE_MAPS)))


def create_summary_metadata(description, metadata):
    """Creates summary metadata. Reserved for future use. Required by
    TensorBoard.

    Args:
      description: The description to show in TensorBoard.

    Returns:
      A `SummaryMetadata` protobuf object.
    """
    ln_proto = LabelToNames()
    if 'label_to_names' in metadata:
        ln_proto.label_to_names.update(metadata['label_to_names'])
    return SummaryMetadata(
        summary_description=description,
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=ln_proto.SerializeToString()),
    )


def parse_plugin_metadata(content):
    """Parse summary metadata to a Python object. Reserved for future use.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the Open3D plugin.
    """
    ln_proto = LabelToNames()
    ln_proto.ParseFromString(content)
    return ln_proto.label_to_names
