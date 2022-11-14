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

import open3d as o3d


def to_mitsuba(name, o3d_mesh, bsdf=None):
    import mitsuba as mi

    # TODO: if bsdf not provided make one from material properties

    # Mesh constructor looks for this specific property for setting BSDF
    bsdf_prop = mi.Properties()
    bsdf_prop["mesh_bsdf"] = bsdf

    # Create Mitsuba mesh shell
    has_normals = 'normals' in o3d_mesh.vertex
    has_uvs = False
    has_colors = 'colors' in o3d_mesh.vertex
    mi_mesh = mi.Mesh(name,
                      vertex_count=o3d_mesh.vertex.positions.shape[0],
                      face_count=o3d_mesh.triangle.indices.shape[0],
                      has_vertex_normals=has_normals,
                      has_vertex_texcoords=has_uvs,
                      props=bsdf_prop)

    # Vertex color is not a 'built-in' attribute. Needs to be added.
    if has_colors:
        mi_mesh.add_attribute("vertex_color", 3,
                              o3d_mesh.vertex.colors.numpy().flatten())

    # TODO: Get texcoords into Mitsuba per-vertex.

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params['vertex_positions'] = o3d_mesh.vertex.positions.numpy().flatten(
    )
    mesh_params['faces'] = o3d_mesh.triangle.indices.numpy().flatten()
    if has_normals:
        mesh_params['vertex_normals'] = o3d_mesh.vertex.normals.numpy().flatten(
        )

    # Let Mitsuba know parameters have been updated
    mesh_params.update()
    return mi_mesh
