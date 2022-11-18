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

import mitsuba
import open3d as o3d


def o3d_material_to_bsdf(mat, vertex_color=False):
    import mitsuba as mi

    def create_bsdf_entry(mat, value, o3d_name, per_vertex):
        if per_vertex:
            return {'type': 'mesh_attribute', 'name': 'vertex_color'}
        elif o3d_name in mat.texture_maps:
            return {
                'type':
                    'bitmap',
                'bitmap':
                    mi.Bitmap(mat.texture_maps[o3d_name].as_tensor().numpy())
            }
        else:
            return {'type': 'rgb', 'value': value}

    base_color = mat.vector_properties['base_color'][0:3]
    roughness = mat.scalar_properties['roughness']
    metallic = mat.scalar_properties['metallic']
    reflectance = mat.scalar_properties['reflectance']
    anisotropy = mat.scalar_properties['anisotropy']

    bsdf_dict = {'type': 'principled'}
    bsdf_dict['base_color'] = create_bsdf_entry(mat, base_color, 'albedo',
                                                vertex_color)
    bsdf_dict['roughness'] = create_bsdf_entry(mat, roughness, 'roughness',
                                               False)
    bsdf_dict['metallic'] = create_bsdf_entry(mat, metallic, 'metallic', False)
    bsdf_dict['anisotropic'] = create_bsdf_entry(mat, anisotropy, 'anisotropy',
                                                 False)
    bsdf_dict['specular'] = reflectance
    bsdf = mi.load_dict(bsdf_dict)
    return bsdf


def to_mitsuba(name, o3d_mesh, bsdf=None):
    import mitsuba as mi
    import numpy as np

    # What features does this mesh have
    has_normals = 'normals' in o3d_mesh.vertex
    has_uvs = 'texture_uvs' in o3d_mesh.triangle
    has_colors = 'colors' in o3d_mesh.vertex

    # Convert Open3D Material to Mitsuba's principled BSDF
    if bsdf is None:
        bsdf = o3d_material_to_bsdf(o3d_mesh.material, vertex_color=has_colors)

    # Mesh constructor looks for this specific property for setting BSDF
    bsdf_prop = mi.Properties()
    bsdf_prop['mesh_bsdf'] = bsdf

    # Create Mitsuba mesh shell
    mi_mesh = mi.Mesh(name,
                      vertex_count=o3d_mesh.vertex.positions.shape[0],
                      face_count=o3d_mesh.triangle.indices.shape[0],
                      has_vertex_normals=has_normals,
                      has_vertex_texcoords=has_uvs,
                      props=bsdf_prop)

    # Vertex color is not a 'built-in' attribute. Needs to be added.
    if has_colors:
        mi_mesh.add_attribute('vertex_color', 3,
                              o3d_mesh.vertex.colors.numpy().flatten())

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params['vertex_positions'] = o3d_mesh.vertex.positions.numpy().flatten(
    )
    mesh_params['faces'] = o3d_mesh.triangle.indices.numpy().flatten()
    if has_normals:
        mesh_params['vertex_normals'] = o3d_mesh.vertex.normals.numpy().flatten(
        )
    if has_uvs:
        # Mitsuba wants UVs per-vertex so copy them into place
        per_vtx_uvs = np.zeros((o3d_mesh.vertex.positions.shape[0], 2))
        for idx, uvs in zip(o3d_mesh.triangle.indices,
                            o3d_mesh.triangle.texture_uvs):
            per_vtx_uvs[idx.numpy()] = uvs.numpy()
        mesh_params['vertex_texcoords'] = np.subtract(1.0,
                                                      per_vtx_uvs,
                                                      out=per_vtx_uvs,
                                                      where=[False,
                                                             True]).flatten()

    # Let Mitsuba know parameters have been updated
    return mi_mesh
