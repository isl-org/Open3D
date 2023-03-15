# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

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

    # check for normal map
    if 'normal' in mat.texture_maps:
        nmap_dict = {'type': 'normalmap'}
        nmap_dict['normalmap'] = {
            'type': 'bitmap',
            'raw': True,
            'bitmap': mi.Bitmap(mat.texture_maps['normal'].as_tensor().numpy())
        }
        nmap_dict['bsdf'] = bsdf_dict
        bsdf = mi.load_dict(nmap_dict)
    else:
        bsdf = mi.load_dict(bsdf_dict)
    return bsdf


def to_mitsuba(self, name, bsdf=None):
    """Convert Open3D TriangleMesh to Mitsuba Mesh.

    Converts an Open3D TriangleMesh to a Mitsuba Mesh which can be used directly
    in a Mitsbua scene. The TriangleMesh's material will be converted to a
    Mitsuba Principled BSDF and assigned to the Mitsuba Mesh. Optionally, the
    user may provide a Mitsuba BSDF to be used instead of converting the Open3D
    material.

    Args:
        name (str): Name for the Mitsuba Mesh. Used by Mitsuba as an identifier

        bsdf (default None): If a Mitsuba BSDF is supplied it will be used as
        the BSDF for the converted mesh. Otherwise, the TriangleMesh's material
        will be converted to Mitsuba Principled BSDF.

    Returns:
        A Mitsuba Mesh (with associated BSDF) ready for use in a Mitsuba scene.
    """

    import mitsuba as mi
    import numpy as np

    # What features does this mesh have
    has_normals = 'normals' in self.vertex
    has_uvs = 'texture_uvs' in self.triangle
    has_colors = 'colors' in self.vertex

    # Convert Open3D Material to Mitsuba's principled BSDF
    if bsdf is None:
        bsdf = o3d_material_to_bsdf(self.material, vertex_color=has_colors)

    # Mesh constructor looks for this specific property for setting BSDF
    bsdf_prop = mi.Properties()
    bsdf_prop['mesh_bsdf'] = bsdf

    # Create Mitsuba mesh shell
    mi_mesh = mi.Mesh(name,
                      vertex_count=self.vertex.positions.shape[0],
                      face_count=self.triangle.indices.shape[0],
                      has_vertex_normals=has_normals,
                      has_vertex_texcoords=has_uvs,
                      props=bsdf_prop)

    # Vertex color is not a 'built-in' attribute. Needs to be added.
    if has_colors:
        mi_mesh.add_attribute('vertex_color', 3,
                              self.vertex.colors.numpy().flatten())

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params['vertex_positions'] = self.vertex.positions.numpy().flatten()
    mesh_params['faces'] = self.triangle.indices.numpy().flatten()
    if has_normals:
        mesh_params['vertex_normals'] = self.vertex.normals.numpy().flatten()
    if has_uvs:
        # Mitsuba wants UVs per-vertex so copy them into place
        per_vtx_uvs = np.zeros((self.vertex.positions.shape[0], 2))
        for idx, uvs in zip(self.triangle.indices, self.triangle.texture_uvs):
            per_vtx_uvs[idx.numpy()] = uvs.numpy()
        mesh_params['vertex_texcoords'] = np.subtract(1.0,
                                                      per_vtx_uvs,
                                                      out=per_vtx_uvs,
                                                      where=[False,
                                                             True]).flatten()

    # Let Mitsuba know parameters have been updated
    return mi_mesh


# Add to_mitsuba method to TriangleMesh
o3d.t.geometry.TriangleMesh.to_mitsuba = to_mitsuba
