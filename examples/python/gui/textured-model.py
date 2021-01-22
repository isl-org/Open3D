#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys, os


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: texture-model.py [model directory]\n\t This example will load [model direcotry].obj plus any of albedo, normal, ao, metallic and roughness textures present."
        )
        exit()

    model_dir = sys.argv[1]
    model_name = os.path.join(model_dir, os.path.basename(model_dir) + ".obj")
    model = o3d.io.read_triangle_mesh(model_name)
    material = o3d.visualization.rendering.Material()
    material.shader = "defaultLit"

    albedo_name = os.path.join(model_dir, "albedo.png")
    normal_name = os.path.join(model_dir, "normal.png")
    ao_name = os.path.join(model_dir, "ao.png")
    metallic_name = os.path.join(model_dir, "metallic.png")
    roughness_name = os.path.join(model_dir, "roughness.png")
    if os.path.exists(albedo_name):
        material.albedo_img = o3d.io.read_image(albedo_name)
    if os.path.exists(normal_name):
        material.normal_img = o3d.io.read_image(normal_name)
    if os.path.exists(ao_name):
        material.ao_img = o3d.io.read_image(ao_name)
    if os.path.exists(metallic_name):
        material.base_metallic = 1.0
        material.metallic_img = o3d.io.read_image(metallic_name)
    if os.path.exists(roughness_name):
        material.roughness_img = o3d.io.read_image(roughness_name)

    o3d.visualization.draw([{
        "name": "cube",
        "geometry": model,
        "material": material
    }])


if __name__ == "__main__":
    main()
