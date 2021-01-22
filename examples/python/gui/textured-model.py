#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys

def main():
    if len(sys.argv) < 3:
        print ("Usage: texture-model.py [model name] [texture name]")
        exit()

    model = o3d.io.read_triangle_mesh(sys.argv[1])
    cube = o3d.geometry.TriangleMesh.create_box()
    material = o3d.visualization.rendering.Material()
    material.shader = "defaultLit"
    material.albedo_img = o3d.io.read_image(sys.argv[2])
    o3d.visualization.draw([{"name": "cube", "geometry": model, "material": material}])

if __name__ == "__main__":
    main()
