# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/file_io.py

import open3d as o3d

if __name__ == "__main__":

    print("Testing IO for point cloud ...")
    pcd = o3d.io.read_point_cloud("../../TestData/fragment.pcd")
    print(pcd)
    o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

    print("Testing IO for meshes ...")
    mesh = o3d.io.read_triangle_mesh("../../TestData/knot.ply")
    print(mesh)
    o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)

    print("Testing IO for images ...")
    img = o3d.io.read_image("../../TestData/lena_color.jpg")
    print(img)
    o3d.io.write_image("copy_of_lena_color.jpg", img)
