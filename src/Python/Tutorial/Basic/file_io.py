# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

from open3d import *

if __name__ == "__main__":

    print("Testing IO for point cloud ...")
    pcd = read_point_cloud("../../TestData/fragment.pcd")
    print(pcd)
    write_point_cloud("copy_of_fragment.pcd", pcd)

    print("Testing IO for meshes ...")
    mesh = read_triangle_mesh("../../TestData/knot.ply")
    print(mesh)
    write_triangle_mesh("copy_of_knot.ply", mesh)

    print("Testing IO for images ...")
    img = read_image("../../TestData/lena_color.jpg")
    print(img)
    write_image("copy_of_lena_color.jpg", img)
