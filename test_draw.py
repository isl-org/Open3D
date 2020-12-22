import open3d as o3d

pcd = o3d.io.read_point_cloud(
    "/Users/ylao/repo/Open3D/examples/test_data/fragment.pcd")
o3d.visualization.draw_geometries([pcd])
