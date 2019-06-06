import open3d as o3d
import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x * x + y * y + z * z)
    r_x = math.acos(y / r)
    r_y = math.atan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, math.cos(r_x), -math.sin(r_x)],
                        [0, math.sin(r_x), math.cos(r_x)]])
    rot_y = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0],
                        [-math.sin(r_y), 0, math.cos(r_y)]])
    return rot_y.dot(rot_x)


def depth_to_pcd(depth, intrinsic, extrinsic, w, h):
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    uu, vv = np.meshgrid(x, y)
    uu_vector = uu.ravel()
    vv_vector = vv.ravel()
    depth_vector = np.asarray(depth, dtype=np.float32).ravel()

    uvd = np.asarray(
        [uu_vector * depth_vector, vv_vector * depth_vector, depth_vector])
    uvd_roi = uvd[:, depth_vector != 0]
    xyz_3d = np.linalg.inv(intrinsic).dot(uvd_roi)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_3d.transpose())
    pcd.transform(np.linalg.inv(extrinsic))
    return pcd


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= np.matlib.repmat(center, len(model.vertices), 1)
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


sphere = o3d.io.read_triangle_mesh("../../TestData/sphere.ply")
model = o3d.io.read_triangle_mesh("../../TestData/bathtub_0154.ply")
print("visualize model")
o3d.visualization.draw_geometries([model])

# rescale geometry
sphere = preprocess(sphere)
model = preprocess(model)

w = 320
h = 320
vis = o3d.visualization.Visualizer()
vis.create_window(width=w, height=h)
vis.add_geometry(model)
vis.get_render_option().mesh_show_back_face = True

ctr = vis.get_view_control()
param = ctr.convert_to_pinhole_camera_parameters()

pcd_agg = o3d.geometry.PointCloud()
n_pts = len(sphere.vertices)
centers_pts = np.zeros((n_pts, 3))
i = 0
for xyz in sphere.vertices:
    # get new camera pose
    trans = get_extrinsic(xyz)
    param.extrinsic = trans
    c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
    centers_pts[i, :] = c[:3]
    i += 1
    ctr.convert_from_pinhole_camera_parameters(param)

    # capture depth image and make a point cloud
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(False)
    pcd_agg += depth_to_pcd(depth, param.intrinsic.intrinsic_matrix, trans, w,
                            h)

vis.destroy_window()

print("visualize camera center")
centers = o3d.geometry.PointCloud()
centers.points = o3d.utility.Vector3dVector(centers_pts)
o3d.visualization.draw_geometries([centers, model])

print("voxelize dense point cloud")
voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_agg, voxel_size=0.05)
print(voxel)
o3d.visualization.draw_geometries([voxel])

print("save and load o3d.geometry.VoxelGrid")
o3d.io.write_voxel_grid("voxel_grid_test.ply", voxel)
voxel_read = o3d.io.read_voxel_grid("voxel_grid_test.ply")
print(voxel_read)

print("visualize original model and voxels together")
o3d.visualization.draw_geometries([voxel_read, model])
