import open3d as o3d
import numpy as np
import os

import meshes


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
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


def mesh_voxelization(mesh,
                      output_filename,
                      camera_path,
                      cubic_size,
                      voxel_resolution,
                      w=300,
                      h=300,
                      use_depth=True,
                      surface_method='pointcloud',
                      visualization=False):

    mesh.compute_vertex_normals()
    camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0])

    # rescale geometry
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    i = 0
    for cid, xyz in enumerate(camera_sphere.vertices):
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
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
        print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))

    vis.destroy_window()

    print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    else:
        raise Exception('invalid surface method')

    voxel_carving_surface = voxel_surface + voxel_carving

    if (visualization):
        print("visualize camera center")
        centers = o3d.geometry.PointCloud()
        centers.points = o3d.utility.Vector3dVector(centers_pts)
        o3d.visualization.draw_geometries([centers, mesh])

        print("surface voxels from %s" % surface_method)
        print(voxel_surface)
        o3d.visualization.draw_geometries([voxel_surface])

        print("carved voxels")
        print(voxel_carving)
        o3d.visualization.draw_geometries([voxel_carving])

        print("combined voxels (carved + surface from %s) together with mesh" %
              surface_method)
        print(voxel_carving_surface)
        o3d.visualization.draw_geometries([voxel_carving_surface, mesh])

    o3d.io.write_voxel_grid(output_filename, voxel_carving_surface)


if __name__ == '__main__':
    mesh = meshes.armadillo()

    output_filename = os.path.abspath("../../TestData/voxelized.ply")
    camera_path = os.path.abspath("../../TestData/sphere.ply")

    visualization = True
    cubic_size = 2.0
    voxel_resolution = 128.0

    mesh_voxelization(mesh,
                      output_filename,
                      camera_path,
                      cubic_size,
                      voxel_resolution,
                      visualization=True)
