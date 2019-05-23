from open3d import *
from os.path import abspath
import math
import numpy as np
import numpy.matlib

from joblib import Parallel, delayed
import multiprocessing

def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0,0,2]).transpose()
    trans = np.eye(4)
    trans[:3,:3] = r
    trans[:3,3] = t
    return trans

def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x*x+y*y+z*z)
    r_x = math.acos(y/r)
    r_y = math.atan2(z,x)
    return [r, r_x, r_y]

def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray(
           [[1, 0, 0],
            [0, math.cos(r_x), -math.sin(r_x)],
            [0, math.sin(r_x), math.cos(r_x)]])
    rot_y = np.asarray(
           [[math.cos(r_y), 0, math.sin(r_y)],
            [0, 1, 0],
            [-math.sin(r_y), 0, math.cos(r_y)]])
    return rot_y.dot(rot_x)

def depth_to_pcd(depth, intrinsic, extrinsic, w, h):
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    uu, vv = np.meshgrid(x, y)
    uu_vector = uu.ravel()
    vv_vector = vv.ravel()
    depth_vector = np.asarray(depth, dtype=np.float32).ravel()

    uvd = np.asarray([uu_vector * depth_vector,
                      vv_vector * depth_vector,
                      depth_vector])
    uvd_roi = uvd[:, depth_vector != 0]
    xyz_3d = np.linalg.inv(intrinsic).dot(uvd_roi)
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz_3d.transpose())
    pcd.transform(np.linalg.inv(extrinsic))
    return pcd

def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= np.matlib.repmat(center, len(model.vertices), 1)
    model.vertices = Vector3dVector(vertices / scale)
    return model

def mesh_voxelization(mid, input_filename, output_filename, camera_path, 
        cubic_size, voxel_resolution, w=300, h=300, visualization=False):

    camera_sphere = read_triangle_mesh(camera_path)
    mesh = read_triangle_mesh(input_filename)
    mesh.compute_vertex_normals()

    voxel_grid_carving = create_voxel_grid(
            w=cubic_size, h=cubic_size, d=cubic_size, 
            voxel_size=cubic_size/voxel_resolution,
            origin=[-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0])

    # rescale geometry 
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    vis = Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    pcd_agg = PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    i = 0
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0,0,0,1]).transpose())
        centers_pts[i,:] = c[:3]
        i += 1
        ctr.convert_from_pinhole_camera_parameters(param)
        
        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += depth_to_pcd(depth,
                param.intrinsic.intrinsic_matrix, trans, w, h)
        
        # depth map carving method
        voxel_grid_carving = carve_voxel_grid_using_depth_map(
                voxel_grid_carving, Image(depth), param)
        print("Depth carving %05d, view %03d/%03d" %
              (mid, cid, len(camera_sphere.vertices)))

    vis.destroy_window()

    voxel_surface = create_surface_voxel_grid_from_point_cloud(
            pcd_agg, voxel_size=cubic_size/voxel_resolution,
            min_bound=[-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0],
            max_bound=[-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0])

    voxel_combine = voxel_surface + voxel_grid_carving

    if (visualization):
        print("visualize camera center")
        centers = PointCloud()
        centers.points = Vector3dVector(centers_pts)
        draw_geometries([centers, mesh])

        print("surface voxels")
        print(voxel_surface)
        draw_geometries([voxel_surface])

        print("carved voxels")
        print(voxel_grid_carving)
        draw_geometries([voxel_grid_carving])

        print("combined voxels")
        print(voxel_combine)
        draw_geometries([voxel_combine])

        print("visualize original model and voxels together")
        draw_geometries([voxel_combine, mesh])
    
    write_voxel_grid(output_filename, voxel_combine)

if __name__ == '__main__':

    n_parallel = 10
    input_filename = [abspath("../../TestData/bathtub_0154.ply")] * n_parallel
    output_filename = [abspath("../../TestData/bathtub_0154_voxel.ply")] * n_parallel
    camera_path = abspath("../../TestData/sphere.ply")
    
    visualization = True
    cubic_size = 2.0
    voxel_resolution = 128.0

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(mesh_voxelization)(i, input_filename[i], output_filename[i], camera_path,
                                cubic_size, voxel_resolution, visualization=False) for i in range(n_parallel))
