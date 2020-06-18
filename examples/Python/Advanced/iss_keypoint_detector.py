# @file      iss_keypoint_detector.py
# @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
#
# Copyright (c) 2020 Ignacio Vizzo, all rights reserved
import os
import sys
import time

import open3d as o3d

import meshes

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../Misc"))


def compute_iss_and_visualize(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    tic = time.time()
    keypoints = pcd.compute_iss_keypoints()
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints.paint_uniform_color([1.0, 0.75, 0.0])
    o3d.visualization.draw_geometries([keypoints, mesh])


def compute_iss_and_visualize_class(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    tic = time.time()
    keypoints = pcd.compute_iss_keypoints(salient_radius=0.005,
                                          non_maxima_radius=0.005,
                                          gamma_21=0.5,
                                          gamma_32=0.5)
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints.paint_uniform_color([1.0, 0.75, 0.0])
    o3d.visualization.draw_geometries([keypoints, mesh])


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.Debug)
    # Download the data
    bunny = meshes.bunny()
    armadillo = meshes.armadillo()

    print("Computing ISS Keypoint on Standford Bunny...")
    compute_iss_and_visualize(bunny)

    print("Computing ISS Keypoint on Armadillo model...")
    compute_iss_and_visualize(armadillo)

    print("Computing ISS (with params) Keypoint on Standford Bunny...")
    compute_iss_and_visualize_class(bunny)
