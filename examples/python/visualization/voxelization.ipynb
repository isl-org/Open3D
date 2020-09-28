{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "nbsphinx": "hidden"
   },
   "outputs": [],
   "source": [
    "import open3d as o3d\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import copy\n",
    "import os\n",
    "import sys\n",
    "\n",
    "# only needed for tutorial, monkey patches visualization\n",
    "sys.path.append('..')\n",
    "import open3d_tutorial as o3dtut\n",
    "# change to True if you want to interact with the visualization windows\n",
    "o3dtut.interactive = not \"CI\" in os.environ"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Voxelization\n",
    "Point clouds and triangle meshes are very flexible, but irregular, geometry types. The voxel grid is another geometry type in 3D that is defined on a regular 3D grid, whereas a voxel can be thought of as the 3D counterpart to the pixel in 2D. Open3D has the geometry type `VoxelGrid` that can be used to work with voxel grids."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## From triangle mesh\n",
    "Open3D provides the method `create_from_triangle_mesh` that creates a voxel grid from a triangle mesh. It returns a voxel grid where all voxels that are intersected by a triangle are set to `1`, all others are set to `0`. The argument `voxel_size` defines the resolution of the voxel grid."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('input')\n",
    "mesh = o3dtut.get_bunny_mesh()\n",
    "# fit to unit cube\n",
    "mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),\n",
    "           center=mesh.get_center())\n",
    "o3d.visualization.draw_geometries([mesh])\n",
    "\n",
    "print('voxelization')\n",
    "voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,\n",
    "                                                              voxel_size=0.05)\n",
    "o3d.visualization.draw_geometries([voxel_grid])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## From point cloud\n",
    "The voxel grid can also be created from a point cloud using the method `create_from_point_cloud`. A voxel is occupied if at least one point of the point cloud is within the voxel. The color of the voxel is the average of all the points within the voxel. The argument `voxel_size` defines the resolution of the voxel grid."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('input')\n",
    "N = 2000\n",
    "pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)\n",
    "# fit to unit cube\n",
    "pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),\n",
    "          center=pcd.get_center())\n",
    "pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))\n",
    "o3d.visualization.draw_geometries([pcd])\n",
    "\n",
    "print('voxelization')\n",
    "voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,\n",
    "                                                            voxel_size=0.05)\n",
    "o3d.visualization.draw_geometries([voxel_grid])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Inclusion test\n",
    "The voxel grid can also be used to test if points are within an occupied voxel. The method `check_if_included` takes a `(n,3)` array as input and outputs a `bool` array."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "queries = np.asarray(pcd.points)\n",
    "output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))\n",
    "print(output[:10])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Voxel carving\n",
    "The methods `create_from_point_cloud` and `create_from_triangle_mesh` create occupied voxels only on the surface of the geometry. It is however possible to carve a voxel grid from a number of depth maps or silhouettes. Open3D provides the methods `carve_depth_map` and `carve_silhouette` for voxel carving.\n",
    "\n",
    "The code below demonstrates the usage by first rendering depthmaps from a geometry and using those depthmaps to carve a dense voxel grid. The result is a filled voxel grid of the given shape."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def xyz_spherical(xyz):\n",
    "    x = xyz[0]\n",
    "    y = xyz[1]\n",
    "    z = xyz[2]\n",
    "    r = np.sqrt(x * x + y * y + z * z)\n",
    "    r_x = np.arccos(y / r)\n",
    "    r_y = np.arctan2(z, x)\n",
    "    return [r, r_x, r_y]\n",
    "\n",
    "\n",
    "def get_rotation_matrix(r_x, r_y):\n",
    "    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],\n",
    "                        [0, np.sin(r_x), np.cos(r_x)]])\n",
    "    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],\n",
    "                        [-np.sin(r_y), 0, np.cos(r_y)]])\n",
    "    return rot_y.dot(rot_x)\n",
    "\n",
    "\n",
    "def get_extrinsic(xyz):\n",
    "    rvec = xyz_spherical(xyz)\n",
    "    r = get_rotation_matrix(rvec[1], rvec[2])\n",
    "    t = np.asarray([0, 0, 2]).transpose()\n",
    "    trans = np.eye(4)\n",
    "    trans[:3, :3] = r\n",
    "    trans[:3, 3] = t\n",
    "    return trans\n",
    "\n",
    "\n",
    "def preprocess(model):\n",
    "    min_bound = model.get_min_bound()\n",
    "    max_bound = model.get_max_bound()\n",
    "    center = min_bound + (max_bound - min_bound) / 2.0\n",
    "    scale = np.linalg.norm(max_bound - min_bound) / 2.0\n",
    "    vertices = np.asarray(model.vertices)\n",
    "    vertices -= center\n",
    "    model.vertices = o3d.utility.Vector3dVector(vertices / scale)\n",
    "    return model\n",
    "\n",
    "\n",
    "def voxel_carving(mesh,\n",
    "                  output_filename,\n",
    "                  camera_path,\n",
    "                  cubic_size,\n",
    "                  voxel_resolution,\n",
    "                  w=300,\n",
    "                  h=300,\n",
    "                  use_depth=True,\n",
    "                  surface_method='pointcloud'):\n",
    "    mesh.compute_vertex_normals()\n",
    "    camera_sphere = o3d.io.read_triangle_mesh(camera_path)\n",
    "\n",
    "    # setup dense voxel grid\n",
    "    voxel_carving = o3d.geometry.VoxelGrid.create_dense(\n",
    "        width=cubic_size,\n",
    "        height=cubic_size,\n",
    "        depth=cubic_size,\n",
    "        voxel_size=cubic_size / voxel_resolution,\n",
    "        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],\n",
    "        color=[1.0, 0.7, 0.0])\n",
    "\n",
    "    # rescale geometry\n",
    "    camera_sphere = preprocess(camera_sphere)\n",
    "    mesh = preprocess(mesh)\n",
    "\n",
    "    # setup visualizer to render depthmaps\n",
    "    vis = o3d.visualization.Visualizer()\n",
    "    vis.create_window(width=w, height=h, visible=False)\n",
    "    vis.add_geometry(mesh)\n",
    "    vis.get_render_option().mesh_show_back_face = True\n",
    "    ctr = vis.get_view_control()\n",
    "    param = ctr.convert_to_pinhole_camera_parameters()\n",
    "\n",
    "    # carve voxel grid\n",
    "    pcd_agg = o3d.geometry.PointCloud()\n",
    "    centers_pts = np.zeros((len(camera_sphere.vertices), 3))\n",
    "    for cid, xyz in enumerate(camera_sphere.vertices):\n",
    "        # get new camera pose\n",
    "        trans = get_extrinsic(xyz)\n",
    "        param.extrinsic = trans\n",
    "        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())\n",
    "        centers_pts[cid, :] = c[:3]\n",
    "        ctr.convert_from_pinhole_camera_parameters(param)\n",
    "\n",
    "        # capture depth image and make a point cloud\n",
    "        vis.poll_events()\n",
    "        vis.update_renderer()\n",
    "        depth = vis.capture_depth_float_buffer(False)\n",
    "        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(\n",
    "            o3d.geometry.Image(depth),\n",
    "            param.intrinsic,\n",
    "            param.extrinsic,\n",
    "            depth_scale=1)\n",
    "\n",
    "        # depth map carving method\n",
    "        if use_depth:\n",
    "            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)\n",
    "        else:\n",
    "            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)\n",
    "        print(\"Carve view %03d/%03d\" % (cid + 1, len(camera_sphere.vertices)))\n",
    "    vis.destroy_window()\n",
    "\n",
    "    # add voxel grid survace\n",
    "    print('Surface voxel grid from %s' % surface_method)\n",
    "    if surface_method == 'pointcloud':\n",
    "        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(\n",
    "            pcd_agg,\n",
    "            voxel_size=cubic_size / voxel_resolution,\n",
    "            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),\n",
    "            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))\n",
    "    elif surface_method == 'mesh':\n",
    "        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(\n",
    "            mesh,\n",
    "            voxel_size=cubic_size / voxel_resolution,\n",
    "            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),\n",
    "            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))\n",
    "    else:\n",
    "        raise Exception('invalid surface method')\n",
    "    voxel_carving_surface = voxel_surface + voxel_carving\n",
    "\n",
    "    return voxel_carving_surface, voxel_carving, voxel_surface"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mesh = o3dtut.get_armadillo_mesh()\n",
    "\n",
    "output_filename = os.path.abspath(\"../../test_data/voxelized.ply\")\n",
    "camera_path = os.path.abspath(\"../../test_data/sphere.ply\")\n",
    "visualization = True\n",
    "cubic_size = 2.0\n",
    "voxel_resolution = 128.0\n",
    "\n",
    "voxel_grid, voxel_carving, voxel_surface = voxel_carving(\n",
    "    mesh, output_filename, camera_path, cubic_size, voxel_resolution)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"surface voxels\")\n",
    "print(voxel_surface)\n",
    "o3d.visualization.draw_geometries([voxel_surface])\n",
    "\n",
    "print(\"carved voxels\")\n",
    "print(voxel_carving)\n",
    "o3d.visualization.draw_geometries([voxel_carving])\n",
    "\n",
    "print(\"combined voxels (carved + surface)\")\n",
    "print(voxel_grid)\n",
    "o3d.visualization.draw_geometries([voxel_grid])"
   ]
  }
 ],
 "metadata": {
  "celltoolbar": "Edit Metadata",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
