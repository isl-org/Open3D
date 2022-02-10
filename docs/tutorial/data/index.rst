.. _dataset:

Dataset
=======

Open3D comes with a built-in dataset module for convenient access to commonly
used example datasets. These datasets will downloaded automatically from the
internet.

.. code-block:: python

    import open3d as o3d

    if __name__ == "__main__":
        dataset = o3d.data.EaglePointCloud()
        pcd = o3d.io.read_point_cloud(dataset.path)
        o3d.visualization.draw(pcd)

.. code-block:: cpp

    #include <string>
    #include <memory>
    #include "open3d/Open3D.h"

    int main() {
        using namespace open3d;

        data::EaglePointCloud dataset;
        auto pcd = io::CreatePointCloudFromFile(dataset.GetPath());
        visualization::Draw({pcd});

        return 0;
    }

- Dataset will be downloaded can cached automatically. The default data root is
  ``~/open3d_data``. The data will be downloaded to ``~/open3d_data/download``
  and extracted to ``~/open3d_data/extract``.
- Optionally, you can change the default data root. This can be done by setting
  the environment variable ``OPEN3D_DATA_ROOT`` or passing the ``data_root``
  argument when constructing a dataset object.

Point Cloud
~~~~~~~~~~~

SamplePointCloudPCD
-------------------

Colored point cloud of a living room from the Redwood dataset in PCD format.

.. code-block:: python

    dataset = o3d.data.SamplePointCloudPCD()
    pcd = o3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::SamplePointCloudPCD dataset;
    auto pcd = io::CreatePointCloudFromFile(dataset.GetPath());

SamplePointCloudPLY
-------------------

Colored point cloud of a living room from the Redwood dataset in PLY format.

.. code-block:: python

    dataset = o3d.data.SamplePointCloudPLY()
    pcd = o3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::SamplePointCloudPLY dataset;
    auto pcd = io::CreatePointCloudFromFile(dataset.GetPath());

EaglePointCloud
---------------

Eagle colored point cloud.

.. code-block:: python

    dataset = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::EaglePointCloud dataset;
    auto pcd = io::CreatePointCloudFromFile(dataset.GetPath());

RedwoodLivingRoomPointClouds
----------------------------

57 point clouds of binary PLY format from the Redwood RGB-D Dataset.

.. code-block:: python

    dataset = o3d.data.RedwoodLivingRoomPointCloud()
    pcds = []
    for pcd_path in dataset.paths:
        pcds.append(o3d.io.read_point_cloud(pcd_path))

.. code-block:: cpp

    data::RedwoodLivingRoomPointCloud dataset;
    std::vector<std::shared_ptr<geometry::PointCloud>> pcds;
    for (const std::string& pcd_path: dataset.GetPaths()) {
        pcds.push_back(io::CreatePointCloudFromFile(pcd_path));
    }

RedwoodOfficePointClouds
------------------------

53 point clouds of binary PLY format from Redwood RGB-D Dataset.

.. code-block:: python

    dataset = o3d.data.RedwoodOfficePointCloud()
    pcds = []
    for pcd_path in dataset.paths:
        pcds.append(o3d.io.read_point_cloud(pcd_path))

.. code-block:: cpp

    data::RedwoodOfficePointClouds dataset;
    std::vector<std::shared_ptr<geometry::PointCloud>> pcds;
    for (const std::string& pcd_path: dataset.GetPaths()) {
        pcds.push_back(io::CreatePointCloudFromFile(pcd_path));
    }

Triangle Mesh
~~~~~~~~~~~~~

BunnyMesh
---------

The bunny triangle mesh from Stanford in PLY format.

.. code-block:: python

    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::BunnyMesh dataset;
    geometry::TriangleMesh mesh;
    ReadTriangleMesh(dataset.GetPath(), mesh);

ArmadilloMesh
-------------

The armadillo mesh from Stanford in PLY format.

.. code-block:: python

    dataset = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::ArmadilloMesh dataset;
    geometry::TriangleMesh mesh;
    ReadTriangleMesh(dataset.GetPath(), mesh);

KnotMesh
--------

A 3D Mobius knot mesh in PLY format.

.. code-block:: python

    dataset = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::KnotMesh dataset;
    geometry::TriangleMesh mesh;
    ReadTriangleMesh(dataset.GetPath(), mesh);

RGBD Image
~~~~~~~~~~

SampleRGBDDatasetRedwood
------------------------

Sample set of 5 color images, 5 depth images from the ``Redwood RGBD living-room1`` dataset.
It also contains a ``camera trajectory log``, ``a camera odometry log``, a ``rgbd match`` file,
and a ``point cloud reconstruction`` obtained from TSDF.

.. code-block:: python

    dataset = o3d.data.SampleRGBDDatasetRedwood()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        color_raw = o3d.io.read_image(dataset.color_paths[i])
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                                   color_raw, depth_raw)
        rgbd_images.append(rgbd_image)

    pcd = o3d.io.read_point_cloud(dataset.reconstruction_path)

.. code-block:: cpp

    data::SampleRGBDDatasetRedwood dataset;

    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for(size_t i = 0; i < dataset.GetDepthPaths().size(); ++i) {
        geometry::Image color_raw, depth_raw;
        io::ReadImage(dataset.GetColorPaths[i], color_raw);
        io::ReadImage(dataset.GetDepthPaths[i], depth_raw);

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                                    color_raw, depth_raw,
                                    /*depth_scale =*/ 1000.0,
                                    /*depth_trunc =*/ 3.0,
                                    /*convert_rgb_to_intensity =*/ False);
        rgbd_images.push_back(rgbd_image);
    }

    geometry::PointCloud pcd;
    io::ReadPointCloud(dataset.GetReconstructionPath(), pcd);

SampleFountainRGBDDataset
-------------------------

Sample set of 33 color and depth images from the ``Fountain RGBD dataset``.
It also contains ``camera poses at key frames log`` and ``mesh reconstruction``.
It is used in demo of ``Color Map Optimization``.

.. code-block:: python

    dataset = o3d.data.SampleFountainRGBDDataset()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        depth = o3d.io.read_image(dataset.depth_paths[i])
        color = o3d.io.read_image(dataset.color_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                           color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
                              dataset.keyframe_poses_log_path)
    mesh = o3d.io.read_triangle_mesh(
         dataset.reconstruction_path)

.. code-block:: cpp

    data::SampleFountainRGBDDataset dataset;

    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for(size_t i = 0; i < dataset.GetDepthPaths().size(); ++i) {
        geometry::Image color_raw, depth_raw;
        io::ReadImage(dataset.GetColorPaths[i], color_raw);
        io::ReadImage(dataset.GetDepthPaths[i], depth_raw);

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                                    color_raw, depth_raw,
                                    /*depth_scale =*/ 1000.0,
                                    /*depth_trunc =*/ 3.0,
                                    /*convert_rgb_to_intensity =*/ False);
    }

    camera::PinholeCameraTrajectory camera_trajectory;
    io::ReadPinholeCameraTrajectory(dataset.GetKeyframePosesLogPath(),
                                    camera_trajectory);

    geometry::TriangleMesh mesh;
    io::ReadTriangleMesh(dataset.GetReconstructionPath(), mesh);

SampleRGBDImageNYU
------------------

Color image ``NYU_color.ppm`` and depth image ``NYU_depth.pgm`` sample from NYU RGBD dataset.

Open3D does not support ppm/pgm file yet.

Refer to ``Geometry/RGBD Images/NYU Dataset Tutorial``.

SampleRGBDImageSUN
------------------

Color image ``SUN_color.jpg`` and depth image ``SUN_depth.png`` sample from SUN RGBD dataset.

.. code-block:: python

    dataset = o3d.data.SampleRGBDImageSUN()
    color_raw = o3d.io.read_image(dataset.color_path)
    depth_raw = o3d.io.read_image(dataset.depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

.. code-block:: cpp

    data::SampleRGBDImageSUN dataset;

    geometry::Image color_raw, depth_raw;
    io::ReadImage(dataset.GetColorPath, color_raw);
    io::ReadImage(dataset.GetDepthPath, depth_raw);
    auto rgbd_image = geometry::RGBDImage::CreateFromSUNFormat(
                                color_raw, depth_raw,
                                /*convert_rgb_to_intensity =*/ False);

SampleRGBDImageTUM
------------------

Color image ``TUM_color.png`` and depth image ``TUM_depth.png`` sample from TUM RGBD dataset.

.. code-block:: python

    dataset = o3d.data.SampleRGBDImageTUM()
    color_raw = o3d.io.read_image(dataset.color_path)
    depth_raw = o3d.io.read_image(dataset.depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

.. code-block:: cpp

    data::SampleRGBDImageTUM dataset;

    geometry::Image color_raw, depth_raw;
    io::ReadImage(dataset.GetColorPath, color_raw);
    io::ReadImage(dataset.GetDepthPath, depth_raw);
    auto rgbd_image = geometry::RGBDImage::CreateFromTUMFormat(
                                color_raw, depth_raw,
                                /*convert_rgb_to_intensity =*/ False);

Image
~~~~~

JuneauImage
-----------

`JuneauImage` contains the ``JuneauImage.jpg`` file.

.. code-block:: python

    img_data = o3d.data.JuneauImage()
    img = o3d.io.read_image(img_data.path)

.. code-block:: cpp

    data::JuneauImage img_data;
    geometry::Image img;
    io::ReadImage(img_data.path, img);

Demo
~~~~

DemoICPPointClouds
------------------

3 point cloud fragments of binary PCD format, from living-room1 scene of Redwood RGB-D dataset.
This data is used for ICP demo.

.. code-block:: python

    dataset = o3d.data.DemoICPPointClouds()
    pcd0 = o3d.io.read_point_cloud(dataset.paths[0])
    pcd1 = o3d.io.read_point_cloud(dataset.paths[1])
    pcd2 = o3d.io.read_point_cloud(dataset.paths[2])

.. code-block:: cpp

    data::DemoICPPointClouds dataset;
    geometry::PointCloud pcd0, pcd1, pcd2;
    io::ReadPointCloud(dataset.GetPaths()[0], pcd0);
    io::ReadPointCloud(dataset.GetPaths()[1], pcd1);
    io::ReadPointCloud(dataset.GetPaths()[2], pcd2);

DemoColoredICPPointClouds
-------------------------

2 point cloud fragments of binary PCD format, from apartment scene of Redwood RGB-D dataset.
This data is used for Colored-ICP demo.

.. code-block:: python

    dataset = o3d.data.DemoColoredICPPointClouds()
    pcd0 = o3d.io.read_point_cloud(dataset.paths[0])
    pcd1 = o3d.io.read_point_cloud(dataset.paths[1])

.. code-block:: cpp

    data::DemoColoredICPPointClouds dataset;
    geometry::PointCloud pcd0, pcd1;
    io::ReadPointCloud(dataset.GetPaths()[0], pcd0);
    io::ReadPointCloud(dataset.GetPaths()[1], pcd1);

DemoCropPointCloud
------------------

Point cloud, and ``cropped.json`` (a saved selected polygon volume file).
This data is used for point cloud crop demo.

.. code-block:: python

    dataset = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.pointcloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(
                                dataset.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)

.. code-block:: cpp

    data::DemoCropPointCloud dataset;
    geometry::PointCloud pcd;
    io::ReadPointCloud(dataset.GetPointCloudPath(), pcd);
    SelectionPolygonVolume vol;
    io::ReadIJsonConvertible(dataset.GetCroppedJSONPath(), vol);
    auto chair = vol.CropPointCloud(pcd);

DemoPointCloudFeatureMatching
-----------------------------

Sample set of 2 point cloud fragments and their respective FPFH features and L32D features.
This data is used for point cloud feature matching demo.

.. code-block:: python

    dataset = o3d.data.DemoPointCloudFeatureMatching()

    pcd0 = o3d.io.read_point_cloud(dataset.pointcloud_paths[0])
    pcd1 = o3d.io.read_point_cloud(dataset.pointcloud_paths[1])

    fpfh_feature0 = o3d.io.read_feature(dataset.fpfh_feature_paths[0])
    fpfh_feature1 = o3d.io.read_feature(dataset.fpfh_feature_paths[1])

    l32d_feature0 = o3d.io.read_feature(dataset.l32d_feature_paths[0])
    l32d_feature1 = o3d.io.read_feature(dataset.l32d_feature_paths[1])

.. code-block:: cpp

    data::DemoPointCloudFeatureMatching dataset;

    auto pcd0 = io::CreatePointCloudFromFile(dataset.GetPointCloudPaths()[0]);
    auto pcd1 = io::CreatePointCloudFromFile(dataset.GetPointCloudPaths()[1]);

    pipelines::registration::Feature fpfh_feature0, fpfh_feature1;
    io::ReadFeature(dataset.GetFPFHFeaturePaths()[0], fpfh_feature0);
    io::ReadFeature(dataset.GetFPFHFeaturePaths()[1], fpfh_feature1);

    pipelines::registration::Feature l32d_feature0, l32d_feature1;
    io::ReadFeature(dataset.GetL32DFeaturePaths()[0], l32d_feature0);
    io::ReadFeature(dataset.GetL32DFeaturePaths()[1], l32d_feature1);

DemoPoseGraphOptimization
-------------------------

Sample fragment pose graph, and global pose graph. This data is used for pose graph optimization demo.

.. code-block:: python

    dataset = o3d.data.DemoPoseGraphOptimization()
    pose_graph_fragment = o3d.io.read_pose_graph(
                dataset.pose_graph_fragment_path)
    pose_graph_global = o3d.io.read_pose_graph(
                dataset.pose_graph_global_path)

.. code-block:: cpp

    data::DemoPoseGraphOptimization dataset;
    pipelines::registration::PoseGraph pose_graph_fragment;
    io::ReadPoseGraph(dataset.GetPoseGraphFragmentPath(),
                      pose_graph_fragment);

    pipelines::registration::PoseGraph pose_graph_global;
    io::ReadPoseGraph(dataset.GetPoseGraphGlobalPath(),
                      pose_graph_global);
