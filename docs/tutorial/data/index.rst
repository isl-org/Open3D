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
    auto mesh = io::CreatePointCloudFromFile(dataset.GetPath());

ArmadilloMesh
-------------

The armadillo mesh from Stanford in PLY format.

.. code-block:: python

    dataset = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::ArmadilloMesh dataset;
    auto mesh = io::CreatePointCloudFromFile(dataset.GetPath());

KnotMesh
--------

A 3D Mobius knot mesh in PLY format.

.. code-block:: python

    dataset = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::KnotMesh dataset;
    auto mesh = io::CreatePointCloudFromFile(dataset.GetPath());

TODO: @Rishabh, update the documentation below.

RGBD Image
~~~~~~~~~~

SampleRGBDDatasetRedwood
------------------------

Data from Redwood RGBD living-room1. It contains 5 color images, 5 depth images,
a camera trajectory log, a camera odometry log, a rgbd match file, and a
point cloud reconstruction obtained from TSDF.

TODO: Add code to show the path and how to load.

SampleFountainRGBDDataset
-------------------------

`SampleFountainRGBDDataset` contains a sample set of 33 color and depth images
from the ``Fountain RGBD dataset``. It also contains ``camera poses at keyframes
log`` and ``mesh reconstruction``. It is used in demo of ``Color Map Optimization``.

SampleRGBDImageNYU
------------------

`SampleRGBDImageNYU` contains a color image ``NYU_color.ppm`` and a depth image
``NYU_depth.pgm`` sample from NYU RGBD  dataset.

.. code-block:: python

    rgbd_data = o3d.data.SampleRGBDImageNYU()
    color_raw = o3d.io.read_image(rgbd_data.color_path)
    depth_raw = o3d.io.read_image(rgbd_data.depth_path)

.. code-block:: cpp

    data::SampleRGBDImageNYU rgbd_data;

    geometry::Image im_color;
    io::ReadImage(rgbd_data.color_path, im_color);

    geometry::Image im_depth;
    io::ReadImage(rgbd_data.depth_path, im_depth);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);

SampleRGBDImageSUN
------------------

`SampleRGBDImageSUN` contains a color image ``SUN_color.jpg`` and a depth image
``SUN_depth.png`` sample from SUN RGBD dataset.

.. code-block:: python

    rgbd_data = o3d.data.SampleRGBDImageSUN()
    color_raw = o3d.io.read_image(rgbd_data.color_path)
    depth_raw = o3d.io.read_image(rgbd_data.depth_path)

.. code-block:: cpp

    data::SampleRGBDImageSUN rgbd_data;

    geometry::Image im_color;
    io::ReadImage(rgbd_data.color_path, im_color);

    geometry::Image im_depth;
    io::ReadImage(rgbd_data.depth_path, im_depth);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);

SampleRGBDImageTUM
------------------

`SampleRGBDImageTUM` contains a color image ``TUM_color.png`` and a depth image
``TUM_depth.png`` sample from TUM RGBD dataset.

.. code-block:: python

    rgbd_data = o3d.data.SampleRGBDImageTUM()
    color_raw = o3d.io.read_image(rgbd_data.color_path)
    depth_raw = o3d.io.read_image(rgbd_data.depth_path)

.. code-block:: cpp

    data::SampleRGBDImageSUN rgbd_data;

    geometry::Image im_color;
    io::ReadImage(rgbd_data.color_path, im_color);

    geometry::Image im_depth;
    io::ReadImage(rgbd_data.depth_path, im_depth);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);

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

`DemoICPPointClouds` contains 3 point clouds of binary PCD format. This data is
used in Open3D for ICP demo.

DemoColoredICPPointClouds
-------------------------

`DemoColoredICPPointClouds` contains 2 point clouds of PLY format. This data is
used in Open3D for Colored-ICP demo.

DemoCropPointCloud
------------------

`DemoCropPointCloud` contains a point cloud, and ``cropped.json`` (a saved
selected polygon volume file). This data is used in Open3D for point cloud crop
demo.

DemoPointCloudFeatureMatching
-----------------------------

`DemoPointCloudFeatureMatching` contains 2 point cloud fragments and their
respective FPFH features and L32D features. This data is used in Open3D for
point cloud feature matching demo.

DemoPoseGraphOptimization
-------------------------

`DemoPoseGraphOptimization` contains an example fragment pose graph, and
global pose graph. This data is used in Open3D for pose graph optimization demo.
