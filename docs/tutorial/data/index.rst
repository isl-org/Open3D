.. _dataset:

Dataset
=======

Open3D comes with a built-in dataset module for convenient access to commonly
used example datasets. These datasets will be downloaded automatically from the
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

- Datasets are downloaded can cached automatically. The default data root is
  ``~/open3d_data``. Data will be downloaded to ``~/open3d_data/download``
  and extracted to ``~/open3d_data/extract``.
- Optionally, you can change the default data root. This can be done by setting
  the environment variable ``OPEN3D_DATA_ROOT`` or passing the ``data_root``
  argument when constructing a dataset object.

PointCloud
~~~~~~~~~~

PCDPointCloud
-------------

Colored point cloud of a living room from the Redwood dataset in PCD format.

.. code-block:: python

    dataset = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::PCDPointCloud dataset;
    auto pcd = io::CreatePointCloudFromFile(dataset.GetPath());

PLYPointCloud
-------------

Colored point cloud of a living room from the Redwood dataset in PLY format.

.. code-block:: python

    dataset = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::PLYPointCloud dataset;
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

LivingRoomPointClouds
---------------------

57 point clouds of binary PLY format from the Redwood RGB-D Dataset.

.. code-block:: python

    dataset = o3d.data.LivingRoomPointClouds()
    pcds = []
    for pcd_path in dataset.paths:
        pcds.append(o3d.io.read_point_cloud(pcd_path))

.. code-block:: cpp

    data::LivingRoomPointClouds dataset;
    std::vector<std::shared_ptr<geometry::PointCloud>> pcds;
    for (const std::string& pcd_path: dataset.GetPaths()) {
        pcds.push_back(io::CreatePointCloudFromFile(pcd_path));
    }

OfficePointClouds
-----------------

53 point clouds of binary PLY format from Redwood RGB-D Dataset.

.. code-block:: python

    dataset = o3d.data.OfficePointClouds()
    pcds = []
    for pcd_path in dataset.paths:
        pcds.append(o3d.io.read_point_cloud(pcd_path))

.. code-block:: cpp

    data::OfficePointClouds dataset;
    std::vector<std::shared_ptr<geometry::PointCloud>> pcds;
    for (const std::string& pcd_path: dataset.GetPaths()) {
        pcds.push_back(io::CreatePointCloudFromFile(pcd_path));
    }

TriangleMesh
~~~~~~~~~~~~

BunnyMesh
---------

The bunny triangle mesh from Stanford in PLY format.

.. code-block:: python

    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::BunnyMesh dataset;
    auto mesh = io::CreateMeshFromFile(dataset.GetPath());

ArmadilloMesh
-------------

The armadillo mesh from Stanford in PLY format.

.. code-block:: python

    dataset = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::ArmadilloMesh dataset;
    auto mesh = io::CreateMeshFromFile(dataset.GetPath());

KnotMesh
--------

A 3D Mobius knot mesh in PLY format.

.. code-block:: python

    dataset = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::KnotMesh dataset;
    auto mesh = io::CreateMeshFromFile(dataset.GetPath());

TriangleModel with PBR texture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MonkeyModel
-----------

The monkey model with PBR texture.

.. code-block:: python

    dataset = o3d.data.MonkeyModel()
    model = o3d.io.read_triangle_model(dataset.path)

.. code-block:: cpp

    data::BunnyMesh dataset;
    visualization::rendering::TriangleMeshModel model;
    auto model = io::ReadTriangleModel(dataset.GetPath(), model);

SwordModel
----------

The sword model with PBR texture.

.. code-block:: python

    dataset = o3d.data.SwordModel()
    model = o3d.io.read_triangle_model(dataset.path)

.. code-block:: cpp

    data::SwordModel dataset;
    visualization::rendering::TriangleMeshModel model;
    io::ReadTriangleModel(dataset.GetPath(), model);

CrateModel
----------

The crate model with PBR texture.

.. code-block:: python

    dataset = o3d.data.CrateModel()
    model = o3d.io.read_triangle_model(dataset.path)

.. code-block:: cpp

    data::CrateModel dataset;
    visualization::rendering::TriangleMeshModel model;
    io::ReadTriangleModel(dataset.GetPath(), model);

FlightHelmetModel
-----------------

The flight helmet gltf model with PBR texture.

.. code-block:: python

    dataset = o3d.data.FlightHelmetModel()
    model = o3d.io.read_triangle_model(dataset.path)

.. code-block:: cpp

    data::FlightHelmetModel dataset;
    visualization::rendering::TriangleMeshModel model;
    io::ReadTriangleModel(dataset.GetPath(), model);

AvocadoModel
------------

The Avocado glb model with PNG format embedded textures.

.. code-block:: python

    dataset = o3d.data.AvocadoModel()
    model = o3d.io.read_triangle_model(dataset.path)

.. code-block:: cpp

    data::AvocadoModel dataset;
    visualization::rendering::TriangleMeshModel model;
    io::ReadTriangleModel(dataset.GetPath(), model);

DamagedHelmetModel
------------------

The damaged helmet glb model with JPG format embedded textures.

.. code-block:: python

    dataset = o3d.data.DamagedHelmetModel()
    model = o3d.io.read_triangle_model(dataset.path)

.. code-block:: cpp

    data::DamagedHelmetModel dataset;
    visualization::rendering::TriangleMeshModel model;
    io::ReadTriangleModel(dataset.GetPath(), model);

Texture material images
~~~~~~~~~~~~~~~~~~~~~~~

MetalTexture
------------

Albedo, normal, roughness and metallic texture files for metal based material.

.. code-block:: python

    mat_data = o3d.data.MetalTexture()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)
    mat.metallic_img = o3d.io.read_image(mat_data.metallic_texture_path)

.. code-block:: cpp

    data::MetalTexture mat_data;

    auto mat = visualization::rendering::MaterialRecord();
    mat.shader = "defaultUnlit";
    mat.albedo_img = io::CreateImageFromFile(mat_data.albedo_texture_path);
    mat.normal_img = io::CreateImageFromFile(mat_data.normal_texture_path);
    mat.roughness_img = io::CreateImageFromFile(mat_data.roughness_texture_path);
    mat.metallic_img = io::CreateImageFromFile(mat_data.metallic_texture_path);

PaintedPlasterTexture
---------------------

Albedo, normal and roughness texture files for painted plaster based material.

.. code-block:: python

    mat_data = o3d.data.PaintedPlasterTexture()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

.. code-block:: cpp

    data::PaintedPlasterTexture mat_data;

    auto mat = visualization::rendering::MaterialRecord();
    mat.shader = "defaultUnlit";
    mat.albedo_img = io::CreateImageFromFile(mat_data.albedo_texture_path);
    mat.normal_img = io::CreateImageFromFile(mat_data.normal_texture_path);
    mat.roughness_img = io::CreateImageFromFile(mat_data.roughness_texture_path);

TilesTexture
------------

Albedo, normal and roughness texture files for tiles based material.

.. code-block:: python

    mat_data = o3d.data.TilesTexture()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

.. code-block:: cpp

    data::TilesTexture mat_data;

    auto mat = visualization::rendering::MaterialRecord();
    mat.shader = "defaultUnlit";
    mat.albedo_img = io::CreateImageFromFile(mat_data.albedo_texture_path);
    mat.normal_img = io::CreateImageFromFile(mat_data.normal_texture_path);
    mat.roughness_img = io::CreateImageFromFile(mat_data.roughness_texture_path);

TerrazzoTexture
---------------

Albedo, normal and roughness texture files for terrazzo based material.

.. code-block:: python

    mat_data = o3d.data.TerrazzoTexture()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

.. code-block:: cpp

    data::TerrazzoTexture mat_data;

    auto mat = visualization::rendering::MaterialRecord();
    mat.shader = "defaultUnlit";
    mat.albedo_img = io::CreateImageFromFile(mat_data.albedo_texture_path);
    mat.normal_img = io::CreateImageFromFile(mat_data.normal_texture_path);
    mat.roughness_img = io::CreateImageFromFile(mat_data.roughness_texture_path);

WoodTexture
-----------

Albedo, normal and roughness texture files for wood based material.

.. code-block:: python

    mat_data = o3d.data.WoodTexture()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

.. code-block:: cpp

    data::WoodTexture mat_data;

    auto mat = visualization::rendering::MaterialRecord();
    mat.shader = "defaultUnlit";
    mat.albedo_img = io::CreateImageFromFile(mat_data.albedo_texture_path);
    mat.normal_img = io::CreateImageFromFile(mat_data.normal_texture_path);
    mat.roughness_img = io::CreateImageFromFile(mat_data.roughness_texture_path);

WoodFloorTexture
----------------

Albedo, normal and roughness texture files for wooden floor based material.

.. code-block:: python

    mat_data = o3d.data.WoodFloorTexture()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

.. code-block:: cpp

    data::WoodFloorTexture mat_data;

    auto mat = visualization::rendering::MaterialRecord();
    mat.shader = "defaultUnlit";
    mat.albedo_img = io::CreateImageFromFile(mat_data.albedo_texture_path);
    mat.normal_img = io::CreateImageFromFile(mat_data.normal_texture_path);
    mat.roughness_img = io::CreateImageFromFile(mat_data.roughness_texture_path);

Image
~~~~~

JuneauImage
-----------

The RGB image ``JuneauImage.jpg`` file.

.. code-block:: python

    img_data = o3d.data.JuneauImage()
    img = o3d.io.read_image(img_data.path)

.. code-block:: cpp

    data::JuneauImage img_data;
    auto img = io::CreateImageFromFile(img_data.path);

RGBDImage
~~~~~~~~~

SampleRedwoodRGBDImages
-----------------------

Sample set of 5 color images, 5 depth images from the Redwood RGBD
living-room1 dataset. It also contains a camera trajectory log, a camera
odometry log, an rgbd match file, and a point cloud reconstruction obtained from
TSDF.

.. code-block:: python

    dataset = o3d.data.SampleRedwoodRGBDImages()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        color_raw = o3d.io.read_image(dataset.color_paths[i])
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                                   color_raw, depth_raw)
        rgbd_images.append(rgbd_image)

    pcd = o3d.io.read_point_cloud(dataset.reconstruction_path)

.. code-block:: cpp

    data::SampleRedwoodRGBDImages dataset;

    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for(size_t i = 0; i < dataset.GetDepthPaths().size(); ++i) {
        auto color_raw = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto depth_raw = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color_raw, *depth_raw,
                /*depth_scale =*/1000.0,
                /*depth_trunc =*/3.0,
                /*convert_rgb_to_intensity =*/false);
        rgbd_images.push_back(rgbd_image);
    }

    auto pcd = io::CreatePointCloudFromFile(dataset.GetReconstructionPath());

SampleFountainRGBDImages
------------------------

Sample set of 33 color and depth images from the Fountain RGBD dataset.
It also contains camera poses at key frames log and mesh reconstruction.

.. code-block:: python

    dataset = o3d.data.SampleFountainRGBDImages()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        depth = o3d.io.read_image(dataset.depth_paths[i])
        color = o3d.io.read_image(dataset.color_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                           color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
                              dataset.keyframe_poses_log_path)
    mesh = o3d.io.read_triangle_mesh(dataset.reconstruction_path)

.. code-block:: cpp

    data::SampleFountainRGBDImages dataset;

    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for(size_t i = 0; i < dataset.GetDepthPaths().size(); ++i) {
        auto color_raw = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto depth_raw = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color_raw, *depth_raw,
                /*depth_scale =*/1000.0,
                /*depth_trunc =*/3.0,
                /*convert_rgb_to_intensity =*/false);
        rgbd_images.push_back(rgbd_image);
    }

    camera::PinholeCameraTrajectory camera_trajectory;
    io::ReadPinholeCameraTrajectory(dataset.GetKeyframePosesLogPath(),
                                    camera_trajectory);
    auto mesh = io::CreateMeshFromFile(dataset.GetReconstructionPath());

SampleNYURGBDImage
------------------

Color image ``NYU_color.ppm`` and depth image ``NYU_depth.pgm`` sample from NYU
RGBD dataset.

.. code-block:: python

    import matplotlib.image as mpimg

    def read_nyu_pgm(filename, byteorder='>'):
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)
        img = np.frombuffer(buffer,
                            dtype=byteorder + 'u2',
                            count=int(width) * int(height),
                            offset=len(header)).reshape((int(height), int(width)))
        img_out = img.astype('u2')
        return img_out

    dataset = o3d.data.SampleNYURGBDImage()
    color_raw = mpimg.imread(dataset.color_path)
    depth_raw = read_nyu_pgm(dataset.depth_path)
    color = o3d.geometry.Image(color_raw)
    depth = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(
        color, depth, convert_rgb_to_intensity=False)

SampleSUNRGBDImage
------------------

Color image ``SUN_color.jpg`` and depth image ``SUN_depth.png`` sample from SUN
RGBD dataset.

.. code-block:: python

    dataset = o3d.data.SampleSUNRGBDImage()
    color_raw = o3d.io.read_image(dataset.color_path)
    depth_raw = o3d.io.read_image(dataset.depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

.. code-block:: cpp

    data::SampleSUNRGBDImage dataset;

    auto color_raw = io::CreateImageFromFile(dataset.GetColorPath());
    auto depth_raw = io::CreateImageFromFile(dataset.GetDepthPath());

    auto rgbd_image = geometry::RGBDImage::CreateFromSUNFormat(
        *color_raw, *depth_raw, /*convert_rgb_to_intensity =*/ false);

SampleTUMRGBDImage
------------------

Color image ``TUM_color.png`` and depth image ``TUM_depth.png`` sample from TUM
RGBD dataset.

.. code-block:: python

    dataset = o3d.data.SampleTUMRGBDImage()
    color_raw = o3d.io.read_image(dataset.color_path)
    depth_raw = o3d.io.read_image(dataset.depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

.. code-block:: cpp

    data::SampleTUMRGBDImage dataset;

    auto color_raw = io::CreateImageFromFile(dataset.GetColorPath());
    auto depth_raw = io::CreateImageFromFile(dataset.GetDepthPath());
    auto rgbd_image = geometry::RGBDImage::CreateFromTUMFormat(
        *color_raw, *depth_raw, /*convert_rgb_to_intensity =*/ false);

LoungeRGBDImages
------------------

Lounge RGBD dataset from Stanford containing ``color`` and ``depth``
sequence of 3000 images, along with ``camera trajectory`` and ``reconstruction``.

.. code-block:: python

    dataset = o3d.data.LoungeRGBDImages()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        color_raw = o3d.io.read_image(dataset.color_paths[i])
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                                   color_raw, depth_raw)
        rgbd_images.append(rgbd_image)

    mesh = o3d.io.read_triangle_mesh(dataset.reconstruction_path)

.. code-block:: cpp

    data::LoungeRGBDImages dataset;

    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for(size_t i = 0; i < dataset.GetDepthPaths().size(); ++i) {
        auto color_raw = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto depth_raw = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color_raw, *depth_raw,
                /*depth_scale =*/1000.0,
                /*depth_trunc =*/3.0,
                /*convert_rgb_to_intensity =*/false);
        rgbd_images.push_back(rgbd_image);
    }

    auto mesh = io::CreateTriangleMeshFromFile(dataset.GetReconstructionPath());

BedroomRGBDImages
------------------

Bedroom RGBD dataset from Redwood containing ``color`` and ``depth``
sequence of 21931 images, along with ``camera trajectory`` and ``reconstruction``.

.. code-block:: python

    dataset = o3d.data.BedroomRGBDImages()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        color_raw = o3d.io.read_image(dataset.color_paths[i])
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                                   color_raw, depth_raw)
        rgbd_images.append(rgbd_image)

    mesh = o3d.io.read_triangle_mesh(dataset.reconstruction_path)

.. code-block:: cpp

    data::BedroomRGBDImages dataset;

    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for(size_t i = 0; i < dataset.GetDepthPaths().size(); ++i) {
        auto color_raw = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto depth_raw = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color_raw, *depth_raw,
                /*depth_scale =*/1000.0,
                /*depth_trunc =*/3.0,
                /*convert_rgb_to_intensity =*/false);
        rgbd_images.push_back(rgbd_image);
    }

    auto mesh = io::CreateTriangleMeshFromFile(dataset.GetReconstructionPath());

Demo
~~~~

DemoICPPointClouds
------------------

3 point cloud fragments of binary PCD format, from living-room1 scene of Redwood
RGB-D dataset. This data is used for ICP demo.

.. code-block:: python

    dataset = o3d.data.DemoICPPointClouds()
    pcd0 = o3d.io.read_point_cloud(dataset.paths[0])
    pcd1 = o3d.io.read_point_cloud(dataset.paths[1])
    pcd2 = o3d.io.read_point_cloud(dataset.paths[2])

.. code-block:: cpp

    data::DemoICPPointClouds dataset;
    auto pcd0 = io::CreatePointCloudFromFile(dataset.GetPaths()[0]);
    auto pcd1 = io::CreatePointCloudFromFile(dataset.GetPaths()[1]);
    auto pcd2 = io::CreatePointCloudFromFile(dataset.GetPaths()[2]);

DemoColoredICPPointClouds
-------------------------

2 point cloud fragments of binary PCD format, from apartment scene of Redwood
RGB-D dataset. This data is used for Colored-ICP demo.

.. code-block:: python

    dataset = o3d.data.DemoColoredICPPointClouds()
    pcd0 = o3d.io.read_point_cloud(dataset.paths[0])
    pcd1 = o3d.io.read_point_cloud(dataset.paths[1])

.. code-block:: cpp

    data::DemoColoredICPPointClouds dataset;
    auto pcd0 = io::CreatePointCloudFromFile(dataset.GetPaths()[0]);
    auto pcd1 = io::CreatePointCloudFromFile(dataset.GetPaths()[1]);

DemoCropPointCloud
------------------

Point cloud and ``cropped.json`` (a saved selected polygon volume file).
This data is used for point cloud crop demo.

.. code-block:: python

    dataset = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(dataset.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)

.. code-block:: cpp

    data::DemoCropPointCloud dataset;
    auto pcd = io::CreatePointCloudFromFile(dataset.GetPointCloudPath());
    visualization::SelectionPolygonVolume vol;
    io::ReadIJsonConvertible(dataset.GetCroppedJSONPath(), vol);
    auto chair = vol.CropPointCloud(*pcd);

DemoFeatureMatchingPointClouds
------------------------------

Sample set of 2 point cloud fragments and their respective FPFH features and
L32D features. This data is used for point cloud feature matching demo.

.. code-block:: python

    dataset = o3d.data.DemoFeatureMatchingPointClouds()

    pcd0 = o3d.io.read_point_cloud(dataset.point_cloud_paths[0])
    pcd1 = o3d.io.read_point_cloud(dataset.point_cloud_paths[1])

    fpfh_feature0 = o3d.io.read_feature(dataset.fpfh_feature_paths[0])
    fpfh_feature1 = o3d.io.read_feature(dataset.fpfh_feature_paths[1])

    l32d_feature0 = o3d.io.read_feature(dataset.l32d_feature_paths[0])
    l32d_feature1 = o3d.io.read_feature(dataset.l32d_feature_paths[1])

.. code-block:: cpp

    data::DemoFeatureMatchingPointClouds dataset;

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

Sample fragment pose graph, and global pose graph. This data is used for pose
graph optimization demo.

.. code-block:: python

    dataset = o3d.data.DemoPoseGraphOptimization()
    pose_graph_fragment = o3d.io.read_pose_graph(dataset.pose_graph_fragment_path)
    pose_graph_global = o3d.io.read_pose_graph(dataset.pose_graph_global_path)

.. code-block:: cpp

    data::DemoPoseGraphOptimization dataset;
    auto pose_graph_fragment = io::CreatePoseGraphFromFile(
                        dataset.GetPoseGraphFragmentPath());
    auto pose_graph_global = io::CreatePoseGraphFromFile(
                        dataset.GetPoseGraphGlobalPath());

RedwoodIndoorLivingRoom1
------------------------

The living room 1 scene of the Redwood indoor dataset (Augmented ICL-NUIM
Dataset). The dataset contains a dense point cloud, a rgb sequence, a clean
depth sequence, a noisy depth sequence, an oni file, and camera trajectory.


.. code-block:: python

    dataset = o3d.data.RedwoodIndoorLivingRoom1()
    assert Path(gt_download_dir).is_dir()
    pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)

    im_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_rgbds.append(im_rgbd)

    im_noisy_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths,
                                      dataset.noisy_depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_noisy_rgbds.append(im_rgbd)

.. code-block:: cpp

    data::RedwoodIndoorLivingRoom1 dataset;

    auto pcd = io::CreatePointCloudFromFile(dataset.GetPointCloudPath());

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_rgbds.push_back(im_rgbd);
    }

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_noisy_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth =
                io::CreateImageFromFile(dataset.GetNoisyDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_noisy_rgbds.push_back(im_rgbd);
    }

RedwoodIndoorLivingRoom2
------------------------

The living room 2 scene of the Redwood indoor dataset (Augmented ICL-NUIM
Dataset). The dataset contains a dense point cloud, a rgb sequence, a clean
depth sequence, a noisy depth sequence, an oni file, and camera trajectory.


.. code-block:: python

    dataset = o3d.data.RedwoodIndoorLivingRoom2()
    assert Path(gt_download_dir).is_dir()
    pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)

    im_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_rgbds.append(im_rgbd)

    im_noisy_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths,
                                      dataset.noisy_depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_noisy_rgbds.append(im_rgbd)

.. code-block:: cpp

    data::RedwoodIndoorLivingRoom2 dataset;

    auto pcd = io::CreatePointCloudFromFile(dataset.GetPointCloudPath());

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_rgbds.push_back(im_rgbd);
    }

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_noisy_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth =
                io::CreateImageFromFile(dataset.GetNoisyDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_noisy_rgbds.push_back(im_rgbd);
    }


RedwoodIndoorOffice1
--------------------

The office 1 scene of the Redwood indoor dataset (Augmented ICL-NUIM
Dataset). The dataset contains a dense point cloud, a rgb sequence, a clean
depth sequence, a noisy depth sequence, an oni file, and camera trajectory.


.. code-block:: python

    dataset = o3d.data.RedwoodIndoorOffice1()
    assert Path(gt_download_dir).is_dir()
    pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)

    im_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_rgbds.append(im_rgbd)

    im_noisy_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths,
                                      dataset.noisy_depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_noisy_rgbds.append(im_rgbd)

.. code-block:: cpp

    data::RedwoodIndoorOffice1 dataset;

    auto pcd = io::CreatePointCloudFromFile(dataset.GetPointCloudPath());

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_rgbds.push_back(im_rgbd);
    }

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_noisy_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth =
                io::CreateImageFromFile(dataset.GetNoisyDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_noisy_rgbds.push_back(im_rgbd);
    }

RedwoodIndoorOffice2
--------------------

The office 2 scene of the Redwood indoor dataset (Augmented ICL-NUIM
Dataset). The dataset contains a dense point cloud, a rgb sequence, a clean
depth sequence, a noisy depth sequence, an oni file, and camera trajectory.


.. code-block:: python

    dataset = o3d.data.RedwoodIndoorOffice2()
    assert Path(gt_download_dir).is_dir()
    pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)

    im_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_rgbds.append(im_rgbd)

    im_noisy_rgbds = []
    for color_path, depth_path in zip(dataset.color_paths,
                                      dataset.noisy_depth_paths):
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth)
        im_noisy_rgbds.append(im_rgbd)

.. code-block:: cpp

    data::RedwoodIndoorOffice2 dataset;

    auto pcd = io::CreatePointCloudFromFile(dataset.GetPointCloudPath());

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth = io::CreateImageFromFile(dataset.GetDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_rgbds.push_back(im_rgbd);
    }

    std::vector<std::shared_ptr<geometry::RGBDImage>> im_noisy_rgbds;
    for (size_t i = 0; i < dataset.GetColorPaths().size(); ++i) {
        auto im_color = io::CreateImageFromFile(dataset.GetColorPaths()[i]);
        auto im_depth =
                io::CreateImageFromFile(dataset.GetNoisyDepthPaths()[i]);
        auto im_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*im_color,
                                                                    *im_depth);
        im_noisy_rgbds.push_back(im_rgbd);
    }
