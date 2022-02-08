.. _data:

Data
=========

Open3D provides `open3d.data` module for convenient access to built-in
example and test data. You'll need internet access to use the data module.
The downloaded data will be stored in the Open3D's data root directory.

A dataset class locates the data root directory in the following order:

1. User-specified by ``data_root`` when instantiating a dataset object.
2. OPEN3D_DATA_ROOT environment variable.
3. $HOME/open3d_data.

By default, (3) will be used, and it is also the recommended way.

.. tabs::
    .. code-tab:: python

        import open3d as o3d

        # The default data_root for the following example is `$HOME/open3d_data`.
        eagle_data = o3d.data.EaglePointCloud()
        print("Prefix: ", eagle_data.prefix)

        # So, this will download the `EaglePointCloud.ply` data in 
        # `$HOME/open3d_data/download/EaglePointCloud/`, and copy the same to
        # `$HOME/open3d_data/extract/EaglePointCloud/`.
        print("Data root: ", eagle_data.data_root)
        print("Download dir: ", eagle_data.download_dir)
        print("Extract dir: ", eagle_data.extract_dir)

    .. code-tab:: cpp

        #include "open3d/Open3D.h"
        #include <iostream>
        using namespace open3d;

        // The default data_root for the following example is `$HOME/open3d_data`.
        data::EaglePointCloud eagle_data();
        std::cout << "Prefix: " << eagle_data.GetPrefix() << std::endl;

        // So, this will download the `EaglePointCloud.ply` data in 
        // `$HOME/open3d_data/download/EaglePointCloud/`, and copy the same to
        // `$HOME/open3d_data/extract/EaglePointCloud/`.
        std::cout << "Data root: " << eagle_data.GetDataRoot() << std::endl;
        std::cout << "Download dir: " << eagle_data.GetDownloadDir() << std::endl;
        std::cout << "Extract dir: " << eagle_data.GetExtractDir() << std::endl;

.. code-block:: bash

    [Open3D INFO] Downloading https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/EaglePointCloud.ply
    [Open3D INFO] Downloaded to /home/rey/open3d_data/download/EaglePointCloud/EaglePointCloud.ply
    Prefix:  EaglePointCloud
    Data root:  /home/rey/open3d_data
    Download dir:  /home/rey/open3d_data/download/EaglePointCloud
    Extract dir:  /home/rey/open3d_data/extract/EaglePointCloud


When a dataset object is instantiated, the corresponding data will be downloaded 
in ${data_root}/download/prefix/ and extracted or copied to ${data_root}/extract/prefix/. 
The default ${data_root} location is $HOME/open3d_data. If the extracted data 
directory exists, the files will be used without validation. If it does not 
exist, and the valid downloaded file exists, the data will be extracted from 
the downloaded file. If the downloaded file does not exist or validates against 
the provided MD5, it will be re-downloaded.

After the data is downloaded and extracted, the dataset object will NOT load the 
data for you. Instead, you will get the paths to the data files and use Open3D’s 
I/O functions to load the data. This design exposes where the data is stored and 
how the data is loaded, allowing users to modify the code and load their own data 
in a similar way. Please check the documentation of the specific dataset to know 
more about the specific functionalities provided for it.


Point Cloud Data
~~~~~~~~~~~~~~~~


SamplePointCloudPCD
-----------------------

`SamplePointCloudPCD` contains the `fragment.pcd` colored point cloud mesh from 
the `Redwood Living Room` dataset.

See :ref:`reference </tutorial/geometry/pointcloud.html#Plane-segmentation>`.

.. tabs::

    .. code-tab:: python

        sample_data = o3d.data.SamplePointCloudPCD()
        pcd = o3d.io.read_point_cloud(sample_data.path)

    .. code-tab:: cpp

        data::SamplePointCloudPCD sample_data();
        geometry::PointCloud pcd;
        io::ReadPointCloud(sample_data.path, pcd);


SamplePointCloudPLY
----------------------------------------

`SamplePointCloudPLY` contains the `fragment.ply` colored point cloud mesh from 
the `Redwood Living Room` dataset.

See :ref:`reference </tutorial/geometry/pointcloud.html#Visualize-point-cloud>`.

.. tabs::

    .. code-tab:: python

        sample_data = o3d.data.SamplePointCloudPLY()
        pcd = o3d.io.read_point_cloud(sample_data.path)

    .. code-tab:: cpp

        data::SamplePointCloudPLY sample_data();
        geometry::PointCloud pcd;
        io::ReadPointCloud(sample_data.path, pcd);


EaglePointCloud
----------------------------------------

`EaglePointCloud` contains the `EaglePointCloud.ply` colored point cloud mesh.

.. tabs::

    .. code-tab:: python

            eagle_data = o3d.data.EaglePointCloud()
            pcd = o3d.io.read_point_cloud(eagle_data.path)

    .. code-tab:: cpp

            data::EaglePointCloud eagle_data();
            geometry::PointCloud pcd;
            io::ReadPointCloud(eagle_data.path, pcd);


RedwoodLivingRoomPointClouds
----------------------------------------

`RedwoodLivingRoomPointClouds` contains 57 point clouds of binary PLY format, 
from Redwood RGB-D Dataset.

Content:

.. code-block:: bash

        livingroom1-fragments-ply.zip
                ├── cloud_bin_0.ply
                ├── cloud_bin_1.ply
                ├── ...
                └── cloud_bin_56.ply

`paths` returns the list of paths to these poin cloud fragments. 
Example: Use `paths[0]` to access `cloud_bin_0.ply`.

.. tabs::

    .. code-tab:: python

            pcd_fragments_data = o3d.data.RedwoodLivingRoomPointCloud()
            for path in pcd_fragments_data.paths:
                pcd = open3d.io.read_point_cloud(pcd_fragments_data.path)

    .. code-tab:: cpp

            data::RedwoodLivingRoomPointCloud pcd_fragments_data();
            for(const std::string& path : pcd_fragments_data.path) {
                geometry::PointCloud pcd;
                io::ReadPointCloud(path, pcd);
            }


RedwoodOfficePointClouds
----------------------------------------

`RedwoodOfficePointClouds` contains 53 point clouds of binary PLY format, 
from Redwood RGB-D Dataset.

Content:

.. code-block:: bash

        office1-fragments-ply.zip
                ├── cloud_bin_0.ply
                ├── cloud_bin_1.ply
                ├── ...
                └── cloud_bin_52.ply

`paths` returns the list of paths to these poin cloud fragments. 
Example: Use paths[0] to access `cloud_bin_0.ply`.

.. tabs::

    .. code-tab:: python

            pcd_fragments_data = o3d.data.RedwoodOfficePointCloud()
            for path in pcd_fragments_data.paths:
                pcd = open3d.io.read_point_cloud(pcd_fragments_data.path)
                o3d.visualization.draw([pcd])

    .. code-tab:: cpp

            data::RedwoodOfficePointClouds pcd_fragments_data();
            for(const std::string& path : pcd_fragments_data.path) {
                geometry::PointCloud pcd;
                io::ReadPointCloud(path, pcd);
            }


Triangle Mesh Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


BunnyMesh
----------------------------------------

`BunnyMesh` contains the `BunnyMesh.ply` triangle mesh from Stanford University Computer Graphics Laboratory.

See :ref:`reference </tutorial/geometry/mesh.html#Connected-components>`.

.. tabs::

    .. code-tab:: python

            mesh_data = o3d.data.BunnyMesh()
            mesh = o3d.io.read_triangle_mesh(mesh_data.path)

    .. code-tab:: cpp

            data::BunnyMesh bunny_data();
            geometry::TriangleMesh mesh; 
            io::ReadTriangleMesh(bunny_data.path);


ArmadilloMesh
----------------------------------------

`ArmadilloMesh` contains the `ArmadilloMesh.ply` triangle mesh from Stanford University Computer Graphics Laboratory.

See :ref:`reference </tutorial/geometry/pointcloud.html#Visualize-point-cloud>`.

.. tabs::

    .. code-tab:: python

            mesh_data = open3d.data.ArmadilloMesh()
            mesh = open3d.io.read_triangle_mesh(mesh_data.path)
            o3d.visualization.draw([mesh])


    .. code-tab:: cpp

            data::ArmadilloMesh armadillo_data();
            geometry::TriangleMesh mesh; 
            io::ReadTriangleMesh(armadillo_data.path);


KnotMesh
----------------------------------------

`KnotMesh` contains the `KnotMesh.ply` triangle mesh.

See :ref:`reference </tutorial/geometry/mesh.html#Mesh>`.

.. tabs::

    .. code-tab:: python

        mesh_data = open3d.data.KnotMesh()
        mesh = open3d.io.read_triangle_mesh(mesh_data.path)
        o3d.visualization.draw([mesh])


    .. code-tab:: cpp

        data::KnotMesh knot_data();
        geometry::TriangleMesh mesh; 
        io::ReadTriangleMesh(knot_data.path);


RGB-D Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


SampleRGBDDatasetRedwood
----------------------------------------

`SampleRGBDDatasetRedwood` contains a sample set of 5 color and depth images from Redwood RGBD dataset living-room1. Additionally it also contains camera trajectory log, camera odometry log, rgbd match, and point cloud reconstruction obtained using TSDF.

See :ref:`reference </tutorial/geometry/rgbd_image.html#Redwood-dataset>`.


SampleFountainRGBDDataset
----------------------------------------

`SampleFountainRGBDDataset` contains a sample set of 33 color and depth images from the `Fountain RGBD dataset`. It also contains `camera poses at keyframes log` and `mesh reconstruction`. It is used in demo of `Color Map Optimization`.

See :ref:`reference </tutorial/pipelines/color_map_optimization.html#Input>`.


SampleRGBDImageNYU
----------------------------------------

`SampleRGBDImageNYU` contains a color image `NYU_color.ppm` and a depth image `NYU_depth.pgm` sample from NYU RGBD  dataset.

See :ref:`reference </tutorial/geometry/rgbd_image.html#NYU-dataset>`.

.. tabs::

    .. code-tab:: python

            rgbd_data = o3d.data.SampleRGBDImageNYU()
            color_raw = o3d.io.read_image(rgbd_data.color_path)
            depth_raw = o3d.io.read_image(rgbd_data.depth_path)


    .. code-tab:: cpp

            data::SampleRGBDImageNYU rgbd_data();

            geometry::Image im_color;
            io::ReadImage(rgbd_data.color_path, im_color);

            geometry::Image im_depth;
            io::ReadImage(rgbd_data.depth_path, im_depth);

            std::shared_ptr<geometry::RGBDImage> im_rgbd = 
                    geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);


SampleRGBDImageSUN
----------------------------------------

`SampleRGBDImageSUN` contains a color image `SUN_color.jpg` and a depth image 
`SUN_depth.png` sample from SUN RGBD dataset.

See :ref:`reference </tutorial/geometry/rgbd_image.html#SUN-dataset>`.

.. tabs::

    .. code-tab:: python

            rgbd_data = open3d.data.SampleRGBDImageSUN()
            color_raw = open3d.io.read_image(rgbd_data.color_path)
            depth_raw = open3d.io.read_image(rgbd_data.depth_path)


    .. code-tab:: cpp

            data::SampleRGBDImageSUN rgbd_data();

            geometry::Image im_color;
            io::ReadImage(rgbd_data.color_path, im_color);

            geometry::Image im_depth;
            io::ReadImage(rgbd_data.depth_path, im_depth);

            std::shared_ptr<geometry::RGBDImage> im_rgbd = 
                    geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);


SampleRGBDImageTUM
----------------------------------------

`SampleRGBDImageTUM` contains a color image `TUM_color.png` and a depth image 
`TUM_depth.png` sample from TUM RGBD dataset.

See :ref:`reference </tutorial/geometry/rgbd_image.html#TUM-dataset>`.

.. tabs::

    .. code-tab:: python

            rgbd_data = open3d.data.SampleRGBDImageTUM()
            color_raw = open3d.io.read_image(rgbd_data.color_path)
            depth_raw = open3d.io.read_image(rgbd_data.depth_path)

    .. code-tab:: cpp

            data::SampleRGBDImageSUN rgbd_data();

            geometry::Image im_color;
            io::ReadImage(rgbd_data.color_path, im_color);

            geometry::Image im_depth;
            io::ReadImage(rgbd_data.depth_path, im_depth);

            std::shared_ptr<geometry::RGBDImage> im_rgbd = 
                    geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);


Image Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


JuneauImage
----------------------------------------

`JuneauImage` contains the `JuneauImage.jpg` file.

See :ref:`reference </tutorial/geometry/file_io.html#Image>`.

.. tabs::

    .. code-tab:: python

            img_data = o3d.data.JuneauImage()
            img = o3d.io.read_image(img_data.path)

    .. code-tab:: python

            data::JuneauImage img_data();
            geometry::Image img;
            io::ReadImage(img_data.path, img);


Demo Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


DemoICPPointClouds
----------------------------------------

`DemoICPPointClouds` contains 3 point clouds of binary PCD format. This data is 
used in Open3D for ICP demo.

See :ref:`reference </tutorial/pipelines/icp_registration.html#Input>`.


DemoColoredICPPointClouds
----------------------------------------

`DemoColoredICPPointClouds` contains 2 point clouds of PLY format. This data is 
used in Open3D for Colored-ICP demo.

See :ref:`reference </tutorial/pipelines/colored_pointcloud_registration.html#Input>`.


DemoCropPointCloud
----------------------------------------

`DemoCropPointCloud` contains a point cloud, and `cropped.json` (a saved 
selected polygon volume file). This data is used in Open3D for point cloud crop 
demo.

See :ref:`reference </tutorial/geometry/pointcloud.html#Crop-point-cloud>`.


DemoPointCloudFeatureMatching
----------------------------------------

`DemoPointCloudFeatureMatching` contains 2 point cloud fragments and their 
respective FPFH features and L32D features. This data is used in Open3D for 
point cloud feature matching demo.


DemoPoseGraphOptimization
----------------------------------------

`DemoPoseGraphOptimization` contains an example fragment pose graph, and 
global pose graph. This data is used in Open3D for pose graph optimization demo.

