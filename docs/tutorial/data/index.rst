.. _data:

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

`SamplePointCloudPCD` contains the `fragment.pcd` colored point cloud mesh from the `Redwood Living Room` dataset.

(See :ref:`/tutorial/geometry/pointcloud.html#Plane-segmentation` for reference)

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

`SamplePointCloudPLY` contains the `fragment.ply` colored point cloud mesh from the `Redwood Living Room` dataset.

(See :ref:`/tutorial/geometry/pointcloud.html#Visualize-point-cloud` for reference)

.. tabs::

    .. code-tab:: python

        sample_data = o3d.data.SamplePointCloudPCD()
        pcd = o3d.io.read_point_cloud(sample_data.path)

    .. code-tab:: cpp

        data::SamplePointCloudPCD sample_data();
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

`RedwoodLivingRoomPointClouds` contains 57 point clouds of binary PLY format, from Redwood RGB-D Dataset.

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

`RedwoodOfficePointClouds` contains 53 point clouds of binary PLY format, from Redwood RGB-D Dataset.

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


    .. code-block:: cpp

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

.. code-block:: python

        mesh_data = o3d.data.BunnyMesh()
        mesh = o3d.io.read_triangle_mesh(mesh_data.path)

.. code-block:: cpp

        data::BunnyMesh bunny_data();
        geometry::TriangleMesh mesh; 
        io::ReadTriangleMesh(bunny_data.path);


ArmadilloMesh
----------------------------------------

`ArmadilloMesh` contains the `ArmadilloMesh.ply` triangle mesh from Stanford University Computer Graphics Laboratory.

.. code-block:: python

        mesh_data = open3d.data.ArmadilloMesh()
        mesh = open3d.io.read_triangle_mesh(mesh_data.path)
        o3d.visualization.draw([mesh])


.. code-block:: cpp

        data::ArmadilloMesh armadillo_data();
        geometry::TriangleMesh mesh; 
        io::ReadTriangleMesh(armadillo_data.path);


KnotMesh
----------------------------------------

`KnotMesh` contains the `KnotMesh.ply` triangle mesh.

.. code-block:: python

        mesh_data = open3d.data.KnotMesh()
        mesh = open3d.io.read_triangle_mesh(mesh_data.path)
        o3d.visualization.draw([mesh])


.. code-block:: cpp

        data::KnotMesh knot_data();
        geometry::TriangleMesh mesh; 
        io::ReadTriangleMesh(knot_data.path);


RGB-D Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sample NYU RGB-D Dataset Image
----------------------------------------

Loading data:

.. code-block:: python

        rgbd_data = open3d.data.SampleRGBDImageNYU()
        color_raw = open3d.io.read_image(rgbd_data.color_path)
        depth_raw = open3d.io.read_image(rgbd_data.depth_path)


.. code-block:: cpp

        rgbd_data = open3d::data::SampleRGBDImageNYU()
        color_raw = open3d::io::read_image(rgbd_data.color_path)
        depth_raw = open3d::io::read_image(rgbd_data.depth_path)

Mirror(s):
    - `Mirror 1 <https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/fragment.ply>`_
Contents:
    fragment.ply
Source:
    Living Room point cloud fragment from Redwood RGB-D livingroom1 sequence.
Licence:
    Creative Commons 3.0 (CC BY 3.0).


Sample SUN RGB-D Dataset Image
----------------------------------------

Loading data:

.. code-block:: python

        rgbd_data = open3d.data.SampleRGBDImageSUN()
        color_raw = open3d.io.read_image(rgbd_data.color_path)
        depth_raw = open3d.io.read_image(rgbd_data.depth_path)


.. code-block:: cpp

        rgbd_data = open3d::data::SampleRGBDImageSUN()
        color_raw = open3d::io::read_image(rgbd_data.color_path)
        depth_raw = open3d::io::read_image(rgbd_data.depth_path)

Mirror(s):
    - `Mirror 1 <https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/fragment.ply>`_
Contents:
    fragment.ply
Source:
    Living Room point cloud fragment from Redwood RGB-D livingroom1 sequence.
Licence:
    Creative Commons 3.0 (CC BY 3.0).


Sample TUM RGB-D Dataset Image
----------------------------------------

Loading data:

.. code-block:: python

        rgbd_data = open3d.data.SampleRGBDImageTUM()
        color_raw = open3d.io.read_image(rgbd_data.color_path)
        depth_raw = open3d.io.read_image(rgbd_data.depth_path)


.. code-block:: cpp

        rgbd_data = open3d::data::SampleRGBDImageTUM()
        color_raw = open3d::io::read_image(rgbd_data.color_path)
        depth_raw = open3d::io::read_image(rgbd_data.depth_path)

Mirror(s):
    - `Mirror 1 <https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/fragment.ply>`_
Contents:
    fragment.ply
Source:
    Living Room point cloud fragment from Redwood RGB-D livingroom1 sequence.
Licence:
    Creative Commons 3.0 (CC BY 3.0).


Image Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Demo Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

