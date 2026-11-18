Geometry
========

.. toctree::
    :caption: Basics

    pointcloud
    mesh
    rgbd_image
    kdtree

.. toctree::
    :caption: Processing

    file_io
    pointcloud_outlier_removal
    voxelization
    octree
    surface_reconstruction
    transformation
    mesh_deformation
    iss_keypoint_detector
    ray_casting
    distance_queries
    uvmaps

.. toctree::
    :caption: Interface

    python_interface
    working_with_numpy

File path types
---------------

File IO functions accept filenames as ``str``, ``bytes``, or ``os.PathLike``,
including ``pathlib.Path``. Their generated signatures preserve this union and
describe accepted string filenames alongside ``os.PathLike`` inputs.

.. code-block:: python

    from pathlib import Path
    import tempfile
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
    with tempfile.TemporaryDirectory() as folder:
        filename = Path(folder) / "cloud.ply"
        assert o3d.io.write_point_cloud(filename, cloud)
        restored = o3d.io.read_point_cloud(str(filename))
        assert len(restored.points) == 1
