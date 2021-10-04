.. _optimized_integration:

TSDF Integration
-------------------------------------
Truncated Signed Distance Function (TSDF) integration is the key of dense volumetric scene reconstruction. It receives relatively noisy depth images from RGB-D sensors such as Kinect and RealSense, and integrates depth readings into the :ref:`voxel_block_grid` given known camera poses. TSDF integration reduces noise and generates smooth surfaces. 

The integration process mainly consists of two steps, (sparse) **block** selection and activation, and (dense) **voxel** value integration. An example can be found at ``examples/python/t_reconstruction_system/integrate.py``.

The system may run at a minimal dataset in ``examples/test_data/RGBD``, but it is recommended to run on real-world longer sequences to demonstrate the functionality. Please refer to :ref:`tutorial/geometry/rgbd_image.ipynb` for more available datasets.

Activation
``````````
In the activation step, coordinates of **blocks** that contain measurements from the depth frame are selected via a frustum hash map, where duplicates are removed:


.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate.py
   :language: python
   :lineno-start: 69
   :lines: 69-71
   :linenos:
   :dedent:


Integration
``````````
Now we can process the voxels in the blocks at ``frustum_block_coords``. This is done by projecting all such related voxels to the input images and perform weight average. A 

Optimized function, along with raw depth images with calibration parameters to activate and perform TSDF integration, optionally with colors:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate.py
   :language: python
   :lineno-start: 73
   :lines: 73-79
   :linenos:
   :dedent:


At current, to use our optimized function, we assume the below combinations of data types, in the order of ``depth image``, ``color image``, ``tsdf in voxel grid``, ``weight in voxel grid``, ``color in voxel grid`` in CPU

.. literalinclude:: ../../../cpp/open3d/t/geometry/kernel/VoxelBlockGridCPU.cpp
   :language: cpp
   :lineno-start: 230
   :lines: 230-236
   :linenos:
   :dedent:

and CUDA

.. literalinclude:: ../../../cpp/open3d/t/geometry/kernel/VoxelBlockGridCUDA.cu
   :language: cpp
   :lineno-start: 256
   :lines: 256-262
   :linenos:
   :dedent:

To generalize the functionality, you may extend the macros and/or the kernel functions and compile from scratch achieve the maximal performance (~100Hz on a GTX 1070), or follow :ref:`customized_integration` and implement a fast prototype (~25Hz).

Surface extraction
``````````
You may use the provided APIs to extract surface points. 

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate.py
   :language: python
   :lineno-start: 126
   :lines: 126-130
   :linenos:
   :dedent:

Note ``extract_triangle_mesh`` applies marching cubes and generate mesh. ``extract_point_cloud`` uses the similar algorithm, but skips the triangle face generation step.


Save and load
``````````
The voxel block grids can be saved to and loaded from `.npz` files that are accessible via numpy.

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate.py
   :language: python
   :lineno-start: 43
   :lines: 43,84
   :linenos:
   :dedent:

The `.npz` file of the aforementioned voxel block grid contains the following entries:

- ``attr_name_tsdf``: stores the value buffer index: 0
- ``attr_name_weight``: stores the value buffer index: 1
- ``attr_name_color``: stores the value buffer index: 2
- ``value_000``: the tsdf value buffer
- ``value_001``: the weight value buffer
- ``value_002``: the color value buffer
- ``key``: all the active keys
- ``block_resolution``: 8
- ``voxel_size``: 0.0059 = 3.0 / 512
- ``CUDA:0``: the device

