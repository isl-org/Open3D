.. _customized_integration:

Customized Integration
-------------------------------------
You can prototype a new RGB-D volumetric reconstruction algorithm with additional properties (e.g. semantic labels) while maintaining a reasonable performance. An example can be found at ``examples/python/t_reconstruction_system/integrate_custom.py``.

Activation
``````````
The frustum **block** selection remains the same, but then we manually activate these blocks and obtain their buffer indices in the :ref:`/tutorial/core/hashmap.ipynb`:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 78
   :lines: 27,79-87

Voxel Indices
``````````````
We can then unroll **voxel** indices in these **blocks** into a flattened array, along with their corresponding voxel coordinates.

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 91
   :lines: 27,92-93

Up to now we have finished preparation. Then we can perform customized geometry transformation in the Tensor interface, with the same fashion as we conduct in numpy or pytorch.

Geometry transformation
````````````````````````
We first transform the voxel coordinates to the frame's coordinate system, project them to the image space, and filter out-of-bound correspondences:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 99
   :lines: 27,100-118

Customized integration
````````````````````````
With the data association, we are able to conduct integration. In this example, we show the conventional TSDF integration written in vectorized Python code:

- Read the associated RGB-D properties from the color/depth images at the associated ``u, v`` indices;
- Read the voxels from the voxel buffer arrays (``vbg.attribute``) at masked ``voxel_indices``;
- Perform in-place modification

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 118
   :lines: 27,119-128,133-151

You may follow the example and adapt it to your customized properties. Open3D supports conversion from and to PyTorch tensors without memory any copy, see :ref:`/tutorial/core/tensor.ipynb#PyTorch-I/O-with-DLPack-memory-map`. This can be use to leverage PyTorch's capabilities such as automatic differentiation and other operators.
