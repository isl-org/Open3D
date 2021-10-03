.. _customized_integration:

Customized Integration
-------------------------------------
You can prototype a new RGB-D volumetric reconstruction algorithm with additional properties (e.g. semantic labels) while maintaining a reasonable performance.

Activation
``````````
The frustum **block** selection remains the same, but then we manually activate these blocks and obtain their buffer indices in the :ref:`/tutorial/core/hashmap.ipynb`:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 74
   :lines: 74-82
   :linenos:
   :dedent:

Voxel Indices
``````````
We can then unroll **voxel** indices in these **blocks** into a flattened array, along with their corresponding voxel coordinates.

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 87
   :lines: 87-88
   :linenos:
   :dedent:

Up to now we have finished preparation. Then we can perform customized geometry transformation in the Tensor interface, with the same fashion as we conduct in numpy or pytorch.

Geometry transformation
``````````
We first transform the voxel coordinates to the frame's coordinate system, project them to the image space, and filter out-of-bound correspondences:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 95
   :lines: 95-113
   :linenos:
   :dedent:

Customized integration
``````````
With the data association, we are able to conduct integration. In this example, we show the conventional TSDF integration written in vectorized python:

- Read the associated RGB-D properties from the color/depth images at the associated ``u, v`` indices;
- Read the voxels from the voxel buffer arrays (``vbg.attribute``) at masked `voxel_indices`;
- Perform in-place modification

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate_custom.py
   :language: python
   :lineno-start: 114
   :lines: 114-123,128-148
   :linenos:
   :dedent:

You may follow the example and adapt it to your customized properties. Note you can always reuse such buffers in PyTorch without copying using DLPack, see :ref:`/tutorial/core/tensor.ipynb#PyTorch-I/O-with-DLPack-memory-map`, for potentially differentiable operations and advanced operators that are at current not available in Open3D's tensor engine.
