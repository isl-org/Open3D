.. _ray_casting_voxel_block_grid:

Ray Casting in a Voxel Block Grid
---------------------------------

.. note::
   This is NOT ray casting for triangle meshes. Please refer to :ref:`/python_api/open3d.t.geometry.RayCastingScene.rst` for that use case.

Ray casting can be performed in a voxel block grid to generate depth and color images at specific view points without extracting the entire surface. It is useful for frame-to-model tracking, and for differentiable volume rendering.

We provide optimized conventional rendering, and basic support for customized rendering that may be used in differentiable rendering. An example can be found at ``examples/python/t_reconstruction_system/ray_casting.py``.

Conventional rendering
``````````````````````
From a reconstructed voxel block grid from :ref:`optimized_integration`, we can efficiently render the scene given the input depth as a rough range estimate.

.. literalinclude:: ../../../examples/python/t_reconstruction_system/ray_casting.py
   :language: python
   :lineno-start: 76
   :lines: 27,77-92

The results could be directly obtained and visualized by

.. literalinclude:: ../../../examples/python/t_reconstruction_system/ray_casting.py
   :language: python
   :lineno-start: 90
   :lines: 27,91,93-95,105-112

Customized rendering
`````````````````````
In customized rendering, we manually perform trilinear-interpolation by accessing properties at 8 nearest neighbor voxels with respect to the found surface point per pixel:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/ray_casting.py
   :language: python
   :lineno-start: 97
   :lines: 27,98-103,114-115

Since the output is rendered via indices, the rendering process could be rewritten in differentiable engines like PyTorch seamlessly via :ref:`/tutorial/core/tensor.ipynb#PyTorch-I/O-with-DLPack-memory-map`.
