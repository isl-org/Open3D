.. _voxel_block_grid:

Voxel Block Grid
-------------------------------------
A voxel block grid is a globally sparse and locally dense data structure to represent 3D scenes.
It is globally sparse since 2D object surfaces are usually occupying a small portion of the 3D space; it is locally dense as physical surfaces are often contiguous.

To represent such a structure, we first coarsely divide the 3D space into **block** grids. Blocks containing surfaces are organized in a hash map by 3D coordinates (sparse globally), and are further divided into dense **voxels** that can be accessed by array indices (dense locally). The reason why we do not maintain a voxel hash map is that we can preserve the data locality instead of scattering adjacent data uniformly into the memory.

Please first check :ref:`/tutorial/core/hashmap.ipynb`, especially section :ref:`/tutorial/core/hashmap.ipynb#Multi-valued-hash-map` to acquire a basic understanding of the underlying data structure. Please refer to [Dong2021]_ for more explanation.

Construction
````````````````````
A voxel block grid can be constructed by

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate.py
   :language: python
   :lineno-start: 52
   :lines: 52-60
   :linenos:
   :dedent:


In this example, internally, the key of the hash map is trivially of element shape ``(3,)`` and dtype ``int32``.
The multiple values, on the other hand, are organized as a structure of array (SoA). It by default contains

- Truncated Signed Distance Function (TSDF) of element shape ``(8, 8, 8, 1)``,
- Weight of element shape ``(8, 8, 8, 1)``,
- (Optionally) RGB color of element shape ``(8, 8, 8, 3)``,

where ``8`` stands for the local voxel block grid resolution.

By convention we use ``3.0 / 512`` as the voxel resolution. This spatial resolution is equivalent to representing a ``3m x 3m x 3m`` room with a dense ``512x512x512`` voxel grid.

The voxel block grid is recommended to be used on GPU (optimized), but should also work on CPU (slow but works).
Empirically, we reserve ``100000`` such blocks for a living room-scale scene to avoid time and memory consuming rehashing. 

You can always customize your own properties, e.g., ``intensity`` of element shape ``(8, 8, 8, 1)`` in ``float32``, ``label`` of element shape ``(8, 8, 8, 1)`` in ``int32``, etc. To know how to process the data, please refer to :ref:`customized_integration`.
