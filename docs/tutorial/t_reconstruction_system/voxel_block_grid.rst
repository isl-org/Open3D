.. _voxel_block_grid:

Voxel Block Grid
-------------------------------------
A voxel block grid is a globally sparse and locally dense data structure to represent 3D scenes. It is globally sparse since 2D object surfaces are usually occupying a small portion of the 3D space; it is locally dense in order to represent contiguous surfaces.

To represent such a structure, we first coarsely divide the 3D space into **block** grids. Blocks containing surfaces are organized in a hash map by 3D coordinates (sparse globally), and are further divided into dense **voxels** that can be accessed by array indices (dense locally). The reason why we do not maintain a voxel hash map is that we can preserve the data locality instead of scattering adjacent data uniformly into the memory.

Please first check the :ref:`/tutorial/core/hashmap.ipynb`, especially section the :ref:`/tutorial/core/hashmap.ipynb#Multi-valued-hash-map` to acquire a basic understanding of the underlying data structure. Please refer to [Dong2021]_ for more explanation.

Construction
````````````````````
A voxel block grid can be constructed by:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/integrate.py
   :language: python
   :lineno-start: 56
   :lines: 27,57-74

In this example, the multi-value hash map has key shape shape ``(3,)`` and dtype ``int32``. The hash map values are organized as a structure of array (SoA). The hash map values include:

By default it contains:

- Truncated Signed Distance Function (TSDF) of element shape ``(8, 8, 8, 1)``
- Weight of element shape ``(8, 8, 8, 1)``
- (Optionally) RGB color of element shape ``(8, 8, 8, 3)``

where ``8`` stands for the local voxel block grid resolution.

By convention, we use ``3.0 / 512`` as the voxel resolution. This spatial resolution is equivalent to representing a ``3m x 3m x 3m`` (m = meter) room with a dense ``512 x 512 x 512`` voxel grid.

The voxel block grid is optimized to run fast on GPU. On CPU the it runs slower. Empirically, we reserve ``100000`` such blocks for a living room-scale scene to avoid frequent rehashing.

You can always customize your own properties, e.g., ``intensity`` of element shape ``(8, 8, 8, 1)`` in ``float32``, ``label`` of element shape ``(8, 8, 8, 1)`` in ``int32``, etc. To know how to process the data, please refer to :ref:`customized_integration`.
