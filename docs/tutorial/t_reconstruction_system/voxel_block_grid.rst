.. _voxel_block_grid

Voxel Block Grid
===================================================================
A voxel block grid is a globally sparse and locally dense data structure in 3D.
It can be initialized by:
.. code-block:: python
    vbg = o3d.t.geometry.VoxelBlockGrid(('tsdf', 'weight', 'color'),
                                        (o3c.float32, o3c.float32, o3c.float32),
                                        ((1), (1), (3)),
                                        3.0 / 512,
                                        8,
                                        100000,
                                        o3d.core.Device('CUDA:0'))

Formulation
````````````````````
Locally, a voxel block is an plain 3D array that can be accessed by simple indexing. The 3D array 
Globally, these blocks are allocated on demand around surfaces, which are comparatively sparse in the 3D space. We use a hash map to store such representation.
