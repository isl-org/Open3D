.. _voxel_block_grid

Voxel Block Grid
===================================================================
A voxel block grid is a globally sparse and locally dense data structure in 3D.

Formulation
````````````````````
Locally, a voxel block is an plain 3D array that can be accessed by simple indexing.
Globally, these blocks are allocated on demand around surfaces, which are comparatively sparse in the 3D space. We use a hash map to store such representation.
