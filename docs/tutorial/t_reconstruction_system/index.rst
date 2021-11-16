.. _t_reconstruction_system:

Reconstruction system (Tensor)
===================================================================

This tutorial demonstrates volumetric RGB-D reconstruction and dense RGB-D SLAM with the Open3D :ref:`/tutorial/core/tensor.ipynb` interface and the Open3D :ref:`/tutorial/core/hashmap.ipynb` backend.

It is possible to run the tutorial with the minimalistic dataset in ``examples/test_data/RGBD``, but it is recommended to run the tutorial with real-world datasets with longer sequences to demonstrate its capability. Please refer to :ref:`/tutorial/geometry/rgbd_image.ipynb` for more available datasets. The ``Redwood`` dataset can be a good starting point.

If you use any part of the tensor-based reconstruction system or the hash map backend in Open3D, please cite [Dong2021]_::

  @article{Dong2021,
      author    = {Wei Dong, Yixing Lao, Michael Kaess, and Vladlen Koltun}
      title     = {{ASH}: A Modern Framework for Parallel Spatial Hashing in {3D} Perception},
      journal   = {arXiv:2110.00511},
      year      = {2021},
  }

.. note::
   As of now the tutorial is only for **online** dense SLAM, and **offline** integration **with** provided poses. The tutorials for tensor-based **offline** reconstruction system, Simultaneous localization and calibration (SLAC), and shape from shading (SfS) tutorials as mentioned in [Dong2021]_ are still under construction. At current, please refer to :ref:`reconstruction_system` for the legacy versions.

.. toctree::

   voxel_block_grid
   integration
   customized_integration
   ray_casting
   dense_slam
