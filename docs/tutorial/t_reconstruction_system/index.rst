.. _t_reconstruction_system:

Reconstruction system (Tensor)
===================================================================

This tutorial demonstrates volumetric RGB-D reconstruction and dense RGB-D SLAM with the :ref:`/tutorial/core/tensor.ipynb` interface and the :ref:`/tutorial/core/hashmap.ipynb` backend.

The tutorial may run at a minimal dataset in ``examples/test_data/RGBD``, but it is recommended to run on real-world longer sequences to demonstrate the functionality. Please refer to :ref:`/tutorial/geometry/rgbd_image.ipynb` for available datasets. The ``Redwood`` dataset is recommended.

If you use any part of the Tensor-based reconstruction system, please cite [Dong2021]_::

  @article{Dong2021,
      author    = {Wei Dong, Yixing Lao, Michael Kaess, and Vladlen Koltun}
      title     = {{ASH}: A Modern Framework for Parallel Spatial Hashing in {3D} Perception},
      journal   = {arXiv:2110.00511},
      year      = {2021},
  }

.. note::
   The tutorials for tensor-based offline reconstruction system, simultaneous localization and calibration (SLAC), and shape from shading (SfS) as mentioned in [Dong2021]_ are under construction. At current, please refer to :ref:`reconstruction_system` for the legacy versions.

.. toctree::

   voxel_block_grid
   integration
   customized_integration
   ray_casting
   dense_slam
