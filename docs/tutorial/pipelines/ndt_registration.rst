Normal Distributions Transform
==============================

Normal Distributions Transform (NDT) registration aligns a source point cloud to
a target point cloud represented as a voxel grid of local Gaussian
distributions. The method can be useful when a smooth target distribution is
preferred over point-to-point nearest-neighbor correspondences.

This implementation follows the Normal Distributions Transform introduced by
Biber and Straßer [BiberAndStrasser2003]_ and the 3D NDT formulation described
by Gao [Gao2023]_.

Open3D exposes NDT through
``open3d.pipelines.registration.registration_ndt``. The main parameters are
collected in ``NormalDistributionsTransformOption``, including both voxel
Gaussian model parameters and convergence criteria. Optimization stops when
the pose update is small or the relative change in mean Mahalanobis objective
falls below the configured threshold:

.. literalinclude:: ../../../examples/python/pipelines/ndt_registration.py
   :language: python
   :start-after: option =
   :end-before: print("Apply Normal Distributions Transform registration")

The full runnable script is available at
``examples/python/pipelines/ndt_registration.py``.
