.. _t_pipelines:

Pipelines (Tensor) 
==================

.. toctree::

    t_icp_registration
    t_robust_kernel

Point-to-plane error
-------------------

``TransformationEstimationPointToPlane.compute_rmse`` computes the root mean
square of each point displacement projected onto its corresponding target
normal. A displacement tangent to the target plane contributes zero error,
including when the plane is oblique to the coordinate axes. Normals are used
as provided, without normalization.

.. code-block:: python

    import open3d as o3d

    source = o3d.t.geometry.PointCloud(
        o3d.core.Tensor([[4.0, -3.0, 0.0]], o3d.core.float64))
    target = o3d.t.geometry.PointCloud(
        o3d.core.Tensor([[0.0, 0.0, 0.0]], o3d.core.float64))
    target.point.normals = o3d.core.Tensor([[0.6, 0.8, 0.0]], o3d.core.float64)
    correspondences = o3d.core.Tensor([0], o3d.core.int64)
    estimator = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    error = estimator.compute_rmse(source, target, correspondences)
    assert abs(error) < 1e-12
