Degeneracy-Aware ICP
====================

Open3D provides ``TransformationEstimationPointToPlaneDCReg`` for legacy CPU
ICP registration in scenes where point-to-plane ICP may contain weak geometric
directions. The implementation is based on DCReg [Hu2025]_, which detects,
characterizes, and mitigates ill-conditioning through Schur-complement
subspaces and preconditioned conjugate gradient.

``TransformationEstimationPointToPlaneDCReg`` uses the same correspondences,
target normals, robust kernels, and ICP convergence criteria as
``TransformationEstimationPointToPlane``. The difference is the 6D linear solve:
DCReg detects weak directions from the Schur complements of the normal equation,
builds a block preconditioner, and falls back to a dense solve if the
preconditioned solve is not reliable.

The pose update follows Open3D's legacy ICP perturbation convention. At each
iteration, ``registration_icp`` has already transformed the source by the
current pose, the estimator solves a target/world-frame point-to-plane update,
and Open3D applies it as ``transformation = update @ transformation``.

The target point cloud must have normals, just like ordinary point-to-plane
ICP.

.. code-block:: python

    import copy
    import numpy as np
    import open3d as o3d

    source = o3d.io.read_point_cloud("source.pcd")
    target = o3d.io.read_point_cloud("target.pcd")
    target.estimate_normals()

    option = o3d.pipelines.registration.DCRegOption(
        degeneracy_condition_threshold=10.0,
        kappa_target=10.0,
        pcg_tolerance=1e-6,
        pcg_max_iteration=10,
    )
    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlaneDCReg(
            option
        )
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=0.05,
        init=np.eye(4),
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50
        ),
    )

    print(result.transformation)

``DCRegOption`` exposes the Schur condition threshold, the target condition
number after weak-eigenvalue clamping, and the PCG stopping parameters. The
defaults are intended to be conservative for the same data scale used by
ordinary point-to-plane ICP.

Standalone-Compatible Local ICP
--------------------------------

For reproducing the standalone DCReg examples, Open3D also provides
``registration_icp_dcreg_local``. This is a separate ICP loop because the
standalone path needs the original source point, the current pose state, and
the current rotation for the local-frame Jacobian:

.. code-block:: python

    option = o3d.pipelines.registration.DCRegOption(
        local_plane_knn=5,
        local_plane_max_thickness=0.2,
        local_plane_weight_slope=0.9,
        local_plane_min_weight=0.1,
        local_plane_use_weight_derivative=True,
        local_frame_convergence_rotation=1e-5,
        local_frame_convergence_translation=1e-3,
    )
    result = o3d.pipelines.registration.registration_icp_dcreg_local(
        source,
        target,
        max_correspondence_distance=0.5,
        init=init,
        option=option,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=30
        ),
    )
    analysis = (
        o3d.pipelines.registration.compute_dcreg_local_degeneracy_analysis(
            source,
            target,
            max_correspondence_distance=0.5,
            transformation=result.transformation,
            option=option,
        )
    )

    print(result.transformation)
    print("DCReg degeneracy description:")
    print(analysis.degeneracy_description)
    print("weak rotation axes:", analysis.weak_rotation_axes_description)
    print("weak translation axes:", analysis.weak_translation_axes_description)
    print("condition number rotation:", analysis.condition_number_rotation)
    print("condition number translation:", analysis.condition_number_translation)
    print("coordinate frame:", analysis.coordinate_frame)

This compatibility path fits a local plane from each transformed source point's
target k-nearest neighbors and uses the standalone SO(3) update
``R <- R Exp(omega), t <- t + R v``. It does not require target normals.

Degeneracy Diagnostics
----------------------

``registration_icp`` still returns the standard Open3D ``RegistrationResult``.
For DCReg-specific diagnostics, call
``compute_dcreg_degeneracy_analysis`` on a point cloud pair and a
correspondence set. A typical pattern is to evaluate the final ICP
linearization after applying ``result.transformation``:

.. code-block:: python

    source_aligned = copy.deepcopy(source)
    source_aligned.transform(result.transformation)

    analysis = o3d.pipelines.registration.compute_dcreg_degeneracy_analysis(
        source_aligned,
        target,
        result.correspondence_set,
        option,
    )

    print(analysis.is_degenerate)
    print(analysis.is_rank_deficient)
    print(analysis.condition_number_full)
    print(analysis.condition_number_rotation)
    print(analysis.condition_number_translation)
    print(analysis.weak_rotation_axes)
    print(analysis.weak_translation_axes)
    print(analysis.coordinate_frame)
    print(analysis.degeneracy_description)
    print(analysis.solver_type)

For the standalone-compatible local-plane path, use
``compute_dcreg_local_degeneracy_analysis`` with the same source, target,
maximum correspondence distance, transformation, and ``DCRegOption``:

.. code-block:: python

    analysis = (
        o3d.pipelines.registration.compute_dcreg_local_degeneracy_analysis(
            source,
            target,
            max_correspondence_distance=0.5,
            transformation=result.transformation,
            option=option,
        )
    )

    print(analysis.degeneracy_description)
    print(analysis.weak_rotation_axes_description)
    print(analysis.weak_translation_axes_description)
    print(analysis.condition_number_rotation)
    print(analysis.condition_number_translation)
    print(analysis.coordinate_frame)

The eigenvalue and weak-axis fields use the physical x/y/z axis order after
DCReg aligns the Schur eigenvectors to coordinate axes. For the Open3D legacy
ICP estimator, these axes are the target/world-frame incremental axes used by
Open3D's left-multiplied SE(3) point-to-plane linearization. They are not the
local body-frame translation axes used by the standalone DCReg SO(3) examples.
The weak-axis
arrays use ``1`` for weak and ``0`` for not weak. If the estimator uses a robust
kernel, pass the same kernel to ``compute_dcreg_degeneracy_analysis`` so the
diagnostic normal equation matches the ICP linearization.

For ``registration_icp_dcreg_local`` and
``compute_dcreg_local_degeneracy_analysis``, the weak translation axes are the
standalone DCReg local body-frame axes.

For exact rank-deficient cases, the Schur complements may be impossible to form
because one Hessian block is singular. In the Open3D-native point-to-plane
estimator, ``schur_factorization_ok`` remains ``False``, but the diagnostic
function fills the eigenvalue, condition-number, and weak-axis fields from a
block-Hessian eigensolver fallback. This is intended for interpretation only;
the transformation update still uses the minimum-norm rank-deficient solve. The
standalone-compatible local-plane ICP follows the original DCReg behavior and
falls back to the QR solve when Schur/PCG is not reliable.

Implementation Notes
--------------------

This Open3D implementation is intentionally smaller than the original DCReg
research code. It is designed to fit the existing
``open3d.pipelines.registration`` API rather than to reproduce the full
standalone experiment system.

The main differences are:

* Open3D owns the ICP loop, correspondence search, robust kernel, point cloud
  containers, and ``RegistrationResult``. DCReg is implemented only as a
  ``TransformationEstimation`` method.
* The estimator returns Open3D-style left-multiplied SE(3) increments. This
  intentionally follows ``TransformationEstimationPointToPlane`` rather than
  the standalone DCReg examples' local-frame update convention.
* For ``TransformationEstimationPointToPlaneDCReg``, target normals must
  already be available, as in ordinary point-to-plane ICP.
* ``registration_icp_dcreg_local`` is the standalone-compatible exception: it
  ports the original kNN local-plane residual, piecewise-linear residual
  weighting, and local body-frame SO(3) update for example reproduction.
* Weak-axis diagnostics describe the Open3D point-to-plane normal equation.
  They should not be compared axis-by-axis with standalone DCReg parking-lot
  logs unless the same normals, correspondences, robust weights, and local
  body-frame Jacobian are used.
* The eigenvalue clamping is applied only inside the preconditioner. The
  original point-to-plane least-squares objective is not modified.
* Exact rank-deficient systems use a minimum-norm fallback before the Schur/PCG
  path. This keeps pure-null motions, such as translation along a cylinder axis,
  from receiving arbitrary updates.
* Degeneracy diagnostics are available through
  ``compute_dcreg_degeneracy_analysis`` instead of being added to the generic
  ``RegistrationResult``.
* Dataset runners, YAML presets, parking-lot-specific preprocessing, detailed
  experiment logs, and paper benchmark scripts are not included.
* This first integration targets the legacy CPU API. Tensor and CUDA
  registration APIs are not implemented here.

References
----------

.. [Hu2025] Xiangcheng Hu, Xieyuanli Chen, Mingkai Jia, Jin Wu, Ping Tan,
   Steven L. Waslander, "DCReg: Decoupled Characterization for Efficient
   Degenerate LiDAR Registration," arXiv:2509.06285, 2025.
   https://arxiv.org/abs/2509.06285
