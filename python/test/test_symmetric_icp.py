# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np


class TestSymmetricICP:

    def test_transformation_estimation_symmetric_constructor(self):
        """Test TransformationEstimationSymmetric constructor."""
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )
        assert estimation is not None

    def test_transformation_estimation_symmetric_with_kernel(self):
        """Test TransformationEstimationSymmetric with robust kernel."""
        kernel = o3d.pipelines.registration.HuberLoss(0.1)
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
            kernel)
        assert estimation is not None
        assert estimation.kernel is not None

    def test_transformation_estimation_symmetric_compute_rmse(self):
        """Test compute_rmse method."""
        # Create simple test point clouds with normals
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0]])
        source.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1]])

        target.points = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1],
                                                    [1.1, 0.1, 0.1],
                                                    [0.1, 1.1, 0.1]])
        target.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1]])

        corres = o3d.utility.Vector2iVector([[0, 0], [1, 1], [2, 2]])
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )

        rmse = estimation.compute_rmse(source, target, corres)
        assert rmse > 0.0

    def test_transformation_estimation_symmetric_compute_rmse_empty_corres(
            self):
        """Test compute_rmse with empty correspondences."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        corres = o3d.utility.Vector2iVector()
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )

        rmse = estimation.compute_rmse(source, target, corres)
        assert rmse == 0.0

    def test_transformation_estimation_symmetric_compute_rmse_no_normals(self):
        """Test compute_rmse without normals."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]])
        target.points = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1],
                                                    [1.1, 0.1, 0.1]])
        # No normals

        corres = o3d.utility.Vector2iVector([[0, 0], [1, 1]])
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )

        rmse = estimation.compute_rmse(source, target, corres)
        assert rmse == 0.0

    def test_transformation_estimation_symmetric_compute_transformation(self):
        """Test compute_transformation method."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Create test point clouds with normals
        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0], [1, 1, 0]])
        source.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1], [0, 0, 1]])

        target.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0], [1, 1, 0]])
        target.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1], [0, 0, 1]])

        corres = o3d.utility.Vector2iVector([[0, 0], [1, 1], [2, 2], [3, 3]])
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )

        transformation = estimation.compute_transformation(
            source, target, corres)

        # Should be close to identity for perfect correspondence
        expected = np.eye(4)
        np.testing.assert_allclose(transformation, expected, atol=1e-3)

    def test_transformation_estimation_symmetric_compute_transformation_empty_corres(
            self):
        """Test compute_transformation with empty correspondences."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        corres = o3d.utility.Vector2iVector()
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )

        transformation = estimation.compute_transformation(
            source, target, corres)

        expected = np.eye(4)
        np.testing.assert_allclose(transformation, expected)

    def test_transformation_estimation_symmetric_compute_transformation_no_normals(
            self):
        """Test compute_transformation without normals."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]])
        target.points = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1],
                                                    [1.1, 0.1, 0.1]])
        # No normals

        corres = o3d.utility.Vector2iVector([[0, 0], [1, 1]])
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )

        transformation = estimation.compute_transformation(
            source, target, corres)

        expected = np.eye(4)
        np.testing.assert_allclose(transformation, expected)

    def test_registration_symmetric_icp(self):
        """Test registration_symmetric_icp function."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Create test point clouds with normals
        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0]])
        source.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1]])

        # Target is slightly translated
        target.points = o3d.utility.Vector3dVector([[0.05, 0.05, 0.05],
                                                    [1.05, 0.05, 0.05],
                                                    [0.05, 1.05, 0.05]])
        target.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1]])

        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()

        result = o3d.pipelines.registration.registration_symmetric_icp(
            source, target, 0.1, np.eye(4), estimation, criteria)

        assert len(result.correspondences) > 0
        assert result.fitness > 0.0
        assert result.inlier_rmse >= 0.0

    def test_registration_symmetric_icp_with_robust_kernel(self):
        """Test registration_symmetric_icp with robust kernel."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Create test point clouds with normals
        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0], [1, 1, 0]])
        source.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1], [0, 0, 1]])

        target.points = o3d.utility.Vector3dVector([[0.02, 0.02, 0.02],
                                                    [1.02, 0.02, 0.02],
                                                    [0.02, 1.02, 0.02],
                                                    [1.02, 1.02, 0.02]])
        target.normals = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1],
                                                     [0, 0, 1], [0, 0, 1]])

        kernel = o3d.pipelines.registration.HuberLoss(0.1)
        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
            kernel)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            1e-6, 1e-6, 30)

        result = o3d.pipelines.registration.registration_symmetric_icp(
            source, target, 0.1, np.eye(4), estimation, criteria)

        assert len(result.correspondences) > 0
        assert result.fitness > 0.0

    def test_registration_symmetric_icp_convergence(self):
        """Test registration_symmetric_icp convergence with known transformation."""
        np.random.seed(42)

        # Create a more complex test case
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Generate random points with normals
        num_points = 50
        points = np.random.rand(num_points, 3) * 10.0
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0  # Simple normal for testing

        source.points = o3d.utility.Vector3dVector(points)
        source.normals = o3d.utility.Vector3dVector(normals)

        # Create target by transforming source with known transformation
        true_transformation = np.eye(4)
        true_transformation[0, 3] = 0.1  # Small translation in x
        true_transformation[1, 3] = 0.05  # Small translation in y

        target = source.transform(true_transformation)

        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            1e-6, 1e-6, 30)

        result = o3d.pipelines.registration.registration_symmetric_icp(
            source, target, 0.5, np.eye(4), estimation, criteria)

        # Check that registration converged to reasonable result
        assert result.fitness > 0.5
        assert result.inlier_rmse < 1.0

    def test_registration_symmetric_icp_requires_normals(self):
        """Test that symmetric ICP requires normals."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]])
        target.points = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1],
                                                    [1.1, 0.1, 0.1]])
        # No normals set - should handle gracefully

        estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        )
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()

        # This should not crash, but may not produce meaningful results
        result = o3d.pipelines.registration.registration_symmetric_icp(
            source, target, 0.1, np.eye(4), estimation, criteria)

        # The function should complete without error
        assert result is not None
