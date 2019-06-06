.. _multiway_registration:

Multiway registration
-------------------------------------

Multiway registration is the process to align multiple pieces of geometry in a global space. Typically, the input is a set of geometries (e.g., point clouds or RGBD images) :math:`\{\mathbf{P}_{i}\}`. The output is a set of rigid transformations :math:`\{\mathbf{T}_{i}\}`, so that the transformed point clouds :math:`\{\mathbf{T}_{i}\mathbf{P}_{i}\}` are aligned in the global space.

Open3D implements multiway registration via pose graph optimization. The backend implements the technique presented in [Choi2015]_.

.. literalinclude:: ../../../examples/Python/Advanced/multiway_registration.py
   :language: python
   :lineno-start: 5
   :lines: 5-
   :linenos:

Input
````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/multiway_registration.py
   :language: python
   :lineno-start: 15
   :lines: 15-21
   :linenos:

The first part of the tutorial script reads three point clouds from files. The point clouds are downsampled and visualized together. They are misaligned.

.. image:: ../../_static/Advanced/multiway_registration/initial.png
    :width: 400px

.. _build_a_posegraph:

.. literalinclude:: ../../../examples/Python/Advanced/multiway_registration.py
   :language: python
   :lineno-start: 24
   :lines: 24-68
   :linenos:

A pose graph has two key elements: nodes and edges. A node is a piece of geometry :math:`\mathbf{P}_{i}` associated with a pose matrix :math:`\mathbf{T}_{i}` which transforms :math:`\mathbf{P}_{i}` into the global space. The set :math:`\{\mathbf{T}_{i}\}` are the unknown variables to be optimized. ``PoseGraph.nodes`` is a list of ``PoseGraphNode``. We set the global space to be the space of :math:`\mathbf{P}_{0}`. Thus :math:`\mathbf{T}_{0}` is identity matrix. The other pose matrices are initialized by accumulating transformation between neighboring nodes. The neighboring nodes usually have large overlap and can be registered with :ref:`point_to_plane_icp`.

A pose graph edge connects two nodes (pieces of geometry) that overlap. Each edge contains a transformation matrix :math:`\mathbf{T}_{i,j}` that aligns the source geometry :math:`\mathbf{P}_{i}` to the target geometry :math:`\mathbf{P}_{j}`. This tutorial uses :ref:`point_to_plane_icp` to estimate the transformation. In more complicated cases, this pairwise registration problem should be solved via :ref:`global_registration`.

[Choi2015]_ has observed that pairwise registration is error-prone. False pairwise alignments can outnumber correctly
aligned pairs. Thus, they partition pose graph edges into two classes. **Odometry edges** connect temporally close, neighboring nodes. A local registration algorithm such as ICP can reliably align them. **Loop closure edges** connect any non-neighboring nodes. The alignment is found by global registration and is less reliable. In Open3D, these two classes of edges are distinguished by the ``uncertain`` parameter in the initializer of ``PoseGraphEdge``.

In addition to the transformation matrix :math:`\mathbf{T}_{i}`, the user can set an information matrix :math:`\mathbf{\Lambda}_{i}` for each edge. If :math:`\mathbf{\Lambda}_{i}` is set using function ``get_information_matrix_from_point_clouds``, the loss on this pose graph edge approximates the RMSE of the corresponding sets between the two nodes, with a line process weight. Refer to Eq (3) to (9) in [Choi2015]_ and `the Redwood registration benchmark <http://redwood-data.org/indoor/registration.html>`_ for details.

The script creates a pose graph with three nodes and three edges. Among the edges, two of them are odometry edges (``uncertain = False``) and one is a loop closure edge (``uncertain = True``).

.. _optimize_a_posegraph:

.. literalinclude:: ../../../examples/Python/Advanced/multiway_registration.py
   :language: python
   :lineno-start: 82
   :lines: 82-89
   :linenos:

Open3D uses function ``global_optimization`` to perform pose graph optimization. Two types of optimization methods can be chosen: ``GlobalOptimizationGaussNewton`` or ``GlobalOptimizationLevenbergMarquardt``. The latter is recommended since it has better convergence property. Class ``GlobalOptimizationConvergenceCriteria`` can be used to set the maximum number of iterations and various optimization parameters.

Class ``GlobalOptimizationOption`` defines a couple of options. ``max_correspondence_distance`` decides the correspondence threshold. ``edge_prune_threshold`` is a threshold for pruning outlier edges. ``reference_node`` is the node id that is considered to be the global space.

.. code-block:: sh

    Optimizing PoseGraph ...
    [GlobalOptimizationLM] Optimizing PoseGraph having 3 nodes and 3 edges.
    Line process weight : 3.745800
    [Initial     ] residual : 6.741225e+00, lambda : 6.042803e-01
    [Iteration 00] residual : 1.791471e+00, valid edges : 3, time : 0.000 sec.
    [Iteration 01] residual : 5.133682e-01, valid edges : 3, time : 0.000 sec.
    [Iteration 02] residual : 4.412544e-01, valid edges : 3, time : 0.000 sec.
    [Iteration 03] residual : 4.408356e-01, valid edges : 3, time : 0.000 sec.
    [Iteration 04] residual : 4.408342e-01, valid edges : 3, time : 0.000 sec.
    Delta.norm() < 1.000000e-06 * (x.norm() + 1.000000e-06)
    [GlobalOptimizationLM] total time : 0.000 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 3 nodes and 3 edges.
    Line process weight : 3.745800
    [Initial     ] residual : 4.408342e-01, lambda : 6.064910e-01
    Delta.norm() < 1.000000e-06 * (x.norm() + 1.000000e-06)
    [GlobalOptimizationLM] total time : 0.000 sec.
    CompensateReferencePoseGraphNode : reference : 0

The global optimization performs twice on the pose graph. The first pass optimizes poses for the original pose graph taking all edges into account and does its best to distinguish false alignments among uncertain edges. These false alignments have small line process weights, and they are pruned after the first pass. The second pass runs without them and produces a tight global alignment. In this example, all the edges are considered as true alignments, hence the second pass terminates immediately.

.. _visualize_optimization:

Visualize optimization
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/multiway_registration.py
   :language: python
   :lineno-start: 91
   :lines: 91-95
   :linenos:

Ouputs:

.. image:: ../../_static/Advanced/multiway_registration/optimized.png
    :width: 400px

The transformed point clouds are listed and visualized using ``draw_geometries``.

.. _make_a_combined_point_cloud:

Make a combined point cloud
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/multiway_registration.py
   :language: python
   :lineno-start: 97
   :lines: 97-105
   :linenos:

.. image:: ../../_static/Advanced/multiway_registration/combined.png
    :width: 400px

``PointCloud`` has convenient operator ``+`` that can merge two point clouds into single one.
After merging, the points are uniformly resampled using ``voxel_down_sample``.
This is recommended post-processing after merging point cloud since this can relieve duplicating or over-densified points.

.. note:: Although this tutorial demonstrates multiway registration for point clouds. The same procedure can be applied to RGBD images. See :ref:`reconstruction_system_make_fragments` for an example.
