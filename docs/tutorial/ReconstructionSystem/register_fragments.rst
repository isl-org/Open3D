.. _reconstruction_system_register_fragments:

Register fragments
-------------------------------------

Once the fragments of the scene are created, the next step is to align them in a global space.

Input arguments
``````````````````````````````````````

This script runs with ``python run_system.py [config] --register``. In ``[config]``, ``["path_dataset"]`` should have subfolders ``fragments`` which stores fragments in ``.ply`` files and a pose graph in a ``.json`` file.

The main function runs ``make_posegraph_for_scene`` and ``optimize_posegraph_for_scene``. The first function performs pairwise registration. The second function performs multiway registration.


Preprocess point cloud
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 17
   :lines: 5,18-28
   :linenos:

This function downsamples point cloud to make a point cloud sparser and regularly distributed. Normals and FPFH feature are precomputed. See :ref:`/tutorial/Basic/pointcloud.ipynb#voxel-downsampling`, :ref:`/tutorial/Basic/pointcloud.ipynb#vertex-normal-estimation`, and :ref:`/tutorial/Advanced/global_registration.ipynb#extract-geometric-feature` for more details.


Compute initial registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 54
   :lines: 5,55-81
   :linenos:

This function computes a rough alignment between two fragments. If the fragments are neighboring fragments, the rough alignment is determined by an aggregating RGBD odometry obtained from :ref:`reconstruction_system_make_fragments`. Otherwise, ``register_point_cloud_fpfh`` is called to perform global registration. Note that global registration is less reliable according to [Choi2015]_.


.. _reconstruction_system_feature_matching:

Pairwise global registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 30
   :lines: 5,31-52
   :linenos:

This function uses :ref:`/tutorial/Advanced/global_registration.ipynb#RANSAC` or :ref:`/tutorial/Advanced/global_registration.ipynb#fast-global-registration` for pairwise global registration.


.. _reconstruction_system_compute_initial_registration:

Multiway registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 83
   :lines: 5,84-103
   :linenos:

This script uses the technique demonstrated in :ref:`/tutorial/Advanced/multiway_registration.ipynb`. Function ``update_posegrph_for_scene`` builds a pose graph for multiway registration of all fragments. Each graph node represents a fragment and its pose which transforms the geometry to the global space.

Once a pose graph is built, function ``optimize_posegraph_for_scene`` is called for multiway registration.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/optimize_posegraph.py
   :language: python
   :lineno-start: 42
   :lines: 5,43-50
   :linenos:

Main registration loop
``````````````````````````````````````

The function ``make_posegraph_for_scene`` below calls all the functions introduced above. The main workflow is: pairwise global registration -> multiway registration.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 135
   :lines: 5,136-176
   :linenos:

Results
``````````````````````````````````````

The following is messages from pose graph optimization.

.. code-block:: sh

    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 42 edges.
    Line process weight : 55.885667
    [Initial     ] residual : 7.791139e+04, lambda : 1.205976e+00
    [Iteration 00] residual : 6.094275e+02, valid edges : 22, time : 0.001 sec.
    [Iteration 01] residual : 4.526879e+02, valid edges : 22, time : 0.000 sec.
    [Iteration 02] residual : 4.515039e+02, valid edges : 22, time : 0.000 sec.
    [Iteration 03] residual : 4.514832e+02, valid edges : 22, time : 0.000 sec.
    [Iteration 04] residual : 4.514825e+02, valid edges : 22, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.003 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 35 edges.
    Line process weight : 60.762800
    [Initial     ] residual : 6.336097e+01, lambda : 1.324043e+00
    [Iteration 00] residual : 6.334147e+01, valid edges : 22, time : 0.000 sec.
    [Iteration 01] residual : 6.334138e+01, valid edges : 22, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.001 sec.
    CompensateReferencePoseGraphNode : reference : 0


There are 14 fragments and 52 valid matching pairs among the fragments. After 23 iteration, 11 edges are detected to be false positive. After they are pruned, pose graph optimization runs again to achieve tight alignment.
