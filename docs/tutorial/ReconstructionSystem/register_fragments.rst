.. _reconstruction_system_register_fragments:

Register fragments
-------------------------------------

Once the fragments of the scene are created, the next step is to align them in a global space.

Input arguments
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 190
   :lines: 190-197
   :linenos:

This script runs with ``python run_system.py [config] --register``. In ``[config]``, ``["path_dataset"]`` should have subfolders *fragments* which stores fragments in .ply files and a pose graph in a .json file.

The main function runs ``make_posegraph_for_scene`` and ``optimize_posegraph_for_scene``. The first function performs pairwise registration. The second function performs multiway registration.


Preprocess point cloud
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 15
   :lines: 15-22
   :linenos:

This function downsample point cloud to make a point cloud sparser and regularly distributed. Normals and FPFH feature are precomputed. See :ref:`voxel_downsampling`, :ref:`vertex_normal_estimation`, and :ref:`extract_geometric_feature` for more details.


.. _reconstruction_system_feature_matching:

Pairwise global registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 25
   :lines: 25-35
   :linenos:

This function uses :ref:`feature_matching` for pairwise global registration.


.. _reconstruction_system_compute_initial_registration:

Compute initial registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 38
   :lines: 38-64
   :linenos:

This function computes a rough alignment between two fragments. The rough alignments are used to initialize ICP refinement. If the fragments are neighboring fragments, the rough alignment is determined by an aggregating RGBD odometry obtained from :ref:`reconstruction_system_make_fragments`. Otherwise, ``register_point_cloud_fpfh`` is called to perform global registration. Note that global registration is less reliable according to [Choi2015]_.


Fine-grained registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 67
   :lines: 67-129
   :linenos:

Two options are given for the fine-grained registration. The ``registration_colored_icp`` is recommended since it uses color information to prevent drift. Details see [Park2017]_.


Multiway registration
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 132
   :lines: 132-146
   :linenos:

This script uses the technique demonstrated in :ref:`multiway_registration`. Function ``update_posegrph_for_scene`` builds a pose graph for multiway registration of all fragments. Each graph node represents a fragments and its pose which transforms the geometry to the global space.

Once a pose graph is built, function ``optimize_posegraph_for_scene`` is called for multiway registration.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/optimize_posegraph.py
   :language: python
   :lineno-start: 12
   :lines: 12-26
   :linenos:

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/optimize_posegraph.py
   :language: python
   :lineno-start: 39
   :lines: 39-45
   :linenos:

Main registration loop
``````````````````````````````````````

The function ``make_posegraph_for_scene`` below calls all the functions introduced above.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/register_fragments.py
   :language: python
   :lineno-start: 149
   :lines: 149-187
   :linenos:

The main workflow is: pairwise global registration -> local refinement -> multiway registration.

Results
``````````````````````````````````````

The following is messages from pose graph optimization.

.. code-block:: sh

    PoseGraph with 14 nodes and 52 edges.
    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 52 edges.
    Line process weight : 49.899808
    [Initial     ] residual : 1.307073e+06, lambda : 8.415505e+00
    [Iteration 00] residual : 1.164909e+03, valid edges : 31, time : 0.000 sec.
    [Iteration 01] residual : 1.026223e+03, valid edges : 34, time : 0.000 sec.
    [Iteration 02] residual : 9.263710e+02, valid edges : 41, time : 0.000 sec.
    [Iteration 03] residual : 8.434943e+02, valid edges : 40, time : 0.000 sec.
    :
    [Iteration 22] residual : 8.002788e+02, valid edges : 41, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.006 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 41 edges.
    Line process weight : 52.121020
    [Initial     ] residual : 3.490871e+02, lambda : 1.198591e+01
    [Iteration 00] residual : 3.409909e+02, valid edges : 40, time : 0.000 sec.
    [Iteration 01] residual : 3.393578e+02, valid edges : 40, time : 0.000 sec.
    [Iteration 02] residual : 3.390909e+02, valid edges : 40, time : 0.000 sec.
    [Iteration 03] residual : 3.390108e+02, valid edges : 40, time : 0.000 sec.
    :
    [Iteration 08] residual : 3.389679e+02, valid edges : 40, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.002 sec.
    CompensateReferencePoseGraphNode : reference : 0


There are 14 fragments and 52 valid matching pairs between fragments. After 23 iteration, 11 edges are detected to be false positive. After they are pruned, pose graph optimization runs again to achieve tight alignment.
