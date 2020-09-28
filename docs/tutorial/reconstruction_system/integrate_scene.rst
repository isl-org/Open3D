.. _reconstruction_system_integrate_scene:

Integrate scene
-------------------------------------

The final step of the system is to integrate all RGBD images into a single TSDF
volume and extract a mesh as the result.

Input arguments
``````````````````````````````````````

The script runs with ``python run_system.py [config] --integrate``. In
``[config]``, ``["path_dataset"]`` should have subfolders *image* and *depth*
in which frames are synchronized and aligned. In ``[config]``, the optional
argument ``["path_intrinsic"]`` specifies path to a json file that has a camera
intrinsic matrix (See :ref:`/tutorial/pipelines/rgbd_odometry.ipynb#read-camera-intrinsic` for
details). If it is not given, the PrimeSense factory setting is used instead.

Integrate RGBD frames
``````````````````````````````````````

.. literalinclude:: ../../../examples/python/reconstruction_system/integrate_scene.py
   :language: python
   :lineno-start: 13
   :lines: 5,17-54
   :linenos:

This function first reads the alignment results from both
:ref:`reconstruction_system_make_fragments` and
:ref:`reconstruction_system_register_fragments`, then computes the pose of each
RGBD image in the global space. After that, RGBD images are integrated using
:ref:`/tutorial/pipelines/rgbd_integration.ipynb`.


Results
``````````````````````````````````````
This is a printed log from the volume integration script.

.. code-block:: sh

    Fragment 000 / 013 :: integrate rgbd frame 0 (1 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 1 (2 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 2 (3 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 3 (4 of 100).
    :
    Fragment 013 / 013 :: integrate rgbd frame 1360 (61 of 64).
    Fragment 013 / 013 :: integrate rgbd frame 1361 (62 of 64).
    Fragment 013 / 013 :: integrate rgbd frame 1362 (63 of 64).
    Fragment 013 / 013 :: integrate rgbd frame 1363 (64 of 64).
    Writing PLY: [========================================] 100%

The following image shows the final scene reconstruction.

.. image:: ../../_static/reconstruction_system/integrate_scene/scene.png
    :width: 500px
