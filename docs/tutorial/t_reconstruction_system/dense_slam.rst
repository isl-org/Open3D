.. _dense_slam:

Dense RGB-D SLAM
-------------------------------------
Equipped with the fast volumetric reconstruction backend, we in addition provide a dense RGB-D SLAM system using frame-to-model tracking.

Disclaimer
``````````
This SLAM prototype is mainly a showcase of the real-time volumetric processing. Please be aware that the tracking/RGB-D odometry module is **not fully optimized for accuracy**, and there is **no relocalization module** implemented at current. In general, it works for room-scale scenes with relatively moderate motion, and may fail on more challenging sequences. More robust and reliable localization is our future work.


Model and frame intialization
``````````
In a SLAM system, we maintain a ``model`` built upon a :ref:`voxel_block_grid`, an input ``frame`` containing raw RGB-D input, and a synthesized ``frame`` generated from the ``model`` with volumetric ray casting.

.. literalinclude:: ../../../examples/python/t_reconstruction_system/dense_slam.py
   :language: python
   :lineno-start: 41
   :lines: 41-49
   :linenos:
   :dedent:

Frame-to-model tracking
``````````
The frame-to-model tracking runs in a loop:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/dense_slam.py
   :language: python
   :lineno-start: 53
   :lines: 53-73
   :linenos:
   :dedent:

where we iteratively update the synthesized frame via ray-casting from the model, and perform the tensor version of :ref:`/tutorial/pipelines/rgbd_odometry.ipynb` between the input frame and the synthesized frame.

The reconstruction results can be saved following :ref:`optimized_integration`, and the trajectory of the camera pose in the world coordinate system can be obtained by accumulating ``T_frame_to_model``.
