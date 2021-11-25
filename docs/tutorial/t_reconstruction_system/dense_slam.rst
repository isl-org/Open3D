.. _dense_slam:

Dense RGB-D SLAM
-------------------------------------
Equipped with the fast volumetric reconstruction backend, we in addition provide a dense RGB-D SLAM system using frame-to-model tracking. The example can be found at ``examples/python/t_reconstruction_system/dense_slam.py`` for the command line version and ``examples/python/t_reconstruction_system/dense_slam_gui.py`` for a GUI demo. Similar C++ versions can be found at ``examples/cpp/DenseSLAM.cpp`` and ``examples/cpp/DenseSLAMGUI.cpp``.

.. note::
   This SLAM prototype is mainly a showcase of the real-time volumetric processing. Please be aware that the tracking/RGB-D odometry module is **not fully optimized for accuracy**, and there is **no relocalization module** implemented currently. In general, it should work for room-scale scenes with relatively moderate motion, and may fail on more challenging sequences. More robust and reliable localization is our future work.


Model and frame initialization
````````````````````````````````
In a SLAM system, we maintain a ``model`` built upon a :ref:`voxel_block_grid`, an input ``frame`` containing raw RGB-D input, and a synthesized ``frame`` generated from the ``model`` with volumetric ray casting.

.. literalinclude:: ../../../examples/python/t_reconstruction_system/dense_slam.py
   :language: python
   :lineno-start: 45
   :lines: 27,46-54

Frame-to-model tracking
````````````````````````
The frame-to-model tracking runs in a loop:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/dense_slam.py
   :language: python
   :lineno-start: 57
   :lines: 27,58-78

where we iteratively update the synthesized frame via ray-casting from the model, and perform the tensor version of :ref:`/tutorial/pipelines/rgbd_odometry.ipynb` between the input frame and the synthesized frame.

The reconstruction results can be saved following :ref:`optimized_integration`, and the trajectory of the camera pose in the world coordinate system can be obtained by accumulating ``T_frame_to_model``.


FAQ
``````````
**Q**: Why did my tracking fail?

**A**: If it fails after successfully tracking multiple frames, it could be caused by the instability of the system. Please refer to the disclaimer. If it fails at frame 0 on initialization, please check these important factors:

- Depth/RGB images are correctly loaded.
- Image **intrinsic matrix** is correctly set, see the calibration factors. If it is not correctly set, the point cloud from the depth will be distorted, resulting in failure in the following operations.
- Depth **scale** factor is correctly set (e.g., 1000 for PrimeSense and RealSense D435, 5000 for the TUM dataset). It is responsible for the correct transformation from the raw depth pixel value to the metric distance.
- Depth **max** factor (mainly set for 3.0, but could be insufficient for larger scales). It will prune all the faraway points.

If all above have been correctly set but still no luck, please file an issue.


**Q**: So WHY did my tracking fail?

**A**: For the front end, we are using direct RGB-D odometry. Comparing to feature-based odometry, RGB-D odometry is more accurate when it completes successfully but is less robust. We will add support for feature-based tracking in the future. For the backend, unlike our offline reconstruction system, we do not detect loop closures, and do not perform pose graph optimization or bundle adjustment at the moment.


**Q**: Why don't you implement loop closure or relocalization?

**A**: Relocalization is challenging for volumetric reconstruction, as active real-time volume deformation and/or reintegration is needed. Since we are using direct odometry, we do not keep track of sparse features over the frames. A non-trivial system upgrade that addresses all the problems will be future work.
