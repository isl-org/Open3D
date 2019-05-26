.. _capture_your_own_dataset:

Capture your own dataset
-------------------------------------

If you have a `RealSense camera <https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html>`_, capturing RGBD frames is easy by using ``sensors/realsense_recoder.py``.

.. note:: This tutorial assumes that valid RealSense Python package and OpenCV Python package are installed in your environment. Please follow `this instruction <https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python>`_ to install RealSense Python package.

Input arguments
``````````````````````````````````````

The script runs with one of the following three options:

.. code-block:: bash

    python realsense_recoder.py --record_imgs
    python realsense_recoder.py --record_rosbag
    python realsense_recoder.py --playback_rosbag

In either ``record_imgs`` and ``record_rosbag`` mode, the script displays the following capturing preview.

.. image:: ../../_static/ReconstructionSystem/capture_your_own_dataset/recorder.png
    :width: 400px

The left side shows color image with invalid depth region markup (in gray color), and the right side shows jet color coded depth map. Invalid depth pixels are object boundary, uncertain region, or distant region (more than 3m). Capturing frames without too many gray pixels is recommended for the good reconstruction quality.

By default, ``record_imgs`` mode saves aligned color and depth images in ``dataset/realsense`` folder that can be used for reconstruction system.

.. code-block:: bash

    dataset
    └── realsense
        ├── camera_intrinsic.json
        ├── color
        │   ├── 000000.jpg
        │   ├── :
        └── depth
            ├── 000000.png
            ├── :

``camera_intrinsic.json`` has intrinsic parameter of the used RealSense camera. This parameter set should be used with the dataset.

Make a new configuration file
``````````````````````````````````````

A new configuration file is required to specify path to the new dataset. ``config/realsense.json`` is provided for this purpose.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/config/realsense.json
   :language: json
   :lineno-start: 1
   :lines: 1-
   :linenos:

Note that ``path_dataset`` and ``path_intrinsic`` indicates paths of dataset and intrinsic parameters.


Run reconstruction system
``````````````````````````````````````

Run the system by using the new configuration file.

.. code-block:: sh

    cd examples/Python/ReconstructionSystem/
    python run_system.py config/realsense.json [--make] [--register] [--refine] [--integrate]
