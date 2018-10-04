.. _capture_your_own_dataset:

Capture your own dataset
-------------------------------------

If you have a `Realsense camera <https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html>`_, capturing RGBD frames is easy by using ``recoder_realsense.py``.

Input arguments
``````````````````````````````````````

The script runs with one of the following three options:

.. code-block:: bash

    python recoder_realsense.py --record_imgs
    python recoder_realsense.py --record_rosbag
    python recoder_realsense.py --playback_rosbag

In either ``record_imgs`` and ``record_rosbag`` mode, the script will display the following capturing preview and save frames.

.. image:: ../../_static/ReconstructionSystem/capture_your_own_dataset/recorder.png
    :width: 400px

``record_imgs`` mode saves aligned color and depth images in ``dataset/realsense`` folder that can be used for reconstruction system.

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

camera_intrinsic.json has intrinsic parameter of the used realsense camera. This parameter set should be used with the dataset for the good reconstruction result.

Make a new configure file
``````````````````````````````````````

A new configuration file is required to specify path to the new dataset. For example, let's check config/realsense.json.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/config/realsense.json
   :language: json
   :lineno-start: 1
   :lines: 1-
   :linenos:

Note that ``path_dataset`` and ``path_intrinsic`` has paths of dataset and intrinsic parameters.


Run reconstruction system
``````````````````````````````````````

Run the system by using the new configuration file.

.. code-block:: sh

    cd examples/Python/ReconstructionSystem/
    python run_system.py config/realsense.json [--make] [--register] [--refine] [--integrate]
