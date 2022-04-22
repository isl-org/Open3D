.. _azure_kinect:

Azure Kinect with Open3D
------------------------

Azure Kinect is only officially supported on Windows and Ubuntu 18.04.

Installation
============

Install the Azure Kinect SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow `this guide <https://github.com/microsoft/Azure-Kinect-Sensor-SDK>`_
to install the Azure Kinect SDK (K4A).

On Ubuntu, you'll need to set up a udev rule to use the Kinect camera without
``sudo``. Follow
`these instructions <https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#linux-device-setup>`_.

After installation, you may run ``k4aviewer`` from the Linux terminal or
``k4aviewer.exe`` on Windows to make sure that the device is working.

Currently, Open3D supports the Azure Kinect SDK version ``v1.4.1``, though future
versions might also be compatible.

Install Open3D from Pip
~~~~~~~~~~~~~~~~~~~~~~~

If you're using Open3D installed via Pip, Open3D's Azure Kinect features
should work out-of-the box if K4A is installed in the system in the recommended
way. Open3D will try to load the K4A dynamic library automatically at runtime,
when a K4A related feature within Open3D is used.

On Ubuntu, the default search path
follows the Linux `convention <https://unix.stackexchange.com/a/22999/130082>`_.

On Windows, Open3D will try to load the shared library from the default
installation path. For example, for K4A ``v1.4.1``, the default path is
``C:\Program Files\Azure Kinect SDK v1.4.1``. If this doesn't work, copy
``depthengine_x_x.dll``, ``k4a.dll`` and ``k4arecord.dll`` to where Open3D
Python module is installed if you're using Open3D with Python, or to the same
directory as your C++ executable.

You can get Open3D's Python module path with the following command:

.. code-block:: sh

    python -c "import open3d as o3d; import os; print(os.path.dirname(o3d.__file__))"

Compile from Source
~~~~~~~~~~~~~~~~~~~

To build Open3D from source with K4A support, set ``BUILD_AZURE_KINECT=ON`` at
CMake config step. That is,

.. code-block:: sh

    cmake -DBUILD_AZURE_KINECT=ON -DOTHER_FLAGS ..

Open3D Azure Kinect Viewer
==========================

Open3D Azure Kinect Viewer is used for previewing RGB and depth image stream
captured by the Azure Kinect sensor.

Open3D provides Python and C++ example code of Azure Kinect viewer. Please
see ``examples/cpp/AzureKinectViewer.cpp`` and
``examples/python/reconstruction_system/sensors/azure_kinect_viewer.py``
for details.

We'll use the Python version as an example.

.. code-block:: sh

    python examples/python/reconstruction_system/sensors/azure_kinect_viewer.py --align_depth_to_color

.. image:: https://storage.googleapis.com/open3d-bin/docs/images/azure_kinect_viewer_aligned.png
    :alt: azure_kinect_viewer_aligned.png

When recording at a higher resolution at a high framerate, sometimes it is
helpful to use the raw depth image without transformation to reduce computation.

.. code-block:: sh

    python examples/python/reconstruction_system/sensors/azure_kinect_viewer.py

.. image:: https://storage.googleapis.com/open3d-bin/docs/images/azure_kinect_viewer_unaligned.png
    :alt: azure_kinect_viewer_unaligned.png

When the visualizer window is active, press ``ESC`` to quit the viewer.

You may also specify the sensor config with a ``json`` file.

.. code-block:: sh

    python examples/python/reconstruction_system/sensors/azure_kinect_viewer.py --config config.json

An sensor config will look like the following. For the full list of available
configs, refer to `this file <https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/include/k4a/k4atypes.h>`_.

.. code-block:: json

    {
        "camera_fps" : "K4A_FRAMES_PER_SECOND_30",
        "color_format" : "K4A_IMAGE_FORMAT_COLOR_MJPG",
        "color_resolution" : "K4A_COLOR_RESOLUTION_720P",
        "depth_delay_off_color_usec" : "0",
        "depth_mode" : "K4A_DEPTH_MODE_WFOV_2X2BINNED",
        "disable_streaming_indicator" : "false",
        "subordinate_delay_off_master_usec" : "0",
        "synchronized_images_only" : "false",
        "wired_sync_mode" : "K4A_WIRED_SYNC_MODE_STANDALONE"
    }

Open3D Azure Kinect Recorder
============================

Open3D Azure Kinect Viewer is used for recording RGB and depth image stream
to an MKV file.

Open3D provides Python and C++ example code of Azure Kinect viewer. Please
see ``examples/cpp/AzureKinectRecord.cpp`` and
``examples/python/reconstruction_system/sensors/azure_kinect_recorder.py``
for details.

We'll use the Python version as an example.

.. code-block:: sh

    python examples/python/reconstruction_system/sensors/azure_kinect_recorder.py --output record.mkv

You may optionally specify the camera config when running the recorder script.

When the visualizer window is active, press ``SPACE`` to start or pause the
recording or press ``ESC`` to quit the recorder.

.. image:: https://storage.googleapis.com/open3d-bin/docs/images/azure_kinect_recorder.png
    :alt: azure_kinect_recorder.png

Open3D Azure Kinect MKV Reader
==============================

The recorded MKV file uses K4A's custom format which contains both RGB and depth
information. The regular video player may only support playing back the color channel
or not supporting the format at all. To view the customized MKV file, use the
Open3D Azure Kinect MKV Reader.

Open3D provides Python and C++ example code of Open3D Azure Kinect MKV Reader.
Please see ``examples/cpp/AzureKinectMKVReader.cpp`` and
``examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py``
for details.

.. code-block:: sh

    python examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input record.mkv

.. image:: https://storage.googleapis.com/open3d-bin/docs/images/azure_kinect_mkv_reader.png
    :alt: azure_kinect_mkv_reader.png

Note that even though the recorder records the unaligned raw depth image, the
reader can correctly wrap the depth image to align with the color image.

To convert the MKV video to color and depth image frames, specify the ``--output``
flag.

.. code-block:: sh

    python examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input record.mkv --output frames

.. image:: https://storage.googleapis.com/open3d-bin/docs/images/azure_kinect_mkv_reader_extract.png
    :alt: azure_kinect_mkv_reader_extract.png
