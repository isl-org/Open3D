.. _azure_kinect_record:

Azure Kinect Capturing
----------------------

Dependencies
============

Follow `the guide <https://github.com/microsoft/Azure-Kinect-Sensor-SDK>`_
to install Azure Kinect SDK. You'll also need to install OpenCV.

On Ubuntu:

.. code-block:: bash

    sudo apt install libopencv-dev


On Ubuntu, to use ``libusb`` without root, we need to add a ``udev`` rule.
Create a new file ``/etc/udev/rules.d/51-usb-device.rules`` and add the
following contents.

.. code-block::

SUBSYSTEM=="usb", TAG+="uaccess", GROUP="plugdev"

Now, run ``k4aviewer`` and see if it can detect the camera.

.. code-block:: bash

k4aviewer
