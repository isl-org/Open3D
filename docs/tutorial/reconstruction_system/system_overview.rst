.. _reconstruction_system_overview:

System overview
-----------------------------------

The system has 4 main steps:

**Step 1**. :ref:`reconstruction_system_make_fragments`: build local geometric
surfaces (referred to as
fragments) from short subsequences of the input RGBD sequence. This part uses
:ref:`/tutorial/pipelines/rgbd_odometry.ipynb`,
:ref:`/tutorial/pipelines/multiway_registration.ipynb`, and
:ref:`/tutorial/pipelines/rgbd_integration.ipynb`.

**Step 2**. :ref:`reconstruction_system_register_fragments`: the fragments are
aligned in a global space to detect loop closure. This part uses
:ref:`/tutorial/pipelines/global_registration.ipynb`,
:ref:`/tutorial/pipelines/icp_registration.ipynb`, and
:ref:`/tutorial/pipelines/multiway_registration.ipynb`.

**Step 3**. :ref:`reconstruction_system_refine_registration`: the rough
alignments are aligned more tightly. This part uses
:ref:`/tutorial/pipelines/icp_registration.ipynb`, and
:ref:`/tutorial/pipelines/multiway_registration.ipynb`.

**Step 4**. :ref:`reconstruction_system_integrate_scene`: integrate RGB-D images
to generate a mesh model for
the scene. This part uses :ref:`/tutorial/pipelines/rgbd_integration.ipynb`.

.. _reconstruction_system_dataset:

Example dataset
``````````````````````````````````````

We provide default datasets such as Lounge RGB-D dataset from Stanford, 
Jack Jack RealSense L515 bag file dataset to demonstrate the system in this tutorial.
Other than this, one may user any RGB-D data.

There are lots of excellent RGBD datasets such as 
 `Redwood data <http://redwood-data.org/>`_,
`TUM RGBD data <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_,
`ICL-NUIM data <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>`_, 
`the SceneNN dataset <http://people.sutd.edu.sg/~saikit/projects/sceneNN/>`_ and
`SUN3D data <http://sun3d.cs.princeton.edu/>`_.

.. _reconstruction_system_how_to_run_the_pipeline:

Quick start
``````````````````````````````````````
Running the example code
.. code-block:: sh
    # Activate your conda enviroment, where you have installed open3d pip package.
    # Run

Put all color images in the ``image`` folder, and all depth images in the
``depth`` folder. Run the following commands from the root folder.

.. code-block:: sh

    cd examples/python/reconstruction_system/
    python run_system.py [config_file] [--make] [--register] [--refine] [--integrate]

``config_file`` has parameters and file paths. For example,
``reconstruction_system/config/tutorial.json`` has the following script.

.. literalinclude:: ../../../examples/python/reconstruction_system/config/tutorial.json
   :language: json
   :lineno-start: 1
   :lines: 1-
   :linenos:

We assume that the color images and the depth images are synchronized and
registered. ``"path_intrinsic"`` specifies path to a json file that stores the
camera intrinsic matrix (See
:ref:`/tutorial/pipelines/rgbd_odometry.ipynb#read-camera-intrinsic` for
details). If it is not given, the PrimeSense factory setting is used. For your
own dataset, use an appropriate camera intrinsic and visualize a depth image
(likewise :ref:`/tutorial/geometry/rgbd_image.ipynb`) prior to using the system.

.. note:: ``"python_multi_threading": true`` utilizes ``joblib`` to parallelize
    the system using every CPU cores. With this option, Mac users may encounter
    an unexpected program termination. To avoid this issue, set this flag to
    ``false``.


Capture your own dataset
``````````````````````````````````````

This tutorial provides an example that can record synchronized and aligned RGBD
images using the Intel RealSense camera. For more details, please see
:ref:`capture_your_own_dataset`.
