.. _reconstruction_system_overview:

System overview
-----------------------------------

The system has 4 main steps:

**Step 1**. :ref:`reconstruction_system_make_fragments`: build local geometric surfaces (referred to as
fragments) from short subsequences of the input RGBD sequence. This part uses :ref:`/tutorial/Basic/rgbd_odometry.ipynb`, :ref:`/tutorial/Advanced/multiway_registration.ipynb`, and :ref:`/tutorial/Advanced/rgbd_integration.ipynb`.

**Step 2**. :ref:`reconstruction_system_register_fragments`: the fragments are aligned in a global space to detect loop closure. This part uses :ref:`/tutorial/Advanced/global_registration.ipynb`, :ref:`/tutorial/Basic/icp_registration.ipynb`, and :ref:`/tutorial/Advanced/multiway_registration.ipynb`.

**Step 3**. :ref:`reconstruction_system_refine_registration`: the rough alignments are aligned more tightly. This part uses :ref:`/tutorial/Basic/icp_registration.ipynb`, and :ref:`/tutorial/Advanced/multiway_registration.ipynb`.

**Step 4**. :ref:`reconstruction_system_integrate_scene`: integrate RGB-D images to generate a mesh model for
the scene. This part uses :ref:`/tutorial/Advanced/rgbd_integration.ipynb`.

.. _reconstruction_system_dataset:

Example dataset
``````````````````````````````````````

We use `the SceneNN dataset <http://people.sutd.edu.sg/~saikit/projects/sceneNN/>`_ to demonstrate the system in this tutorial. Alternatively, there are lots of excellent RGBD datasets such as `Redwood data <http://redwood-data.org/>`_, `TUM RGBD data <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_, `ICL-NUIM data <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>`_, and `SUN3D data <http://sun3d.cs.princeton.edu/>`_.

The tutorial uses sequence ``016`` from the SceneNN dataset. This is a `quick link <https://drive.google.com/open?id=11U8jEDYKvB5lXsK3L1rQcGTjp0YmRrzT>`_ to download the RGBD sequence used in this tutorial. Alternatively, you can download from the original dataset from `SceneNN oni file archive <https://drive.google.com/drive/folders/0B-aa7y5Ox4eZUmhJdmlYc3BQSG8>`_, and then extract the ``oni`` file into color and depth image sequence using `OniParser from the Redwood reconstruction system <http://redwood-data.org/indoor/tutorial.html>`_ or other convertion tools. Some helper scripts can be found from ``ReconstructionSystem/scripts``.

.. _reconstruction_system_how_to_run_the_pipeline:

Quick start
``````````````````````````````````````

Put all color images in the ``image`` folder, and all depth images in the ``depth`` folder. Run following commands from the root folder.

.. code-block:: sh

    cd examples/Python/ReconstructionSystem/
    python run_system.py [config_file] [--make] [--register] [--refine] [--integrate]

``config_file`` has parameters and file paths. For example, ReconstructionSystem/config/tutorial.json has the following script.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/config/tutorial.json
   :language: json
   :lineno-start: 1
   :lines: 1-
   :linenos:

We assume that the color images and the depth images are synchronized and registered. ``"path_intrinsic"`` specifies path to a json file that stores the camera intrinsic matrix (See :ref:`/tutorial/Basic/rgbd_odometry.ipynb#read-camera-intrinsic` for details). If it is not given, the PrimeSense factory setting is used. For your own dataset, use an appropriate camera intrinsic and visualize a depth image (likewise :ref:`/tutorial/Basic/rgbd_image.ipynb`) prior to use the system.

.. note:: ``"python_multi_threading": true`` utilizes ``joblib`` to parallelize the system using every CPU cores. With this option, Mac users may encounter an unexpected program termination. To avoid this issue, set this flag as ``false``.


Capture your own dataset
``````````````````````````````````````

This tutorial provides an example that can records synchronized and aligned RGBD images using Intel Realsense camera. For more details, please see :ref:`capture_your_own_dataset`.
