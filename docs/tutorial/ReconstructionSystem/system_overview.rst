.. _reconstruction_system_overview:

System overview
-----------------------------------

The system has three main steps:

**Step 1**. :ref:`reconstruction_system_make_fragments`: build local geometric surfaces (referred to as
fragments) from short subsequences of the input RGBD sequence. This part uses :ref:`rgbd_odometry`, :ref:`multiway_registration`, and :ref:`rgbd_integration`.

**Step 2**. :ref:`reconstruction_system_register_fragments`: the fragments are aligned in a global space. This part uses :ref:`global_registration`, :ref:`icp_registration`, and :ref:`multiway_registration`.

**Step 3**. :ref:`reconstruction_system_integrate_scene`: integrate RGB-D images to generate a mesh model for
the scene. This part uses :ref:`rgbd_integration`.

.. _reconstruction_system_dataset:

Example dataset
``````````````````````````````````````

We use `the SceneNN dataset <http://people.sutd.edu.sg/~saikit/projects/sceneNN/>`_ to demonstrate the system in this tutorial. Alternatively, there are lots of excellent RGBD datasets such as `Redwood data <http://redwood-data.org/>`_, `TUM RGBD data <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_, `ICL-NUIM data <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>`_, and `SUN3D data <http://sun3d.cs.princeton.edu/>`_.

The tutorial uses the 016 sequence from the SceneNN dataset. The sequence is from `SceneNN oni file archieve <https://drive.google.com/drive/folders/0B-aa7y5Ox4eZUmhJdmlYc3BQSG8>`_. The oni file can be extracted into color and depth image sequence using `OniParser from the Redwood reconstruction system <http://redwood-data.org/indoor/tutorial.html>`_. Alternatively, any tool that can convert an .oni file into a set of synchronized RGBD images will work. This is a `quick link <https://drive.google.com/open?id=11U8jEDYKvB5lXsK3L1rQcGTjp0YmRrzT>`_ to download the 016 sequence.

.. _reconstruction_system_how_to_run_the_pipeline:

Quick start
``````````````````````````````````````

Put all color images in the *image* folder, and all depth images in the *depth* folder. Run following commands from the root folder.

.. code-block:: sh

    cd examples/Python/ReconstructionSystem/
    python run_system.py [config_file] [--make] [--register] [--integrate]

``config_file`` has parameters and file paths. For example, ReconstructionSystem/config/redwood.json has the following script.

.. literalinclude:: ../../../examples/Python/ReconstructionSystem/config/redwood.json
   :language: json
   :lineno-start: 1
   :lines: 1-
   :linenos:

We assume the color images and the depth images are synchronized and registered. ``"path_intrinsic"`` specifies path to a json file that stores the camera intrinsic matrix (See :ref:`reading_camera_intrinsic` for details). If it is not given, the PrimeSense factory setting is used. For your own dataset, use an appropriate camera intrinsic and visualize a depth image (likewise :ref:`rgbd_redwood`) prior to use the system.
