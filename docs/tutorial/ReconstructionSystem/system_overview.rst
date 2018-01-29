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

The tutorial uses the 016 sequence from the SceneNN dataset. The sequence can be found from `SceneNN oni file archieve <https://drive.google.com/drive/folders/0B-aa7y5Ox4eZUmhJdmlYc3BQSG8>`_. The oni file can be extracted into color and depth image sequence using `OniParser from the Redwood reconstruction system <http://redwood-data.org/indoor/tutorial.html>`_. Alternatively, any tool that can convert an .oni file into a set of synchronized RGBD images will work.

.. _reconstruction_system_how_to_run_the_pipeline:

Quick start
``````````````````````````````````````

Put all color images in the *image* folder, and all depth images in the *depth* folder. Run following commands from the root folder.

.. code-block:: sh

	cd <your_path_to_py3d_lib>/Tutorial/ReconstructionSystem/
	python make_fragments.py [path_to_dataset] [-path_intrinsic (optional)]
	python register_fragments.py [path_to_dataset]
	python integrate_scene.py [path_to_dataset] [-path_intrinsic (optional)]
