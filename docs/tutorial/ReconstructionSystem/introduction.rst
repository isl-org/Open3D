.. _reconstruction_system_:

Introduction
-------------------------------------

System overview
``````````````````````````````````````

This is brief introduction of each core module.

**Step 1**, :ref:`reconstruction_system_make_fragments`: from RGBD sequence, the pipeline builds a hierarchical registration scheme to reconstruct scene. It first builds small fragments that is alignment of 100 RGBD frame, which is corresponds to about 3.3 seconds of length with 30fps RGBD camera.

**Step 2**, :ref:`reconstruction_system_register_fragments`: the fragments are then registered in global manner to locate the fragment onto right place. The principal purpose of building fragments and register fragments is to recover accurate RGBD camera trajectory.

**Step 3**, :ref:`reconstruction_system_integrate_scene`: as a final step, the pipeline generates beautiful mesh model of the scene by integrating RGBD sequence and estimated RGBD camera trajectory.

The system includes :ref:`utilities` a python helper functions for utilities.


.. _reconstruction_system_dataset:

Example dataset
``````````````````````````````````````

The example dataset for this tutorial is one of SceneNN RGBD sequence. The tutorial uses 016 sequence.
The sequence can be found from `SceneNN oni file archieve <https://drive.google.com/drive/folders/0B-aa7y5Ox4eZUmhJdmlYc3BQSG8>`_.

The oni file can be extracted into color and depth image sequence using OniParser.
OniParser is part of `Reconstruction system <http://redwood-data.org/indoor/data/indoor-executables-1.1.zip>`_ proposed by [Choi2015]_, and its tutorial about command line arguments can be found from `here <http://redwood-data.org/indoor/tutorial.html>`_.


.. _reconstruction_system_how_to_run_the_pipeline:

How to run the pipeline
``````````````````````````````````````

If the dataset folder having *image* and *depth* is prepared, simply run following commands.

.. code-block:: shell

	cd [your_path_to_py3d_lib]/Tutorial/ReconstructionSystem/make_fragments.py

	python make_fragments.py [path_to_dataset] [-path_intrinsic (optional)]

	python register_fragments.py [path_to_dataset]

	python integrate_scene.py [path_to_dataset] [-path_intrinsic (optional)]

The tutorial explains each script in the next.
