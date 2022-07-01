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

We provide default datasets such as Lounge RGB-D dataset from Stanford, Bedroom RGB-D dataset from Redwood,
Jack Jack RealSense L515 bag file dataset to demonstrate the system in this tutorial.
Other than this, one may user any RGB-D data.
There are lots of excellent RGBD datasets such as: 
`Redwood data <http://redwood-data.org/>`_, `TUM RGBD data <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_, 
`ICL-NUIM data <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>`_, 
`the SceneNN dataset <http://people.sutd.edu.sg/~saikit/projects/sceneNN/>`_ and `SUN3D data <http://sun3d.cs.princeton.edu/>`_.

.. _reconstruction_system_how_to_run_the_pipeline:

Quick start
``````````````````````````````````````
Getting the example code

.. code-block:: sh

    # Activate your conda enviroment, where you have installed open3d pip package.
    # Clone the Open3D github repository and go to the example.
    cd examples/python/reconstruction_system/

    # Show CLI help for `run_system.py`
    python dense_slam_gui.py --help

Running the example with default dataset.

.. code-block:: sh

    # The following command, will download and use the default dataset,
    # which is ``lounge`` dataset from stanford. 
    # --make will make fragments from RGBD sequence.
    # --register will register all fragments to detect loop closure.
    # --refine flag will refine rough registrations.
    # --integrate flag will integrate the whole RGBD sequence to make final mesh.
    # [Optional] Use --slac and --slac_integrate flags to perform SLAC optimisation.
    python run_system.py --make --register --refine --integrate

Changing the default dataset.
One may change the default dataset to other avaialble datasets. 
Currently the following datasets are available:

1. Lounge (keyword: ``lounge``) (Default)

2. Bedroom (keyword: ``bedroom``)

3. Jack Jack (keyword: ``jack_jack``)


.. code-block:: sh

    # Using jack_jack as the default dataset.
    python run_system.py --default_dataset 'bedroom' --make --register --refine --integrate

Running the example with custom dataset using config file.
Manually download or store the data in a folder and store all the color images 
in the ``image`` sub-folder, and all the depth images in the ``depth`` sub-folder. 
Create a ``config.json`` file and set the ``path_dataset`` to the data directory.
Override the parameters for which you want to change the default values.

Example config file for offline reconstruction system has been provided in 
``examples/python/reconstruction_system/config/tutorial.json``, which looks like the following:

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
