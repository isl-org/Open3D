.. _t_reconstruction_system:

Reconstruction system (Tensor)
===================================================================

This tutorial demonstrates volumetric RGB-D reconstruction and dense RGB-D SLAM with the Open3D :ref:`/tutorial/core/tensor.ipynb` interface and the Open3D :ref:`/tutorial/core/hashmap.ipynb` backend.

It is possible to run the tutorial with the minimalistic dataset ``SampleRedwoodRGBDImages``, but it is recommended to run the tutorial with real-world datasets with longer sequences to demonstrate its capability. Please refer to :ref:`/tutorial/geometry/rgbd_image.ipynb` for more available datasets. The ``Redwood`` dataset can be a good starting point.

If you use any part of the tensor-based reconstruction system or the hash map backend in Open3D, please cite [Dong2021]_::

  @article{Dong2021,
      author    = {Wei Dong, Yixing Lao, Michael Kaess, and Vladlen Koltun}
      title     = {{ASH}: A Modern Framework for Parallel Spatial Hashing in {3D} Perception},
      journal   = {arXiv:2110.00511},
      year      = {2021},
  }

.. note::
   As of now the tutorial is only for **online** dense SLAM, and **offline** integration **with** provided poses. The tutorials for tensor-based **offline** reconstruction system, Simultaneous localization and calibration (SLAC), and shape from shading (SfS) tutorials as mentioned in [Dong2021]_ are still under construction. At current, please refer to :ref:`reconstruction_system` for the legacy versions.

Quick start
``````````````````````````````````````
Getting the example code

.. code-block:: sh

    # Activate your conda enviroment, where you have installed open3d pip package.
    # Clone the Open3D github repository and go to the example.
    cd examples/python/t_reconstruction_system/

    # Show CLI help for ``dense_slam_gui.py``
    python dense_slam_gui.py --help

Running the example with default dataset.

.. code-block:: sh

    # The following command, will download and use the default dataset,
    # which is ``lounge`` dataset from stanford. 
    python dense_slam_gui.py 

It is recommended to use CUDA if avaialble.

.. code-block:: sh

    # The following command, will download and use the default dataset,
    # which is ``lounge`` dataset from stanford. 
    python dense_slam_gui.py --device 'cuda:0'

Changing the default dataset.
One may change the default dataset to other avaialble datasets. 
Currently the following datasets are available:

1. Lounge (keyword: ``lounge``) (Default)

2. Bedroom (keyword: ``bedroom``)

3. Jack Jack (keyword: ``jack_jack``)

.. code-block:: sh

    # Using jack_jack as the default dataset.
    python dense_slam_gui.py --default_dataset 'bedroom'


Running the example with custom dataset using config file.
Manually download or store the data in a folder and store all the color images 
in the ``image`` sub-folder, and all the depth images in the ``depth`` sub-folder. 
Create a ``config.yml`` file and set the ``path_dataset`` to the data directory.
Override the parameters for which you want to change the default values.

Example config file for online reconstruction system has been provided in 
``examples/python/t_reconstruction_system/default_config.yml``, which looks like the following:

.. literalinclude:: ../../../examples/python/t_reconstruction_system/default_config.yml
   :language: yml
   :lineno-start: 1
   :lines: 1-
   :linenos:

Capture your own dataset
``````````````````````````````````````

This tutorial provides an example that can record synchronized and aligned RGBD
images using the Intel RealSense camera. For more details, please see
:ref:`capture_your_own_dataset`.

Getting started with online reconstruction system
``````````````````````````````````````

.. toctree::

   voxel_block_grid
   integration
   customized_integration
   ray_casting
   dense_slam
