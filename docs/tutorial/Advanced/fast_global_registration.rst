.. _fast_global_registration:

Fast global registration
-------------------------------------

The RANSAC based :ref:`global_registration` solution may take a long time due to countless model proposals and evaluations.
[Zhou2016]_ introduced a faster approach that quickly optimizes line process weights of few correspondences.
As there is no model proposal and evaluation involved for each iteration, the approach proposed in [Zhou2016]_ can save a lot of computational time.

This script compares the running time of RANSAC based :ref:`global_registration` and implementation of [Zhou2016]_.

.. literalinclude:: ../../../examples/Python/Advanced/fast_global_registration.py
   :language: python
   :lineno-start: 5
   :lines: 5-
   :linenos:

Input
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/fast_global_registration.py
   :language: python
   :lineno-start: 27
   :lines: 27-29
   :linenos:

For the pair comparison, the script reuses the ``prepare_dataset`` function defined in :ref:`global_registration`.
It produces a pair of downsampled point clouds as well as FPFH features.

Baseline
``````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/fast_global_registration.py
   :language: python
   :lineno-start: 31
   :lines: 31-37
   :linenos:

This script calls RANSAC based :ref:`global_registration` as a baseline. After registration it displays the following result.

.. image:: ../../_static/Advanced/fast_global_registration/ransac.png
    :width: 400px

.. code-block:: shell

    RANSAC based global registration took 2.538 sec.

Fast global registration
``````````````````````````````````````

With the same input used for a baseline, the next script calls the implementation of [Zhou2016]_.

.. literalinclude:: ../../../examples/Python/Advanced/fast_global_registration.py
   :language: python
   :lineno-start: 14
   :lines: 14-22
   :linenos:

This script displays the following result.

.. image:: ../../_static/Advanced/fast_global_registration/fgr.png
    :width: 400px

.. code-block:: shell

    Fast global registration took 0.193 sec.

With proper configuration, the accuracy of fast global registration is even comparable with ICP.
Please refer to [Zhou2016]_ for more experimental results.
