.. _builddocs:

Building Documentation
======================

You can build the documentation of Open3D and contribute to the documentation process.

**Prerequisites:**

Make sure you have cloned the repository and have built both the C++ and Python documentation from source.
Visit `compiling from source <http://www.open3d.org/docs/compilation.html>`_ on how to build the source code.

Documentation:
``````````````

The Python documentation is written in
`reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_ and compiled
with `sphinx <http://www.sphinx-doc.org/>`_.

Documentation for C++ API is made with `Doxygen <http://www.doxygen.nl/>`_


Installing Sphinx
`````````````````

.. tip:: Make sure you have pip installed. You can check by typing the 'pip' command in the terminal.

.. code-block:: bash

    pip install sphinx sphinx-autobuild sphinx-rtd-theme

Installing Doxygen
``````````````````

**Ubuntu:**

.. code-block:: bash

    sudo apt-get -y install doxygen

**Windows:**

Visit `Doxygen downloads <http://www.doxygen.nl/download.html>`_ page to checkout the source or binaries.
You can download the latest `binaries <https://sourceforge.net/projects/doxygen/files/snapshots/>`_.

.. tip:: Make sure you have Python x86 installed as the current version of doxygen has issues with x64

You can install a version of `GNU Windows <http://gnuwin32.sourceforge.net/>`_ to perform the 'make html' command in Windows.
You can download make for Windows `here <http://gnuwin32.sourceforge.net/packages/make.htm>`_

Alternatively you can simply use 'python make.py html' command to build if you don't have GNU Windows installed.

Building the Documentation
``````````````````````````

In your Open3D source directory goto ``docs``

To build the Python documentation present at `Open3D <http://www.open3d.org/docs>`_:

.. code-block:: bash

    make html

Or if you are using Windows without GNU Windows installed

.. code-block:: bash

    python make.py html

To build the C++ documentation present at `Open3D C++ API <http://open3d.org/cppapi/index.html>`_:

.. code-block:: bash

    doxygen doxyfile

The documentation is built on CI with `make documentation script <https://github.com/intel-isl/Open3D/blob/master/util/scripts/make-documentation.sh>`_