.. _headless_rendering:

Headless rendering
-------------------------------------

This tutorial introduces how to render and save images from terminal without any display device.
This feature is experimental. Tested with Ubuntu 16.04 environment.

Install OSMesa
````````````````````````

To generate a headless context, it is necessary to install `OSMesa <https://www.mesa3d.org/osmesa.html>`_.

.. code-block:: shell

	$ sudo apt-get install libosmesa6-dev

Otherwise, recent version of OSMesa can be built from source.

.. code-block:: shell

	# download OSMesa 2018 release
	$ cd
	$ wget ftp://ftp.freedesktop.org/pub/mesa/mesa-18.0.0.tar.gz
	$ tar xf mesa-18.0.0.tar.gz
	$ cd mesa-18.0.0/

	# configure compile option and build
	$ ./configure --enable-osmesa --disable-driglx-direct --disable-gbm --enable-dri --with-gallium-drivers=swrast
	$ make

	# add OSMesa to local path.
	$ export PATH=$PATH:~/mesa-18.0.0


Install virtualenv
````````````````````````

The next step is to make a virtual environment for Python.

.. code-block:: shell

	$ sudo apt-get install python3-tk virtualenv python-pip
	$ virtualenv -p /usr/bin/python3 py3env
	$ source py3env/bin/activate
	(py3env) $ pip install numpy matplotlib

This script installs and activates ``py3env``. Necessary modules, ``numpy`` and ``matplotlib`` are installed on ``py3env``.

.. Error:: Anaconda users recommended to use this configuration as ``conda install matplotlib`` installs additional modules that is not based on OSMesa. This will make **segmentation fault error** at the runtime.


Build Open3D with OSMesa
````````````````````````

To notify cmake to link OSMesa, open ``~/Open3D/src/CMakeLists.txt`` and change ``Open3D_HEADLESS_RENDERING`` flag to ``ON``.

.. code-block:: shell

	(py3env) $ vi ~/Open3D/src/CMakeLists.txt
	# option(Open3D_HEADLESS_RENDERING "Use OSMesa for headless rendering" ON)

Let's move to build folder.

.. code-block:: shell

	(py3env) $ cd ~/Open3D/
	(py3env) $ mkdir build && cd build

As ``Open3D_HEADLESS_RENDERING`` is ``ON``, ``-DOpen3D_USE_NATIVE_DEPENDENCY_BUILD=OFF`` will build glew and glfw from source. It will link them to OSMesa.

.. code-block:: shell

	# force to build dependencies from source
	(py3env) $ cmake -DOpen3D_USE_NATIVE_DEPENDENCY_BUILD=OFF -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 ../src
	(py3env) $ make # or make -j for multi-core machine

Note that ``-DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3`` is the same path what was set for python virtual environment.


Test headless rendering
````````````````````````

As a final step, test a python script that saves depth and surface normal sequences.

.. code-block:: shell

	(py3env) $ cd ~/Open3D/build/lib/Tutorial/Advanced/
	(py3env) $ python headless_rendering.py

This should print the following:

.. code-block:: shell

	Capture image 00000
	Capture image 00001
	Capture image 00002
	Capture image 00003
	Capture image 00004
	Capture image 00005
	:
	Capture image 00030

Rendered images are at ~/Open3D/build/lib/TestData/depth and image folder.

.. Note:: ``headless_rendering.py`` saves png files. This may take some time. Try tweak the script for your purpose.

.. Error:: If glew and glfw did not correctly linked with OSMesa, it may crash with following error. **GLFW Error: X11: The DISPLAY environment variable is missing. Failed to initialize GLFW**
