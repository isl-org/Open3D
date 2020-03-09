.. _style_guide:

Open3D style guide
#####################

Coding style
=============

Consistent coding style is an important factor of code readability. Some principles:

1. Code itself is a document. Name functions and variables in a way they are self explanatory.
2. Be consistent with existing code and documents. Be consistent with C++ conventions.
3. Use common sense.

We generally follow the `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_, with a few modifications:

* Use 4 spaces for indent. Use two indents for a forced line break (usually due to the 80 character length limit).
* Use ``#pragma once`` for header guard.
* All Open3D classes and functions are nested in namespace ``open3d``.
* Avoid using naked pointers. Use ``std::shared_ptr`` and ``std::unique_ptr`` instead.
* C++11 features are recommended, but C++14 and C++17 are also accepted.

We also recommend reading the `C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_.

For Python, please use Google style guidelines, as shown `here <http://google.github.io/styleguide/pyguide.html>`_.


Automated style Checker
========================

Open3D's CI checks for code formatting based on the style specified in
``.clang-format`` for C++ files and ``.style.yapf`` for Python files.
Please build the ``check-style`` and ``apply-style``
CMake target before submitting a pull request, or use your editor's
``clang-format`` and ``yapf`` integration to format the source code automatically.

Different ``clang-format`` versions may produce slightly different
formatting results. For standardization, ``clang-format`` version
``5.0`` shall be used.

.. _1-installing-clang-format-50:

Install clang-format
--------------------

By default, the make system tries to detect either ``clang-format-5.0``
or ``clang-format`` from PATH.

.. _11-ubuntu:

Ubuntu
~~~~~~~~~~

.. code:: bash

   # Ubuntu 16.04
   sudo apt update
   sudo apt install clang-format-5.0
   clang-format-5.0 --version

   # Ubuntu 18.04
   sudo apt update
   sudo apt install clang-format-5.0
   clang-format-5.0 --version

.. _12-macos:

macOS
~~~~~~~~~

.. code:: bash

   curl https://raw.githubusercontent.com/intel-isl/Open3D-3rdparty/master/clang-format/clang-format%405.rb -o $(brew --repo)/Library/Taps/homebrew/homebrew-core/Formula/clang-format@5.rb
   brew install clang-format@5
   clang-format --version

   # (Optional) If another clang-format version was previously installed, we can keep
   # both versions and switch the default to version 5
   brew unlink clang-format
   brew link clang-format@5

   # (Optional) If you'd like to uninstall
   brew uninstall clang-format@5

Alternatively, download the clang-5.0 macOS package from `LLVM Download Page`_,
unzip and add the directory containing ``clang-format`` to ``PATH``.

.. _13-windows:

Windows
~~~~~~~~~~~

Download clang-5.0 Windows package from `LLVM Download Page`_. During
installation, select the option which allows adding clang toolchains to
``PATH``. After installation, open a CMD terminal and try

.. code:: batch

   clang-format --version


.. _14-check-version:

Checking clang-format version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After installation, check ``clang-format``'s version with

.. code:: bash

   # In most cases
   clang-format --version

   # Or, when installed as clang-format-5.0, e.g. on Ubuntu
   clang-format-5.0 --version

and make sure that version ``5.0`` is installed.


.. _2-install-yapf:

Install YAPF
-------------------------------

We use `YAPF <https://github.com/google/yapf.git>`_ for Python formatting.
Different YAPF versions may produce slightly different formatting results, thus
we choose version ``0.28.0`` as the standard version to be used.

Install YAPF with

.. code:: bash

   # For Pip
   pip install yapf==0.28.0

   # For conda
   conda install yapf=0.28.0

You can also download `YAPF <https://github.com/google/yapf.git>`_ and install
it from source.


.. _3-checking-and-applying-format:

Checking and applying format
-------------------------------

.. _31-ubuntu--macos:

Ubuntu & macOS
~~~~~~~~~~~~~~~~~~

After CMake config, to check style, run

.. code:: bash

   make check-style

After CMake config, to apply proper style, run

.. code:: bash

   make apply-style

.. _32-windows:

Windows
~~~~~~~~~~~

After CMake config, to check style, run

.. code:: batch

   cmake --build . --target check-style

After CMake config, to apply the proper style, run

.. code:: batch

   cmake --build . --target apply-style

.. _LLVM Download Page: http://releases.llvm.org/download.html
