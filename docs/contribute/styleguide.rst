.. _style_guide:

Open3D style guide
##################

Style checker
=============

Install dependencies
--------------------

.. code:: bash

   conda activate <your-virtual-env>

   # The version of the style checker is critical.
   pip install -U clang-format==10.0.1.1 yapf==0.30.0 nbformat==5.1.3

Check or apply style
--------------------

Option 1: Run the style checker directly.

.. code:: bash

   python util/check_style.py
   python util/check_style.py --apply

Option 2: Configure the project and run ``make``.

.. code:: bash

   mkdir build
   cd build
   cmake ..

   # Ubuntu/macOS
   make check-style
   make apply-style

   # Windows
   cmake --build . --target check-style
   cmake --build . --target apply-style

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
