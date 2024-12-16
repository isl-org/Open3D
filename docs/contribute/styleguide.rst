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
   # cd to the root of the Open3D folder first.
   pip install -r python/requirements_style.txt

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
* C++17 features are acceptable. Do not use C++20 or later features.

We also recommend reading the `C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_.

For Python, please use Google style guidelines, as shown `here <http://google.github.io/styleguide/pyguide.html>`_.
