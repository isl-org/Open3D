.. _contribute:

Contributing to Open3D
##########################

The Open3D project was started by `Qianyi Zhou <http://qianyi.info>`_ and `Jaesik Park <http://jaesik.info>`_. The project is developed and maintained by a community effort. To contribute to the project, you need to

* Know how to use Open3D;
* Know basic development rules such as coding style, issues, and pull requests;
* Be willing to follow the guidelines in this page.

Issues and pull requests
===========================

The ``master`` branch is used only for stable development versions of Open3D. Any code change is made through four steps using the `issues and pull requests system <https://help.github.com/categories/collaborating-with-issues-and-pull-requests/>`_.

1. An `issue <https://github.com/IntelVCL/Open3D/issues>`_ is opened for a feature request or a bug fix.
2. A contributor starts a new branch or forks the repository, makes changes, and submits a `pull request <https://github.com/IntelVCL/Open3D/pulls>`_.
3. Code change is reviewed and discussed in the pull request. Modifications are made to address the issues raised in the discussion.
4. One of the admins merges the pull request to the master branch.

.. Tip:: Code review is known to be the best practice to maintain the usability and consistency of a large project. Though it takes some time at the beginning, it saves a lot of time in the long run. For new contributors, it can be viewed as a training procedure, in which an experienced contributor, as a reviewer, walks the new contributor through everything he needs to know.

.. Note:: There is an exception for this rule. Small changes can be made directly to the ``master`` branch, such as fixing typos, formatting documents, and fixing an obvious bug.

Maintain sanity of the project
===============================

Most importantly, do not break the build. Before submitting a pull request, make sure the project builds **without any error or warning** under the following toolchains:

* Windows, Visual Studio 2015+, CMake 3.0+
* OS X, Clang included in the latest Xcode, CMake 3.0+
* Ubuntu 16.04, native gcc (4.8+ or 5.x), CMake 3.0+

For C++ code, it is recommended to use C++11 features. However, do not use C++14 or C++17 features since some of them are not properly supported by mainstream compilers.

For Python code, make sure it runs on both Python 2.7 and Python 3.x.

.. note:: The easiest way to start coding as a new contributor is to take an existing code snippet as reference and write some code similar to it.

Coding style
=============

Consistent coding style is an important factor of code readability. Some principles:

1. Code itself is a document. Name functions and variables in a way they are self explanatory.
2. Be consistent with existing code and documents. Be consistent with C++ conventions.
3. Use common sense.

We generally follow the `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_, with a few modifications:

* Use tab for indent. In IDE it should be 4 spaces wide. Use two indents for a forced line break (usually due to the 80 character length limit).
* Use ``#pragma once`` for header guard.
* All Open3D classes and functions are nested in namespace ``three``.
* Avoid using naked pointers. Use ``std::shared_ptr`` and ``std::unique_ptr`` instead.
* C++11 features are recommended.

Another good reading for modern C++ coding style is `C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_
