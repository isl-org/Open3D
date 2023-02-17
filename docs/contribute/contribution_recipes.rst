.. _contribution_recipes:

Contribution methods
####################

Code contribution
=================

You have used Open3D. You are familiar with the Open3D C++ and/or Python interface(s). You want to contribute code for a new feature or a bug fix. The easiest way to start as a new contributor is to take an existing code snippet as a reference and write some code similar to it. Follow the procedure below.

Recommended procedure
---------------------

1. Download, build from source, and familiarize yourself with Open3D.
2. Read the :ref:`style_guide` and install required tools.
3. Check existing classes, examples, and related code.
4. Fork Open3D on `GitHub <https://github.com/isl-org/Open3D>`_.
5. Create new features in your fork. Do not forget unit tests and documentation. Double-check the :ref:`style_guide`.
6. Make a pull request to the `master branch <https://github.com/isl-org/Open3D/tree/master>`_.
7. Make sure your PR passes the CI tests. If it doesn’t, fix the code until it builds and passes the CI tests.
8. Your PR will be assigned to reviewers.
9. Engage with your reviewers during the review process. Address issues or concerns raised during the review. Don’t let the review die.

Congratulations, your pull request has been merged. You are now part of Open3D!

Dos
---

+-------------------------------------------------------------------------------------------------------------+
| [DO] Follow the :ref:`style_guide` and install the required tools                                           |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Use C++14 features when contributing C++ code                                                          |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Remember to provide Python bindings when adding new C++ core functionalities                           |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Use Python 3 when contributing Python code                                                             |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Be aware of OS and compiler versions. Please check :ref:`compiler_version` for details.                |
+-------------------------------------------------------------------------------------------------------------+
| | [DO] Provide suitable unit tests                                                                          |
| |  - Your contribution must come with unit tests of the most important functionality                        |
| |  - If you are modifying existing code, make sure that the unit tests are updated accordingly              |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Use object-oriented design                                                                             |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Minimize dependencies. Do not pull in heavy dependencies. Make the code as self-contained as possible  |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Keep your pull request as small and self-contained as possible                                         |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Write simple code that is clean and easy to understand                                                 |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Minimize redundancies in your implementation                                                           |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Avoid “premature optimization” that may hinder code readability                                        |
+-------------------------------------------------------------------------------------------------------------+

Don’ts
------

+-------------------------------------------------------------------------------------------------------------------------+
| [DON'T]  Do not contribute Python 2.7 code. Python 2.7 has been deprecated                                              |
+-------------------------------------------------------------------------------------------------------------------------+
| | [DON'T]  Do not break the library                                                                                     |
| |  - When submitting a pull request, make sure that CI is able to build your PR without any error or warning            |
| |  - We know that eliminating all warnings can be hard. Please try. This increases stability and speeds up compilation  |
+-------------------------------------------------------------------------------------------------------------------------+
| [DON'T]  Do not create a massive pull request with dozens of commits and files                                          |
+-------------------------------------------------------------------------------------------------------------------------+

.. _review_contribution:

Code reviews
============

You want to contribute to Open3D by reviewing code. Your mission is to help developers comply with Open3D standards. If you are new to this, make sure you have a good understanding of code review. (See this `excellent introduction <https://google.github.io/eng-practices/review/reviewer/>`_.) Follow the procedure below.

Recommended procedure
---------------------

 1. Check the list of `open pull requests <https://github.com/isl-org/Open3D/pulls>`_ and pick one that doesn’t yet have a reviewer. Leave a comment on the PR mentioning your interest: e.g., “I could help review this PR.”
 2. A project maintainer will assign you to the PR as a reviewer.
 3. Use `Reviewable <https://reviewable.io/reviews>`_ to perform the code review.
 4. When you begin the review, post a comment to indicate that you have started: e.g., “Starting to review this PR.”
 5. Make sure that CI is able to build the PR and that tests pass.
 6. For features that are not covered by tests and CI (e.g., visualizer), download the code, run it, and inspect.
 7. Look for unit tests. If unit tests are not provided, ask the author to add them.
 8. Check for good documentation: doxygen/docstring comments.
 9. Check for good design.

    a. Object-oriented design.
    b. Code is simple and clear without premature optimizations.
    c. The implementation is not redundant.

 10. Verify compliance with the :ref:`style_guide`.
 11. Provide clear feedback to the author and make suggestions for improving the PR.
 12. If the PR gets frozen for a while (more than a week), ping the author to revive the process.
 13. When everything is correct, give your **:LGTM:**.

Congratulations, you have improved Open3D with your review. You are now part of Open3D!

Dos
---

+-------------------------------------------------------------------------------------------------------------+
| [DO] Uphold the highest standards of quality                                                                |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Implement the :ref:`principles`                                                                        |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Help the code’s author improve their contribution                                                      |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Point out algorithm/API/design/style/build/other issues that need to be improved                       |
+-------------------------------------------------------------------------------------------------------------+
| [DO] Make sure the contribution comes with documentation / updates documentation                            |
+-------------------------------------------------------------------------------------------------------------+

Don’ts
------

+-----------------------------------------------------------------------------------------------------------------------------------+
| [DON’T] Do not approve just to be nice. Do not compromise on quality. Do not compromise the :ref:`principles` of Open3D           |
+-----------------------------------------------------------------------------------------------------------------------------------+

.. _report_contribution:

Bug reports
===========

You are using Open3D. You are not getting the results you want. You think there is a bug, or a missing feature. You want to get support. Good! Please follow the procedure below.

Recommended procedure
---------------------

 1. Check the Open3D GitHub repository to see if there is already a related issue

    a. If there is an existing issue, add a comment explaining the problem you encountered
    b. You can also join our `discord channel <https://discord.gg/D35BGvn>`_ to ask questions. Other community members may have encountered the same issue and may be able to provide a solution

 2. If you cannot find an existing ticket, please file your bug report on the GitHub issues board. Your report should include the following elements:

    a. A description of the problem
    b. A description of your environment: OS, Python version, compiler, Open3D version, installation method.
    c. A minimal example to reproduce the problem.
    d. The obtained output. Feel free to include screenshots.
    e. A description of the expected result.

 3. The Open3D team will explicitly acknowledge the receipt of the bug report by commenting on the issue.

Congratulations, you have improved Open3D with your report. Thanks for making Open3D better!

Dos
---

+---------------------------------------------------------------------------------------------------------------------------------+
| [DO] Always include a minimal example that reproduces the error                                                                 |
+---------------------------------------------------------------------------------------------------------------------------------+
| [DO] Provide information about your environment, so that we can detect problems related to compilers, dependencies, etc.        |
+---------------------------------------------------------------------------------------------------------------------------------+
| | [DO] Indicate the output you were expecting                                                                                   |
| |   - Sometimes there are misunderstandings and the library provides you with a different output than the expected one          |
+---------------------------------------------------------------------------------------------------------------------------------+

Don’ts
------

+---------------------------------------------------------------------------------------------------------------------------------------+
| [DON’T] Do not open a new issue without double-checking whether there is already an existing issue that deals with the same problem   |
+---------------------------------------------------------------------------------------------------------------------------------------+

.. _documentation_contribution:

Documentation
=============

Recommended procedure
---------------------

1. Follow the general code contribution guidelines.
2. Follow the :ref:`builddocs` instructions to build both C++ and Python documentation. Make sure you can view the generated local web pages with a browser.
3. Adhere to the  cases presented below.
4. Iterate steps 2 and 3 to build the docs and see the generated results. Make sure the syntax is correct so that the expected web page is generated.

Case 1: When documenting C++ code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* In header files, use `Doxygen syntax <http://www.doxygen.nl/manual/docblocks.html>`_. In C++ header files we use the in-line style with `///` for docstring blocks. Make sure to use `Doxygen commands <http://www.doxygen.nl/manual/commands.html>`_ whenever possible. For instance, use `\brief` to denote the brief summary, `\param` to define a parameter, `\return` to define the return value,  `\p` to reference a parameter, `\ref` to reference another function, etc.
* Use `Markdown <https://www.doxygen.nl/manual/markdown.html>`__ for formatting longer descriptions, such as lists and example code. Use LaTeX syntax for Math equations.
* See for example this `Calculator` class:

.. _calculator_class:

.. code:: cpp

    class Calculator {
    public:
        /// \brief Computes summation.
        ///
        /// Performs addition of \p a and \p b, i.e. \f$ c=a+b \f$. Unlike \ref sub, \ref add is commutative.
        /// For example code, leave a blank line and indent 4 spaces for Markdown code formatting.
        /// \param a LHS operand for summation.
        /// \param b RHS operand for summation.
        /// \return The sum of \p a and \p b.
        /// \example
        ///
        ///     c = add(5, 4);
        int add(int a, int b) { return a + b; }

        /// \brief Computes subtraction.
        ///
        /// If detailed description is needed, add a blank line after the "brief"
        /// section. Subtracts \p b from \p a, i.e. \f$ c=a-b \f$.
        /// \param a LHS operand for subtraction.
        /// \param b RHS operand for subtraction.
        /// \return The difference of \p a and \p b.
        /// \example
        ///
        ///     c = sub(5, 4);
        int sub(int a, int b) { return a - b; }
    };

* Add in-line comments to cpp files to explain complex or non-intuitive parts of your algorithm.

Case 2: When documenting Python bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* First, complete the Python binding code following the guides from `Pybind11 Docs <https://pybind11.readthedocs.io/en/stable/basics.html>`_. Make sure to write the high-level docstrings for the classes and functions. Also use ``"param_name"_a`` to denote function parameters.  Use standard RST based docstring syntax (`Google style <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`__) as explained `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ and `there <https://www.python.org/dev/peps/pep-0257/>`_.
* Use the ``docstring::ClassMethodDocInject()`` or ``docstring::FunctionDocInject()`` to insert parameter docs.
* Example binding and docstrings for the ``Calculator`` class:

..  code:: cpp

    py::class_<Calculator> calculator(
                m, "Calculator",
                "Calculator class performs numerical computations.");
    calculator.def("add", &Calculator::Add,
                   "Performs ``a`` plus ``b``, i.e. :math:`c=a+b` Unlike "
                   ":math:`open3d.Calculator.sub`, "
                   ":math:`open3d.Calculator.add` is "
                   "commutative.",
                                "a"_a, "b"_a);
    calculator.def("sub", &Calculator::Add, "Subtracts ``b`` from ``a``,"
                   " i.e. :math:`c=a-b`",
                   "a"_a,
                   "b"_a);
    docstring::ClassMethodDocInject(m, "Calculator", "add",
                                    {{"a", "LHS operand for summation."},
                                     {"b", "RHS operand for summation."}});
    docstring::ClassMethodDocInject(m, "Calculator", "sub",
                                    {{"a", "LHS operand for subtraction."},
                                     {"b", "RHS operand for subtraction."}});

Case 3: When documenting pure Python code (no bindings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Use standard docstring syntax (`Google style <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`__) as explained `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ and `there <https://www.python.org/dev/peps/pep-0257/>`_.

Case 4: When adding a Python tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Place your tutorial notebook within ``Open3D/docs/jupyter/``
* Inside ``Open3D/docs/tutorial``, update the ``toctree`` directive within the
  appropriate ``index.rst`` file
* Update the ``index.rst`` file to include your new tutorial

.. note::
   When you commit a ipynb notebook file make sure to remove the output cells
   to keep the commit sizes small.
   You can use the script ``docs/jupyter/jupyter_strip_output.sh`` for
   stripping the output cells of all tutorials.

Dos
---

+---------------------------------------------------------------------------------------------------------------------------------+
| [DO] Always use a spell checker when writing documentation (e.g. `Grammarly <https://app.grammarly.com/>`_).                    |
+---------------------------------------------------------------------------------------------------------------------------------+
