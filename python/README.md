Open3D
===============================

Open3D: A Modern Library for 3D Data Processing

Installation
------------

To install use pip:

    $ pip install open3d

For a development installation (requires [Node.js](https://nodejs.org) and [Yarn version 1](https://classic.yarnpkg.com/)),

    $ git clone https://github.com/intel-isl/Open3D.git
    $ cd Open3D
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --overwrite --sys-prefix open3d
    $ jupyter nbextension enable --py --sys-prefix open3d

When actively developing your extension for JupyterLab, run the command:

    $ jupyter labextension develop --overwrite open3d

Then you need to rebuild the JS when you make a code change:

    $ cd js
    $ yarn run build

You then need to refresh the JupyterLab page when your javascript changes.
