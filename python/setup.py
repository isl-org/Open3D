from __future__ import print_function
from setuptools import setup, find_packages
import os
from os.path import join as pjoin
from distutils import log

from jupyter_packaging import (
    create_cmdclass,
    install_npm,
    ensure_targets,
    combine_commands,
)

here = os.path.dirname(os.path.abspath(__file__))

log.set_verbosity(log.DEBUG)
log.info('setup.py entered')
log.info('$PATH=%s' % os.environ['PATH'])

name = 'open3d'

js_dir = pjoin(here, 'js')

# Representative files that should exist after a successful build
jstargets = [
    pjoin(js_dir, 'dist', 'index.js'),
]

data_files_spec = [
    ('share/jupyter/nbextensions/open3d', 'open3d/nbextension', '*.*'),
    ('share/jupyter/labextensions/open3d', 'open3d/labextension', '**'),
    ('share/jupyter/labextensions/open3d', '.', 'install.json'),
    ('etc/jupyter/nbconfig/notebook.d', '.', 'open3d.json'),
]

cmdclass = create_cmdclass('jsdeps', data_files_spec=data_files_spec)
cmdclass['jsdeps'] = combine_commands(
    install_npm(js_dir, npm=['yarn'], build_cmd='build:prod'),
    ensure_targets(jstargets),
)

# Force platform specific wheel
# https://stackoverflow.com/a/45150383/1255535
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

    cmdclass['bdist_wheel'] = bdist_wheel
except ImportError:
    print("Warning: cannot import `wheel` to build platform-specific wheel. "
          "Run `pip install wheel` to fix this warning.")

# Read requirements.txt
with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f.readlines() if line]

# Read requirements for ML
if '@BUNDLE_OPEN3D_ML@' == 'ON':
    with open('@OPEN3D_ML_ROOT@/requirements.txt', 'r') as f:
        install_requires += [line.strip() for line in f.readlines() if line]

setup_args = dict(
    name="@PYPI_PACKAGE_NAME@",
    version='@PROJECT_VERSION@',
    description='Open3D: A Modern Library for 3D Data Processing.',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    include_package_data=True,
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    cmdclass=cmdclass,
    author='Open3D Team',
    author_email='@PROJECT_EMAIL@',
    url="@PROJECT_HOME@",
    project_urls={
        'Documentation': '@PROJECT_DOCS@',
        'Source code': '@PROJECT_CODE@',
        'Issues': '@PROJECT_ISSUES@',
    },
    keywords="3D reconstruction point cloud mesh RGB-D visualization",
    license="MIT",
    classifiers=[
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Environment :: X11 Applications",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Multimedia :: Graphics :: Capture",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
)

setup(**setup_args)
