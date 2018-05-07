# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

from setuptools import setup, find_packages

setup(
    author = 'IntelVCL',
    author_email = 'info@open3d.org',
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Programming Language :: Python :: 3 :: Only",
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
    description = ("Open3D is an open-source library that supports rapid development of software that deals with 3D data.."),
    install_requires=['numpy', 'matplotlib', 'opencv-python', ],
    include_package_data=True,
    keywords = "3D reconstruction pointcloud",
    license = "MIT",
    long_description=open('README.md').read(),
    name = "open3d",
    packages=['open3d', ],
    url = "http://www.open3d.org",
    project_urls={
        'Documentation': 'http://www.open3d.org/docs',
        'Source code': 'https://github.com/IntelVCL/Open3D',
        'Issues': 'https://github.com/IntelVCL/Open3D/issues',
        },
    version = '0.1.0',
)
