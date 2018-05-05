# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

from setuptools import setup, find_packages


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    author = 'IntelVCL',
    author_email = 'info@open3d.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers, Developers',
        'Topic :: Software Development :: 3D Data manipulation and visualization',
        'License :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        "Operating System :: POSIX :: Linux",
        "Core features :: Basic 3D data structures",
        "Core features :: Basic 3D data processing algorithms",
        "Core features :: Scene reconstruction",
        "Core features :: Surface alignment",
        "Core features :: 3D visualization",
        "Core features :: Python binding",
        "Supported compilers :: GCC 4.8 and later on Linux",
        "Supported compilers :: XCode 8.0 and later on OS X",
        "Supported compilers :: Visual Studio 2015 and later on Windows",
        "Resources :: Website :: www.open3d.org",
        "Resources :: Code :: github.com/IntelVCL/Open3D",
        "Resources :: Document :: www.open3d.org/docs",
        "Resources :: License :: The MIT license",
    ],
    description = ("Open3D is an open-source library that supports rapid development of software that deals with 3D data.."),
    # distclass=BinaryDistribution,
    install_requires=['numpy', 'matplotlib', 'opencv-python', ],
    include_package_data=True,
    keywords = "3D reconstruction pointcloud",
    license = "MIT",
    long_description=open('README.md').read(),
    # long_description_content_type='text/x-rst',
    name = "open3d",
    packages=['open3d', ],
    # packages=find_packages(),
    url = "http://www.open3d.org",
    project_urls={
        'Documentation': 'http://www.open3d.org/docs',
        'Source code': 'https://github.com/IntelVCL/Open3D',
        'Issues': 'https://github.com/IntelVCL/Open3D/issues',
        },
    version = '0.1.0',
)
