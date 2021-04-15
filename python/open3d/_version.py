# Module version
# TODO (yixing): fix me with CMake config
version_info = (0, 1, 12, 'final', 0)

# Module version stage suffix map
_specifier_ = {'alpha': 'a', 'beta': 'b', 'candidate': 'rc', 'final': ''}

# Module version accessible using open3d.__version__
__version__ = '%s.%s.%s%s' % (
    version_info[0], version_info[1], version_info[2], '' if version_info[3]
    == 'final' else _specifier_[version_info[3]] + str(version_info[4]))
