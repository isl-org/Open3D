import os as _os
if 'OPEN3D_ML_ROOT' in _os.environ:
    from ml3d.datasets import *
else:
    from open3d._ml3d import datasets
