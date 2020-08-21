import os as _os
from . import layers
from . import ops

if 'OPEN3D_ML_ROOT' in _os.environ:
    from ml3d import datasets  # this is for convenience to have everything on the same level
    from ml3d.tf import dataloaders
    from ml3d.tf import models
    from ml3d.tf import pipelines
else:
    # import from the bundled ml3d module
    from open3d._ml3d import datasets  # this is for convenience to have everything on the same level
    from open3d._ml3d.tf import dataloaders
    from open3d._ml3d.tf import models
    from open3d._ml3d.tf import pipelines
