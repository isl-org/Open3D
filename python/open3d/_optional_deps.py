# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import importlib.util

_ML_EXTRA_IMPORTS = (
    ("addict", "addict"),
    ("PIL", "pillow"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("pyquaternion", "pyquaternion"),
)


def require_ml_extra():
    """Raise ImportError if Open3D-ML Python dependencies are not installed."""
    for module_name, pip_name in _ML_EXTRA_IMPORTS:
        if importlib.util.find_spec(module_name) is None:
            raise ImportError(
                f"Open3D-ML requires the '{pip_name}' package. Install ML "
                f"dependencies with: pip install 'open3d[ml]' "
                f"(or open3d-cpu[ml] / open3d-xpu[ml] as appropriate)."
            ) from None
