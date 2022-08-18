import numpy as np
from typing import Any


class TensorMap(dict):

    def __init__(self):
        super().__init__()

    def __setattr__(self, key: str, value: Any) -> None:
        sup = super()
        sup.__setitem__(key, value)

    def __getattr__(self, key: str) -> None:
        sup = super()
        if sup.__contains__(key):
            return sup.__getitem__(key)
        else:
            return None


if __name__ == "__main__":
    tm = TensorMap()
    tm.normals = 100
    print(tm.normals)
    print(tm.points)
