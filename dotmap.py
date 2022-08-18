import numpy as np
from typing import Any


class TensorMap(dict):
    pass


class TensorMapWrapper(TensorMap):

    def __init__(self):
        super(TensorMapWrapper, self).__init__()

    def __setattr__(self, key: str, value: Any) -> None:
        print(f"setattr {key} {value}")
        super(TensorMapWrapper, self)[key] = value

    def __getattr__(self, key: str) -> None:
        d = super(TensorMapWrapper, self)
        if key in d:
            return d[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from None


if __name__ == "__main__":
    tm = TensorMap()
    tm.normals = 100
    print(tm.normals)
    print(tm.points)
