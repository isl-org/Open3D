import numpy as np
from typing import Any


class TensorMap(dict):

    def __init__(self):
        super().__init__()

    def __setattr__(self, key: str, value: Any) -> None:
        print(f"setattr {key} {value}")
        super()[key] = value

    def __getattr__(self, key: str) -> None:
        d = super()
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
