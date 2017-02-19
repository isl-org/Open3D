from py3d import *
import numpy as np
import sys

def test_py3d_eigen():
    print("Test eigen in py3d")

if __name__ == "__main__":
    if len(sys.argv) == 1 or "eigen" in sys.argv:
        test_py3d_eigen()
