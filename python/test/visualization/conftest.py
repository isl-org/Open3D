import os
import sys
import shutil
import pytest


def pytest_runtest_setup(item):
    testable = (sys.platform.startswith('linux') and 'DISPLAY' in os.environ)
    if not testable:
        pytest.skip("Cannot run GUI tests without Linux and X11.")
