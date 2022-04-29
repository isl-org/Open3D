import pytest


def pytest_addoption(parser):
    parser.addoption("--skip_sycl_failed_tests",
                     action="store_true",
                     default=False,
                     help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "skip_sycl_failed_tests: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip_sycl_failed_tests"):
        marker = pytest.mark.skip(reason="Temporarily skipped test for SYCL")
        for item in items:
            if "skip_sycl_failed_tests" in item.keywords:
                item.add_marker(marker)
    else:
        return
