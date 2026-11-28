# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "config_device, cli_device, expected",
    [
        ("CUDA:0", None, "CUDA:0"),
        ("CUDA:0", "CPU:0", "CPU:0"),
        (None, None, "CPU:0"),
    ],
)
def test_slac_device_precedence(tmp_path, config_device, cli_device, expected):
    script = (Path(__file__).resolve().parents[3] /
              "examples/python/reconstruction_system/run_system.py")
    (tmp_path / "image").mkdir()
    (tmp_path / "depth").mkdir()
    (tmp_path / "fragments").mkdir()
    config = {"path_dataset": str(tmp_path)}
    if config_device is not None:
        config["device"] = config_device
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = [sys.executable, str(script), "--config", str(config_path), "--slac"]
    if cli_device is not None:
        args.extend(["--device", cli_device])

    # The empty fragments directory stops the real stage before backend setup,
    # so the CLI configuration can be checked without a GPU or RGB-D dataset.
    result = subprocess.run(args,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=60)
    assert result.returncode == 1
    assert "No fragment found" in result.stderr
    devices = re.findall(r"^\s*device\s*:\s*(\S+)$", result.stdout,
                         re.MULTILINE)
    assert [device.upper() for device in devices] == [expected]
