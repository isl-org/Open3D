# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
import pytest

pytest.importorskip("tensorboard")

from open3d.visualization.tensorboard_plugin import summary


def test_async_data_writer_idle_restart(tmp_path):
    """The writer exits when idle and restarts without stranding data."""
    writer = summary._AsyncDataWriter(idle_secs=0.01)
    tagfilepath = str(tmp_path / "geometry")

    filename, first_offset = writer.enqueue(tagfilepath, b"first")
    first_thread = writer._writer_thread
    first_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not writer._writer_running

    second_filename, second_offset = writer.enqueue(tagfilepath, b"second")
    second_thread = writer._writer_thread
    assert second_thread is not first_thread
    assert not second_thread.daemon

    writer.close()

    assert filename == second_filename
    assert first_offset == 0
    assert second_offset == len(b"first")
    assert (tmp_path / filename).read_bytes() == b"firstsecond"