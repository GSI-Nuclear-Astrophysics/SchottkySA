"""Check native Qt reference counting and normal interpreter shutdown."""

import subprocess
import sys
import textwrap

import pytest


def test_qt_void_calls_preserve_none_references():
    pytest.importorskip("PySide6")
    # Isolate native failures so a faulty binding cannot abort the pytest process.
    probe = textwrap.dedent(
        """
        import gc
        import sys

        import PySide6
        from PySide6.QtCore import QTimer

        timer = QTimer()
        for _ in range(20):
            timer.stop()
        gc.collect()
        before = sys.getrefcount(None)
        for _ in range(200):
            timer.stop()
        after = sys.getrefcount(None)
        assert after == before, (
            f"PySide6 {PySide6.__version__}: void Qt calls changed None's "
            f"reference count by {after - before}"
        )
        """
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
