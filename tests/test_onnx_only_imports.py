"""Check that the ONNX-serving import path does not eagerly import PyTorch."""

import os
import subprocess
import sys


def test_onnx_api_import_does_not_import_torch():
    env = dict(os.environ, ISSUE_INTENT_BACKEND="onnx-int8")
    script = (
        "import sys; "
        "import src.api.main; "
        "assert 'torch' not in sys.modules, 'ONNX API eagerly imported torch'; assert 'transformers' not in sys.modules, 'ONNX API eagerly imported transformers'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
