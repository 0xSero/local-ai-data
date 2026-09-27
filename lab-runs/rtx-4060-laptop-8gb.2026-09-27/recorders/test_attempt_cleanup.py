"""No GPU: failed launch + failed log read must still remove the recorded container."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import run_tabby_attempt as attempt

calls = []
container = "a" * 64


def fake_command(argv, **kwargs):
    if argv[:2] == ["docker", "run"]:
        Path(argv[argv.index("--cidfile") + 1]).write_text(container)
        raise KeyboardInterrupt("Injected interruption after container creation")
    if argv[:2] == ["docker", "inspect"]:
        return json.dumps({"Running": False, "Pid": 0, "OOMKilled": False})
    assert argv[0] == "nvidia-smi", argv
    return "simulated GPU sample"


def fake_run(argv, **kwargs):
    calls.append(argv)
    if argv[:2] == ["docker", "logs"]:
        raise OSError("Injected log read failure")
    assert argv[:2] in (["docker", "stop"], ["docker", "rm"]), argv
    assert argv[-1] == container


with tempfile.TemporaryDirectory(dir=attempt.work / "campaign") as directory:
    out = Path(directory) / "attempt"
    # Only small pinned metadata is needed; never require or read model weights.
    fake_work = Path(directory) / "work"
    for name in ["campaign/qwen-launch.json", "campaign/qwen-files-verified.json", "campaign/config/qwen.yml"]:
        target = fake_work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(attempt.work / name, target)
    (fake_work / "run_tabby_validation.py").touch()
    (fake_work / "large-assets/models/Qwen3.5-9B-exl3-22ef1303").mkdir(parents=True)
    with patch.object(attempt, "command", fake_command), patch.object(attempt.sp, "run", fake_run), \
         patch.object(attempt, "work", fake_work), \
         patch.object(attempt.sys, "argv", ["run_tabby_attempt.py", "qwen", "--out", str(out)]), \
         contextlib.redirect_stdout(io.StringIO()):
        assert attempt.main() == 1
    result = json.loads((out / "attempt.json").read_text())
    assert "Injected interruption" in result["error"]
    assert any("Injected log read failure" in value for value in result["cleanup_errors"])
    assert ["docker", "stop", "-t", "10", container] in calls
    assert ["docker", "rm", "--force", container] in calls
print("Cleanup self-check passed; no Docker or GPU commands executed.")
