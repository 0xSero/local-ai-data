"""Run one pinned owner validation, retaining container and resource evidence."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess as sp
import sys
import threading
import time
import urllib.request

work = Path(__file__).resolve().parent
gpu_fields = "name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.default_limit,enforced.power.limit,clocks.current.graphics,clocks.current.memory,pstate"


def command(argv, **kwargs):
    return sp.check_output(argv, text=True, **kwargs).strip()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("qwen", "gemma"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert (work / "run_tabby_validation.py").is_file(), "Validation wrapper must exist before launch."
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    launch = json.loads((work / f"campaign/{args.model}-launch.json").read_text())
    config = work / f"campaign/config/{args.model}.yml"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == launch["config"]["sha256"]
    verified = json.loads((work / f"campaign/{args.model}-files-verified.json").read_text())
    assert verified["revision"] == launch["weights"]["revision"]
    model = work / "large-assets/models" / Path(launch["weights"]["at"]).name
    assert model.is_dir()
    cidfile = out / "container-id.local.txt"
    argv = ["docker", "run", "--detach", "--cidfile", str(cidfile), "--pull=never", "--runtime=nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "--memory=8g", "--memory-swap=16g", "--shm-size=" + launch["shm"],
            "-p", f"127.0.0.1:18089:{launch['port']}",
            "--mount", f"type=bind,src={config},dst={launch['config']['at']},readonly",
            "--mount", f"type=bind,src={model.resolve()},dst={launch['weights']['at']},readonly",
            "--entrypoint", launch["entrypoint"], launch["image"], *launch["args"]]
    assert not launch.get("env"), "Pass any newly introduced profile environment explicitly."
    write(out / "command.local.json", argv)
    write(out / "launch.json", launch)
    write(out / "weights-verified.json", verified)
    write(out / "host.json", {
        "kernel": os.uname().release, "architecture": os.uname().machine,
        "os_release": Path("/etc/os-release").read_text(),
        "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        "ac_online": Path("/sys/class/power_supply/ACAD/online").read_text().strip(),
        "platform_profile": Path("/sys/firmware/acpi/platform_profile").read_text().strip(),
        "memory_limit_bytes": 8 * 2**30, "memory_plus_swap_limit_bytes": 16 * 2**30,
        "gpu_csv_fields": gpu_fields.split(","), "resource_sample_interval_seconds": 2,
    })
    container = child = cgroup = None
    stop = threading.Event()
    result = {"model": args.model, "lab_started": False}

    def sample():
        with (out / "resources.jsonl").open("w") as stream:
            while not stop.is_set():
                record = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "monotonic": time.monotonic()}
                try:
                    record["gpu"] = command(["nvidia-smi", "--query-gpu=" + gpu_fields, "--format=csv,noheader,nounits"], timeout=10)
                    record["meminfo"] = Path("/proc/meminfo").read_text()
                    record["memory_pressure"] = Path("/proc/pressure/memory").read_text()
                    if cgroup:
                        record["cgroup"] = {name: (cgroup / name).read_text() for name in (
                            "memory.current", "memory.peak", "memory.max", "memory.swap.current", "memory.swap.max",
                            "memory.events", "memory.stat", "memory.pressure") if (cgroup / name).exists()}
                except Exception as exc:
                    record["error"] = repr(exc)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                stop.wait(2)

    def save_server_log():
        with (out / "server.log").open("w") as stream:
            sp.run(["docker", "logs", "--timestamps", container], stdout=stream, stderr=sp.STDOUT, check=True, timeout=30)

    def interrupted(*_):
        raise KeyboardInterrupt("Attempt interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        container = command(argv)
        state = json.loads(command(["docker", "inspect", "--format", "{{json .State}}", container]))
        group = Path(f"/proc/{state['Pid']}/cgroup").read_text().splitlines()
        cgroup = Path("/sys/fs/cgroup") / next(line.split(":", 2)[2].lstrip("/") for line in group if line.startswith("0::"))
        deadline = time.monotonic() + 3600
        while True:
            state = json.loads(command(["docker", "inspect", "--format", "{{json .State}}", container]))
            if not state["Running"]:
                raise RuntimeError(f"Server stopped before readiness: {state}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18089/v1/model", timeout=5) as response:
                    loaded = json.load(response)
                write(out / "loaded-model.json", loaded)
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model readiness exceeded one hour")
                time.sleep(2)
        save_server_log()
        packages = "import importlib.metadata as m,json; print(json.dumps({x.metadata['Name']:x.version for x in m.distributions() if any(k in x.metadata['Name'].lower() for k in ['torch','exllama','flash','transformers','tabby'])},indent=2))"
        (out / "packages.json").write_text(command(["docker", "exec", container, launch["entrypoint"], "-c", packages]) + "\n")
        result["lab_started"] = True
        with (out / "lab-console.log").open("w") as stream:
            child = sp.Popen([sys.executable, str(work / "run_tabby_validation.py"), args.model,
                              "--endpoint", "http://127.0.0.1:18089", "--out", str(out / "lab"),
                              "--server-log", str(out / "server.log")], stdout=stream, stderr=sp.STDOUT)
            result["lab_exit_code"] = child.wait()
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        cleanup_errors = []
        if child and child.poll() is None:
            try:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except sp.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            except Exception as exc:
                cleanup_errors.append(f"lab termination: {exc!r}")
        if not container and cidfile.exists():
            container = cidfile.read_text().strip()
        if container:
            # A failed evidence read must never skip cleanup of this attempt's container.
            actions = [
                ("server log", save_server_log),
                ("state before stop", lambda: result.update(container_state_before_stop=json.loads(command(["docker", "inspect", "--format", "{{json .State}}", container], timeout=30)))),
                ("final cgroup", lambda: result.update(cgroup_final={name: (cgroup / name).read_text() for name in ("memory.events", "memory.peak", "memory.swap.current", "memory.pressure") if cgroup and (cgroup / name).exists()})),
                ("stop", lambda: sp.run(["docker", "stop", "-t", "10", container], check=True, stdout=sp.DEVNULL, timeout=30)),
                ("state after stop", lambda: result.update(container_state_after_stop=json.loads(command(["docker", "inspect", "--format", "{{json .State}}", container], timeout=30)))),
                ("remove", lambda: sp.run(["docker", "rm", "--force", container], check=True, stdout=sp.DEVNULL, timeout=30)),
            ]
            for label, action in actions:
                try:
                    action()
                except Exception as exc:
                    cleanup_errors.append(f"{label}: {exc!r}")
        stop.set()
        monitor.join(timeout=15)
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
        write(out / "attempt.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("lab_exit_code") == 0 and not result.get("error") and not cleanup_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
