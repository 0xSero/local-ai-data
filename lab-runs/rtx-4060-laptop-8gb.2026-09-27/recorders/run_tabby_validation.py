#!/usr/bin/env python3
"""Run the pinned upstream lab unchanged; retain traffic and audit its evidence.

python3 work/run_tabby_validation.py --self-check
python3 work/run_tabby_validation.py qwen --endpoint http://127.0.0.1:18080 \
    --out work/attempt-qwen-001 --server-log work/qwen-startup.log
"""

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

REGISTRY = Path(__file__).resolve().parent / "registry"
CARD = "rtx-4060-laptop-8gb"
FILES = {"qwen": "qwen3.5-9b.tabbyapi.64k.json", "gemma": "gemma-4-12b.tabbyapi.32k.json"}
WEIGHTS = {"qwen": "TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938",
           "gemma": "turboderp/gemma-4-12B-it-exl3@38309570753c5fde71a818feb513b1e09a81fc18"}
SETTINGS = {"qwen": {"ctx": 65536, "draft": "mtp"},
            "gemma": {"vision": True, "tools": "null", "vars": "{enable_thinking: true}"}}
PINS = {
    "lab/lab.py": "59832ef8b15c4677354dd8d7d7fdb6b273c803ed51eaaa6af772d540680c1143",
    "registry/engines/tabbyapi-exl3.json": "92d4e012c911fc5c1787a887330203e6b7d3d450ebca24938de18e8c657f73e3",
}
MTP_MARKER = "Using main model MTP component for drafting"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Stream:
    """Tee exactly the bytes consumed by the lab, without read-ahead or re-framing."""

    def __init__(self, response, out, number, emit):
        self.response, self.active, self.emit = response, response, emit
        self.number = number
        self.raw = (out / f"stream-{number}.sse").open("ab", buffering=0)
        self.times = (out / f"stream-{number}.jsonl").open("a", buffering=1)
        self.iterator = iter(response)
        self.offset = 0

    def __getattr__(self, name):
        return getattr(self.active, name)

    def __enter__(self):
        self.active = self.response.__enter__()
        self.iterator = iter(self.active)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.iterator)
        received = time.monotonic()
        self.raw.write(line)
        self.times.write(json.dumps({"monotonic": received, "offset": self.offset, "bytes": len(line)}) + "\n")
        self.offset += len(line)
        return line

    def __exit__(self, kind, error, tb):
        try:
            return self.response.__exit__(kind, error, tb)
        finally:
            self.raw.close()
            self.times.close()
            self.emit("stream_closed", id=self.number, error=repr(error) if error else None)


def audit_stream(path):
    done, terminal, content = False, None, False
    for line in path.read_bytes().splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        require(not done, "SSE data followed [DONE]")
        if payload == b"[DONE]":
            done = True
            continue
        chunk = json.loads(payload)
        require("error" not in chunk and "choices" in chunk, "SSE error or unknown event")
        for choice in chunk["choices"]:
            if choice.get("finish_reason") is not None:
                terminal = choice["finish_reason"]
            delta = choice.get("delta") or {}
            content |= bool(delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"))
    require(done and terminal == "stop" and content, "SSE needs content, finish_reason=stop and [DONE]")
    return {"done": done, "finish_reason": terminal}


def validate(name, endpoint, out, server_log=None, registry=REGISTRY):
    parsed = urllib.parse.urlsplit(endpoint)
    require(parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.strip("/")),
            "Endpoint must be a plain local HTTP origin without credentials")
    endpoint = endpoint.rstrip("/")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {"recipe": name, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "phase": "preflight", "lab_exit": None, "audit": {"passed": False}, "recipe_restored": False}
    dump(out / "status.json", status)
    recipe_path = registry / "registry/recipes/nvidia" / CARD / FILES[name]
    before = None
    original_open, original_argv = urllib.request.urlopen, sys.argv
    original_term = signal.getsignal(signal.SIGTERM)
    lab, original_call = None, None
    sequence, token_calls, streams = itertools.count(1), [], []
    journal = (out / "http.jsonl").open("a", buffering=1)
    log = (out / "lab.log").open("a", buffering=1)

    def emit(event, **fields):
        journal.write(json.dumps({"monotonic": time.monotonic(), "phase": status["phase"],
                                  "event": event, **fields}, ensure_ascii=False) + "\n")

    def error_record(error):
        record = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, urllib.error.HTTPError):
            record.update(status=error.code, body=error.read().decode("utf-8", errors="replace"))
        return record

    def recorded_call(ep, path, body=None, timeout=3600):
        number = next(sequence)
        emit("request", id=number, path=path, body=body, timeout=timeout)
        try:
            response, seconds = original_call(ep, path, body, timeout)
        except BaseException as error:
            emit("error", id=number, path=path, **error_record(error))
            if "token" in path:
                token_calls.append({"id": number, "path": path, "valid": False})
            raise
        emit("response", id=number, path=path, seconds=seconds, body=response)
        if "token" in path:
            valid = (path == "/v1/token/encode" and isinstance(response, dict)
                     and type(response.get("length")) is int and response["length"] >= 0
                     and isinstance(response.get("tokens"), list)
                     and response["length"] == len(response["tokens"]))
            token_calls.append({"id": number, "path": path, "valid": valid,
                                "length": response.get("length") if isinstance(response, dict) else None})
        return response, seconds

    def recorded_open(request, *args, **kwargs):
        body = json.loads(request.data) if isinstance(request, urllib.request.Request) and request.data else {}
        if not body.get("stream"):
            return original_open(request, *args, **kwargs)
        number = next(sequence)
        emit("stream_request", id=number, path=urllib.parse.urlsplit(request.full_url).path, body=body)
        try:
            response = original_open(request, *args, **kwargs)
        except BaseException as error:
            emit("error", id=number, **error_record(error))
            raise
        streams.append(number)
        return Stream(response, out, number, emit)

    def interrupted(signum, _frame):
        raise SystemExit(f"signal {signum}")

    try:
        signal.signal(signal.SIGTERM, interrupted)
        for file, digest in PINS.items():
            require(hashlib.sha256((registry / file).read_bytes()).hexdigest() == digest, f"Pinned source changed: {file}")
        spec = importlib.util.spec_from_file_location("recorded_lab", registry / "lab/lab.py")
        lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab)
        before = recipe_path.read_bytes()
        (out / "recipe.before.json").write_bytes(before)
        recipe = json.loads(before)
        launch = lab.render(recipe)
        require(recipe["card"] == CARD and recipe["engine"] == "tabbyapi-exl3@0f83e6198dc3", "Unexpected recipe")
        require(recipe["weights"] == WEIGHTS[name] and recipe["set"] == SETTINGS[name], "Pinned recipe settings changed")
        require(lab.recipe_path(recipe, launch) == recipe_path, "Unexpected recipe path")
        expected_id = lab.dirname(recipe["weights"])
        argv = [str(registry / "lab/lab.py"), "try", recipe["weights"], "--model", recipe["model"],
                "--engine", recipe["engine"], "--card", CARD, "--on", "endpoint", "--endpoint", endpoint,
                "--gpu", "NVIDIA GeForce RTX 4060 Laptop GPU"]
        for key, value in recipe["set"].items():
            argv += ["--set", f"{key}={str(value).lower() if isinstance(value, bool) else value}"]
        dump(out / "launch.json", {"argv": argv, "source_sha256": PINS, "recipe": recipe, "launch": launch})
        (out / "config.yml").write_text(launch["config"]["text"])
        original_call = lab.call
        lab.call, urllib.request.urlopen = recorded_call, recorded_open
        current, _ = lab.call(endpoint, "/v1/model", timeout=30)
        params = current.get("parameters") or {}
        expected = {"max_seq_len": launch["ctx"], "cache_size": launch["ctx"] + 1024,
                    "cache_mode": "Q4", "max_batch_size": 1, "chunk_size": 2048, "use_vision": launch["vision"]}
        require(current.get("id") == expected_id, "Wrong loaded model")
        require(all(params.get(k) == v for k, v in expected.items()), f"Loaded parameters differ: expected {expected}")
        available, _ = lab.call(endpoint, "/v1/models", timeout=30)
        require([m.get("id") for m in available.get("data", [])] == [expected_id], "Mount exactly one model folder")
        startup = Path(server_log).read_bytes() if server_log else b""
        if server_log:
            (out / "server-startup.log").write_bytes(startup)
        draft = recipe["set"].get("draft") == "mtp"
        if draft:
            require(server_log and MTP_MARKER.encode() in startup, "Qwen requires --server-log with the MTP startup marker")
        else:
            require(params.get("draft") is None and MTP_MARKER.encode() not in startup
                    and b"Using draft model:" not in startup, "Unexpected draft model")
        status["draft_check"] = {"api": params.get("draft"), "expected_mtp": draft,
                                 "startup_marker": MTP_MARKER.encode() in startup,
                                 "limitation": "Pinned Tabby model_info does not populate parameters.draft"}
        lab.call(endpoint, "/v1/token/encode", {"text": "Tokenizer preflight."}, timeout=30)
        require(token_calls[-1]["valid"] and token_calls[-1]["length"] > 0, "Tokenizer preflight failed")
        pattern = f"{CARD}.{recipe['model']}.tabbyapi-exl3.{launch['ctx'] // 1024}k.*.json"
        old_runs = {run: run.read_bytes() for run in lab.RUNS.glob(pattern)}
        status["phase"] = "lab"
        dump(out / "status.json", status)
        sys.argv = argv
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                status["lab_exit"] = lab.main()
            except SystemExit as error:
                status["lab_exit"] = error.code if isinstance(error.code, int) else 1
                raise
            except BaseException:
                status["lab_exit"] = 1
                raise
            finally:
                for run in lab.RUNS.glob(pattern):
                    if run.read_bytes() != old_runs.get(run):
                        shutil.copyfile(run, out / run.name)
        status["phase"] = "audit"
        require(status["lab_exit"] == 0, f"Upstream lab failed with exit {status['lab_exit']}")
        require(len(streams) == 1, "Expected exactly one speed stream")
        stream_audit = audit_stream(out / f"stream-{streams[0]}.sse")
        require(all(c["valid"] for c in token_calls), "Tokenizer failure or fallback in run")
        speed_counts = [c for c in token_calls if c["id"] > streams[0]]
        require(len(speed_counts) == 2 and all(c["length"] > 0 for c in speed_counts),
                "Missing speed sample/whole-answer tokenization")
        status["audit"] = {"passed": True, "stream": stream_audit, "tokenizer_backed": True}
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc(file=log)
    finally:
        urllib.request.urlopen, sys.argv = original_open, original_argv
        signal.signal(signal.SIGTERM, original_term)
        if lab is not None and original_call is not None:
            lab.call = original_call
        if before is not None:
            after = recipe_path.read_bytes()
            (out / "recipe.after_lab.json").write_bytes(after)
            if not status["audit"]["passed"] and after != before:
                recipe_path.write_bytes(before)
                status["recipe_restored"] = True
        status.update(phase="finished", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), token_calls=token_calls)
        dump(out / "status.json", status)
        journal.close()
        log.close()
    print(json.dumps({"out": str(out), "lab_exit": status["lab_exit"], "audit": status["audit"], "error": status.get("error")}))
    return 0 if status["audit"]["passed"] else 2


def self_check():
    """Exercise the real upstream CLI and rollback using in-memory HTTP responses."""
    with tempfile.TemporaryDirectory(prefix="tabby-recorder-check-") as tmp:
        root = Path(tmp) / "registry"
        paths = [*PINS, f"registry/cards/nvidia/{CARD}.json", *[f"registry/recipes/nvidia/{CARD}/{f}" for f in FILES.values()]]
        for path in paths:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REGISTRY / path, target)
        original_open = urllib.request.urlopen
        try:
            for name, scenario in [("qwen", "ok"), ("gemma", "ok"), ("qwen", "truncated"),
                                   ("qwen", "fallback"), ("qwen", "wrong_model"), ("qwen", "http_error")]:
                recipe_path = root / f"registry/recipes/nvidia/{CARD}" / FILES[name]
                before = recipe_path.read_bytes()
                recipe = json.loads(before)
                repo, revision = recipe["weights"].split("@")
                served = repo.split("/")[1] + "-" + revision[:8]
                ctx = 65536 if name == "qwen" else 32768
                token_requests = 0

                def fake_open(request, *args, **kwargs):
                    nonlocal token_requests
                    path = urllib.parse.urlsplit(request.full_url).path
                    body = json.loads(request.data) if request.data else {}
                    if path == "/v1/model":
                        response = {"id": "wrong" if scenario == "wrong_model" else served,
                                    "parameters": {"max_seq_len": ctx, "cache_size": ctx + 1024, "cache_mode": "Q4",
                                                   "max_batch_size": 1, "chunk_size": 2048, "use_vision": name == "gemma", "draft": None}}
                    elif path == "/v1/models":
                        response = {"data": [{"id": served}]}
                    elif "token" in path:
                        token_requests += 1
                        if scenario == "fallback" and token_requests > 1:
                            raise urllib.error.HTTPError(request.full_url, 500, "tokenizer failed", {}, io.BytesIO(b'{"error":"fake tokenizer failure"}'))
                        tokens = [7] * ((len(body["text"]) + 3) // 4)
                        response = {"tokens": tokens, "length": len(tokens)}
                    elif body.get("stream"):
                        require(body["temperature"] == 0.8 and "max_tokens" not in body, "Sampling changed")
                        chunks = [{"choices": [{"delta": {"content": "A lighthouse keeper watched the sea. " * 40}}]},
                                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
                        raw = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
                        return io.BytesIO(raw + (b"" if scenario == "truncated" else b"data: [DONE]\n\n"))
                    else:
                        require(body["temperature"] == 0.6 and "max_tokens" not in body, "Sampling changed")
                        prompt = body["messages"][-1]["content"]
                        message = {"content": "red, yellow, blue"}
                        if "17 * 23" in prompt:
                            message = {"content": "391", "reasoning_content": "17 * 20 + 17 * 3 = 391"}
                        elif "weather in Paris" in prompt:
                            if scenario == "http_error":
                                raise urllib.error.HTTPError(request.full_url, 503, "fake overload", {}, io.BytesIO(b'{"error":"fake overload"}'))
                            message = {"content": "", "tool_calls": [{"id": "one", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}
                        elif body["messages"][-1]["role"] == "tool":
                            message = {"content": "Paris is 17 degrees and overcast."}
                        elif "access code" in prompt:
                            message = {"content": "58213"}
                        response = {"choices": [{"message": message, "finish_reason": "stop"}],
                                    "usage": ({"prompt_tokens": int(ctx * 0.8), "completion_tokens": 10}
                                              if scenario == "fallback" else {})}
                    return io.BytesIO(json.dumps(response).encode())

                urllib.request.urlopen = fake_open
                startup = Path(tmp) / "startup.log"
                startup.write_text(MTP_MARKER if name == "qwen" else "Loading Gemma")
                out = Path(tmp) / f"{name}-{scenario}"
                result = validate(name, "http://127.0.0.1:18080", out, startup, root)
                status = json.loads((out / "status.json").read_text())
                assert (result == 0) == (scenario == "ok"), status
                assert "Authorization" not in (out / "http.jsonl").read_text()
                if scenario != "ok":
                    assert recipe_path.read_bytes() == before, "Failure changed recipe"
                if scenario in {"truncated", "fallback"}:
                    assert status["lab_exit"] == 0 and status["recipe_restored"], status
                if scenario == "http_error":
                    assert "fake overload" in (out / "http.jsonl").read_text()
        finally:
            urllib.request.urlopen = original_open
    print("self-check passed: exact CLI, both recipes, HTTP/SSE capture, fallback rejection and rollback")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("recipe", choices=FILES)
        parser.add_argument("--endpoint", required=True)
        parser.add_argument("--out", required=True, help="New attempt directory; must not already exist")
        parser.add_argument("--server-log", help="Complete startup log from this container; required for Qwen MTP verification")
        args = parser.parse_args()
        sys.exit(validate(args.recipe, args.endpoint, args.out, args.server_log))
