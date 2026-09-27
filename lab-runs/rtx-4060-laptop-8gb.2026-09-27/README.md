# RTX 4060 Laptop owner validation — 2026-09-27

Qwen3.5-9B EXL3 and Gemma 4 12B EXL3 each passed all six registry gates in one attempt on an ASUS ROG Zephyrus G14 GA403UV: RTX 4060 Laptop GPU (8,188 MiB), Ryzen 9 8945HS, CachyOS kernel 7.2.6, NVIDIA driver 615.71.09, AC power, `performance` profile. The desktop remained active.

| Measurement | Qwen3.5-9B | Gemma 4 12B |
| --- | ---: | ---: |
| Configured context / cache tokens | 65,536 / 66,560 | 32,768 / 33,792 |
| Draft mode | MTP | None |
| Context plain-content tokens | 52,573 (80.220%) | 25,731 (78.5248%) |
| Server context count, including template/control tokens | 52,583 | 25,746 |
| Decode, including reasoning | 52.4 tok/s | 24.0 tok/s |
| Timed sample | 1,573 tokens / ~30.021 s | 720 tokens / ~30.005 s |
| Complete returned text | 2,247 tokens | 1,418 tokens |
| Speed-window GPU temperature | 81–83°C | 81–86°C |
| Speed-window GPU power draw | 57.25–60.69 W | 57.22–75.12 W |
| Speed-window device-wide memory use | 5,764 MiB | 6,330 MiB |

Both loaded the expected model, answered the chat gate, returned `391` with separate reasoning, called the Paris weather tool and used its 17°C/overcast result, recalled code `58213`, and exceeded the 15 tok/s threshold. Both streams ended naturally with `finish_reason: stop` and `[DONE]`; no client output cap was set.

Speed includes reasoning and visible text. **Gemma's timed sample was almost entirely reasoning**, so 24.0 tok/s is not an answer-only rate. Counts use the model's tokenizer without character-count fallback. Context counts cover plain input content; the full configured windows were not exercised. The lab's `prefill` fields (878 and 631) divide those input counts by complete context-request latency (59.9 and 40.8 seconds), rather than timing the prefill phase alone. Qwen MTP was confirmed by config and startup logs because the pinned model-info endpoint returns `draft: null`. Gemma loaded its enabled vision modules, but no image input was tested. These synthetic checks do not establish broad model quality.

The default power value was 55 W; enforced limits varied across 49.17–74.97 W during the two attempts. Sampled peak temperatures were 90°C and 88°C; throttle-reason counters were not collected. Both containers had 8 GiB memory / 16 GiB memory-plus-swap limits and recorded zero cgroup `max`, `oom`, and `oom_kill` events. Swap remained allocated. The [Qwen](qwen-resource-summary.json) and [Gemma](gemma-resource-summary.json) resource summaries retain complete phase measurements and sampling limitations.

The unchanged run JSONs and their recipe proof hashes are:

- [Qwen run](../rtx-4060-laptop-8gb.qwen3.5-9b.tabbyapi-exl3.64k.20260927T093915.json): `sha256:3919b5e56629a4ba`.
- [Gemma run](../rtx-4060-laptop-8gb.gemma-4-12b.tabbyapi-exl3.32k.20260927T095103.json): `sha256:a9eb457e5e5d6b09`.

## Reproduction and records

Use registry commit `b94e4255a42fc0c63c2c3279d700114a6f84313e`. Each attempt directory ([Qwen](qwen-001), [Gemma](gemma-001)) supplies `command.json`, `weights-verified.json`, and `lab/config.yml`. Download that exact weights revision, verify its receipts, substitute local mount paths in the recorded container command, and then execute the lab argv in `lab/launch.json` with Python from the pinned checkout. Only the relevant model directory was mounted; the endpoint was localhost-only. Image and installed-package identities are recorded.

`http.jsonl` preserves complete synthetic requests, responses, and tokenizer results. `stream-20.sse` and its JSONL preserve bytes, offsets, and monotonic receipt times. `resources.jsonl` records nominal two-second host/GPU/cgroup samples. Logs, audit status, metadata, and generated recipe output accompany them. Recorder timing overhead was not measured.

`recorders/` contains the collection code and cleanup check for audit. These scripts depend on the original scratch layout and do **not** run directly from this bundle. To use the validation recorder or its fake-response `--self-check`, copy `run_tabby_validation.py` beside a checkout directory named `registry` at the pinned commit. The full attempt runner additionally requires the scratch files and model directories specified in its source; the recorded commands above reproduce acceptance without reconstructing that runner.

## Publication substitutions and integrity

The lab JSONs, HTTP/SSE data, timing records, resource measurements, server logs, configs, and collection scripts are unchanged. Auxiliary absolute host paths become `<REGISTRY>`, `<CONFIG_YAML>`, `<MODEL_DIR>`, or `<ATTEMPT_DIR>`; inspected host process IDs become `<host-pid>`. Image metadata omits an unrelated failed optional Git-binary probe. Container IDs, private command originals, and duplicate captures are excluded. Quantitative data and container-internal paths are retained.

Recipe proof hashes cover the original JSON bytes **without the single final newline**, truncated to 16 hexadecimal characters. `SHA256SUMS` covers complete published files, including that newline, and both sibling run JSONs. From this directory:

```sh
sha256sum -c SHA256SUMS
```
