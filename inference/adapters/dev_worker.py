"""Public Dev-0.4B inference using the recorded checkpoint and native decision interface."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "dev" / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "4")
REVISION = "78f7121608a2ca44dc598bba4d74e4070567fd50"
SOURCE_REVISION = "2ce2563db938e9ec22ffe69e5fb8d4706f61ed1a"
WEIGHTS = ROOT / "adapters" / "weights" / "dev"


def main():
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from safetensors import safe_open
        from dev.inference import Predictor
        from dev.data import encode, SCHEMA_VERSION

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        manifest = json.loads((ROOT / "adapters" / "dev_weights_manifest.json").read_text())
        for entry in manifest["artifacts"]:
            path = WEIGHTS / entry["file"]
            with path.open("rb") as handle:
                actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            if path.stat().st_size != entry["bytes"] or actual_sha != entry["sha256"]:
                raise RuntimeError("Public Dev artifact changed")
        predictor = Predictor(WEIGHTS, device="cpu")
        # The author's loader uses strict=False. Independently verify all keys
        # and tensor shapes so that no missing/random head can go unnoticed.
        expected = predictor.model.state_dict()
        with safe_open(WEIGHTS / "model.safetensors", framework="pt", device="cpu") as weights:
            if set(weights.keys()) != set(expected):
                raise RuntimeError("Public Dev checkpoint tensor keys do not match the model")
            if any(tuple(weights.get_slice(k).get_shape()) != tuple(expected[k].shape) for k in expected):
                raise RuntimeError("Public Dev checkpoint tensor shapes do not match")
        del expected
        print("Public Dev-0.4B ready (CPU, pinned weights, strict key/shape check).", file=sys.stderr, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            supplied = request["questions"]["move"]
            keys = list(supplied["criteria"])
            question = {"type": "choice", "instructions": supplied["instructions"],
                        "criteria": [f"{key}: {supplied['criteria'][key]}" for key in keys]}
            with contextlib.redirect_stdout(sys.stderr):
                row = encode({"schema_version": SCHEMA_VERSION, "group": "inference", "state": request["state"],
                              "command": "", "question": question}, predictor.tokenizer, predictor.max_length)
                started = time.perf_counter()
                answer = predictor.answer(request["state"], {"move": question})["move"]
                elapsed = (time.perf_counter() - started) * 1000
            response = {"choice": keys[answer["index"]],
                        "probabilities": dict(zip(keys, answer["probabilities"])),
                        "metadata": {"model": "mpnikhil/dev-0.4b", "revision": REVISION,
                                     "source_revision": SOURCE_REVISION, "device": "cpu", "dtype": "float32",
                                     "tokens": len(row["input_ids"]), "state_truncated": False,
                                     "temperature": predictor.temperatures.get("choice", 1.0),
                                     "inference_ms": round(elapsed, 3)}}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
