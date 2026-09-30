"""Pinned Kev inference; JSONL requests on stdin, responses only on stdout."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "kev"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
REVISION = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
ADAPTERS = Path(__file__).resolve().parent


def verify_local_artifacts():
    """Fail before tensor loading unless every downloaded artifact matches its pin."""
    manifest = json.loads((ADAPTERS / "kev_weights_manifest.json").read_text())
    if (manifest.get("base_repo"), manifest.get("base_revision"),
            manifest.get("adapter_repo"), manifest.get("adapter_revision")) != (
            "Qwen/Qwen3.5-4B-Base", BASE_REVISION, "jaredpalmer/kev-4b", REVISION):
        raise RuntimeError("Unexpected Kev artifact identity")
    required = {
        "weights/kev-4b/base/model.safetensors-00001-of-00002.safetensors",
        "weights/kev-4b/base/model.safetensors-00002-of-00002.safetensors",
        "weights/kev-4b/base/model.safetensors.index.json",
        "weights/kev-4b/base/config.json", "weights/kev-4b/base/tokenizer.json",
        "weights/kev-4b/base/tokenizer_config.json",
        "weights/kev-4b/adapter/head.pt", "weights/kev-4b/adapter/adapter_config.json",
        "weights/kev-4b/adapter/adapter_model.safetensors",
    }
    entries = manifest["files"]
    if not required.issubset({entry["file"] for entry in entries}):
        raise RuntimeError("Missing required Kev artifact")
    base_dir = ADAPTERS / "weights/kev-4b/base"
    if {path.name for path in base_dir.glob("*.safetensors")} != {
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors"}:
        raise RuntimeError("Unexpected base safetensors files")
    for entry in entries:
        path = (ADAPTERS / entry["file"]).resolve()
        if not path.is_relative_to(ADAPTERS / "weights/kev-4b"):
            raise RuntimeError("Artifact path is outside the pinned weights directory")
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Artifact size mismatch: {path.name}")
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(4 * 1024**2):
                sha.update(block)
        if sha.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Artifact SHA mismatch: {path.name}")
    return manifest


def load_local_model():
    """Native Checkpoint MLX operations, with verified local paths instead of Hub IO.

    Matches Checkpoint.load/_load_mlx at source revision
    fdfdfd2c7a98225715a9cc8f4d9e1ef29ebbe79c: native base/tokenizer,
    native scale-1 LoRA merge, unchanged pointer head and calibration.
    """
    verify_local_artifacts()
    import torch
    import mlx.core as mx
    from transformers import AutoConfig
    from kev.checkpoint import Checkpoint
    from kev.model import is_hybrid, load_tokenizer, pad_id
    from kev.mlx_model import MLXDecisionModel, merge_lora

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    mx.set_memory_limit(24 * 1024**3)
    base_dir = ADAPTERS / "weights/kev-4b/base"
    checkpoint = Checkpoint(str(ADAPTERS / "weights/kev-4b/adapter"))
    if (checkpoint.meta.base, checkpoint.meta.base_revision) != (
            "Qwen/Qwen3.5-4B-Base", BASE_REVISION):
        raise RuntimeError("Unexpected Kev base metadata")
    if checkpoint.full or checkpoint.meta.option_isolation:
        raise RuntimeError("Checkpoint is incompatible with the prescribed MLX path")
    if not is_hybrid(AutoConfig.from_pretrained(str(base_dir), local_files_only=True).get_text_config()):
        raise RuntimeError("Expected a hybrid Qwen base")
    tokenizer = load_tokenizer(str(base_dir))
    model = MLXDecisionModel(str(base_dir), pad_id(tokenizer), head_dim=checkpoint.meta.head_dim)
    merged_tensors = merge_lora(model.lm, checkpoint.path, 1.0)
    if merged_tensors != 248:
        raise RuntimeError("Unexpected LoRA merge coverage")
    model.head.load_state_dict(checkpoint.meta.head)
    model.eval()
    model.head.temperature = checkpoint.meta.temperature
    mx.set_cache_limit(512 * 1024**2)
    return tokenizer, model, checkpoint


def main():
    with contextlib.redirect_stdout(sys.stderr):
        from kev.api import SystemOneRequest, to_record, to_answers
        tokenizer, model, checkpoint = load_local_model()
        print("Kev ready (MLX, pinned weights).", file=sys.stderr, flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                record, meta = to_record(SystemOneRequest.model_validate(request))
                encoded = model.encode(tokenizer, record, max_state=512, max_branch=1024, strict=True)
                start = time.perf_counter()
                probabilities = [p.tolist() for p in model.probs(encoded)]
                elapsed = (time.perf_counter() - start) * 1000
                answer = to_answers(probabilities, meta)["move"]
            response = {
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
                "metadata": {
                    "model": "jaredpalmer/kev-4b",
                    "revision": REVISION,
                    "base_revision": BASE_REVISION,
                    "backend": "mlx",
                    "dtype": model.dtype,
                    "temperature": checkpoint.meta.temperature,
                    "tokens": len(encoded["ids"]),
                    "state_truncated": encoded["state_truncated"],
                    "truncated": encoded["state_truncated"],
                    "legal_options": len(encoded["opt_idx"][0]),
                    "inference_ms": round(elapsed, 3),
                },
            }
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
