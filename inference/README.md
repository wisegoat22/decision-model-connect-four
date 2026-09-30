# Original inference adapters

These are the adapters and launcher used for the recorded model decisions. The public copy replaces local package paths with pinned Git sources, adjusts the launcher to import the portable harness from `../src`, and defaults fresh runs to `../runs/new-experiment`. The original game records are never overwritten by that default.

**This directory is an implementation record, not a one-command fresh-inference installation.** Replaying and analyzing the included evidence works with Python's standard library alone; a fresh model run additionally needs the large external weights, exact source checkouts and isolated runtimes listed below. Those dependencies are not bundled, and a fresh installation from this public copy has not been end-to-end tested.

## Recorded directory layout

- `vendor/dev`, `vendor/kev`, `vendor/von`, `vendor/laya`: upstream checkouts at the commits in `models.json`.
- `dev-env/bin/python`: Dev runtime. Package versions are in `adapters/dev-runtime-requirements.txt`.
- `kev-env/bin/python`: Kev runtime: Torch 2.8.0, Transformers 5.17.0, MLX 0.32.3, MLX-LM 0.31.3; pinned Kev source plus dependencies.
- `adapters/.venv-decision/bin/python`: Von/Laya runtime. Sanitized package versions and pinned source URLs are in `adapters/runtime-requirements.txt`.
- `adapters/weights/dev`, `adapters/weights/kev-4b`, `adapters/weights/von`, `adapters/weights/laya`: downloaded artifacts matching the adjacent manifest files. The workers verify hashes before inference. Kev also needs the base revision recorded in `models.json` and its manifest.

The workers read a JSON request per stdin line and return a JSON choice/probability response per stdout line. They do not compute tactical features or search the game tree. Recorded runs used native highest-probability choices, no additional game training, Von chains disabled, CPU float32 for Dev/Von/Laya, and MLX bf16 with a float32 pointer head for Kev.

Once the dependencies are independently set up in that layout, `python3 inference/run_local.py --run-dir runs/new-experiment` uses the harness for fresh records. A different software or hardware setup may produce different results. Keep fresh results separate from `evidence/`, which holds the original experiment.

The adapters use native input encodings that differ across models. Timing fields inside the adapters cover different work; the public results report uses separately labeled harness request/response measurements and makes no intrinsic speed claim.
