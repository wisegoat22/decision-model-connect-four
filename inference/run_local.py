"""Run/resume the four public local decision models, with no hosted API calls."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tournament import JsonlAdapter, run_tournament

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT.parent / "runs" / "new-experiment"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["dev", "kev", "von", "laya"], default=["dev", "kev", "von", "laya"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    models = json.loads((ROOT / "models.json").read_text())
    commands = {
        "dev": [ROOT / "dev-env/bin/python", ROOT / "adapters/dev_worker.py"],
        "kev": [ROOT / "kev-env/bin/python", ROOT / "adapters/kev_worker.py"],
        "von": [ROOT / "adapters/.venv-decision/bin/python", ROOT / "adapters/decision_adapter.py", "--model", "von", "--device", "cpu"],
        "laya": [ROOT / "adapters/.venv-decision/bin/python", ROOT / "adapters/decision_adapter.py", "--model", "laya", "--device", "cpu"],
    }
    logs = ROOT / "run-logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    with ExitStack() as stack:
        adapters = {mid: stack.enter_context(JsonlAdapter([str(x) for x in commands[mid]],
                       timeout_seconds=60, startup_timeout_seconds=180,
                       stderr_path=logs / f"{mid}.log", env=env)) for mid in args.models}
        result = run_tournament(args.run_dir, models, adapters, stop_requested=lambda: stopping)
    print(json.dumps({"complete": result["complete"], "completed_games": result["completed_games"],
                      "planned_games": result["planned_games"], "blocked_models": result["blocked_models_this_pass"],
                      "elapsed_seconds": round(time.monotonic() - started, 2), "standings": result["standings"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
