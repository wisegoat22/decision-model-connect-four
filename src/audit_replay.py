"""Read-only audit and text replay of the archived Connect Four games.

Uses only the Python standard library. It never loads weights, calls a model,
contacts a service, or changes the recorded evidence.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from analyze_observed import analyze
from engine import digest
from tournament import audit_match, calculate_standings

DEFAULT_EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def audit(evidence=DEFAULT_EVIDENCE, source_dir=None):
    """Validate all frozen records and recompute statistics without model calls."""
    evidence = Path(evidence)
    source_dir = Path(source_dir) if source_dir else Path(__file__).resolve().parent
    frozen = read_json(evidence / "protocol.json")
    spec = frozen["specification"]
    results = read_json(evidence / "results.json")
    require(frozen["protocol_hash"] == digest(spec) == results["protocol_hash"],
            "Frozen protocol digest mismatch")
    require(results["complete"] and results["completed_games"] == results["planned_games"] == 60,
            "Expected the complete 60-game recorded tournament")
    require(not results["blocked_models_this_pass"], "Recorded tournament is blocked")
    require(results["models"] == spec["models"] == read_json(source_dir / "models.json"),
            "Frozen model descriptors differ")
    for name, expected_sha in spec["harness_sources_sha256"].items():
        require(name in {"engine.py", "tournament.py"}, "Unexpected harness source name")
        require(hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == expected_sha,
                f"Frozen harness source changed: {name}")

    expected_ids = {match["id"] for match in spec["schedule"]}
    actual_ids = {path.stem for path in (evidence / "matches").glob("*.json")}
    require(len(expected_ids) == 60 and expected_ids == actual_ids,
            "Missing, extra or duplicate scheduled match")
    models = {model["id"]: model for model in spec["models"]}
    records, calls, forced, tokens = [], Counter(), Counter(), defaultdict(list)
    for expected in spec["schedule"]:
        record = read_json(evidence / "matches" / (expected["id"] + ".json"))
        # Checks opening placements, every board, exact prompts and choice IDs,
        # gravity, turns, terminal state, source attribution and declared outcome.
        state = audit_match(record, expected, frozen["protocol_hash"])
        require(record["status"] == "complete" and state.terminal,
                f"Incomplete game: {record['id']}")
        require(not record["attempts"] and record["pending_decision"] is None,
                f"Unexpected pending or failed attempt: {record['id']}")
        records.append(record)
        for move in record["moves"]:
            if move["opening"]:
                continue
            decision = move["decision"]
            model_id = decision["model_id"]
            if decision["source"] == "forced_by_rules":
                forced[model_id] += 1
                continue
            response = decision["response"]
            meta = response["metadata"]
            require(not meta.get("truncated") and not meta.get("state_truncated"),
                    "Truncated recorded request")
            revision = meta.get("revision", meta.get("weights_revision"))
            require(revision == models[model_id]["weights_revision"],
                    f"Recorded weights revision mismatch: {model_id}")
            source_revision = meta.get("source_revision", meta.get("source_commit"))
            if source_revision is not None:
                require(source_revision == models[model_id]["source_commit"],
                        f"Recorded source revision mismatch: {model_id}")
            if model_id == "kev":
                require(meta["base_revision"] == models[model_id]["base_revision"]
                        and meta["backend"] == "mlx" and meta["dtype"] == "bfloat16",
                        "Recorded Kev configuration mismatch")
            else:
                require(meta["device"] == "cpu", "Recorded CPU configuration mismatch")
            if model_id in {"von", "laya"}:
                require(meta["chains_enabled"] is False and meta["model_forward_passes"] == 1,
                        "Recorded peer inference mode mismatch")
            probabilities = response.get("probabilities", meta.get("probabilities"))
            criteria = decision["request"]["questions"]["move"]["criteria"]
            require(isinstance(probabilities, dict) and set(probabilities) == set(criteria),
                    "Probability labels do not match legal choices")
            require(all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                        for p in probabilities.values())
                    and abs(sum(probabilities.values()) - 1) < .002,
                    "Probabilities are not finite and normalized")
            # Some native APIs record probabilities rounded to four decimals.
            require(probabilities[response["choice"]] >= max(probabilities.values()) - .00011,
                    "Chosen option is not a recorded maximum")
            calls[model_id] += 1
            tokens[model_id].append(meta.get("tokens", meta.get("input_tokens")))

    require(records == results["matches"], "Individual games differ from results.json")
    require(calculate_standings(spec["models"], records, complete=True) == results["standings"],
            "Recomputed standings differ")
    require(all(row["games"] == 30 and row["first_player_games"] == row["second_player_games"] == 15
                for row in results["standings"]), "Seat balance differs")
    require(sum(calls.values()) == 581 and not forced, "Decision counts differ")
    archived_audit = read_json(evidence / "AUDIT.json")
    require(archived_audit["model_calls"] == dict(calls)
            and archived_audit["completed_games"] == len(records), "Archived audit count mismatch")
    observed = read_json(evidence / "OBSERVED_STATISTICS.json")
    require(analyze(evidence) == observed, "Recomputed observed statistics differ")
    analysis_audit = read_json(evidence / "OBSERVED_ANALYSIS_AUDIT.json")
    for filename, key in [("results.json", "source_results_sha256"),
                          ("OBSERVED_STATISTICS.json", "statistics_sha256"),
                          ("INDEPENDENT_OBSERVED_ANALYSIS.json", "independent_analysis_sha256")]:
        require(hashlib.sha256((evidence / filename).read_bytes()).hexdigest() == analysis_audit[key],
                f"Archived analysis provenance differs: {filename}")
    return {"status": "passed", "scope": "Recorded-action replay and analysis; no fresh model inference",
            "games": len(records), "model_decisions": sum(calls.values()),
            "recorded_piece_placements": sum(len(record["moves"]) for record in records),
            "wins": {row["model_id"]: row["wins"] for row in results["standings"]},
            "protocol_hash": frozen["protocol_hash"], "statistics_reproduced": True}


def show_match(evidence, match_id):
    """Print each actual recorded move and board, without generating new moves."""
    frozen = read_json(Path(evidence) / "protocol.json")
    known = {match["id"] for match in frozen["specification"]["schedule"]}
    require(match_id in known, "Match ID is not in the frozen schedule")
    record = read_json(Path(evidence) / "matches" / (match_id + ".json"))
    print(f"\n{match_id}: X={record['first_player']} O={record['second_player']}")
    for move in record["moves"]:
        label = "preset opening" if move["opening"] else move["decision"]["model_id"]
        print(f"\nPly {move['ply']}: {label}, {'XO'[move['player'] - 1]} -> column {move['column']}")
        print("1234567")
        for row in range(6):
            print("".join(".XO"[v] for v in move["board"][row * 7:(row + 1) * 7]))
    print(f"\nWinner: {record['winner']}; termination: {record['termination']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--match", help="After the audit, print the recorded boards for this match ID.")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.evidence), indent=2))
    if args.match:
        show_match(args.evidence, args.match)


if __name__ == "__main__":
    main()
