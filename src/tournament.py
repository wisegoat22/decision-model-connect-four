"""Resumable model-neutral Connect Four tournament; no model/API dependencies."""
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from engine import State, EMPTY_BOARD, canonical_json, digest, drop, model_request, normalize_choice, replay

OPENINGS = (
    {"id": "empty", "columns": []},
    {"id": "a", "columns": [4, 3, 5, 4]},
    {"id": "a-mirror", "columns": [4, 5, 3, 4]},
    {"id": "b", "columns": [2, 4, 5, 3]},
    {"id": "b-mirror", "columns": [6, 4, 3, 5]},
)


class AdapterError(RuntimeError):
    pass


class AdapterTimeout(AdapterError):
    pass


class DecisionPaused(AdapterError):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class JsonlAdapter:
    """One persistent process per isolated model runtime; stdout is JSONL only.

    The harness does not inspect or record environment variables. Caller-owned
    adapters enforce provider billing caps and normalize their model outputs.
    A timeout terminates this adapter process, never guesses a game action.
    """
    def __init__(self, command, *, timeout_seconds=60, startup_timeout_seconds=120, stderr_path=None, env=None):
        if timeout_seconds <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.completed_calls = 0
        self.buffer = b""
        self.stderr_file = open(stderr_path, "ab") if stderr_path else None
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr_file or subprocess.DEVNULL, env=env)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def __call__(self, request):
        if self.process.poll() is not None:
            raise AdapterError("Adapter process is unavailable")
        try:
            self.process.stdin.write((canonical_json(request) + "\n").encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise AdapterError("Adapter input closed") from error
        deadline = time.monotonic() + (self.timeout_seconds if self.completed_calls else self.startup_timeout_seconds)
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                self.close()
                raise AdapterTimeout("Adapter response timed out")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise AdapterError("Adapter output closed before a response")
            self.buffer += chunk
            if len(self.buffer) > 4 * 1024 * 1024:
                self.close()
                raise AdapterError("Adapter response exceeds4MB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        if self.buffer.strip():
            raise AdapterError("Adapter emitted unsolicited extra output")
        try:
            response = json.loads(line)
            canonical_json(response)
        except (ValueError, UnicodeDecodeError) as error:
            raise AdapterError("Adapter response is not finite JSON") from error
        if not isinstance(response, dict):
            raise AdapterError("Adapter response must be a JSON object")
        if response.get("error") is not None:
            raise AdapterError("Adapter reported an infrastructure error")
        self.completed_calls += 1
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout):
            if stream and not stream.closed:
                stream.close()
        if self.stderr_file and not self.stderr_file.closed:
            self.stderr_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def create_protocol(models, *, openings=OPENINGS):
    """Construct deterministic pairings; model order determines the video match."""
    models = json.loads(canonical_json(models))
    identifiers = [model.get("id") for model in models]
    if len(models) < 2 or any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(models):
        raise ValueError("At least two distinct named model IDs are required")
    if any(not model.get("name") or not model.get("source") for model in models):
        raise ValueError("Each model needs its name and source identity")
    openings = json.loads(canonical_json(openings))
    if not openings or len({opening["id"] for opening in openings}) != len(openings):
        raise ValueError("Openings must have distinct IDs")
    opening_boards = set()
    for opening in openings:
        state = replay(opening["columns"])
        if state.terminal or len(opening["columns"]) % 2:
            raise ValueError("Openings must be legal, nonterminal and have equal piece counts")
        if state.board in opening_boards:
            raise ValueError("Openings must produce unique board positions")
        opening_boards.add(state.board)
    schedule = []
    for pair_index, pair in enumerate(itertools.combinations(identifiers, 2), 1):
        for opening_index, opening in enumerate(openings):
            for leg in (1, 2):
                first, second = pair if leg == 1 else pair[::-1]
                schedule.append({"id": f"pair-{pair_index:02d}-opening-{opening_index:02d}-leg-{leg}",
                                 "pair": list(pair), "opening_id": opening["id"],
                                 "opening_columns": opening["columns"],
                                 "first_player": first, "second_player": second})
    specification = {
        "schema_version": "connect-four-round-robin-v1", "game": "connect_four",
        "models": models, "openings": openings, "schedule": schedule,
        "planned_games": len(schedule), "games_per_pair": len(openings) * 2,
        "games_per_model": (len(models) - 1) * len(openings) * 2,
        "maximum_model_decisions_total": sum(42 - len(match["opening_columns"]) for match in schedule),
        "video_match_id": schedule[0]["id"],
        "video_match_ids": [next(match["id"] for match in schedule if match["first_player"] == identifiers[index]
                                  and match["second_player"] == identifiers[index + 1])
                            for index in range(0, len(identifiers) - 1, 2)],
        "video_selection": "First scheduled match for each adjacent pair in model order, fixed before results; covers all four models in a four-model tournament",
        "harness_sources_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                    for name in ("engine.py", "tournament.py")},
        "board": "42 row-major integers, top row first;0 empty,1 X,2 O; columns1..7",
        "rules": "6x7 board, gravity, X first, connect4 horizontal/vertical/diagonal; full board without winner is a draw; no moves after a win",
        "fairness": "Every pair receives each fixed opening in both color assignments. Openings have equal piece counts; no solver-balanced value claim.",
        "request_schema": "state + questions.move(type=choice,instructions,criteria map c1..c7); only legal columns; plain board/rules; no solver features, history or opponent identity",
        "choice_order": "Legal columns ascending left to right for every model",
        "mandatory_moves": "When exactly one legal column remains, every entrant automatically plays it without a model call; logged source=forced_by_rules and model_calls=0. This is legality only, not strategic lookahead.",
        "response_schema": "JSON object with choice:c1..c7 (equivalent numeric strings accepted), optional metadata; no fallback action",
        "ranking": {"primary": "wins descending", "secondary": "points descending", "win_points": 3, "draw_points": 1, "loss_points": 0,
                    "ties": "Exact equal wins and points share competition rank; model ID is only a stable display order"},
        "invalid_choice": "A model response naming an illegal/missing choice forfeits that game; logged separately from board wins",
        "service_failure": "Timeout, budget block, unavailable adapter or malformed transport pauses the game without a loss; unaffected pairs may continue",
        "resume": "Saved moves and completed responses are reused; pending requests without responses may require caller-managed retry/idempotency",
        "publish_guard": "Final ranks only when every scheduled game is complete; partial results explicitly incomplete",
        "billing": "No purchases; provider adapter must enforce its authorized cap before calls; optional before_decision guard pauses cleanly",
        "randomness": "Engine and schedule deterministic; model sampling settings/version must be pinned in model descriptors",
    }
    return specification


def initialize_run(directory, specification):
    directory = Path(directory)
    path = directory / "protocol.json"
    expected_hash = digest(specification)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("protocol_hash") != expected_hash or digest(saved.get("specification")) != expected_hash:
            raise ValueError("Run protocol differs from frozen model identities, openings or rules")
    else:
        write_json(path, {"created_at_utc": utc_now(), "protocol_hash": expected_hash, "specification": specification})
    return expected_hash


def _new_match(match, protocol_hash):
    state = State()
    frames = []
    for column in match["opening_columns"]:
        player = state.to_move
        state = drop(state, column)
        frames.append({"ply": state.ply, "player": player, "column": column,
                       "board": list(state.board), "opening": True})
    return {**match, "protocol_hash": protocol_hash, "initial_board": list(EMPTY_BOARD),
            "status": "pending", "winner": None, "winner_color": None, "draw": False,
            "termination": None, "moves": frames, "attempts": [], "pending_decision": None}


def _finish(record, state):
    record["status"] = "complete"
    record["draw"] = bool(state.draw)
    record["winner_color"] = state.winner
    record["winner"] = ({1: record["first_player"], 2: record["second_player"]}.get(state.winner))
    record["termination"] = "board_full" if state.draw else "connect_four"
    record["pending_decision"] = None


def audit_match(record, match, protocol_hash):
    if record.get("status") not in {"pending", "running", "paused", "complete"}:
        raise ValueError("Unknown saved match status")
    if record["status"] != "complete" and (record.get("winner") is not None or record.get("draw")
                                              or record.get("termination") is not None):
        raise ValueError("An incomplete match cannot claim a final result")
    for key, value in match.items():
        if record.get(key) != value:
            raise ValueError(f"Saved match identity mismatch:{key}")
    if record.get("protocol_hash") != protocol_hash or record.get("initial_board") != list(EMPTY_BOARD):
        raise ValueError("Saved match protocol or initial board mismatch")
    state = State()
    opening = match["opening_columns"]
    if len(record["moves"]) < len(opening):
        raise ValueError("Saved match is missing opening plies")
    for index, frame in enumerate(record["moves"]):
        if frame["ply"] != index + 1 or frame["player"] != state.to_move:
            raise ValueError("Saved move turn mismatch")
        is_opening = index < len(opening)
        if frame["opening"] is not is_opening or is_opening and frame["column"] != opening[index]:
            raise ValueError("Saved opening mismatch")
        if not is_opening:
            decision = frame["decision"]
            expected_model = record["first_player"] if state.to_move == 1 else record["second_player"]
            request = model_request(state)
            if decision["model_id"] != expected_model or decision["request"] != request or decision["request_sha256"] != digest(request):
                raise ValueError("Saved model request mismatch")
            if decision["ply"] != frame["ply"] or decision["player"] != state.to_move:
                raise ValueError("Saved decision turn mismatch")
            if normalize_choice(decision["response"], state) != frame["column"]:
                raise ValueError("Saved response/action mismatch")
            response = decision["response"]
            metadata = response.get("metadata")
            if response.get("error") is not None or isinstance(metadata, dict) and metadata.get("truncated") is True:
                raise ValueError("Infrastructure failure cannot supply a scored move")
            forced = len(request["questions"]["move"]["criteria"]) == 1
            if decision.get("source") != ("forced_by_rules" if forced else "model") or decision.get("model_calls") != (0 if forced else 1):
                raise ValueError("Saved mandatory-move/model-call attribution mismatch")
        state = drop(state, frame["column"])
        if frame["board"] != list(state.board):
            raise ValueError("Saved board differs from exact replay")
    if record["status"] == "complete":
        if record["termination"] == "invalid_choice":
            if state.terminal:
                raise ValueError("Forfeit cannot replace a finished board result")
            decision = record["invalid_decision"]
            request = model_request(state)
            if len(request["questions"]["move"]["criteria"]) == 1:
                raise ValueError("A mandatory move cannot be a model forfeit")
            player_model = record["first_player"] if state.to_move == 1 else record["second_player"]
            if decision["request"] != request or decision["request_sha256"] != digest(request) or decision["model_id"] != player_model:
                raise ValueError("Forfeit request identity mismatch")
            if decision.get("source") != "model" or decision.get("model_calls") != 1:
                raise ValueError("Forfeit must come from an actual model decision")
            metadata = decision["response"].get("metadata")
            if decision["response"].get("error") is not None or isinstance(metadata, dict) and metadata.get("truncated") is True:
                raise ValueError("Infrastructure errors or truncated requests cannot be forfeits")
            try:
                normalize_choice(decision["response"], state)
            except ValueError:
                pass
            else:
                raise ValueError("A legal response cannot be a forfeit")
            winner_color = 3 - state.to_move
            winner_model = record["first_player"] if winner_color == 1 else record["second_player"]
            if record["winner"] != winner_model or record["winner_color"] != winner_color or record["draw"]:
                raise ValueError("Forfeit result mismatch")
        else:
            if not state.terminal:
                raise ValueError("Completed match has no terminal board")
            winner_model = record["first_player"] if state.winner == 1 else record["second_player"] if state.winner == 2 else None
            if record["winner"] != winner_model or record["winner_color"] != state.winner or record["draw"] != bool(state.draw):
                raise ValueError("Completed match WDL mismatch")
            if record["termination"] != ("board_full" if state.draw else "connect_four"):
                raise ValueError("Completed match termination mismatch")
    pending = record.get("pending_decision")
    if pending:
        request = model_request(state)
        expected_model = record["first_player"] if state.to_move == 1 else record["second_player"]
        if pending["request"] != request or pending["request_sha256"] != digest(request) or pending["model_id"] != expected_model:
            raise ValueError("Pending request no longer matches the saved position")
        if pending["ply"] != state.ply + 1 or pending["player"] != state.to_move:
            raise ValueError("Pending request turn mismatch")
    return state


def _run_match(path, record, adapters, *, before_decision=None, stop_requested=None):
    state = replay([frame["column"] for frame in record["moves"]])
    while not state.terminal:
        if stop_requested is not None and stop_requested():
            record.update(status="paused", pause_reason="stop_requested")
            write_json(path, record)
            return None
        player = state.to_move
        model_id = record["first_player"] if player == 1 else record["second_player"]
        request = model_request(state)
        pending = record.get("pending_decision")
        if pending is None:
            pending = {"model_id": model_id, "ply": state.ply + 1, "player": player,
                       "request": request, "request_sha256": digest(request), "started_at_utc": utc_now()}
            record["pending_decision"] = pending
        record["status"] = "running"
        write_json(path, record)
        choices = request["questions"]["move"]["criteria"]
        if "response" not in pending and len(choices) == 1:
            pending.update(source="forced_by_rules", model_calls=0, elapsed_ns=0,
                           response={"choice": next(iter(choices)), "metadata": {"forced_by_rules": True}})
            write_json(path, record)
        if "response" not in pending:
            try:
                if before_decision is not None and before_decision(model_id, request) is False:
                    raise DecisionPaused("Decision budget unavailable")
                started = time.perf_counter_ns()
                response = adapters[model_id](request)
                elapsed = time.perf_counter_ns() - started
                if isinstance(response, str):
                    response = {"choice": response}
                if not isinstance(response, dict):
                    raise AdapterError("Adapter response must be an object")
                if response.get("error") is not None:
                    raise AdapterError("Adapter reported an infrastructure error")
                if isinstance(response.get("metadata"), dict) and response["metadata"].get("truncated") is True:
                    raise AdapterError("Adapter truncated the common game request")
                canonical_json(response)
                pending.update(response=response, elapsed_ns=elapsed, source="model", model_calls=1)
                # Persist the paid/computed response before applying it, so a
                # resumed run need not repeat a completed adapter decision.
                write_json(path, record)
            except Exception as error:
                record["attempts"].append({"model_id": model_id, "ply": state.ply + 1,
                                           "request_sha256": digest(request), "error_type": type(error).__name__,
                                           "recorded_at_utc": utc_now()})
                record.update(status="paused", pause_reason="adapter_unavailable")
                write_json(path, record)
                return model_id
        try:
            column = normalize_choice(pending["response"], state)
        except ValueError:
            winner_color = 3 - player
            record.update(status="complete", termination="invalid_choice", draw=False,
                          winner_color=winner_color,
                          winner=record["first_player"] if winner_color == 1 else record["second_player"],
                          invalid_decision=pending, pending_decision=None)
            write_json(path, record)
            return None
        state = drop(state, column)
        record["moves"].append({"ply": state.ply, "player": player, "column": column,
                                "board": list(state.board), "opening": False, "decision": pending})
        record["pending_decision"] = None
        write_json(path, record)
    _finish(record, state)
    write_json(path, record)
    return None


def calculate_standings(models, records, *, complete=False):
    table = {model["id"]: {"model_id": model["id"], "name": model["name"], "source": model["source"],
                          "wins": 0, "draws": 0, "losses": 0, "games": 0, "points": 0,
                          "first_player_games": 0, "second_player_games": 0,
                          "model_decisions": 0, "forced_moves": 0,
                          "board_wins": 0, "forfeit_wins": 0, "invalid_choices": 0}
             for model in models}
    seen = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError("Duplicate match result")
        seen.add(record["id"])
        if record["status"] != "complete":
            continue
        first, second = record["first_player"], record["second_player"]
        if first == second or first not in table or second not in table:
            raise ValueError("Unknown or repeated participant in match")
        table[first]["first_player_games"] += 1
        table[second]["second_player_games"] += 1
        for identifier in (first, second):
            table[identifier]["games"] += 1
        for frame in record.get("moves", []):
            if not frame["opening"]:
                identifier = first if frame["player"] == 1 else second
                decision = frame["decision"]
                table[identifier]["model_decisions"] += decision.get("model_calls", 0)
                table[identifier]["forced_moves"] += decision.get("source") == "forced_by_rules"
        if record["draw"]:
            if record["winner"] is not None:
                raise ValueError("A drawn game cannot have a winner")
            for identifier in (first, second):
                table[identifier]["draws"] += 1
                table[identifier]["points"] += 1
        else:
            winner = record["winner"]
            if winner not in (first, second):
                raise ValueError("A decisive match requires its participant as winner")
            loser = second if winner == first else first
            table[winner]["wins"] += 1
            table[winner]["points"] += 3
            table[loser]["losses"] += 1
            table[winner]["forfeit_wins" if record["termination"] == "invalid_choice" else "board_wins"] += 1
            if record["termination"] == "invalid_choice":
                table[loser]["invalid_choices"] += 1
    rows = sorted(table.values(), key=lambda row: (-row["wins"], -row["points"], row["model_id"]))
    if complete and len({row["games"] for row in rows}) != 1:
        raise ValueError("Final rankings require equal completed game counts")
    previous_key, previous_rank = None, None
    for index, row in enumerate(rows, 1):
        key = (row["wins"], row["points"])
        if key != previous_key:
            previous_rank = index
        row["rank"] = previous_rank if complete else None
        row["provisional"] = not complete
        previous_key = key
    return rows


def run_tournament(directory, models, adapters, *, pairs=None, openings=OPENINGS,
                   before_decision=None, stop_requested=None):
    """Run only available requested pairs; safely resume a frozen tournament.

    Returns renderer-ready results. Unavailable pairs stay pending, and API
    failures never become losses. Pass the same four model descriptors on each
    resume; adding available adapters does not change the schedule or protocol.
    """
    directory = Path(directory)
    specification = create_protocol(models, openings=openings)
    protocol_hash = initialize_run(directory, specification)
    identifiers = {model["id"] for model in models}
    if set(adapters) - identifiers:
        raise ValueError("Adapter IDs do not belong to this tournament")
    requested = None if pairs is None else {frozenset(pair) for pair in pairs}
    if requested is not None and any(len(pair) != 2 or not pair <= identifiers for pair in requested):
        raise ValueError("Requested pairs must name two different tournament models")
    blocked = set()
    for match in specification["schedule"]:
        path = directory / "matches" / (match["id"] + ".json")
        record = json.loads(path.read_text()) if path.exists() else _new_match(match, protocol_hash)
        audit_match(record, match, protocol_hash)
        if record["status"] == "complete":
            continue
        participants = set(match["pair"])
        if (not participants <= set(adapters) or participants & blocked
                or requested is not None and frozenset(participants) not in requested):
            if not path.exists():
                write_json(path, record)
            continue
        if stop_requested is not None and stop_requested():
            break
        unavailable = _run_match(path, record, adapters, before_decision=before_decision, stop_requested=stop_requested)
        if unavailable:
            blocked.add(unavailable)
    records = []
    for match in specification["schedule"]:
        path = directory / "matches" / (match["id"] + ".json")
        record = json.loads(path.read_text()) if path.exists() else _new_match(match, protocol_hash)
        audit_match(record, match, protocol_hash)
        records.append(record)
    complete = all(record["status"] == "complete" for record in records)
    result = {"schema_version": "connect-four-results-v1", "game": "connect_four",
              "complete": complete, "protocol_hash": protocol_hash, "models": specification["models"],
              "planned_games": len(records), "completed_games": sum(record["status"] == "complete" for record in records),
              "video_match_id": specification["video_match_id"], "video_match_ids": specification["video_match_ids"],
              "ranking_rules": specification["ranking"],
              "matches": records, "standings": calculate_standings(models, records, complete=complete),
              "blocked_models_this_pass": sorted(blocked), "updated_at_utc": utc_now()}
    write_json(directory / "results.json", result)
    return result
