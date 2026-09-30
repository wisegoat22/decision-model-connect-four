from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from engine import State, digest, drop, legal_columns, model_request, replay
import tournament
from tournament import (OPENINGS, AdapterError, AdapterTimeout, JsonlAdapter, audit_match,
                        calculate_standings, create_protocol, run_tournament)

HERE = Path(__file__).resolve().parent
MODELS = [{"id": letter, "name": letter.upper(), "source": "unit-test scripted fixture"} for letter in "abcd"]
ONE_OPENING = (OPENINGS[0],)


def scripted(request):
    """A test fixture only; no external model inference or game strategy."""
    return {"choice": next(iter(request["questions"]["move"]["criteria"])), "metadata": {"fixture": True}}


class TournamentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HERE)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_prespecified60_schedule_balances_every_pair_opening_and_color(self):
        protocol = create_protocol(MODELS)
        schedule = protocol["schedule"]
        self.assertEqual(len(schedule), 60)
        self.assertEqual(protocol["maximum_model_decisions_total"], 2328)
        self.assertEqual(Counter(tuple(row["pair"]) for row in schedule),
                         {("a", "b"): 10, ("a", "c"): 10, ("a", "d"): 10,
                          ("b", "c"): 10, ("b", "d"): 10, ("c", "d"): 10})
        self.assertEqual(Counter(row["first_player"] for row in schedule), dict.fromkeys("abcd", 15))
        self.assertEqual(Counter(row["second_player"] for row in schedule), dict.fromkeys("abcd", 15))
        self.assertEqual(len({(row["first_player"], row["second_player"], row["opening_id"]) for row in schedule}), 60)
        self.assertEqual(protocol["video_match_ids"], [schedule[0]["id"], schedule[50]["id"]])
        self.assertEqual(OPENINGS[2]["columns"], [8 - column for column in OPENINGS[1]["columns"]])
        self.assertEqual(OPENINGS[4]["columns"], [8 - column for column in OPENINGS[3]["columns"]])
        self.assertEqual(len({replay(opening["columns"]).board for opening in OPENINGS}), 5)
        self.assertEqual(create_protocol(MODELS), protocol)

    def test_distinct_opening_sequences_with_identical_boards_are_rejected(self):
        duplicates = [{"id": "first", "columns": [3, 4, 5, 4]},
                      {"id": "different-sequence-same-board", "columns": [5, 4, 3, 4]}]
        with self.assertRaisesRegex(ValueError, "unique board"):
            create_protocol(MODELS, openings=duplicates)

    def test_local_peer_subset_is_incomplete_then_resumes_without_replaying_finished_games(self):
        adapters = {identifier: scripted for identifier in "bcd"}
        first = run_tournament(self.directory, MODELS, adapters, openings=ONE_OPENING)
        self.assertFalse(first["complete"])
        self.assertEqual(first["completed_games"], 6)
        self.assertTrue(all(row["rank"] is None and row["provisional"] for row in first["standings"]))
        finished = {row["id"]: (self.directory / "matches" / (row["id"] + ".json")).read_bytes()
                    for row in first["matches"] if row["status"] == "complete"}
        second = run_tournament(self.directory, MODELS, dict.fromkeys("abcd", scripted), openings=ONE_OPENING)
        self.assertTrue(second["complete"])
        self.assertEqual(second["completed_games"], 12)
        self.assertTrue(all(row["games"] == 6 for row in second["standings"]))
        self.assertTrue(all(row["first_player_games"] == row["second_player_games"] == 3 for row in second["standings"]))
        for identifier, saved in finished.items():
            self.assertEqual((self.directory / "matches" / (identifier + ".json")).read_bytes(), saved)

    def test_pair_filter_preserves_entire_frozen_schedule(self):
        result = run_tournament(self.directory, MODELS, dict.fromkeys("abcd", scripted),
                                pairs=[("c", "d")], openings=ONE_OPENING)
        self.assertEqual(result["planned_games"], 12)
        self.assertEqual(result["completed_games"], 2)
        self.assertFalse(result["complete"])
        self.assertEqual({tuple(row["pair"]) for row in result["matches"] if row["status"] == "complete"}, {("c", "d")})

    def test_identity_and_opening_changes_are_rejected_before_adapter_calls(self):
        run_tournament(self.directory, MODELS, {}, openings=ONE_OPENING)
        changed = copy.deepcopy(MODELS)
        changed[0]["source"] = "different weights"
        with self.assertRaisesRegex(ValueError, "frozen"):
            run_tournament(self.directory, changed, dict.fromkeys("abcd", scripted), openings=ONE_OPENING)
        with self.assertRaisesRegex(ValueError, "frozen"):
            run_tournament(self.directory, MODELS, {}, openings=OPENINGS)

    def test_infrastructure_error_pauses_model_but_runs_other_pairs(self):
        def unavailable(_):
            return {"error": "fixture unavailable"}
        result = run_tournament(self.directory, MODELS[:3], {"a": unavailable, "b": scripted, "c": scripted}, openings=ONE_OPENING)
        self.assertEqual(result["completed_games"], 2)
        self.assertEqual(result["blocked_models_this_pass"], ["a"])
        self.assertFalse(result["complete"])
        a = next(row for row in result["standings"] if row["model_id"] == "a")
        self.assertEqual((a["games"], a["losses"], a["invalid_choices"]), (0, 0, 0))
        resumed = run_tournament(self.directory, MODELS[:3], dict.fromkeys("abc", scripted), openings=ONE_OPENING)
        self.assertTrue(resumed["complete"])

    def test_billing_guard_stops_before_adapter_and_preserves_incomplete_game(self):
        calls = []
        def counted(request):
            calls.append(request)
            return scripted(request)
        result = run_tournament(self.directory, MODELS[:2], dict.fromkeys("ab", counted),
                                openings=ONE_OPENING, before_decision=lambda *_: False)
        self.assertEqual(calls, [])
        self.assertEqual(result["completed_games"], 0)
        self.assertFalse(result["complete"])
        self.assertTrue(all(row["losses"] == 0 for row in result["standings"]))

    def test_invalid_action_is_explicit_forfeit_not_a_board_win(self):
        result = run_tournament(self.directory, MODELS[:2], {"a": lambda _: {"choice": "c9"}, "b": scripted}, openings=ONE_OPENING)
        self.assertTrue(result["complete"])
        self.assertTrue(all(row["termination"] == "invalid_choice" for row in result["matches"]))
        b = next(row for row in result["standings"] if row["model_id"] == "b")
        self.assertEqual((b["wins"], b["forfeit_wins"], b["board_wins"], b["points"]), (2, 2, 0, 6))

    def test_completed_response_survives_interruption_without_duplicate_call(self):
        calls = Counter()
        def counted(identifier):
            def choose(request):
                calls[(identifier, digest(request))] += 1
                return scripted(request)
            return choose
        adapters = {identifier: counted(identifier) for identifier in "ab"}
        with patch.object(tournament, "drop", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_tournament(self.directory, MODELS[:2], adapters, openings=ONE_OPENING)
        first_hash = digest(model_request(State()))
        self.assertEqual(calls[("a", first_hash)], 1)
        result = run_tournament(self.directory, MODELS[:2], adapters, openings=ONE_OPENING)
        self.assertTrue(result["complete"])
        self.assertEqual(calls[("a", first_hash)], 1)

    def test_replay_audit_rejects_board_request_and_result_tampering(self):
        result = run_tournament(self.directory, MODELS[:2], dict.fromkeys("ab", scripted), openings=ONE_OPENING)
        protocol = create_protocol(MODELS[:2], openings=ONE_OPENING)
        record = result["matches"][0]
        match = protocol["schedule"][0]
        self.assertTrue(audit_match(record, match, digest(protocol)).terminal)
        for mutate in (lambda row: row["moves"][0]["board"].__setitem__(0, 1),
                       lambda row: row["moves"][0]["decision"]["request"].__setitem__("state", "changed"),
                       lambda row: row.__setitem__("winner", row["second_player"])):
            altered = copy.deepcopy(record)
            mutate(altered)
            with self.assertRaises(ValueError):
                audit_match(altered, match, digest(protocol))

    def test_wdl_points_and_shared_competition_ranks(self):
        def record(identifier, first, second, winner=None):
            return {"id": identifier, "status": "complete", "first_player": first, "second_player": second,
                    "winner": winner, "draw": winner is None, "termination": "connect_four" if winner else "board_full"}
        records = [record("1", "a", "b", "a"), record("2", "b", "a", "b"),
                   record("3", "c", "d", "c"), record("4", "d", "c", "d"),
                   record("5", "a", "b"), record("6", "c", "d", "c")]
        rows = calculate_standings(MODELS, records, complete=True)
        by_id = {row["model_id"]: row for row in rows}
        self.assertEqual((by_id["c"]["wins"], by_id["c"]["points"], by_id["c"]["rank"]), (2, 6, 1))
        self.assertEqual((by_id["a"]["points"], by_id["b"]["points"]), (4, 4))
        self.assertEqual((by_id["a"]["rank"], by_id["b"]["rank"], by_id["d"]["rank"]), (2, 2, 4))
        self.assertTrue(all(row["games"] == row["wins"] + row["draws"] + row["losses"] for row in rows))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            calculate_standings(MODELS, records + records[:1])

    def test_truncated_request_response_pauses_without_forfeit(self):
        def truncated(_):
            return {"choice": "c1", "metadata": {"truncated": True}}
        result = run_tournament(self.directory, MODELS[:2], dict.fromkeys("ab", truncated), openings=ONE_OPENING)
        self.assertEqual(result["completed_games"], 0)
        self.assertTrue(all(row["losses"] == 0 for row in result["standings"]))

    def test_sole_legal_column_is_automatic_for_both_colors_and_auditable(self):
        # Reach a nearly full drawn pattern through legal recorded moves. This
        # fixture construction is not exposed to adapters or game requests.
        columns = (3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                   4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4)
        protocol = create_protocol(MODELS[:2], openings=ONE_OPENING)
        match = protocol["schedule"][0]
        record = tournament._new_match(match, digest(protocol))
        state = State()
        for column in columns:
            player = state.to_move
            request = model_request(state)
            decision = {"model_id": "a" if player == 1 else "b", "request": request,
                        "ply": state.ply + 1, "player": player,
                        "request_sha256": digest(request), "response": {"choice": f"c{column}"},
                        "source": "model", "model_calls": 1, "elapsed_ns": 1}
            state = drop(state, column)
            record["moves"].append({"ply": state.ply, "player": player, "column": column,
                                    "board": list(state.board), "opening": False, "decision": decision})
        self.assertEqual(legal_columns(state), [7])
        def forbidden(_):
            self.fail("A mandatory move must not call any adapter")
        path = self.directory / "forced.json"
        tournament._run_match(path, record, {"a": forbidden, "b": forbidden})
        self.assertTrue(record["draw"])
        self.assertEqual(record["termination"], "board_full")
        forced = record["moves"][36:]
        self.assertEqual(len(forced), 6)
        self.assertEqual({frame["player"] for frame in forced}, {1, 2})
        self.assertTrue(all(frame["decision"]["source"] == "forced_by_rules"
                            and frame["decision"]["model_calls"] == 0 for frame in forced))
        self.assertTrue(audit_match(record, match, digest(protocol)).draw)
        altered = copy.deepcopy(record)
        altered["moves"][0]["decision"].update(source="forced_by_rules", model_calls=0)
        with self.assertRaisesRegex(ValueError, "attribution"):
            audit_match(altered, match, digest(protocol))


class JsonlTests(unittest.TestCase):
    def test_persistent_process_two_requests_and_cold_timeout_allowance(self):
        code = "import sys,json,time; time.sleep(.04)\nfor line in sys.stdin:\n q=json.loads(line); print(json.dumps({'choice':next(iter(q['questions']['move']['criteria']))}),flush=True)"
        with JsonlAdapter([sys.executable, "-u", "-c", code], timeout_seconds=.2, startup_timeout_seconds=1) as adapter:
            self.assertEqual(adapter(model_request(State()))["choice"], "c1")
            self.assertEqual(adapter(model_request(State()))["choice"], "c1")
            self.assertEqual(adapter.completed_calls, 2)

    def test_partial_line_timeout_does_not_block_forever(self):
        code = "import sys,time; sys.stdin.readline(); sys.stdout.write('{'); sys.stdout.flush(); time.sleep(5)"
        with JsonlAdapter([sys.executable, "-u", "-c", code], timeout_seconds=.05, startup_timeout_seconds=.05) as adapter:
            with self.assertRaises(AdapterTimeout):
                adapter(model_request(State()))
            self.assertIsNotNone(adapter.process.poll())

    def test_worker_error_is_infrastructure_failure(self):
        code = "import sys,json; sys.stdin.readline(); print(json.dumps({'error':'fixture'}),flush=True)"
        with JsonlAdapter([sys.executable, "-u", "-c", code]) as adapter:
            with self.assertRaises(AdapterError):
                adapter(model_request(State()))


if __name__ == "__main__":
    unittest.main()
