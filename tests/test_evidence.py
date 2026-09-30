"""Portable replay and post-hoc analysis checks against the recorded evidence."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_observed import analyze, winning_columns, put
from audit_replay import audit
from engine import replay


class EvidenceTests(unittest.TestCase):
    def test_all_saved_games_and_statistics_replay_without_models(self):
        report = audit(ROOT / "evidence")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["games"], 60)
        self.assertEqual(report["model_decisions"], 581)
        self.assertEqual(report["recorded_piece_placements"], 773)

    def test_statistics_match_archived_json_exactly(self):
        expected = json.loads((ROOT / "evidence/OBSERVED_STATISTICS.json").read_text())
        self.assertEqual(analyze(ROOT / "evidence"), expected)

    def test_one_ply_win_and_block_respect_gravity(self):
        # X threatens column 4 on the bottom row; O has no immediate win.
        state = replay([1, 7, 2, 7, 3])
        self.assertEqual(winning_columns(state.board, 1), {3})  # zero-based analysis API
        self.assertEqual(winning_columns(state.board, 2), set())
        self.assertEqual(winning_columns(put(state.board, 3, 2), 1), set())
        self.assertEqual(winning_columns(put(state.board, 5, 2), 1), {3})


if __name__ == "__main__":
    unittest.main()
