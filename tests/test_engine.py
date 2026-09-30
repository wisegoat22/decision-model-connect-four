from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import unittest

from engine import State, WINDOWS, drop, legal_columns, model_request, normalize_choice, replay


class RulesTests(unittest.TestCase):
    def test_drop_gravity_turns_and_full_column(self):
        state = replay([1, 1, 1, 1, 1, 1])
        self.assertEqual([state.board[row * 7] for row in range(6)], [2, 1, 2, 1, 2, 1])
        self.assertEqual(state.to_move, 1)
        self.assertEqual(legal_columns(state), [2, 3, 4, 5, 6, 7])
        with self.assertRaisesRegex(ValueError, "full"):
            drop(state, 1)

    def test_four_win_directions_and_stop_after_win(self):
        diagonal = [1, 2, 2, 3, 4, 3, 3, 4, 5, 4, 4]
        for columns, winner in (([1, 7, 2, 7, 3, 6, 4], 1),
                                ([1, 7, 1, 7, 2, 7, 2, 7], 2),
                                (diagonal, 1), ([8 - value for value in diagonal], 1)):
            with self.subTest(columns=columns):
                before = replay(columns[:-1])
                self.assertFalse(before.terminal)
                state = drop(before, columns[-1])
                self.assertEqual(state.winner, winner)
                self.assertEqual(legal_columns(state), [])
                self.assertFalse(state.draw)
                with self.assertRaisesRegex(ValueError, "ended"):
                    drop(state, 1)

    def test_full_board_draw_is_not_a_win(self):
        board = tuple([1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1] * 3)
        state = State(board, 1)
        self.assertTrue(state.draw)
        self.assertTrue(state.terminal)
        self.assertIsNone(state.winner)
        self.assertEqual(legal_columns(state), [])
        with self.assertRaises(ValueError):
            model_request(state)

    def test_no_wraparound_wins_and_exact_window_count(self):
        self.assertEqual(len(WINDOWS), 69)
        state = replay([6, 3, 7, 3, 1, 4, 2])
        self.assertIsNone(state.winner)

    def test_invalid_piece_counts_gravity_turns_and_action_types(self):
        bad = list(State().board)
        bad[0] = 1
        with self.assertRaisesRegex(ValueError, "float"):
            State(bad, 2)
        with self.assertRaises(ValueError):
            State((0,) * 41 + (2,), 1)
        with self.assertRaises(ValueError):
            State((0,) * 41 + (1,), 1)
        with self.assertRaises(ValueError):
            State((0,) * 41 + (True,), 2)
        for action in (0, 8, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                drop(State(), action)

    def test_both_winning_colors_are_rejected(self):
        board = (0,) * 28 + (2, 2, 2, 2, 0, 0, 0) + (1, 1, 1, 1, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "Both"):
            State(board, 1)

    def test_plain_model_request_identical_state_and_legal_choice_order(self):
        state = replay([1] * 6)
        request = model_request(state)
        self.assertEqual(request, model_request(State(state.board, state.to_move)))
        self.assertEqual(set(request), {"state", "questions"})
        self.assertEqual(list(request["questions"]["move"]["criteria"]), ["c2", "c3", "c4", "c5", "c6", "c7"])
        self.assertIn("You play X", request["state"])
        self.assertIn("Rows top to bottom; columns left to right", request["state"])
        self.assertIn("best chance to win", request["questions"]["move"]["instructions"])
        self.assertLess(len(request["state"]), 400)
        self.assertNotIn("opponent", request["state"])
        self.assertEqual(normalize_choice({"choice": "c2"}, state), 2)
        self.assertEqual(normalize_choice({"choice": "2"}, state), 2)
        for response in ({"choice": "c1"}, {"choice": "c8"}, {"choice": 2}, {}, " c2 ", "column2"):
            with self.assertRaises(ValueError):
                normalize_choice(response, state)


if __name__ == "__main__":
    unittest.main()
