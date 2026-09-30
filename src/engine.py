"""Deterministic Connect Four rules. No search, evaluation, or model features."""
from dataclasses import dataclass
import hashlib
import json

ROWS, COLUMNS = 6, 7
EMPTY_BOARD = (0,) * (ROWS * COLUMNS)
WINDOWS = tuple(
    tuple((row + step * dr) * COLUMNS + column + step * dc for step in range(4))
    for row in range(ROWS)
    for column in range(COLUMNS)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    if 0 <= row + 3 * dr < ROWS and 0 <= column + 3 * dc < COLUMNS
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def winners(board):
    return {board[window[0]] for window in WINDOWS
            if board[window[0]] and all(board[index] == board[window[0]] for index in window)}


@dataclass(frozen=True)
class State:
    board: tuple = EMPTY_BOARD
    to_move: int = 1

    def __post_init__(self):
        board = tuple(self.board)
        object.__setattr__(self, "board", board)
        if len(board) != 42 or any(type(value) is not int or value not in (0, 1, 2) for value in board):
            raise ValueError("Board must contain42 integers:0 empty,1 X,2 O")
        if type(self.to_move) is not int or self.to_move not in (1, 2):
            raise ValueError("to_move must be1 or2")
        for column in range(COLUMNS):
            occupied = False
            for row in range(ROWS):
                value = board[row * COLUMNS + column]
                if value:
                    occupied = True
                elif occupied:
                    raise ValueError("A piece cannot float above an empty cell")
        first, second = board.count(1), board.count(2)
        if first not in (second, second + 1) or self.to_move != (1 if first == second else 2):
            raise ValueError("Piece counts and player turn are inconsistent")
        winning = winners(board)
        if len(winning) > 1:
            raise ValueError("Both colors cannot win in a legal position")
        if winning == {1} and first != second + 1 or winning == {2} and first != second:
            raise ValueError("Winner is inconsistent with the last player")

    @property
    def ply(self):
        return sum(value != 0 for value in self.board)

    @property
    def winner(self):
        return next(iter(winners(self.board)), None)

    @property
    def terminal(self):
        return self.winner is not None or all(self.board)

    @property
    def draw(self):
        return self.winner is None and all(self.board)


def legal_columns(state):
    if state.terminal:
        return []
    return [column + 1 for column in range(COLUMNS) if state.board[column] == 0]


def drop(state, column):
    if type(column) is not int or not 1 <= column <= COLUMNS:
        raise ValueError("Column must be an integer in1..7")
    if state.terminal:
        raise ValueError("Cannot move after the game ended")
    if column not in legal_columns(state):
        raise ValueError("Column is full")
    board = list(state.board)
    index = next(row * COLUMNS + column - 1 for row in range(ROWS - 1, -1, -1)
                 if board[row * COLUMNS + column - 1] == 0)
    board[index] = state.to_move
    return State(tuple(board), 3 - state.to_move)


def replay(columns):
    state = State()
    for column in columns:
        state = drop(state, column)
    return state


def model_request(state):
    """Only plain coordinates/rules; identical states produce identical requests."""
    legal = legal_columns(state)
    if not legal:
        raise ValueError("Terminal positions have no decision request")
    symbols = ".XO"
    rows = ["".join(symbols[value] for value in state.board[row * 7:(row + 1) * 7]) for row in range(6)]
    context = ("Connect Four. Drop a piece into a column; gravity applies. Connect 4 horizontally, vertically or diagonally to win. "
               f"X moves first. You play {symbols[state.to_move]}. . is empty.\n"
               "Rows top to bottom; columns left to right.\nColumns: 1234567\n"
               + "\n".join(rows) + "\nLegal columns: " + ",".join(map(str, legal)))
    return {"state": context,
            "questions": {"move": {"type": "choice", "instructions": "Choose the legal column that gives you the best chance to win.",
                                    "criteria": {f"c{column}": f"Drop in column {column}." for column in legal}}}}


def normalize_choice(response, state):
    """Accept canonical c1..c7 or the equivalent numeric string; never guess."""
    choice = response.get("choice") if isinstance(response, dict) else response
    if not isinstance(choice, str):
        raise ValueError("Decision must name a legal choice ID")
    if len(choice) == 1 and choice in "1234567":
        choice = "c" + choice
    if choice not in model_request(state)["questions"]["move"]["criteria"]:
        raise ValueError("Decision must name a legal choice ID")
    return int(choice[1])
