"""
Tic-Tac-Toe game engine.

Board is represented as a tuple of 9 values:
  0 = empty, 1 = player X, -1 = player O

Positions are indexed 0-8, left to right, top to bottom:
  0 1 2
  3 4 5
  6 7 8
"""

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def new_board():
    return (0,) * 9


def available_moves(board):
    return [i for i, v in enumerate(board) if v == 0]


def apply_move(board, position, player):
    board = list(board)
    board[position] = player
    return tuple(board)


def check_winner(board):
    """Returns 1 or -1 if that player has won, 0 for draw, None if game ongoing."""
    for a, b, c in WIN_LINES:
        line_sum = board[a] + board[b] + board[c]
        if line_sum == 3:
            return 1
        if line_sum == -3:
            return -1
    if 0 not in board:
        return 0  # draw
    return None  # game not finished


def print_board(board):
    symbols = {0: ".", 1: "X", -1: "O"}
    rows = []
    for r in range(3):
        row = board[r * 3:(r + 1) * 3]
        rows.append(" ".join(symbols[v] for v in row))
    print("\n".join(rows))
