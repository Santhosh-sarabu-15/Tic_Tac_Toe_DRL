import numpy as np
from typing import List, Tuple, Optional


class TicTacToeGame:
    """
    Tic-Tac-Toe Game Engine.

    Board representation:
    3x3 grid encoded as a 1D array of 9 integers:
      0 = Empty
      1 = Player X
     -1 = Player O

    Grid indices:
      0 | 1 | 2
     ---+---+---
      3 | 4 | 5
     ---+---+---
      6 | 7 | 8
    """

    WINNING_COMBINATIONS = [
        # Rows
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        # Columns
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        # Diagonals
        (0, 4, 8), (2, 4, 6)
    ]

    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.winner = 0          # 1 for X, -1 for O, 0 for none/draw
        self.done = False

    def reset(self) -> np.ndarray:
        """Reset game to initial state."""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.winner = 0
        self.done = False
        return self.get_state()

    def get_valid_moves(self) -> List[int]:
        """Return list of legal action indices (0 to 8)."""
        if self.done:
            return []
        return [i for i in range(9) if self.board[i] == 0]

    def make_move(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute move for current_player.

        Returns:
            (state, reward, done, info)
        """
        if self.done:
            raise ValueError("Game is already over.")
        if action < 0 or action > 8 or self.board[action] != 0:
            raise ValueError(f"Illegal move: cell {action} is invalid or occupied.")

        player_who_moved = self.current_player
        self.board[action] = player_who_moved

        self.winner = self.check_winner()
        if self.winner != 0:
            self.done = True
            reward = 1.0  # Positive reward for player who just moved and won
        elif self.is_draw():
            self.done = True
            reward = 0.0
        else:
            self.done = False
            reward = 0.0
            self.current_player = -self.current_player  # Switch player

        info = {
            "player_who_moved": player_who_moved,
            "winner": self.winner,
            "is_draw": self.is_draw()
        }
        return self.get_state(), reward, self.done, info

    def check_winner(self) -> int:
        """
        Check if there is a winner.
        Returns:
            1 if X won, -1 if O won, 0 if no winner yet.
        """
        for a, b, c in self.WINNING_COMBINATIONS:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return int(self.board[a])
        return 0

    def is_draw(self) -> bool:
        """Return True if board is full and no player has won."""
        return (self.check_winner() == 0) and np.all(self.board != 0)

    def is_terminal(self) -> bool:
        """Return True if game has ended (win or draw)."""
        return self.done or self.check_winner() != 0 or self.is_draw()

    def get_state(self) -> np.ndarray:
        """Return copy of board state."""
        return self.board.copy()

    def clone(self) -> 'TicTacToeGame':
        """Return deep copy of current game instance."""
        new_game = TicTacToeGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.done = self.done
        return new_game

    def render_ascii(self) -> str:
        """Return ASCII representation of board."""
        symbols = {0: "-", 1: "X", -1: "O"}
        lines = []
        for r in range(3):
            row_str = " | ".join(symbols[self.board[r * 3 + c]] for c in range(3))
            lines.append(row_str)
        return "\n---------\n".join(lines)
