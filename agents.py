import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from game import TicTacToeGame


class BaseAgent(ABC):
    """Abstract base class for all Tic-Tac-Toe agents."""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def select_move(self, game: TicTacToeGame) -> int:
        """Given current game state, select legal action index (0 to 8)."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects moves uniform-randomly among valid actions."""

    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name=name)

    def select_move(self, game: TicTacToeGame) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")
        return random.choice(valid_moves)


class MinimaxAgent(BaseAgent):
    """
    Classical Minimax Agent.
    Evaluates full search tree to select optimal move.
    """

    def __init__(self, name: str = "MinimaxAgent"):
        super().__init__(name=name)
        self.nodes_searched = 0
        self.decision_time = 0.0

    def select_move(self, game: TicTacToeGame) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        self.nodes_searched = 0
        start_time = time.perf_counter()
        root_player = game.current_player

        _, best_move = self._minimax(game, depth=0, root_player=root_player)
        self.decision_time = time.perf_counter() - start_time

        if best_move is None:
            best_move = valid_moves[0]
        return best_move

    def _minimax(self, game: TicTacToeGame, depth: int, root_player: int) -> Tuple[float, Optional[int]]:
        self.nodes_searched += 1

        if game.is_terminal():
            winner = game.check_winner()
            if winner == root_player:
                return 10.0 - depth, None
            elif winner == -root_player:
                return -10.0 + depth, None
            else:
                return 0.0, None

        valid_moves = game.get_valid_moves()
        is_max = (game.current_player == root_player)

        if is_max:
            best_score = -float('inf')
            best_move = None
            for action in valid_moves:
                child = game.clone()
                child.make_move(action)
                score, _ = self._minimax(child, depth + 1, root_player)
                if score > best_score:
                    best_score = score
                    best_move = action
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for action in valid_moves:
                child = game.clone()
                child.make_move(action)
                score, _ = self._minimax(child, depth + 1, root_player)
                if score < best_score:
                    best_score = score
                    best_move = action
            return best_score, best_move


class AlphaBetaAgent(BaseAgent):
    """
    Minimax Agent with Alpha-Beta Pruning.
    Returns identical optimal moves as Minimax, but searches fewer nodes.
    """

    def __init__(self, name: str = "AlphaBetaAgent"):
        super().__init__(name=name)
        self.nodes_searched = 0
        self.branches_pruned = 0
        self.decision_time = 0.0

    def select_move(self, game: TicTacToeGame) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        self.nodes_searched = 0
        self.branches_pruned = 0
        start_time = time.perf_counter()
        root_player = game.current_player

        _, best_move = self._alphabeta(
            game=game,
            depth=0,
            root_player=root_player,
            alpha=-float('inf'),
            beta=float('inf')
        )
        self.decision_time = time.perf_counter() - start_time

        if best_move is None:
            best_move = valid_moves[0]
        return best_move

    def _alphabeta(
        self,
        game: TicTacToeGame,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float
    ) -> Tuple[float, Optional[int]]:
        self.nodes_searched += 1

        if game.is_terminal():
            winner = game.check_winner()
            if winner == root_player:
                return 10.0 - depth, None
            elif winner == -root_player:
                return -10.0 + depth, None
            else:
                return 0.0, None

        valid_moves = game.get_valid_moves()
        is_max = (game.current_player == root_player)

        if is_max:
            best_score = -float('inf')
            best_move = None
            for action in valid_moves:
                child = game.clone()
                child.make_move(action)
                score, _ = self._alphabeta(child, depth + 1, root_player, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for action in valid_moves:
                child = game.clone()
                child.make_move(action)
                score, _ = self._alphabeta(child, depth + 1, root_player, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break
            return best_score, best_move
