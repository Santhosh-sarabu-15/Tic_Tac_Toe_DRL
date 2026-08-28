import pytest
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent, AlphaBetaAgent


def test_random_agent():
    game = TicTacToeGame()
    agent = RandomAgent()
    move = agent.select_move(game)
    assert move in game.get_valid_moves()


def test_minimax_winning_move():
    game = TicTacToeGame()
    # Setup board state where X has two in top row (0, 1) and cell 2 is empty:
    # X | X | -
    # O | O | -
    # - | - | -
    game.make_move(0)  # X
    game.make_move(3)  # O
    game.make_move(1)  # X
    game.make_move(4)  # O

    # Current player is X. Immediate win at index 2.
    minimax = MinimaxAgent()
    chosen_move = minimax.select_move(game)
    assert chosen_move == 2


def test_minimax_blocking_move():
    game = TicTacToeGame()
    # Setup board state where O has two in top row (0, 1) and cell 2 is empty:
    # O | O | -
    # X | - | -
    # - | - | -
    game.make_move(3)  # X plays 3
    game.make_move(0)  # O plays 0
    game.make_move(6)  # X plays 6
    game.make_move(1)  # O plays 1

    # Current player is X. Opponent O threatens win at index 2. X MUST block at cell 2.
    minimax = MinimaxAgent()
    chosen_move = minimax.select_move(game)
    assert chosen_move == 2


def test_alpha_beta_equivalence():
    """Verify Alpha-Beta selects identical moves as Minimax across various board configurations."""
    minimax = MinimaxAgent()
    alphabeta = AlphaBetaAgent()

    # Test on empty board
    game = TicTacToeGame()
    assert minimax.select_move(game) == alphabeta.select_move(game)

    # Test on multiple mid-game board configurations
    scenarios = [
        [0],
        [4],
        [0, 4, 8],
        [0, 1, 3],
        [4, 0, 2, 6],
        [4, 0, 1, 7, 3]
    ]

    for moves in scenarios:
        g = TicTacToeGame()
        for m in moves:
            g.make_move(m)
        if not g.is_terminal():
            move_mm = minimax.select_move(g)
            move_ab = alphabeta.select_move(g)
            assert move_mm == move_ab, f"Discrepancy on moves {moves}: Minimax={move_mm}, AlphaBeta={move_ab}"


def test_alpha_beta_pruning_performance():
    """Verify Alpha-Beta searches fewer nodes than Minimax and records pruned branches."""
    game = TicTacToeGame()
    minimax = MinimaxAgent()
    alphabeta = AlphaBetaAgent()

    minimax.select_move(game)
    alphabeta.select_move(game)

    print(f"\n[Empty Board] Minimax nodes searched: {minimax.nodes_searched}")
    print(f"[Empty Board] AlphaBeta nodes searched: {alphabeta.nodes_searched}")
    print(f"[Empty Board] AlphaBeta branches pruned: {alphabeta.branches_pruned}")

    assert alphabeta.nodes_searched < minimax.nodes_searched
    assert alphabeta.branches_pruned > 0
