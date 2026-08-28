import pytest
import numpy as np
from game import TicTacToeGame


def test_valid_moves():
    game = TicTacToeGame()
    assert game.get_valid_moves() == list(range(9))

    game.make_move(4)  # X moves center
    assert 4 not in game.get_valid_moves()
    assert len(game.get_valid_moves()) == 8


def test_invalid_moves():
    game = TicTacToeGame()
    game.make_move(0)

    # Move on occupied cell
    with pytest.raises(ValueError, match="Illegal move"):
        game.make_move(0)

    # Move out of bounds
    with pytest.raises(ValueError, match="Illegal move"):
        game.make_move(10)

    with pytest.raises(ValueError, match="Illegal move"):
        game.make_move(-1)


def test_win_detection_horizontal():
    game = TicTacToeGame()
    # X plays top row: 0, 1, 2
    # O plays middle row: 3, 4
    game.make_move(0)  # X
    game.make_move(3)  # O
    game.make_move(1)  # X
    game.make_move(4)  # O
    game.make_move(2)  # X wins

    assert game.is_terminal()
    assert game.check_winner() == 1
    assert not game.is_draw()


def test_win_detection_vertical():
    game = TicTacToeGame()
    # O plays left column: 0, 3, 6
    # X plays non-winning cells: 1, 2, 4
    game.make_move(1)  # X
    game.make_move(0)  # O
    game.make_move(2)  # X
    game.make_move(3)  # O
    game.make_move(4)  # X
    game.make_move(6)  # O wins

    assert game.is_terminal()
    assert game.check_winner() == -1


def test_win_detection_diagonal():
    game = TicTacToeGame()
    # X plays main diagonal: 0, 4, 8
    game.make_move(0)  # X
    game.make_move(1)  # O
    game.make_move(4)  # X
    game.make_move(2)  # O
    game.make_move(8)  # X wins

    assert game.is_terminal()
    assert game.check_winner() == 1


def test_draw_detection():
    game = TicTacToeGame()
    # Sequence leading to a draw:
    # X O X
    # X O O
    # O X X
    # Board indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
    # Moves sequence:
    # X: 0, O: 1, X: 2, O: 4, X: 3, O: 5, X: 7, O: 6, X: 8
    for m in moves:
        game.make_move(m)

    assert game.is_terminal()
    assert game.is_draw()
    assert game.check_winner() == 0


def test_reset():
    game = TicTacToeGame()
    game.make_move(0)
    game.make_move(1)
    game.reset()

    assert np.all(game.board == 0)
    assert game.current_player == 1
    assert game.winner == 0
    assert not game.done
    assert len(game.get_valid_moves()) == 9
