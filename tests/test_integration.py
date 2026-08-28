import os
import pytest
from game import TicTacToeGame
from agents import MinimaxAgent, AlphaBetaAgent
from dqn import DQNAgent
from rag import StrategyRAG
from llm import StrategicExplainer


def test_full_game_dqn_vs_minimax():
    game = TicTacToeGame()
    dqn = DQNAgent(name="DQN_P1")
    if os.path.exists("models/dqn.pt"):
        dqn.load("models/dqn.pt")
    dqn.epsilon = 0.0
    minimax = MinimaxAgent(name="Minimax_P2")

    move_count = 0
    while not game.is_terminal() and move_count < 9:
        active = dqn if game.current_player == 1 else minimax
        action = active.select_move(game)
        game.make_move(action)
        move_count += 1

    assert game.is_terminal()
    assert game.check_winner() in [0, 1, -1]


def test_full_game_minimax_vs_alphabeta():
    game = TicTacToeGame()
    mm = MinimaxAgent()
    ab = AlphaBetaAgent()

    move_count = 0
    while not game.is_terminal() and move_count < 9:
        active = mm if game.current_player == 1 else ab
        action = active.select_move(game)
        game.make_move(action)
        move_count += 1

    assert game.is_terminal()
    assert game.is_draw()  # Optimal play between Minimax & Alpha-Beta MUST draw


def test_explanation_pipeline_integration():
    game = TicTacToeGame()
    dqn = DQNAgent()
    if os.path.exists("models/dqn.pt"):
        dqn.load("models/dqn.pt")
    dqn.epsilon = 0.0
    minimax = MinimaxAgent()
    rag = StrategyRAG()
    explainer = StrategicExplainer()

    # Get DQN action & Q-values
    legal_moves = game.get_valid_moves()
    q_vals = dqn.get_q_values(game).numpy()
    dqn_action = dqn.select_move(game)
    minimax_action = minimax.select_move(game.clone())

    # Retrieve context
    context = rag.query("center control and opening strategy", k=2)
    assert len(context) > 0

    # Generate explanation
    explanation = explainer.explain_move(
        game=game,
        dqn_action=dqn_action,
        q_values=q_vals,
        legal_moves=legal_moves,
        minimax_action=minimax_action,
        retrieved_context=context
    )

    assert explanation is not None
    assert len(explanation) > 50
    assert "Q-value" in explanation or "Selected Move" in explanation
