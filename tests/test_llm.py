import os
import pytest
import numpy as np
from game import TicTacToeGame
from rag import StrategyRAG
from llm import StrategicExplainer


def test_llm_explanation_generation():
    game = TicTacToeGame()
    # X plays 0, 1 -> X threatens 2
    game.make_move(0)
    game.make_move(3)
    game.make_move(1)
    game.make_move(4)

    dqn_action = 2
    minimax_action = 2
    q_values = np.array([0.1, 0.2, 0.95, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    legal_moves = game.get_valid_moves()

    rag = StrategyRAG(kb_path="knowledge_base/strategies.txt")
    context = rag.query("winning move completion", k=2)

    explainer = StrategicExplainer()
    explanation = explainer.explain_move(
        game=game,
        dqn_action=dqn_action,
        q_values=q_values,
        legal_moves=legal_moves,
        minimax_action=minimax_action,
        retrieved_context=context
    )

    assert "Selected Move" in explanation or "DQN Selected Move" in explanation
    assert "0.95" in explanation
    assert "AGREES" in explanation or "Agrees with Minimax" in explanation
    assert len(explanation) > 100


def test_llm_disagreement_explanation():
    game = TicTacToeGame()
    dqn_action = 0
    minimax_action = 4
    q_values = np.array([0.88, 0.1, 0.2, 0.1, 0.75, 0.1, 0.1, 0.1, 0.1])
    legal_moves = game.get_valid_moves()

    explainer = StrategicExplainer()
    explanation = explainer.explain_move(
        game=game,
        dqn_action=dqn_action,
        q_values=q_values,
        legal_moves=legal_moves,
        minimax_action=minimax_action,
        retrieved_context=[]
    )

    assert "Differs from Minimax" in explanation or "NO" in explanation
