import os
# pyrefly: ignore [missing-import]
import pytest
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent, AlphaBetaAgent
from dqn import DQNAgent
from evaluate import evaluate_matchup, run_full_evaluation


def test_evaluate_matchup():
    a1 = RandomAgent("Random1")
    a2 = RandomAgent("Random2")
    res = evaluate_matchup(a1, a2, num_games=10)

    assert res["games_played"] == 10
    assert res["agent1_wins"] + res["agent2_wins"] + res["draws"] == 10
    assert 0.0 <= res["agent1_win_rate"] <= 100.0


def test_minimax_vs_alphabeta():
    mm = MinimaxAgent()
    ab = AlphaBetaAgent()
    res = evaluate_matchup(mm, ab, num_games=2)

    # Minimax vs Alpha-Beta between optimal agents must ALWAYS draw (0 wins for either)
    assert res["agent1_wins"] == 0
    assert res["agent2_wins"] == 0
    assert res["draws"] == 2
    assert res["draw_rate"] == 100.0


def test_policy_agreement_with_dqn():
    dqn = DQNAgent()
    ab = AlphaBetaAgent()
    if os.path.exists("models/dqn.pt"):
        dqn.load("models/dqn.pt")

    res = evaluate_matchup(dqn, ab, num_games=10, track_agreement=True)
    assert res["policy_agreement_rate"] is not None
    assert 0.0 <= res["policy_agreement_rate"] <= 100.0
