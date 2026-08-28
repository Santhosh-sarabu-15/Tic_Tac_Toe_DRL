import os
import tempfile
# pyrefly: ignore [missing-import]
import pytest
import numpy as np
import torch
from game import TicTacToeGame
from dqn import QNetwork, ReplayBuffer, DQNAgent


def test_network_shape():
    net = QNetwork()
    x = torch.randn(1, 9)
    out = net(x)
    assert out.shape == (1, 9)

    batch_x = torch.randn(32, 9)
    batch_out = net(batch_x)
    assert batch_out.shape == (32, 9)


def test_replay_buffer():
    buffer = ReplayBuffer(capacity=100)
    state = np.zeros(9)
    next_state = np.ones(9)
    buffer.push(state, 4, 1.0, next_state, True)

    assert len(buffer) == 1

    states, actions, rewards, next_states, dones = buffer.sample(1)
    assert states.shape == (1, 9)
    assert actions.shape == (1,)
    assert rewards.shape == (1,)
    assert next_states.shape == (1, 9)
    assert dones.shape == (1,)


def test_legal_action_masking():
    game = TicTacToeGame()
    agent = DQNAgent()
    agent.epsilon = 0.0  # Force deterministic greedy selection

    # Fill center and top-left cell
    game.make_move(4)  # X plays 4
    game.make_move(0)  # O plays 0

    valid_moves = game.get_valid_moves()
    assert 4 not in valid_moves
    assert 0 not in valid_moves

    selected_move = agent.select_move(game)
    assert selected_move in valid_moves
    assert selected_move != 4
    assert selected_move != 0


def test_model_save_load():
    agent = DQNAgent()
    game = TicTacToeGame()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_dqn.pt")
        q_before = agent.get_q_values(game).numpy()

        agent.save(model_path)
        assert os.path.exists(model_path)

        new_agent = DQNAgent()
        new_agent.load(model_path)
        q_after = new_agent.get_q_values(game).numpy()

        np.testing.assert_allclose(q_before, q_after, rtol=1e-5)


def test_dqn_short_training():
    agent = DQNAgent()
    game = TicTacToeGame()

    for _ in range(100):
        state = game.get_state()
        valid = game.get_valid_moves()
        if not valid or game.is_terminal():
            game.reset()
            state = game.get_state()
            valid = game.get_valid_moves()
        action = valid[0]
        next_state, reward, done, _ = game.make_move(action)
        agent.memory.push(state, action, reward, next_state, done)

    loss = agent.train_step(batch_size=32)
    assert loss is not None
    assert loss >= 0.0
