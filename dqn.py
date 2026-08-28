import os
import random
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents import BaseAgent
from game import TicTacToeGame


class QNetwork(nn.Module):
    """
    Deep Q-Network for Tic-Tac-Toe.
    Architecture:
      Input (9) -> Linear(9, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 9)
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, output_dim: int = 9):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer for storing transitions."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add experience transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state.copy(), action, float(reward), next_state.copy(), bool(done))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN Agent using PyTorch.
    Uses canonical board state representation (board * current_player)
    to evaluate positions from perspective of the active player.
    """

    def __init__(self, name: str = "DQNAgent", lr: float = 0.0005, gamma: float = 0.99):
        super().__init__(name=name)
        self.gamma = gamma

        self.policy_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=100000)

        self.epsilon = 1.0
        self.epsilon_min = 0.05

    def get_canonical_state(self, game: TicTacToeGame) -> np.ndarray:
        """
        Convert board state to canonical perspective (+1 = self, -1 = opponent, 0 = empty).
        """
        return (game.board * game.current_player).astype(np.float32)

    def get_q_values(self, game: TicTacToeGame) -> torch.Tensor:
        """
        Get Q-values for current game state.
        Returns tensor of shape (9,).
        """
        self.policy_net.eval()
        state = self.get_canonical_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)
        return q_values

    def select_move(self, game: TicTacToeGame) -> int:
        """
        Select move using epsilon-greedy policy with legal action masking.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Greedy action selection with legal move masking
        q_values = self.get_q_values(game).numpy().copy()

        masked_q = np.full(9, -np.inf)
        for move in valid_moves:
            masked_q[move] = q_values[move]

        return int(np.argmax(masked_q))

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """Perform a single mini-batch gradient descent step."""
        if len(self.memory) < batch_size:
            return None

        self.policy_net.train()
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Q(s, a)
        q_eval = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next_target = self.target_net(next_states).clone()
            
            # Mask illegal moves in next states (occupied cells in canonical perspective)
            for i in range(batch_size):
                if not dones[i]:
                    next_board = next_states[i].numpy()
                    for a in range(9):
                        if next_board[a] != 0:  # Cell is occupied (+1 or -1)
                            q_next_target[i, a] = -1e9

            max_q_next = q_next_target.max(dim=1)[0]
            max_q_next = torch.where(dones.bool(), torch.zeros_like(max_q_next), max_q_next)
            q_target = rewards + (self.gamma * max_q_next * (1.0 - dones))

        loss = nn.SmoothL1Loss()(q_eval, q_target)  # Huber Loss for stable Q-learning updates

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str = "models/dqn.pt"):
        """Save PyTorch model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str = "models/dqn.pt"):
        """Load PyTorch model weights."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        self.policy_net.eval()
        self.target_net.eval()
