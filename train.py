import os
import csv
import json
import random
import time
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from game import TicTacToeGame
from dqn import DQNAgent


def train_dqn(
    episodes: int = 50000,
    batch_size: int = 64,
    lr: float = 0.0005,
    gamma: float = 0.99,
    target_update_freq: int = 500,
    log_interval: int = 2500,
    save_path: str = "models/dqn.pt",
    logs_dir: str = "logs"
) -> DQNAgent:
    """
    Train DQN agent through Self-Play with precise 2-player credit assignment.
    No Minimax or heuristics are used during training.
    """
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    agent = DQNAgent(name="DQN_SelfPlay", lr=lr, gamma=gamma)
    agent.epsilon = 1.0
    epsilon_min = 0.05
    # Smooth epsilon decay over 80% of episodes
    epsilon_decay = (epsilon_min / agent.epsilon) ** (1.0 / (episodes * 0.80))

    total_steps = 0
    recent_results = []  # Track (+1 for X win, -1 for O win, 0 for draw)
    recent_losses = []
    metrics_history = []

    start_time = time.time()
    print(f"Starting Corrected DQN Self-Play Training for {episodes:,} episodes...")
    print(f"Hyperparameters: lr={lr}, gamma={gamma}, batch_size={batch_size}, epsilon_start=1.0, epsilon_min={epsilon_min}")
    print("=" * 70, flush=True)

    for ep in range(1, episodes + 1):
        game = TicTacToeGame()
        last_experience = {}  # {player_id: (canonical_state, action)}

        while not game.is_terminal():
            player = game.current_player
            state = agent.get_canonical_state(game)

            action = agent.select_move(game)
            _, _, done, info = game.make_move(action)
            total_steps += 1

            opponent = -player

            # Transition update for opponent's previous move once current move lands on next_board
            if opponent in last_experience:
                opp_state, opp_action = last_experience[opponent]
                opp_next_state = (game.board * opponent).astype(np.float32)

                if done:
                    winner = info["winner"]
                    if winner == player:
                        # Opponent lost
                        agent.memory.push(opp_state, opp_action, -1.0, np.zeros(9, dtype=np.float32), True)
                    elif winner == 0:
                        # Draw
                        agent.memory.push(opp_state, opp_action, 0.0, np.zeros(9, dtype=np.float32), True)
                else:
                    # Opponent's move successfully reached opp_next_state
                    agent.memory.push(opp_state, opp_action, 0.0, opp_next_state, False)

            if done:
                winner = info["winner"]
                if winner == player:
                    # Active player won
                    agent.memory.push(state, action, 1.0, np.zeros(9, dtype=np.float32), True)
                    recent_results.append(1 if player == 1 else -1)
                elif winner == 0:
                    # Draw
                    agent.memory.push(state, action, 0.0, np.zeros(9, dtype=np.float32), True)
                    recent_results.append(0)
                else:
                    # Active player lost (should not happen on active move)
                    agent.memory.push(state, action, -1.0, np.zeros(9, dtype=np.float32), True)
                    recent_results.append(-1 if player == 1 else 1)
            else:
                last_experience[player] = (state, action)

            # Perform gradient step every 4 steps
            if total_steps % 4 == 0:
                loss = agent.train_step(batch_size=batch_size)
                if loss is not None:
                    recent_losses.append(loss)

            if total_steps % target_update_freq == 0:
                agent.update_target_network()

        # Decay epsilon
        if agent.epsilon > epsilon_min:
            agent.epsilon *= epsilon_decay
            agent.epsilon = max(agent.epsilon, epsilon_min)

        if ep % log_interval == 0 or ep == episodes:
            eval_window = recent_results[-log_interval:] if len(recent_results) >= log_interval else recent_results
            x_wins = sum(1 for r in eval_window if r == 1)
            o_wins = sum(1 for r in eval_window if r == -1)
            draws = sum(1 for r in eval_window if r == 0)
            n_eval = len(eval_window) if len(eval_window) > 0 else 1

            win_rate = (x_wins / n_eval) * 100.0
            loss_rate = (o_wins / n_eval) * 100.0
            draw_rate = (draws / n_eval) * 100.0
            avg_reward = sum(eval_window) / n_eval
            avg_loss = sum(recent_losses[-log_interval:]) / max(1, len(recent_losses[-log_interval:])) if recent_losses else 0.0

            log_entry = {
                "episode": ep,
                "win_rate": round(win_rate, 2),
                "draw_rate": round(draw_rate, 2),
                "loss_rate": round(loss_rate, 2),
                "avg_reward": round(avg_reward, 4),
                "epsilon": round(agent.epsilon, 4),
                "loss": round(avg_loss, 6)
            }
            metrics_history.append(log_entry)

            print(f"[TRAINING] Episode: {ep:6d} | Win Rate (X): {win_rate:5.1f}% | Draw Rate: {draw_rate:5.1f}% | Loss Rate (O): {loss_rate:5.1f}% | Avg Reward: {avg_reward:+.3f} | Epsilon: {agent.epsilon:.4f} | Loss: {avg_loss:.5f}", flush=True)

    elapsed_time = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"Training completed in {elapsed_time:.2f} seconds.", flush=True)

    agent.save(save_path)
    print(f"[MODEL] Saved trained weights to {save_path}", flush=True)

    save_metrics(metrics_history, logs_dir)
    plot_metrics(metrics_history, os.path.join(logs_dir, "training_progress.png"))

    return agent


def save_metrics(metrics: List[Dict], logs_dir: str):
    csv_path = os.path.join(logs_dir, "training_metrics.csv")
    json_path = os.path.join(logs_dir, "training_metrics.json")

    fieldnames = ["episode", "win_rate", "draw_rate", "loss_rate", "avg_reward", "epsilon", "loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[LOGS] Saved training metrics to {csv_path} and {json_path}", flush=True)


def plot_metrics(metrics: List[Dict], plot_path: str):
    episodes = [m["episode"] for m in metrics]
    draw_rates = [m["draw_rate"] for m in metrics]
    avg_rewards = [m["avg_reward"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(episodes, draw_rates, color="green", label="Draw Rate (%)")
    axes[0].set_title("Self-Play Convergence (Draw Rate)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Draw Rate (%)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(episodes, avg_rewards, color="blue", label="Avg Reward")
    axes[1].set_title("Average Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(episodes, losses, color="red", label="Training Loss")
    axes[2].set_title("DQN Training Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT] Saved training progress graph to {plot_path}", flush=True)


if __name__ == "__main__":
    train_dqn(episodes=50000)
