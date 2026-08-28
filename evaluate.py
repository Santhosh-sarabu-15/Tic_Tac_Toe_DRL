import os
import json
import time
from typing import Dict, List, Tuple
import numpy as np

from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent, AlphaBetaAgent, BaseAgent
from dqn import DQNAgent


def evaluate_matchup(
    agent1: BaseAgent,
    agent2: BaseAgent,
    num_games: int = 100,
    track_agreement: bool = False,
    minimax_eval_agent: MinimaxAgent = None
) -> Dict:
    """
    Evaluate two agents against each other over a set number of games.
    Alternates starting player (first half: agent1 is X, second half: agent1 is O).

    Returns summary metrics dict.
    """
    a1_wins = 0
    a2_wins = 0
    draws = 0

    a1_times = []
    a2_times = []

    a1_nodes = []
    a2_nodes = []

    ab_pruned = []

    agreements = 0
    total_decisions_tracked = 0

    if track_agreement and minimax_eval_agent is None:
        minimax_eval_agent = AlphaBetaAgent()

    for game_idx in range(num_games):
        game = TicTacToeGame()

        # Alternate starting player
        if game_idx % 2 == 0:
            player_map = {1: agent1, -1: agent2}
        else:
            player_map = {1: agent2, -1: agent1}

        while not game.is_terminal():
            current_player_symbol = game.current_player
            active_agent = player_map[current_player_symbol]

            # If tracking agreement and active_agent is DQN
            if track_agreement and isinstance(active_agent, DQNAgent):
                # Ask Minimax for optimal action on current state
                mm_optimal_action = minimax_eval_agent.select_move(game.clone())
                
                # Get DQN action (deterministic, epsilon=0)
                active_agent.epsilon = 0.0
                dqn_action = active_agent.select_move(game)

                if dqn_action == mm_optimal_action:
                    agreements += 1
                total_decisions_tracked += 1

                action = dqn_action
            else:
                # Regular move selection
                if isinstance(active_agent, DQNAgent):
                    active_agent.epsilon = 0.0
                action = active_agent.select_move(game)

            # Record stats
            if isinstance(active_agent, MinimaxAgent):
                a1_nodes.append(active_agent.nodes_searched) if active_agent == agent1 else a2_nodes.append(active_agent.nodes_searched)
                a1_times.append(active_agent.decision_time) if active_agent == agent1 else a2_times.append(active_agent.decision_time)
            elif isinstance(active_agent, AlphaBetaAgent):
                if active_agent == agent1:
                    a1_nodes.append(active_agent.nodes_searched)
                    a1_times.append(active_agent.decision_time)
                else:
                    a2_nodes.append(active_agent.nodes_searched)
                    a2_times.append(active_agent.decision_time)
                ab_pruned.append(active_agent.branches_pruned)

            game.make_move(action)

        winner = game.check_winner()
        if winner == 1:
            winning_agent = player_map[1]
            if winning_agent == agent1:
                a1_wins += 1
            else:
                a2_wins += 1
        elif winner == -1:
            winning_agent = player_map[-1]
            if winning_agent == agent1:
                a1_wins += 1
            else:
                a2_wins += 1
        else:
            draws += 1

    policy_agreement_rate = (agreements / total_decisions_tracked * 100.0) if total_decisions_tracked > 0 else None

    return {
        "agent1": agent1.name,
        "agent2": agent2.name,
        "games_played": num_games,
        "agent1_wins": a1_wins,
        "agent2_wins": a2_wins,
        "draws": draws,
        "agent1_win_rate": round(a1_wins / num_games * 100.0, 1),
        "agent2_win_rate": round(a2_wins / num_games * 100.0, 1),
        "draw_rate": round(draws / num_games * 100.0, 1),
        "agent1_avg_time_ms": round(np.mean(a1_times) * 1000.0, 3) if a1_times else 0.0,
        "agent2_avg_time_ms": round(np.mean(a2_times) * 1000.0, 3) if a2_times else 0.0,
        "agent1_avg_nodes": round(float(np.mean(a1_nodes)), 1) if a1_nodes else None,
        "agent2_avg_nodes": round(float(np.mean(a2_nodes)), 1) if a2_nodes else None,
        "avg_alphabeta_pruned": round(float(np.mean(ab_pruned)), 1) if ab_pruned else None,
        "policy_agreement_rate": round(policy_agreement_rate, 2) if policy_agreement_rate is not None else None
    }


def run_full_evaluation(model_path: str = "models/dqn.pt", num_games: int = 100) -> Dict[str, Dict]:
    """Run full tournament across all specified agent pairings."""
    print("=" * 70)
    print("            TIC-TAC-TOE AGENT EVALUATION BENCHMARK")
    print("=" * 70)

    # Instantiate agents
    random_agent = RandomAgent()
    minimax_agent = MinimaxAgent()
    alphabeta_agent = AlphaBetaAgent()

    dqn_agent = DQNAgent()
    if os.path.exists(model_path):
        dqn_agent.load(model_path)
        print(f"[MODEL] Successfully loaded trained weights from '{model_path}'.")
    else:
        print(f"[WARNING] Trained model '{model_path}' not found! Using untrained DQN.")

    matchups = [
        ("DQN vs Random", dqn_agent, random_agent, 50, False),
        ("DQN vs Minimax", dqn_agent, minimax_agent, 4, True),
        ("DQN vs Alpha-Beta", dqn_agent, alphabeta_agent, 50, True),
        ("Minimax vs Alpha-Beta", minimax_agent, alphabeta_agent, 4, False),
    ]

    all_results = {}

    for label, a1, a2, games_cnt, track_agree in matchups:
        print(f"\nRunning Matchup: {label} ({games_cnt} games)...", flush=True)
        res = evaluate_matchup(a1, a2, num_games=games_cnt, track_agreement=track_agree, minimax_eval_agent=alphabeta_agent)
        all_results[label] = res

    # Save to JSON
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    display_results_table(all_results)
    return all_results


def display_results_table(results: Dict[str, Dict]):
    """Print clean ASCII summary table of evaluation results."""
    print("\n" + "=" * 70)
    print(f"{'MATCHUP':<24} | {'WINS':<6} | {'DRAWS':<6} | {'LOSSES':<6} | {'AGREEMENT':<10}")
    print("-" * 70)

    for matchup_name, res in results.items():
        w = res['agent1_wins']
        d = res['draws']
        l = res['agent2_wins']
        agree = f"{res['policy_agreement_rate']:.1f}%" if res['policy_agreement_rate'] is not None else "N/A"
        print(f"{matchup_name:<24} | {w:<6} | {d:<6} | {l:<6} | {agree:<10}")

    print("=" * 70)

    # Detailed Search & Timing Metrics
    print("\nSearch & Timing Benchmark Metrics:")
    print("-" * 70)
    for matchup_name, res in results.items():
        if res.get('agent1_avg_nodes') is not None or res.get('agent2_avg_nodes') is not None:
            nodes_info = f"Nodes (A1/A2): {res.get('agent1_avg_nodes', 0)} / {res.get('agent2_avg_nodes', 0)}"
            pruned_info = f"Pruned: {res.get('avg_alphabeta_pruned', 'N/A')}"
            print(f"{matchup_name:<24} -> {nodes_info} | {pruned_info}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_full_evaluation(num_games=100)
