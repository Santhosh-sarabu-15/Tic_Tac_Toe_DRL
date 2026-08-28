import os
import sys
import time
import numpy as np

from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent, AlphaBetaAgent
from dqn import DQNAgent
from train import train_dqn
from evaluate import run_full_evaluation
from rag import StrategyRAG
from llm import StrategicExplainer, POSITION_NAMES


def print_header():
    print("=" * 55)
    print("       AUTONOMOUS TIC-TAC-TOE PLAYER (DRL + RAG + LLM)")
    print("=" * 55)


def get_user_move(game: TicTacToeGame) -> int:
    valid_moves = game.get_valid_moves()
    while True:
        try:
            choice = input(f"Enter your move (0-8, valid: {valid_moves}): ").strip()
            move = int(choice)
            if move in valid_moves:
                return move
            print(f"Invalid move! Cell {move} is occupied or out of range.")
        except ValueError:
            print("Invalid input! Please enter an integer from 0 to 8.")


def play_human_vs_agent(agent_type: str):
    game = TicTacToeGame()
    minimax = MinimaxAgent()
    rag = None
    explainer = None

    if agent_type == "dqn":
        agent = DQNAgent()
        if os.path.exists("models/dqn.pt"):
            agent.load("models/dqn.pt")
            print("[MODEL] Loaded trained DQN weights from 'models/dqn.pt'.")
        else:
            print("[WARNING] 'models/dqn.pt' not found! Using untrained DQN.")
        agent.epsilon = 0.0
        rag = StrategyRAG()
        explainer = StrategicExplainer()
    elif agent_type == "minimax":
        agent = MinimaxAgent()
    elif agent_type == "alphabeta":
        agent = AlphaBetaAgent()

    print(f"\n--- Human (X) vs {agent.name} (O) ---")
    print("Board cell indices reference:")
    print(" 0 | 1 | 2 \n---+---+---\n 3 | 4 | 5 \n---+---+---\n 6 | 7 | 8 \n")

    print("Initial Board:")
    print(game.render_ascii())

    while not game.is_terminal():
        if game.current_player == 1:
            # Human turn
            print("\nYour turn (X):")
            move = get_user_move(game)
            game.make_move(move)
        else:
            # Agent turn
            print(f"\n{agent.name}'s turn (O)...")
            if agent_type == "dqn":
                # Compute state & Q-values
                legal_moves = game.get_valid_moves()
                q_vals = agent.get_q_values(game).numpy()
                action = agent.select_move(game)

                mm_action = minimax.select_move(game.clone())
                agreement = (action == mm_action)

                action_name = POSITION_NAMES.get(action, f"Cell {action}")
                mm_name = POSITION_NAMES.get(mm_action, f"Cell {mm_action}")

                print(f"[DQN LOG] Selected: {action_name} (Index {action}) | Q-value: {q_vals[action]:.4f}")
                print(f"[MINIMAX LOG] Optimal Action: {mm_name} | Agreement: {'YES' if agreement else 'NO'}")

                game.make_move(action)

                # RAG & LLM Explanation
                print("\n[RAG Retrieval & LLM Strategic Explanation]:")
                retrieved_context = rag.query(f"strategy for move {action_name} in position", k=2)
                explanation = explainer.explain_move(
                    game=game,
                    dqn_action=action,
                    q_values=q_vals,
                    legal_moves=legal_moves,
                    minimax_action=mm_action,
                    retrieved_context=retrieved_context
                )
                print(explanation)
            else:
                action = agent.select_move(game)
                print(f"[{agent.name} LOG] Selected Action: {POSITION_NAMES.get(action, f'Cell {action}')}")
                game.make_move(action)

        print("\nCurrent Board State:")
        print(game.render_ascii())

    winner = game.check_winner()
    print("\n" + "=" * 40)
    if winner == 1:
        print("RESULT: Congratulations! You (X) WON!")
    elif winner == -1:
        print(f"RESULT: {agent.name} (O) WON!")
    else:
        print("RESULT: Game ended in a DRAW!")
    print("=" * 40 + "\n")


def play_agent_vs_agent(a1_type: str, a2_type: str):
    game = TicTacToeGame()

    def load_agent(atype, name):
        if atype == "dqn":
            ag = DQNAgent(name=name)
            if os.path.exists("models/dqn.pt"):
                ag.load("models/dqn.pt")
            ag.epsilon = 0.0
            return ag
        elif atype == "minimax":
            return MinimaxAgent(name=name)
        elif atype == "alphabeta":
            return AlphaBetaAgent(name=name)

    agent1 = load_agent(a1_type, f"{a1_type.upper()}_P1(X)")
    agent2 = load_agent(a2_type, f"{a2_type.upper()}_P2(O)")

    print(f"\n--- Autonomous Play: {agent1.name} vs {agent2.name} ---")
    print("Initial Board:")
    print(game.render_ascii())

    while not game.is_terminal():
        current_agent = agent1 if game.current_player == 1 else agent2
        print(f"\n{current_agent.name} thinking...")

        action = current_agent.select_move(game)
        action_name = POSITION_NAMES.get(action, f"Cell {action}")
        print(f"[{current_agent.name}] Move: {action_name} (Index {action})")

        game.make_move(action)
        print(game.render_ascii())
        time.sleep(0.5)

    winner = game.check_winner()
    print("\n" + "=" * 40)
    if winner == 1:
        print(f"RESULT: {agent1.name} WINS!")
    elif winner == -1:
        print(f"RESULT: {agent2.name} WINS!")
    else:
        print("RESULT: Game ended in a DRAW!")
    print("=" * 40 + "\n")


def main_menu():
    while True:
        print_header()
        print("1. Human vs Minimax")
        print("2. Human vs Alpha-Beta")
        print("3. Human vs DQN (with RAG/LLM Explanation)")
        print("4. DQN vs Minimax (Autonomous Play)")
        print("5. DQN vs Alpha-Beta (Autonomous Play)")
        print("6. Minimax vs Alpha-Beta (Autonomous Play)")
        print("7. Train DQN (Self-Play)")
        print("8. Evaluate Agents (Tournament Benchmark)")
        print("9. Exit")
        print("=" * 55)

        choice = input("Select option (1-9): ").strip()

        if choice == "1":
            play_human_vs_agent("minimax")
        elif choice == "2":
            play_human_vs_agent("alphabeta")
        elif choice == "3":
            play_human_vs_agent("dqn")
        elif choice == "4":
            play_agent_vs_agent("dqn", "minimax")
        elif choice == "5":
            play_agent_vs_agent("dqn", "alphabeta")
        elif choice == "6":
            play_agent_vs_agent("minimax", "alphabeta")
        elif choice == "7":
            print("\nStarting DQN Self-Play Training...")
            train_dqn(episodes=25000)
        elif choice == "8":
            run_full_evaluation()
        elif choice == "9":
            print("Exiting application. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice! Please select an option from 1 to 9.")


if __name__ == "__main__":
    main_menu()
