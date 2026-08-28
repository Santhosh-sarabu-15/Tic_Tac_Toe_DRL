"""
Phase 1 Demonstration Script
Demonstrates:
1. A winning Minimax move
2. A blocking Minimax move
3. Alpha-Beta producing the same optimal result
4. Node count comparison & branches pruned
"""

from game import TicTacToeGame
from agents import MinimaxAgent, AlphaBetaAgent


def main():
    print("=" * 60)
    print("           PHASE 1 VERIFICATION & DEMONSTRATION")
    print("=" * 60)

    minimax = MinimaxAgent()
    alphabeta = AlphaBetaAgent()

    # -------------------------------------------------------------
    # 1. Winning Minimax Move
    # -------------------------------------------------------------
    print("\n--- 1. Winning Minimax Move ---")
    g1 = TicTacToeGame()
    g1.make_move(0)  # X plays 0
    g1.make_move(3)  # O plays 3
    g1.make_move(1)  # X plays 1
    g1.make_move(4)  # O plays 4
    print("Board State (X's turn to move):")
    print(g1.render_ascii())

    move_win = minimax.select_move(g1)
    print(f"Minimax Selected Move: {move_win}")
    g1.make_move(move_win)
    print(f"Board after move (Winner: {g1.check_winner()}):")
    print(g1.render_ascii())

    # -------------------------------------------------------------
    # 2. Blocking Minimax Move
    # -------------------------------------------------------------
    print("\n--- 2. Blocking Minimax Move ---")
    g2 = TicTacToeGame()
    g2.make_move(3)  # X plays 3
    g2.make_move(0)  # O plays 0
    g2.make_move(6)  # X plays 6
    g2.make_move(1)  # O plays 1 (O threatens 2)
    print("Board State (X must block O at cell 2):")
    print(g2.render_ascii())

    move_block = minimax.select_move(g2)
    print(f"Minimax Selected Move: {move_block}")
    g2.make_move(move_block)
    print("Board after block move:")
    print(g2.render_ascii())

    # -------------------------------------------------------------
    # 3 & 4. Alpha-Beta Equivalence & Search Efficiency
    # -------------------------------------------------------------
    print("\n--- 3 & 4. Alpha-Beta Equivalence & Node Comparison ---")
    g3 = TicTacToeGame()  # Initial empty board
    print("Initial empty board state:")
    print(g3.render_ascii())

    mm_move = minimax.select_move(g3)
    ab_move = alphabeta.select_move(g3)

    print(f"\n[Minimax]   Optimal Move: {mm_move} | Nodes Searched: {minimax.nodes_searched:,} | Time: {minimax.decision_time*1000:.2f} ms")
    print(f"[AlphaBeta] Optimal Move: {ab_move} | Nodes Searched: {alphabeta.nodes_searched:,} | Pruned: {alphabeta.branches_pruned:,} | Time: {alphabeta.decision_time*1000:.2f} ms")

    assert mm_move == ab_move, "Error: Move choices differ!"
    print("\n[SUCCESS] Minimax and Alpha-Beta returned the exact same optimal move!")
    print(f"[EFFICIENCY] Alpha-Beta pruned {alphabeta.branches_pruned:,} branches and reduced node expansions by {((minimax.nodes_searched - alphabeta.nodes_searched)/minimax.nodes_searched)*100:.2f}%.")
    print("=" * 60)


if __name__ == "__main__":
    main()
