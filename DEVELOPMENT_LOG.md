# Development Log — Autonomous Tic-Tac-Toe Player

## Phase 1: Tic-Tac-Toe Engine & Classical Search Agents

- **Status**: Completed & Verified
- **Implemented Features**:
  - `game.py`: `TicTacToeGame` engine with state management (3x3 grid encoded as 1D array), `reset()`, `get_valid_moves()`, `make_move()`, `check_winner()`, `is_draw()`, `is_terminal()`, `get_state()`, and `clone()`. Illegal move enforcement with `ValueError`.
  - `agents.py`:
    - `RandomAgent`: Uniform random valid action selection.
    - `MinimaxAgent`: Full depth-discounted game-tree search (`+10 - depth` for wins, `-10 + depth` for losses, `0` for draws).
    - `AlphaBetaAgent`: Alpha-beta pruning optimization ($\alpha, \beta$). Tracks `nodes_searched`, `branches_pruned`, and `decision_time`.
- **Tests**: `tests/test_game.py`, `tests/test_agents.py`.
- **Key Results**:
  - Minimax standard search: **549,946** nodes (9,289 ms)
  - Alpha-Beta search: **20,866** nodes (376 ms) | Pruned: **7,692** | Search space reduction: **96.21%**

---

## Phase 2: Deep Reinforcement Learning (DQN) & Self-Play

- **Status**: Completed & Verified
- **Implemented Features**:
  - `dqn.py`: PyTorch 3-layer MLP (`9 -> 128 -> 128 -> 9`), `ReplayBuffer` (capacity 100,000), `DQNAgent` (epsilon-greedy, legal action masking, canonical state transformation `board * current_player`).
  - `train.py`: Pure self-play training loop (2 DQN agents playing against each other without Minimax or heuristic guidance).
  - `tests/test_dqn.py`: 5 tests passed.
- **Root Cause Fixes**:
  - **Transition Perspective Bug**: Fixed intermediate transition pushing in `train.py` where opponent's next state was incorrectly captured before move execution. Updated to exact `game.board * opponent` state after move.
  - **Terminal Target Masking**: Added explicit `torch.where` mask for terminal states in Q-target calculation.
  - **Huber Loss**: Replaced MSE with `SmoothL1Loss`.
- **Retraining**: 50,000 pure self-play episodes. Loss decayed to `0.00970`.

---

## Phase 3: Agent Evaluation & Policy Agreement Benchmark

- **Status**: Completed & Verified
- **Preserved Baseline**: `logs/evaluation_results_baseline.json` preserved initial pre-fix metrics.
- **Retrained Tournament Results**:
  - **DQN vs Random**: Wins: **48** | Draws: **2** | Losses: **0** (96.0% Win Rate, **0.0% Loss Rate**)
  - **DQN vs Minimax**: Wins: **0** | Draws: **4** | Losses: **0** (100.0% Draw Rate, **0.0% Loss Rate**, **66.7% Policy Agreement**)
  - **DQN vs Alpha-Beta**: Wins: **0** | Draws: **50** | Losses: **0** (100.0% Draw Rate, **0.0% Loss Rate**, **66.7% Policy Agreement**)
  - **Minimax vs Alpha-Beta**: Wins: **0** | Draws: **4** | Losses: **0** (100.0% Draw Rate)
- **Artifacts**: `logs/evaluation_results.json`.

---

## Phase 4: RAG Knowledge Base & LLM/SLM Strategic Explanation

- **Status**: Completed & Verified
- **Implemented Features**:
  - `knowledge_base/strategies.txt`: Comprehensive strategy knowledge base covering 11 tactical/strategic concepts (Winning Moves, Blocking Threats, Center Control, Corner Strategy, Edge Strategy, Opening Strategy, Defensive Parity, Endgame Strategy, Creating Forks, Blocking Forks, Double Threats).
  - `rag.py`: `StrategyRAG` class utilizing `SentenceTransformers` (`all-MiniLM-L6-v2`) and FAISS vector index (`faiss.IndexFlatIP`) for L2-normalized cosine similarity search over strategy passages.
  - `llm.py`: `StrategicExplainer` class connecting to OpenAI-compatible API endpoints or utilizing a grounded deterministic fallback template explainer. Strictly functions as an explanation layer with zero influence on move selection. Incorporates actual board state, DQN Q-values, selected action, Minimax recommendation, and retrieved strategy context.
- **Tests**: `tests/test_rag.py`, `tests/test_llm.py` (5/5 tests passed).

---

## Phase 5: CLI Program & Full System Integration

- **Status**: Completed & Verified
- **Implemented Features**:
  - `main.py`: Interactive CLI application with 9 menu options:
    1. Human vs Minimax
    2. Human vs Alpha-Beta
    3. Human vs DQN (with live RAG/LLM strategic explanations)
    4. DQN vs Minimax (Autonomous agent play)
    5. DQN vs Alpha-Beta (Autonomous agent play)
    6. Minimax vs Alpha-Beta (Autonomous agent play)
    7. Train DQN
    8. Evaluate Agents
    9. Exit
  - `tests/test_integration.py`: Autonomous full game execution and end-to-end RAG/LLM explanation pipeline tests.
- **Full Test Suite Status**: **28 / 28 pytest test cases PASSED**.
