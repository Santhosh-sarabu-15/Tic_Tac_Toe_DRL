# Autonomous Tic-Tac-Toe: Deep RL (DQN), Classical Search & RAG-LLM Explainer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end autonomous artificial intelligence project combining **Deep Reinforcement Learning (DQN)**, **Classical Game-Tree Search (Minimax & Alpha-Beta Pruning)**, and a **Retrieval-Augmented Generation (RAG) + LLM Strategic Explainer layer** for interactive Tic-Tac-Toe gameplay and policy analysis.

---

## 🌟 Key Features

- **🎮 Core Game Engine (`game.py`)**  
  Fast, clean 3x3 state representation with 1D array encoding, legal move validation, win/draw detection, and canonical board transformations.

- **🧠 Classical Search Agents (`agents.py`)**  
  - **Random Agent**: Baseline uniform random decision maker.  
  - **Minimax Agent**: Full game-tree search with depth-discounted scoring ($+10 - \text{depth}$ for wins, $-10 + \text{depth}$ for losses).  
  - **Alpha-Beta Pruning Agent**: Optimized game-tree traversal reducing total nodes evaluated from **549,946 to 20,866 (96.21% reduction)**.

- **⚡ Deep Reinforcement Learning (`dqn.py`, `train.py`)**  
  - PyTorch 3-layer MLP architecture (`9 -> 128 -> 128 -> 9`).  
  - Experience Replay Buffer (capacity 100,000) & $\epsilon$-greedy exploration schedule.  
  - **Pure Self-Play Training**: Trained over 50,000 self-play episodes with legal action masking and canonical perspective mapping (`board * current_player`). Zero heuristic guidance during training.

- **📊 Benchmark & Evaluation Engine (`evaluate.py`)**  
  - Automated agent tournaments tracking Win/Draw/Loss rates and Minimax Policy Agreement % across iterations.

- **📚 RAG Strategic Knowledge Base (`rag.py`)**  
  - Vector similarity search built using `SentenceTransformers` (`all-MiniLM-L6-v2`) and FAISS (`faiss.IndexFlatIP`).  
  - Indexing 11 tactical/strategic concepts in `knowledge_base/strategies.txt` (Center Control, Fork Creation, Defensive Parity, Double Threats, Endgame Tactics, etc.).

- **💬 Strategic LLM/SLM Explainer (`llm.py`)**  
  - Generates real-time natural language explanations for DQN moves by combining state context, Q-values, Minimax recommendations, and retrieved strategy vectors.  
  - Supports OpenAI-compatible APIs with a robust offline deterministic template fallback.

- **🖥️ Interactive CLI Application (`main.py`)**  
  Features a full interactive terminal menu for Human vs AI play (with live explanations), Agent vs Agent simulations, model training, and evaluation benchmarks.

---

## 📈 Agent Performance Metrics

### Search Space Optimization
| Agent | Nodes Evaluated | Decision Time | Search Reduction |
|---|---|---|---|
| **Minimax** | 549,946 | 9.28 s | 0% (Baseline) |
| **Alpha-Beta** | 20,866 | 0.37 s | **96.21%** |

### Retrained DQN Benchmark Results (50k Self-Play Episodes)
| Matchup | Wins | Draws | Losses | Win / Draw Rate | Policy Agreement with Minimax |
|---|---|---|---|---|---|
| **DQN vs Random** | 48 | 2 | 0 | **96.0% Win Rate** | N/A |
| **DQN vs Minimax** | 0 | 50 | 0 | **100.0% Draw Rate (0% Loss)** | **66.7%** |
| **DQN vs Alpha-Beta** | 0 | 50 | 0 | **100.0% Draw Rate (0% Loss)** | **66.7%** |

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Tic_Tac_Toe_DRL.git
   cd Tic_Tac_Toe_DRL
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Launch Interactive CLI
```bash
python main.py
```
Menu Options:
1. **Human vs Minimax**
2. **Human vs Alpha-Beta**
3. **Human vs DQN** *(with live RAG/LLM strategic explanations)*
4. **DQN vs Minimax** *(Autonomous agent simulation)*
5. **DQN vs Alpha-Beta** *(Autonomous agent simulation)*
6. **Minimax vs Alpha-Beta** *(Autonomous agent simulation)*
7. **Train DQN Agent**
8. **Evaluate Agents**
9. **Exit**

### 2. Run Test Suite
To verify engine functionality, agent mechanics, DQN logic, RAG retrieval, and full integration:
```bash
pytest
```

---

## 📂 Project Structure

```
Tic_Tac_Toe_DRL/
│
├── game.py                 # Core 3x3 Tic-Tac-Toe Game Engine
├── agents.py               # Random, Minimax, and Alpha-Beta Search Agents
├── dqn.py                  # PyTorch Deep Q-Network Architecture & Replay Buffer
├── train.py                # Pure Self-Play DQN Training Pipeline
├── evaluate.py             # Tournament Evaluation & Policy Agreement Metrics
├── rag.py                  # FAISS + SentenceTransformer Knowledge Base Retriever
├── llm.py                  # RAG-Grounded Natural Language Move Explainer
├── main.py                 # Interactive CLI Application Entrypoint
├── DEVELOPMENT_LOG.md      # Detailed Technical Log of Development Phases
├── requirements.txt        # Project Python Dependencies
│
├── knowledge_base/
│   └── strategies.txt      # 11 Strategic Concepts for RAG Retrieval
│
├── models/
│   └── dqn.pt              # Saved PyTorch DQN Weights
│
├── logs/
│   ├── evaluation_results.json
│   └── evaluation_results_baseline.json
│
└── tests/                  # Unit and Integration Test Suite (28 tests)
    ├── test_game.py
    ├── test_agents.py
    ├── test_dqn.py
    ├── test_evaluation.py
    ├── test_rag.py
    ├── test_llm.py
    └── test_integration.py
```

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
