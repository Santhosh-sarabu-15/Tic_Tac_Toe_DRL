import os
import json
from typing import List, Dict, Optional
import requests

from game import TicTacToeGame


POSITION_NAMES = {
    0: "Top-Left", 1: "Top-Center", 2: "Top-Right",
    3: "Middle-Left", 4: "Center", 5: "Middle-Right",
    6: "Bottom-Left", 7: "Bottom-Center", 8: "Bottom-Right"
}


class StrategicExplainer:
    """
    LLM/SLM Explanation Generator for DQN decisions.
    Receives actual game state, Q-values, Minimax recommendation, and RAG context.
    Strictly functions as an explanation layer (zero influence on move selection).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model

    def explain_move(
        self,
        game: TicTacToeGame,
        dqn_action: int,
        q_values: List[float],
        legal_moves: List[int],
        minimax_action: int,
        retrieved_context: List[Dict]
    ) -> str:
        """
        Generate strategic explanation based strictly on actual game data and RAG context.
        """
        action_name = POSITION_NAMES.get(dqn_action, f"Cell {dqn_action}")
        minimax_name = POSITION_NAMES.get(minimax_action, f"Cell {minimax_action}")
        agreement = (dqn_action == minimax_action)
        agreement_text = "AGREES" if agreement else "DIFFERS"

        dqn_q_val = float(q_values[dqn_action])

        # Formulate grounded prompt context
        board_ascii = game.render_ascii()
        q_summary = ", ".join([f"{POSITION_NAMES[m]}: {q_values[m]:.3f}" for m in legal_moves])

        context_passages = "\n".join([
            f"- [{c.get('title', 'Strategy')}]: {c.get('text', '')} (Similarity: {c.get('similarity_score', 0.0):.2f})"
            for c in retrieved_context
        ])

        # Attempt OpenAI-compatible API call if configured
        if self.api_key:
            try:
                explanation = self._call_llm_api(
                    board_ascii=board_ascii,
                    action_name=action_name,
                    dqn_action=dqn_action,
                    dqn_q_val=dqn_q_val,
                    q_summary=q_summary,
                    minimax_name=minimax_name,
                    agreement_text=agreement_text,
                    context_passages=context_passages
                )
                if explanation:
                    return explanation
            except Exception as e:
                print(f"[LLM WARNING] API call failed: {e}. Falling back to grounded template generator.")

        # Grounded Template Explainer (Guarantees zero hallucination when LLM API key not provided)
        return self._generate_template_explanation(
            board_ascii=board_ascii,
            action_name=action_name,
            dqn_action=dqn_action,
            dqn_q_val=dqn_q_val,
            minimax_name=minimax_name,
            agreement=agreement,
            q_summary=q_summary,
            retrieved_context=retrieved_context
        )

    def _call_llm_api(
        self,
        board_ascii: str,
        action_name: str,
        dqn_action: int,
        dqn_q_val: float,
        q_summary: str,
        minimax_name: str,
        agreement_text: str,
        context_passages: str
    ) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = (
            "You are an expert AI game strategist explaining a Deep Reinforcement Learning (DQN) agent's move in Tic-Tac-Toe. "
            "STRICT RULE: Do NOT invent or fabricate Q-values, board positions, or strategy evidence. "
            "Use only the provided factual data and retrieved strategic concepts."
        )

        user_prompt = f"""
Current Board State:
{board_ascii}

DQN Selected Move: {action_name} (Index {dqn_action})
DQN Assigned Q-value: {dqn_q_val:.4f}
Legal Actions Q-values: {q_summary}

Minimax Optimal Move: {minimax_name}
Classical Policy Agreement: {agreement_text}

Retrieved Strategy Knowledge (RAG):
{context_passages}

Please provide a concise strategic explanation addressing:
1. What move the DQN selected.
2. The strategic context of the move.
3. Why the move is useful (Q-value justification).
4. Agreement with classical Minimax.
5. Evidence from retrieved strategy knowledge.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        return None

    def _generate_template_explanation(
        self,
        board_ascii: str,
        action_name: str,
        dqn_action: int,
        dqn_q_val: float,
        minimax_name: str,
        agreement: bool,
        q_summary: str,
        retrieved_context: List[Dict]
    ) -> str:
        lines = [
            f"### DQN Move Analysis: {action_name} (Index {dqn_action})",
            f"- **Selected Move**: {action_name} (Q-value = {dqn_q_val:.4f})",
            f"- **Classical Minimax Recommendation**: {minimax_name}",
            f"- **Policy Agreement**: {'YES (Agrees with Minimax)' if agreement else 'NO (Differs from Minimax)'}",
            f"- **Q-Value Spectrum**: {q_summary}",
            "",
            "#### Strategic Rationale & Retrieved Evidence:"
        ]

        if retrieved_context:
            for ctx in retrieved_context:
                title = ctx.get("title", "Strategy")
                text = ctx.get("text", "")
                score = ctx.get("similarity_score", 0.0)
                lines.append(f"- **{title}** (Similarity: {score:.2f}): {text}")
        else:
            lines.append("- The DQN selected the legal move yielding the maximum expected cumulative reward.")

        if agreement:
            lines.append(
                f"\n*Summary*: The DQN selected {action_name} because it assigned it the highest Q-value ({dqn_q_val:.4f}) among all legal actions. "
                f"This decision perfectly aligns with classical Minimax game-tree optimization."
            )
        else:
            lines.append(
                f"\n*Summary*: The DQN selected {action_name} with Q-value {dqn_q_val:.4f}. "
                f"While Minimax recommends {minimax_name}, the DQN's learned policy prioritizes {action_name} based on its self-play value estimation."
            )

        return "\n".join(lines)
