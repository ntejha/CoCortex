"""
agents/evaluator.py
===================
EvaluatorAgent with attribution-guided causal tracking.
"""

import json
from uuid import uuid4
from memory.views import get_evaluator_view


class EvaluatorAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def _attribute_relevant_memories(self, memory_view, output: str) -> list:
        """Ask the LLM which memories it actually relied on."""
        if not memory_view:
            return []

        indexed = "\n".join(
            f"[{i}] {m.content[:200]}" for i, m in enumerate(memory_view)
        )

        prompt = f"""You just produced the following evaluation:
"{output[:500]}"

The following verified facts were available to you (by index):
{indexed}

Which fact indices did you actually use to produce that evaluation?
Answer with a JSON array of integer indices only, e.g. [0, 2].
If none were used, answer [].
No explanation, just the JSON array."""

        try:
            raw = self.llm.generate(prompt).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            indices = json.loads(raw)
            return [memory_view[i] for i in indices if 0 <= i < len(memory_view)]
        except Exception:
            return []

    def evaluate(self, output: str):
        decision_id = f"evaluator_{uuid4().hex}"
        memory_view = get_evaluator_view(self.memory_store)

        memory_text = (
            "\n".join(f"- {m.content}" for m in memory_view)
            if memory_view else "No verified facts available."
        )

        prompt = f"""You are an Evaluator Agent.

Verified knowledge:
{memory_text}

Output to evaluate:
{output}

Evaluate correctness and consistency.
"""
        result = self.llm.generate(prompt)

        # Attribution-guided causal linking
        relevant = self._attribute_relevant_memories(memory_view, result)
        for mem in relevant:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        return result, decision_id