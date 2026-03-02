"""
agents/evaluator.py
===================
EvaluatorAgent with attribution-guided causal tracking.

On a passing evaluation, calls repair_on_success() to attempt rehabilitation
of borderline-quarantined memories — making the repair loop bidirectional.
"""

import json
import logging
from uuid import uuid4
from memory.views import get_evaluator_view
from memory.repair import repair_on_success

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def _attribute_relevant_memories(self, memory_view, output: str) -> list:
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
            attributed = [memory_view[i] for i in indices if 0 <= i < len(memory_view)]
            if not attributed and memory_view:
                logger.warning(
                    "EvaluatorAgent attribution returned []: no memories linked to this "
                    "decision. Repair cannot trace this decision if it fails."
                )
            return attributed
        except Exception as e:
            logger.warning(
                f"EvaluatorAgent attribution parse failed ({e}): no memories linked."
            )
            return []

    def evaluate(self, output: str):
        """
        Evaluate agent output against verified knowledge.

        Returns (result_text, decision_id).

        Side effect: if evaluation passes (result contains 'PASS'), triggers
        repair_on_success() to attempt rehabilitation of borderline-quarantined
        memories. This makes the repair loop bidirectional:
          failure → caller runs repair_memories()
          success → evaluator runs repair_on_success() automatically
        """
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

Evaluate correctness and consistency. Start your response with PASS or FAIL.
"""
        result = self.llm.generate(prompt)

        relevant = self._attribute_relevant_memories(memory_view, result)
        for mem in relevant:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        # On a passing evaluation, attempt to rehabilitate borderline-quarantined
        # memories that were not involved in this decision. This is the success-
        # side of the bidirectional repair loop.
        if result and "PASS" in result.upper():
            rehabilitated = repair_on_success(self.memory_store, decision_id)
            if rehabilitated:
                logger.info(
                    f"EvaluatorAgent: {len(rehabilitated)} memory(s) rehabilitated "
                    f"after successful decision {decision_id}"
                )

        return result, decision_id