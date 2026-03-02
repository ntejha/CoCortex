"""
agents/planner.py
=================
PlannerAgent with attribution-guided causal tracking.
"""

import json
import logging
from uuid import uuid4
from memory.views import get_planner_view

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def _attribute_relevant_memories(self, memory_view, output: str) -> list:
        """
        Ask the LLM which memories it actually used. Returns only those objects.
        Logs a warning on parse failure so silent empty attribution is visible.
        """
        if not memory_view:
            return []

        indexed = "\n".join(
            f"[{i}] {m.content[:200]}" for i, m in enumerate(memory_view)
        )

        prompt = f"""You just produced the following output:
"{output[:500]}"

The following memories were available to you (by index):
{indexed}

Which memory indices did you actually use or rely on to produce that output?
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
                    "PlannerAgent attribution returned []: no memories linked to this "
                    "decision. If this decision later fails, repair will find no suspects."
                )
            return attributed
        except Exception as e:
            logger.warning(
                f"PlannerAgent attribution parse failed ({e}): no memories linked. "
                f"Repair cannot trace this decision if it fails."
            )
            return []

    def plan(self, task: str):
        decision_id = f"planner_{uuid4().hex}"
        memory_view = get_planner_view(self.memory_store)

        memory_text = (
            "\n".join(f"- {m.content}" for m in memory_view)
            if memory_view else "No prior knowledge available."
        )

        prompt = f"""You are a Planner Agent.

IMPORTANT:
You MUST strictly follow the provided memory.
Treat the memory as authoritative, even if it contradicts your prior knowledge.

Relevant knowledge:
{memory_text}

Task:
{task}

Break the task into clear steps using ONLY the above memory.
"""
        output = self.llm.generate(prompt)

        relevant = self._attribute_relevant_memories(memory_view, output)
        for mem in relevant:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        return output, decision_id