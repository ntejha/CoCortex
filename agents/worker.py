"""
agents/worker.py
================
WorkerAgent with attribution-guided causal tracking.
"""

import json
import logging
from uuid import uuid4
from memory.views import get_worker_view

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def _attribute_relevant_memories(self, memory_view, output: str) -> list:
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
                    "WorkerAgent attribution returned []: no memories linked to this "
                    "decision. Repair cannot trace this decision if it fails."
                )
            return attributed
        except Exception as e:
            logger.warning(
                f"WorkerAgent attribution parse failed ({e}): no memories linked."
            )
            return []

    def execute(self, plan: str):
        decision_id = f"worker_{uuid4().hex}"
        memory_view = get_worker_view(self.memory_store)

        memory_text = (
            "\n".join(f"- {m.content}" for m in memory_view)
            if memory_view else "No prior procedures available."
        )

        prompt = f"""You are a Worker Agent.

Available procedures and past experiences:
{memory_text}

Plan:
{plan}

Execute the plan carefully.
"""
        output = self.llm.generate(prompt)

        relevant = self._attribute_relevant_memories(memory_view, output)
        for mem in relevant:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        return output, decision_id