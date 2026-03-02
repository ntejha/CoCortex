"""
agents/worker.py
================
WorkerAgent with attribution-guided causal tracking.
"""

import json
from uuid import uuid4
from memory.views import get_worker_view


class WorkerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def _attribute_relevant_memories(self, memory_view, output: str) -> list:
        """
        Ask the LLM which memories it actually relied on to produce its output.
        Only those memories get causally linked to the decision.

        This solves the correlation-vs-causation problem in causal tracking:
        we don't link every memory in the context window (which is correlation),
        we link only memories the model reports using (which is attribution).
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
            return [memory_view[i] for i in indices if 0 <= i < len(memory_view)]
        except Exception:
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

        # Attribution-guided causal linking
        relevant = self._attribute_relevant_memories(memory_view, output)
        for mem in relevant:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        return output, decision_id