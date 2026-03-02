"""
agents/planner.py
=================
PlannerAgent with attribution-guided causal tracking.

Key design: Instead of linking ALL memories in the view to a decision
(correlation, not causation), we ask the LLM which memories it actually
used. Only those get linked. This makes repair far more accurate.
"""

import json
from uuid import uuid4
from memory.views import get_planner_view


class PlannerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def _attribute_relevant_memories(self, memory_view, output: str) -> list:
        """
        Ask the LLM which memories from the view were actually relevant
        to the output it produced. Returns a list of relevant memory objects.

        This solves the correlation-vs-causation problem: we only record
        causal links for memories the LLM reports using, not everything
        that was in the context window.
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
            # On parse failure, conservatively attribute no memories
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

        # Attribution-guided causal linking — only link memories actually used
        relevant = self._attribute_relevant_memories(memory_view, output)
        for mem in relevant:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        return output, decision_id