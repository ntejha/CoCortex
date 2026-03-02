"""
agents/planner.py

Attribution-Guided Causal Tracking:
After generating a plan, the LLM is asked a second time which memory indices
(0-based) it actually used. Only those memories are linked to the decision_id.
If the attribution call fails / returns nothing, no memories are linked.
"""
import json
import logging
from uuid import uuid4
from memory.views import get_planner_view

logger = logging.getLogger(__name__)

_ATTRIBUTION_PROMPT = """
You just generated a plan using the following numbered memory items:

{memory_list}

Which memory indices (0-based) directly influenced your plan?
Respond with a JSON array of integers only, e.g. [0, 2]. 
If none influenced the plan, respond with [].
"""


class PlannerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def plan(self, task: str):
        decision_id = f"planner_{uuid4().hex}"
        memory_view = get_planner_view(self.memory_store)

        memory_text = (
            "\n".join(f"{i}. {m.content}" for i, m in enumerate(memory_view))
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

        # Attribution: ask LLM which memories it actually used
        if memory_view:
            attr_prompt = _ATTRIBUTION_PROMPT.format(
                memory_list=memory_text
            )
            try:
                attr_raw = self.llm.generate(attr_prompt)
                indices = json.loads(attr_raw.strip())
                if isinstance(indices, list):
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(memory_view):
                            self.memory_store.link_memory_to_decision(
                                memory_view[idx].id, decision_id
                            )
                            logger.debug(
                                "Linked memory[%d] id=%s to decision=%s",
                                idx, memory_view[idx].id, decision_id,
                            )
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("planner attribution parse failed — no memories linked")

        return output, decision_id