"""
agents/worker.py

Attribution-Guided Causal Tracking:
After executing the plan, the LLM is asked which memory indices it used.
Only those memories are linked to the decision_id.
"""
import json
import logging
from uuid import uuid4
from memory.views import get_worker_view, aget_worker_view
from agents.base import AsyncAgent

logger = logging.getLogger(__name__)

_ATTRIBUTION_PROMPT = """
You just executed a plan using the following numbered memory items:

{memory_list}

Which memory indices (0-based) directly influenced your execution?
Respond with a JSON array of integers only, e.g. [0, 2]. 
If none influenced execution, respond with [].
"""


class WorkerAgent(AsyncAgent):
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def execute(self, plan: str):
        decision_id = f"worker_{uuid4().hex}"
        memory_view = get_worker_view(self.memory_store)

        memory_text = (
            "\n".join(f"{i}. {m.content}" for i, m in enumerate(memory_view))
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

        # Attribution: ask LLM which memories it actually used
        if memory_view:
            attr_prompt = _ATTRIBUTION_PROMPT.format(memory_list=memory_text)
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
                logger.debug("worker attribution parse failed — no memories linked")

        return output, decision_id

    async def aexecute(self, plan: str) -> tuple[str, str]:
        """
        Async version of execute().
        Attributes memories used via a second self._allm() call.
        """
        decision_id = f"worker_{uuid4().hex[:8]}"
        memory_view = await aget_worker_view(self.memory_store, plan, top_n=5)

        memory_text = "\n".join(
            f"[{i}] {m.content}" for i, m in enumerate(memory_view)
        )

        prompt = f"Available Memory:\n{memory_text}\n\nPlan to execute:\n{plan}\n\nExecute plan."
        output = await self._allm(prompt)

        if memory_view:
            attr_prompt = output + "\n\n" + _ATTRIBUTION_PROMPT.format(
                memory_list=memory_text
            )
            try:
                attr_raw = await self._allm(attr_prompt)
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
                logger.debug("worker async attribution parse failed — no memories linked")

        return output, decision_id