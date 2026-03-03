"""
agents/evaluator.py

Attribution-Guided Causal Tracking for the Evaluator.
"""
import json
import logging
from uuid import uuid4
from memory.views import get_evaluator_view, aget_evaluator_view
from agents.base import AsyncAgent

logger = logging.getLogger(__name__)

_ATTRIBUTION_PROMPT = """
You just evaluated an output using the following numbered memory items:

{memory_list}

Which memory indices (0-based) directly influenced your evaluation?
Respond with a JSON array of integers only, e.g. [0, 2].
If none influenced it, respond with [].
"""


class EvaluatorAgent(AsyncAgent):
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def evaluate(self, output: str):
        decision_id = f"evaluator_{uuid4().hex}"
        memory_view = get_evaluator_view(self.memory_store)

        memory_text = (
            "\n".join(f"{i}. {m.content}" for i, m in enumerate(memory_view))
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

        # Attribution: ask LLM which memories it used
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
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("evaluator attribution parse failed — no memories linked")

        return result, decision_id

    async def aevaluate(self, result_text: str) -> tuple[str, str]:
        """
        Async version of evaluate().
        Attributes memories used via a second self._allm() call.
        """
        decision_id = f"evaluator_{uuid4().hex[:8]}"
        memory_view = await aget_evaluator_view(self.memory_store, result_text, top_n=5)

        memory_text = "\n".join(
            f"[{i}] {m.content}" for i, m in enumerate(memory_view)
        )

        prompt = (
            f"Available Memory (High Confidence):\n{memory_text}\n\n"
            f"Result to evaluate:\n{result_text}\n\n"
            "Evaluate result. Explicitly state PASS or FAIL."
        )
        
        result = await self._allm(prompt)

        # Attribution: ask LLM which memories it used
        if memory_view:
            attr_prompt = result + "\n\n" + _ATTRIBUTION_PROMPT.format(memory_list=memory_text)
            try:
                attr_raw = await self._allm(attr_prompt)
                indices = json.loads(attr_raw.strip())
                if isinstance(indices, list):
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(memory_view):
                            self.memory_store.link_memory_to_decision(
                                memory_view[idx].id, decision_id
                            )
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("evaluator async attribution parse failed — no memories linked")

        return result, decision_id