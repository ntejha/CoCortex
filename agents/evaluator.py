"""
agents/evaluator.py

Attribution-Guided Causal Tracking for the Evaluator.
"""
import json
import logging
from uuid import uuid4
from memory.views import get_evaluator_view

logger = logging.getLogger(__name__)

_ATTRIBUTION_PROMPT = """
You just evaluated an output using the following numbered memory items:

{memory_list}

Which memory indices (0-based) directly influenced your evaluation?
Respond with a JSON array of integers only, e.g. [0, 2].
If none influenced it, respond with [].
"""


class EvaluatorAgent:
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