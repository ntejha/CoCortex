from uuid import uuid4
from memory.views import get_evaluator_view

class EvaluatorAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def evaluate(self, output: str):
        # ---- STEP 5: decision ID ----
        decision_id = f"evaluator_{uuid4().hex}"

        # ---- STEP 5: memory view ----
        memory_view = get_evaluator_view(self.memory_store)

        # ---- STEP 5: log causal influence ----
        for mem in memory_view:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        memory_text = (
            "\n".join(f"- {m.content}" for m in memory_view)
            if memory_view else "No verified facts available."
        )

        prompt = f"""
You are an Evaluator Agent.

Verified knowledge:
{memory_text}

Output to evaluate:
{output}

Evaluate correctness and consistency.
"""
        result = self.llm.generate(prompt)

        return result, decision_id
