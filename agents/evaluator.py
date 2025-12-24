from memory.views import get_evaluator_view

class EvaluatorAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def evaluate(self, output: str):
        memory_view = get_evaluator_view(self.memory_store)

        memory_text = "\n".join(
            f"- {m.content}" for m in memory_view
        ) if memory_view else "No verified facts available."

        prompt = f"""
You are an Evaluator Agent.

Verified knowledge:
{memory_text}

Output to evaluate:
{output}

Evaluate correctness and consistency.
"""
        return self.llm.generate(prompt)
