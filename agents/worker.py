from memory.views import get_worker_view

class WorkerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def execute(self, plan: str):
        memory_view = get_worker_view(self.memory_store)

        memory_text = "\n".join(
            f"- {m.content}" for m in memory_view
        ) if memory_view else "No prior procedures available."

        prompt = f"""
You are a Worker Agent.

Available procedures and past experiences:
{memory_text}

Plan to execute:
{plan}

Execute the plan carefully.
"""
        return self.llm.generate(prompt)
