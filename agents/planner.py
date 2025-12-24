from memory.views import get_planner_view

class PlannerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def plan(self, task: str):
        memory_view = get_planner_view(self.memory_store)

        memory_text = "\n".join(
            f"- {m.content}" for m in memory_view
        ) if memory_view else "No prior knowledge available."

        prompt = f"""
You are a Planner Agent.

Relevant knowledge:
{memory_text}

Task:
{task}

Break the task into clear steps.
"""
        return self.llm.generate(prompt)
