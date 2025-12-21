class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, task, memory_summary=""):
        prompt = f"""
You are a Planner Agent.
Task: {task}

Relevant memory:
{memory_summary}

Break the task into clear ordered steps.
"""
        return self.llm.generate(prompt)
