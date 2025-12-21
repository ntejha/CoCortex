class WorkerAgent:
    def __init__(self, llm):
        self.llm = llm

    def execute(self, step, memory=""):
        prompt = f"""
You are a Worker Agent.
Task step:
{step}

Memory:
{memory}

Execute this step and explain your reasoning.
"""
        return self.llm.generate(prompt)
