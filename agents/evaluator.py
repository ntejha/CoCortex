class EvaluatorAgent:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, output):
        prompt = f"""
You are an Evaluator Agent.
Check the correctness of the output below.

Output:
{output}

Respond strictly in this format:
PASS/FAIL - reason
"""
        return self.llm.generate(prompt)
