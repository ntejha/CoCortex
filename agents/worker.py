from uuid import uuid4
from memory.views import get_worker_view

class WorkerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def execute(self, plan: str):
        # ---- STEP 5: decision ID ----
        decision_id = f"worker_{uuid4().hex}"

        # ---- STEP 5: memory view ----
        memory_view = get_worker_view(self.memory_store)

        # ---- STEP 5: log causal influence ----
        for mem in memory_view:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        memory_text = (
            "\n".join(f"- {m.content}" for m in memory_view)
            if memory_view else "No prior procedures available."
        )

        prompt = f"""
You are a Worker Agent.

Available procedures and past experiences:
{memory_text}

Plan:
{plan}

Execute the plan carefully.
"""
        output = self.llm.generate(prompt)

        return output, decision_id
