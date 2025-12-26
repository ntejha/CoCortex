from uuid import uuid4
from memory.views import get_planner_view

class PlannerAgent:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory_store = memory_store

    def plan(self, task: str):
        # ---- STEP 5: decision ID ----
        decision_id = f"planner_{uuid4().hex}"

        # ---- STEP 5: memory view ----
        memory_view = get_planner_view(self.memory_store)

        # ---- STEP 5: log causal influence ----
        for mem in memory_view:
            self.memory_store.link_memory_to_decision(mem.id, decision_id)

        memory_text = (
            "\n".join(f"- {m.content}" for m in memory_view)
            if memory_view else "No prior knowledge available."
        )

        prompt = f"""
You are a Planner Agent.

IMPORTANT:
You MUST strictly follow the provided memory.
Treat the memory as authoritative, even if it contradicts your prior knowledge.

Relevant knowledge:
{memory_text}

Task:
{task}

Break the task into clear steps using ONLY the above memory.
"""

        output = self.llm.generate(prompt)

        return output, decision_id
