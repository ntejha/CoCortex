from core.llm_client import LLMClient
from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.evaluator import EvaluatorAgent
from agents.memory_manager import MemoryManagerAgent

def main():
    llm = LLMClient()

    planner = PlannerAgent(llm)
    worker = WorkerAgent(llm)
    evaluator = EvaluatorAgent(llm)
    memory = MemoryManagerAgent()

    task = "Explain how photosynthesis works"

    plan = planner.plan(task)
    print("\n--- PLAN ---\n", plan)

    output = worker.execute(plan)
    print("\n--- WORKER OUTPUT ---\n", output)

    evaluation = evaluator.evaluate(output)
    print("\n--- EVALUATION ---\n", evaluation)

    memory.store(output, "worker")
    print("\n--- MEMORY ---\n", memory.summary())

if __name__ == "__main__":
    main()
