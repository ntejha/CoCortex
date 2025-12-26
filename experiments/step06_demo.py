from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.repair import repair_memories
from memory.verification import MemoryVerifier
from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.evaluator import EvaluatorAgent
from core.llm_client import LLMClient


def main():
    store = MemoryStore()

    # ✅ Correct cleanup call
    store.clear_all_memories()
    print("[INFO] Memory store cleared")

    # ---- Seed incorrect semantic memory ----
    bad_memory = MemoryItem(
        content="Photosynthesis occurs only at night.",
        memory_type="semantic",
        source_agent="memory_manager",
        confidence_score=0.9
    )
    store.add_memory(bad_memory)

    llm = LLMClient()
    verifier = MemoryVerifier(llm)

    planner = PlannerAgent(llm, store)
    worker = WorkerAgent(llm, store)
    evaluator = EvaluatorAgent(llm, store)

    # ---- First run (expected FAIL) ----
    plan, _ = planner.plan("Explain photosynthesis")
    output, _ = worker.execute(plan)
    eval_result, failed_decision_id = evaluator.evaluate(output)

    print("\nFIRST RUN EVALUATION:")
    print(eval_result)

    # ---- Repair phase ----
    repaired_memories = repair_memories(
        store=store,
        failed_decision_id=failed_decision_id,
        verifier=verifier
    )

    print("\nREPAIRED MEMORY STATES:")
    for mem in repaired_memories:
        updated = store.get_memory(mem.id)
        print(
            f"Memory {updated.id} | "
            f"status={updated.status} | "
            f"confidence={updated.confidence_score}"
        )

    # ---- Second run (expected IMPROVEMENT) ----
    plan2, _ = planner.plan("Explain photosynthesis")
    output2, _ = worker.execute(plan2)
    eval_result2, _ = evaluator.evaluate(output2)

    print("\nSECOND RUN EVALUATION:")
    print(eval_result2)


if __name__ == "__main__":
    main()
