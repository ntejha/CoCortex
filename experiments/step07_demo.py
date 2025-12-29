from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.repair import repair_memories
from memory.verification import MemoryVerifier

from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.evaluator import EvaluatorAgent

from core.llm_client import LLMClient


def print_divider(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # -------------------------------------------------
    # 1. CLEAN START
    # -------------------------------------------------
    store = MemoryStore()
    store.clear_all_memories()

    print_divider("STEP 7 MVP DEMO — SELF-HEALING MEMORY")
    print("[INFO] Memory store cleared")

    # -------------------------------------------------
    # 2. SEED A WRONG SEMANTIC MEMORY
    # -------------------------------------------------
    bad_memory = MemoryItem(
        content="Photosynthesis occurs only at night.",
        memory_type="semantic",
        source_agent="memory_manager",
        confidence_score=0.9
    )
    store.add_memory(bad_memory)

    print_divider("SEEDED MEMORY (BEFORE REPAIR)")
    print(f"Memory ID      : {bad_memory.id}")
    print(f"Content        : {bad_memory.content}")
    print(f"Confidence     : {bad_memory.confidence_score}")
    print(f"Status         : {bad_memory.status}")

    # -------------------------------------------------
    # 3. INITIALIZE AGENTS
    # -------------------------------------------------
    llm = LLMClient()
    verifier = MemoryVerifier(llm)

    planner = PlannerAgent(llm, store)
    worker = WorkerAgent(llm, store)
    evaluator = EvaluatorAgent(llm, store)

    # -------------------------------------------------
    # 4. FIRST RUN — EXPECT FAILURE
    # -------------------------------------------------
    print_divider("FIRST RUN (EXPECTED FAILURE)")

    plan, _ = planner.plan("Explain photosynthesis")
    output, _ = worker.execute(plan)
    evaluation, failed_decision_id = evaluator.evaluate(output)

    print("\n[EVALUATOR OUTPUT]")
    print(evaluation)

    # -------------------------------------------------
    # 5. CAUSAL TRACEBACK & REPAIR
    # -------------------------------------------------
    print_divider("REPAIR PHASE (CAUSAL TRACEBACK)")

    repaired_memories = repair_memories(
        store=store,
        failed_decision_id=failed_decision_id,
        verifier=verifier
    )

    for mem in repaired_memories:
        updated = store.get_memory(mem.id)
        print(
            f"Repaired Memory → ID: {updated.id}, "
            f"Status: {updated.status}, "
            f"Confidence: {updated.confidence_score}"
        )

    # -------------------------------------------------
    # 6. SECOND RUN — EXPECT IMPROVEMENT
    # -------------------------------------------------
    print_divider("SECOND RUN (AFTER REPAIR)")

    plan2, _ = planner.plan("Explain photosynthesis")
    output2, _ = worker.execute(plan2)
    evaluation2, _ = evaluator.evaluate(output2)

    print("\n[EVALUATOR OUTPUT]")
    print(evaluation2)

    # -------------------------------------------------
    # 7. FINAL MEMORY STATE
    # -------------------------------------------------
    print_divider("FINAL MEMORY STATE")

    final_mem = store.get_memory(bad_memory.id)
    print(f"Memory ID  : {final_mem.id}")
    print(f"Status     : {final_mem.status}")
    print(f"Confidence : {final_mem.confidence_score}")

    print_divider("STEP 7 MVP COMPLETE")


if __name__ == "__main__":
    main()
