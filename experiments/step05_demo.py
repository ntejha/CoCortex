from memory.store import MemoryStore
from memory.schemas import MemoryItem
from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.evaluator import EvaluatorAgent
from core.llm_client import LLMClient

def main():
    store = MemoryStore()
    store.clear_all_memories()

    # ---- Seed semantic memory ----
    semantic = MemoryItem(
        content="Photosynthesis converts CO2 and H2O into glucose using light energy.",
        memory_type="semantic",
        source_agent="evaluator",
        confidence_score=0.95
    )
    store.add_memory(semantic)

    llm = LLMClient()

    planner = PlannerAgent(llm, store)
    worker = WorkerAgent(llm, store)
    evaluator = EvaluatorAgent(llm, store)

    plan, p_id = planner.plan("Explain photosynthesis")
    output, w_id = worker.execute(plan)
    eval_result, e_id = evaluator.evaluate(output)

    print("\nDECISIONS:")
    print(p_id)
    print(w_id)
    print(e_id)

    print("\nMEMORY CAUSAL LINKS:")
    mem = store.get_memory(semantic.id)
    print(f"Memory {mem.id} influenced decisions:")
    for d in mem.influenced_decisions:
        print("-", d)

if __name__ == "__main__":
    main()
