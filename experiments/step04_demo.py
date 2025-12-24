from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.views import (
    get_planner_view,
    get_worker_view,
    get_evaluator_view
)

def main():
    store = MemoryStore()

    # ---- CLEAN START ----
    store.clear_all_memories()
    print("[INFO] Memory store cleared\n")

    # ---- SEED MEMORY ----
    episodic = MemoryItem(
        content="Worker executed steps to explain photosynthesis.",
        memory_type="episodic",
        source_agent="worker",
        confidence_score=0.7
    )

    semantic = MemoryItem(
        content="Photosynthesis converts CO2 and H2O into glucose using light energy.",
        memory_type="semantic",
        source_agent="evaluator",
        confidence_score=0.95
    )

    store.add_memory(episodic)
    store.add_memory(semantic)

    # ---- SHOW VIEWS ----
    print("\n--- PLANNER VIEW ---")
    for m in get_planner_view(store):
        print("-", m.content)

    print("\n--- WORKER VIEW ---")
    for m in get_worker_view(store):
        print("-", m.content)

    print("\n--- EVALUATOR VIEW ---")
    for m in get_evaluator_view(store):
        print("-", m.content)

    # ---- OPTIONAL CLEANUP ----
    store.clear_all_memories()
    print("\n[INFO] Memory store cleaned")

if __name__ == "__main__":
    main()
