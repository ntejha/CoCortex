# experiments/step08_reliability_demo.py

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.scoring import compute_reliability


def print_state(title, memory):
    print(f"\n--- {title} ---")
    print(f"Memory ID       : {memory.id}")
    print(f"Lifecycle State : {memory.lifecycle_state}")
    print(f"Usage Count     : {memory.usage_count}")
    print(f"Failure Count   : {memory.failure_count}")
    print(f"Reliability     : {compute_reliability(memory)}")


def main():
    store = MemoryStore()

    # Ensure clean DB for demo
    store.clear_all_memories()
    print("[INFO] Memory store cleared")

    # 1️⃣ Create memory
    memory = MemoryItem(
        content="Photosynthesis occurs during the day.",
        source_agent="worker",
        confidence_score=0.9,
    )
    store.add_memory(memory)

    # Reload from DB to verify persistence
    memory = store.get_memory(memory.id)
    print_state("INITIAL STATE", memory)

    # 2️⃣ Simulate usage
    for _ in range(5):
        store.mark_memory_used(memory.id)

    memory = store.get_memory(memory.id)
    print_state("AFTER USAGE (TRUST BUILDING)", memory)

    # 3️⃣ Simulate failures
    for _ in range(3):
        store.mark_memory_failed(memory.id)

    memory = store.get_memory(memory.id)
    print_state("AFTER FAILURES (DEMOTION)", memory)

    # 4️⃣ Validate memory (simulates evaluator approval)
    store.validate_memory(memory.id)

    memory = store.get_memory(memory.id)
    print_state("AFTER VALIDATION", memory)

    # Cleanup
    store.clear_all_memories()
    print("\n[INFO] Memory store cleaned")


if __name__ == "__main__":
    main()
