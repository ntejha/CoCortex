# experiments/step09_provenance_demo.py

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.provenance import ProvenanceEngine


def main():
    store = MemoryStore()
    engine = ProvenanceEngine(store)

    # Clean DB
    store.clear_all_memories()
    print("[INFO] Memory store cleared")

    # Create memory
    memory = MemoryItem(
        content="Photosynthesis occurs only at night.",
        source_agent="worker",
        confidence_score=0.9,
    )
    store.add_memory(memory)

    task_id = "task_photosynthesis_001"

    # Link memory to task
    store.link_memory_to_task(memory.id, task_id)

    # Simulate failures
    for _ in range(3):
        store.mark_memory_failed(memory.id)

    # Log repair event
    store.log_repair_event(
        memory.id,
        "Deprecated after repeated factual failures",
    )

    # Reload from DB
    memory = store.get_memory(memory.id)

    # Explain memory
    engine.explain_memory(memory.id)

    # Trace failure
    engine.trace_failure(task_id)

    # Cleanup
    store.clear_all_memories()
    print("\n[INFO] Memory store cleaned")


if __name__ == "__main__":
    main()
