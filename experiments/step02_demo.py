from memory.store import MemoryStore
from memory.schemas import MemoryItem


def run_test():
    store = MemoryStore()

    memory = MemoryItem(
        content="Worker successfully completed step A.",
        memory_type="episodic",
        source_agent="worker",
        confidence_score=0.6,
    )

    store.add_memory(memory)
    store.promote_memory(memory.id)
    store.update_confidence(memory.id, 0.9)
    store.link_memory_to_decision(memory.id, "decision_001")

    print("\n📌 Semantic Memories:")
    for mem in store.get_memory_by_type("semantic"):
        print(mem)

    # Cleanup
    store.clear_all_memories()
    print("\n🧹 Memory store cleaned")


if __name__ == "__main__":
    run_test()
