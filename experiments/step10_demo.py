# experiments/baseline_vs_cocortex_realistic.py

from memory.store import MemoryStore
from memory.schemas import MemoryItem

# -------------------------
# Task Set (Realistic)
# -------------------------
TASKS = [
    "Explain photosynthesis",
    "Answer a biology MCQ",
    "Summarize plant energy conversion",
    "Tutor a student on photosynthesis",
    "Generate biology study notes",
]

# Subtle but misleading memory (real-world style)
BAD_MEMORY = "Photosynthesis can efficiently occur without direct sunlight."


# -------------------------
# Evaluator (Simulated)
# -------------------------
def evaluator(task: str, output: str) -> bool:
    """
    Simulates partial correctness checks.
    Returns False if misleading memory is used.
    """
    return "without direct sunlight" not in output.lower()


# -------------------------
# Baseline System
# -------------------------
def run_baseline():
    failures = 0
    contaminated_tasks = set()

    memory = BAD_MEMORY  # naive global memory

    for task in TASKS:
        output = f"{task}: {memory}"

        if not evaluator(task, output):
            failures += 1
            contaminated_tasks.add(task)

    return {
        "system": "Baseline",
        "failures": failures,
        "contaminated_tasks": len(contaminated_tasks),
        "recovered": False,
    }


# -------------------------
# CoCortex System
# -------------------------
def run_cocortex():
    store = MemoryStore()
    store.clear_all_memories()

    failures = 0
    contaminated_tasks = set()
    recovered_at = None

    memory = MemoryItem(
        content=BAD_MEMORY,
        source_agent="worker",
        confidence_score=0.9,
    )
    store.add_memory(memory)

    for idx, task in enumerate(TASKS, start=1):
        output = f"{task}: {memory.content}"

        if evaluator(task, output):
            if recovered_at is None:
                recovered_at = idx
        else:
            failures += 1
            contaminated_tasks.add(task)

            # mark failure and update lifecycle
            store.mark_memory_failed(memory.id)
            memory = store.get_memory(memory.id)

            # simulate correction after deprecation
            if memory.lifecycle_state == "deprecated":
                memory.content = (
                    "Photosynthesis primarily requires light energy from the sun."
                )

    store.clear_all_memories()

    return {
        "system": "CoCortex",
        "failures": failures,
        "contaminated_tasks": len(contaminated_tasks),
        "recovered_at": recovered_at,
    }


# -------------------------
# Runner
# -------------------------
def main():
    baseline = run_baseline()
    cocortex = run_cocortex()

    print("\nREALISTIC BASELINE vs COCORTEX EVALUATION")
    print("=" * 60)
    print(
        f"{'System':<10} | {'Failures':<8} | "
        f"{'Tasks Contaminated':<18} | {'Recovery'}"
    )
    print("-" * 60)

    print(
        f"{baseline['system']:<10} | "
        f"{baseline['failures']:<8} | "
        f"{baseline['contaminated_tasks']:<18} | "
        f"{baseline['recovered']}"
    )

    print(
        f"{cocortex['system']:<10} | "
        f"{cocortex['failures']:<8} | "
        f"{cocortex['contaminated_tasks']:<18} | "
        f"{cocortex['recovered_at']}"
    )


if __name__ == "__main__":
    main()
