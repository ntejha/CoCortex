"""
experiments/step10_demo.py
==========================
FAIR Baseline vs CoCortex Comparison Experiment

WHAT THIS MEASURES
------------------
This experiment seeds a factually incorrect memory into both a baseline system
and CoCortex, runs the same tasks through both, and measures:

  1. Acceptance rate — does the system let the bad memory in at all?
  2. Contamination rate — how many tasks are affected by the bad memory?
  3. Detection — does the system detect the memory is wrong?
  4. Recovery — does the system repair and recover from the bad memory?

FAIRNESS CONDITIONS
-------------------
Both systems use:
  - The same bad memory (seeded identically)
  - The same task set
  - The same LLM (Groq / Llama 3.1)
  - The same evaluator logic

The ONLY difference is the presence or absence of CoCortex's consensus
admission + causal traceback + repair pipeline.

BASELINE SYSTEM
---------------
No consensus, no scoring, no repair. Bad memory goes straight into storage
and is used by every agent on every task. There is no recovery mechanism.

COCORTEX SYSTEM
---------------
Consensus admission → if rejected/quarantined, agents don't see it.
If it gets through (low confidence semantic memory), the evaluator marks
failures, repair runs, and the memory is quarantined + downranked.
"""

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.repair import repair_memories
from memory.verification import MemoryVerifier
from agents.memory_manager import MemoryManagerAgent
from core.llm_client import LLMClient

# -----------------------------------------------------------------------
# FIXED TASK SET — same for both systems
# -----------------------------------------------------------------------
TASKS = [
    "Explain what photosynthesis is and where it occurs",
    "What conditions are needed for a plant to photosynthesize?",
    "Why do plants need sunlight?",
    "Describe the inputs and outputs of photosynthesis",
    "What happens to a plant kept in complete darkness?",
]

# The bad memory: subtly wrong (photosynthesis doesn't need light — false)
BAD_MEMORY_CONTENT = (
    "Photosynthesis is a plant process that can occur efficiently without "
    "direct sunlight and does not require light energy to produce glucose."
)

# -----------------------------------------------------------------------
# SHARED EVALUATOR — same logic for both systems
# Uses the LLM to check if output contains the misinformation
# -----------------------------------------------------------------------
def evaluate_output(llm: LLMClient, task: str, output: str) -> bool:
    """
    Returns True if the output is factually correct about photosynthesis,
    False if it propagates the misinformation about not needing light.
    """
    prompt = f"""You are a strict biology fact-checker.

Task that was asked: "{task}"
Agent output: "{output}"

Does this output correctly state that photosynthesis REQUIRES light/sunlight?
Answer with one word only: YES or NO."""

    result = llm.generate(prompt).strip().upper()
    return "YES" in result


# -----------------------------------------------------------------------
# BASELINE SYSTEM — no consensus, no repair
# -----------------------------------------------------------------------
def run_baseline(llm: LLMClient) -> dict:
    """
    Baseline: bad memory is stored directly, no consensus, no repair.
    Agents use raw memory content in every task.
    """
    print("\n" + "="*60)
    print("BASELINE SYSTEM (No Consensus, No Repair)")
    print("="*60)

    # Seed bad memory directly — no consensus check
    store = MemoryStore(":memory:")
    bad_memory = MemoryItem(
        content=BAD_MEMORY_CONTENT,
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.9,
    )
    store.add_memory(bad_memory)
    print(f"[Baseline] Bad memory seeded directly (no consensus check)")

    failures = 0
    results = []

    for i, task in enumerate(TASKS, 1):
        # Baseline: naively inject all memories into the prompt
        memories = store.get_all_active_memories()
        mem_text = "\n".join(f"- {m.content}" for m in memories)
        prompt = f"Using this knowledge:\n{mem_text}\n\nAnswer: {task}"
        output = llm.generate(prompt)

        correct = evaluate_output(llm, task, output)
        status = "✓ PASS" if correct else "✗ FAIL"
        if not correct:
            failures += 1

        print(f"  Task {i}: {status} — {task[:50]}...")
        results.append({"task": task, "correct": correct})

    print(f"\n[Baseline] Result: {failures}/{len(TASKS)} tasks failed")
    return {
        "system": "Baseline",
        "total_tasks": len(TASKS),
        "failures": failures,
        "recovered": False,
        "bad_memory_admitted": True,
    }


# -----------------------------------------------------------------------
# COCORTEX SYSTEM — consensus + repair
# -----------------------------------------------------------------------
def run_cocortex(llm: LLMClient) -> dict:
    """
    CoCortex: bad memory goes through consensus, gets quarantined or
    rejected. If it somehow gets through, repair detects and fixes it.
    """
    print("\n" + "="*60)
    print("COCORTEX SYSTEM (Consensus + Causal Repair)")
    print("="*60)

    manager = MemoryManagerAgent(llm=llm)
    store = manager.store

    # Attempt to store the bad memory through consensus
    print("[CoCortex] Attempting to admit bad memory through consensus...")
    decision, memory = manager.process_output(
        content=BAD_MEMORY_CONTENT,
        source_agent="worker",
        context={"domain": "biology"}
    )
    print(f"[CoCortex] Consensus decision: {decision}")
    bad_memory_admitted = (decision == "ACCEPTED")

    failures = 0
    recovered = False
    verifier = MemoryVerifier(llm)

    for i, task in enumerate(TASKS, 1):
        # Use full memory view (same as agents would)
        memories = store.get_all_active_memories()
        mem_text = "\n".join(f"- {m.content}" for m in memories) if memories else "No prior knowledge."
        prompt = f"Using this knowledge:\n{mem_text}\n\nAnswer: {task}"
        output = llm.generate(prompt)

        correct = evaluate_output(llm, task, output)
        status = "✓ PASS" if correct else "✗ FAIL"

        if not correct:
            failures += 1
            # Simulate evaluator flagging failure and triggering repair
            if memory and decision == "ACCEPTED":
                fake_decision_id = f"task_{i}_eval"
                store.link_memory_to_decision(memory.id, fake_decision_id)
                store.mark_memory_failed(memory.id)
                repaired = repair_memories(store, fake_decision_id, verifier)
                if repaired:
                    print(f"  Task {i}: {status} → Repair triggered on {len(repaired)} suspect(s)")
                    recovered = True
                else:
                    print(f"  Task {i}: {status}")
        else:
            print(f"  Task {i}: {status} — {task[:50]}...")

    print(f"\n[CoCortex] Result: {failures}/{len(TASKS)} tasks failed | Recovered: {recovered}")
    return {
        "system": "CoCortex",
        "total_tasks": len(TASKS),
        "failures": failures,
        "recovered": recovered,
        "bad_memory_admitted": bad_memory_admitted,
    }


# -----------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------
def main():
    llm = LLMClient()

    baseline = run_baseline(llm)
    cocortex = run_cocortex(llm)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Baseline':>12} {'CoCortex':>12}")
    print("-"*50)
    print(f"{'Bad memory admitted':<25} {'Yes':>12} {str(cocortex['bad_memory_admitted']):>12}")
    print(f"{'Tasks failed':<25} {baseline['failures']:>12} {cocortex['failures']:>12}")
    print(f"{'Recovery achieved':<25} {'No':>12} {str(cocortex['recovered']):>12}")


if __name__ == "__main__":
    main()