import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from cocortex.engine.memory_engine import MemoryEngine


def main():
    print("\n--- STEP 11 (DOWNGRADED): CoCortex + MemoryEngine + Repair ---\n")

    # Load env
    load_dotenv()
    assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not found in .env"

    session_id = "step11-demo-session"

    # Initialize engine directly (NO LangChain memory)
    engine = MemoryEngine()

    # Groq LLM (only used for generation)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # -------------------------------
    # TURN 1 — WRITE MEMORY
    # -------------------------------
    print("[1] Writing memory")

    user_input_1 = "My favorite programming language is Python."

    response_1 = llm.invoke(
        f"User says: {user_input_1}\nAcknowledge briefly."
    )

    records = engine.load(session_id) or []

    records.append({
        "input": user_input_1,
        "output": response_1.content
    })

    # 🔧 MEMORY REPAIR HOOK
    records = engine.repair_if_needed(records)

    engine.save(session_id, records)

    print("Saved records:")
    print(records)

    # -------------------------------
    # TURN 2 — READ MEMORY
    # -------------------------------
    print("\n[2] Reading memory")

    repaired_records = engine.load(session_id)

    print("Loaded records:")
    print(repaired_records)

    # Build prompt from memory explicitly
    memory_text = "\n".join(
        f"Human: {r['input']}\nAssistant: {r['output']}"
        for r in repaired_records
    )

    final_prompt = f"""
Conversation so far:
{memory_text}

Question:
What is my favorite programming language?
"""

    response_2 = llm.invoke(final_prompt)

    print("\n--- FINAL ANSWER ---")
    print(response_2.content)

    print("\n--- STEP 11 COMPLETE ---\n")


if __name__ == "__main__":
    main()
