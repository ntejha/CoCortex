from engine.memory_engine import MemoryEngine
from memory.cocortex_memory import CoCortexMemory


def cocortex_langchain_memory(session_id: str):
    engine = MemoryEngine()
    return CoCortexMemory(engine=engine, session_id=session_id)
