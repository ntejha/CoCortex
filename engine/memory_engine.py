from cocortex.memory.store import MemoryStore
import cocortex.memory.repair as repair_module


class MemoryEngine:
    """
    Stable, framework-agnostic memory facade.
    Includes a safe in-memory fallback cache when MemoryStore
    does not implement load/save semantics.
    """

    def __init__(self, db_path="cocortex_memory.db"):
        self.store = MemoryStore(db_path)
        self._cache = {}  # fallback memory

    def load(self, session_id):
        # Try real store first
        if hasattr(self.store, "get_session"):
            return self.store.get_session(session_id)

        if hasattr(self.store, "fetch"):
            return self.store.fetch(session_id)

        if hasattr(self.store, "read"):
            return self.store.read(session_id)

        # Fallback cache
        return self._cache.get(session_id, [])

    def save(self, session_id, records):
        # Try real store first
        if hasattr(self.store, "write"):
            return self.store.write(session_id, records)

        if hasattr(self.store, "save_session"):
            return self.store.save_session(session_id, records)

        if hasattr(self.store, "set_session"):
            return self.store.set_session(session_id, records)

        if hasattr(self.store, "add_record"):
            for r in records:
                self.store.add_record(session_id, r)
            return

        # Fallback cache
        self._cache[session_id] = records

    def retrieve(self, session_id, query):
        if hasattr(self.store, "retrieve"):
            return self.store.retrieve(session_id, query)
        return []

    def repair_if_needed(self, records):
        if hasattr(repair_module, "repair_memory"):
            return repair_module.repair_memory(records)

        if hasattr(repair_module, "run"):
            return repair_module.run(records)

        return records
