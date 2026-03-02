class CoCortexMemory:
    """
    LangChain-compatible memory adapter (duck-typed).
    """

    def __init__(self, engine, session_id: str):
        self.engine = engine
        self.session_id = session_id

    @property
    def memory_variables(self):
        return ["history"]

    def load_memory_variables(self, inputs):
        records = self.engine.load(self.session_id) or []

        lines = []
        for r in records:
            if not isinstance(r, dict):
                continue

            user = (
                r.get("input", {}).get("input")
                if isinstance(r.get("input"), dict)
                else r.get("input")
            )
            assistant = (
                r.get("output", {}).get("output")
                if isinstance(r.get("output"), dict)
                else r.get("output")
            )

            if user:
                lines.append(f"Human: {user}")
            if assistant:
                lines.append(f"Assistant: {assistant}")

        return {"history": "\n".join(lines)}

    def save_context(self, inputs, outputs):
        records = self.engine.load(self.session_id) or []

        records.append({
            "input": inputs,
            "output": outputs,
        })

        records = self.engine.repair_if_needed(records)
        self.engine.save(self.session_id, records)

    def clear(self):
        """
        Delete all memory records for this session.

        Previously called engine.save(session_id, []) which loops over an
        empty list and does nothing — existing SQLite rows stayed untouched.
        Now delegates to engine.delete_session() which runs an actual DELETE.
        """
        self.engine.delete_session(self.session_id)