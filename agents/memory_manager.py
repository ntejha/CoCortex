class MemoryManagerAgent:
    def __init__(self):
        self.memory = []

    def store(self, content, source):
        self.memory.append({
            "source": source,
            "content": content
        })

    def summary(self):
        return "\n".join(m["content"] for m in self.memory)
