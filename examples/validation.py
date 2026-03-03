import time
import random
import csv
import numpy as np
import requests
import torch
import faiss
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

# ============================================
# STRICT GPU CHECK
# ============================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available.")

DEVICE = "cuda"

# ============================================
# CONFIG
# ============================================

OLLAMA_MODEL = "llama3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

MAX_TURNS = 100
TEMPERATURE = 0.2
TOP_K = 5
WINDOW_K = 8
GEN_TOKENS = 800
CONTAMINATION_TURN = 40
REPAIR_TURN = 70

random.seed(42)

# ============================================
# OLLAMA CLIENT
# ============================================

class OllamaClient:
    def generate(self, prompt: str):
        start = time.time()

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": GEN_TOKENS,
                },
            },
        )

        latency = time.time() - start
        data = response.json()

        return {
            "text": data["response"],
            "latency": latency,
            "tokens": len(prompt.split()) + len(data["response"].split()),
        }

# ============================================
# VECTOR STORE
# ============================================

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=DEVICE,
        )

        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.texts = []

    def add(self, text):
        emb = self.model.encode([text], convert_to_numpy=True).astype("float32")
        self.index.add(emb)
        self.texts.append(text)

    def search(self, query, k):
        if not self.texts:
            return []
        emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(emb, min(k, len(self.texts)))
        return [self.texts[i] for i in I[0]]

    def clear(self):
        self.index.reset()
        self.texts = []

# ============================================
# SYSTEMS
# ============================================

class WindowOnly:
    def __init__(self, llm):
        self.llm = llm
        self.window = []

    def run(self, task):
        context = "\n".join(self.window[-WINDOW_K:])
        prompt = f"Context:\n{context}\n\nTask:\n{task}"
        result = self.llm.generate(prompt)
        self.window.append(result["text"])
        return result

class RetrievalSystem:
    def __init__(self, llm):
        self.llm = llm
        self.vector = VectorStore()

    def run(self, task):
        retrieved = self.vector.search(task, TOP_K)
        context = "\n".join(retrieved)
        prompt = f"Relevant Memory:\n{context}\n\nTask:\n{task}"
        result = self.llm.generate(prompt)
        self.vector.add(result["text"])
        return result

# ============================================
# METRICS
# ============================================

@dataclass
class Metrics:
    total_tokens: int = 0
    total_latency: float = 0
    turns: int = 0

    def update(self, result):
        self.total_tokens += result["tokens"]
        self.total_latency += result["latency"]
        self.turns += 1

    def summary(self):
        return {
            "avg_tokens": round(self.total_tokens / self.turns, 2),
            "avg_latency": round(self.total_latency / self.turns, 3),
        }

# ============================================
# VALIDATION RUN
# ============================================

def run():
    llm = OllamaClient()
    window = WindowOnly(llm)
    retrieval = RetrievalSystem(llm)

    window_metrics = Metrics()
    retrieval_metrics = Metrics()

    window_growth = []
    retrieval_growth = []

    contamination_detected = 0

    tasks = [
        "Explain neural networks training.",
        "Explain blockchain consensus.",
        "Explain quantum computing principles.",
        "Explain database indexing.",
    ] * (MAX_TURNS // 4)

    for turn in range(MAX_TURNS):
        task = tasks[turn]

        # Inject contamination
        if turn == CONTAMINATION_TURN:
            print("\n>>> Injecting bad memory <<<\n")
            retrieval.vector.add("Neural networks do NOT use backpropagation.")

        # Simulated repair
        if turn == REPAIR_TURN:
            print("\n>>> Repairing memory store <<<\n")
            retrieval.vector.clear()

        w = window.run(task)
        r = retrieval.run(task)

        window_metrics.update(w)
        retrieval_metrics.update(r)

        window_growth.append(w["tokens"])
        retrieval_growth.append(r["tokens"])

        if "do NOT use backpropagation" in r["text"]:
            contamination_detected += 1

        print(f"Turn {turn+1}/{MAX_TURNS}")

    # ============================================
    # RESULTS
    # ============================================

    print("\n=== FINAL RESULTS ===\n")
    print("Window:", window_metrics.summary())
    print("Retrieval:", retrieval_metrics.summary())
    print("Contamination Count:", contamination_detected)

    # Save CSV
    with open("full_validation_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Avg Tokens", "Avg Latency"])
        writer.writerow(["Window", *window_metrics.summary().values()])
        writer.writerow(["Retrieval", *retrieval_metrics.summary().values()])
        writer.writerow(["Contamination Count", contamination_detected])

    # Plot growth
    plt.plot(window_growth, label="Window")
    plt.plot(retrieval_growth, label="Retrieval")
    plt.legend()
    plt.xlabel("Turn")
    plt.ylabel("Tokens")
    plt.title("Context Growth Comparison")
    plt.savefig("context_growth_validation.png")

    print("\nSaved full_validation_results.csv and context_growth_validation.png")

if __name__ == "__main__":
    run()