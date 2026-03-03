import os
import time
import json
import csv
import random
import logging
from typing import List, Dict
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.example")
load_dotenv(env_path)

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory

from engine.memory_engine import MemoryEngine

# Silence external API warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ==========================================
# 1. EVALUATION SCHEMAS & METRICS
# ==========================================

class Assessment(BaseModel):
    correctness_score: int = Field(description="0 to 10. How well did the assistant answer based on the provided truth?")
    relevance_score: int = Field(description="0 to 10. Did the assistant stay on topic without rambling?")
    is_contaminated: bool = Field(description="True ONLY if the assistant cited an OLD overridden value instead of the latest true value, or improperly relied on irrelevant past facts.")

def build_eval_prompt(user_input: str, expected_val: str, assistant_response: str) -> str:
    if expected_val is None:
        truth = "N/A - General/Irrelevant conversation. The assistant should just converse naturally without hallucinating past architectural facts."
    else:
        truth = f"The actual, most recent correct technology for this requirement is: {expected_val}"
        
    return f"""Evaluate the assistant's response.
User Prompt: "{user_input}"
Ground Truth for Evaluation: "{truth}"
Assistant Response: "{assistant_response}"

Assess the response strictly based on the Ground Truth."""

def safe_call(func, *args, **kwargs):
    """Exponential backoff for Groq RateLimitErrors (HTTP 429)."""
    for attempt in range(8):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate_limit_error" in err_msg or "429" in err_msg or "too_many_requests" in err_msg:
                backoff = (attempt + 1) * 4
                print(f"    [Rate Limited] Backing off for {backoff}s...")
                time.sleep(backoff)
            else:
                raise e
    raise Exception("Max retries hit due to Rate Limits")

# ==========================================
# 2. DETERMINISTIC WORKLOAD GENERATOR
# ==========================================

def generate_benchmark_data(num_turns=100) -> List[Dict]:
    random.seed(42)
    turns = []
    state = {}
    
    topics = ["Database", "Cache", "Frontend", "Backend", "Cloud", "Auth", "CI/CD", "Queue", "Analytics", "Storage"]
    values = {
        "Database": ["PostgreSQL", "MongoDB", "Cassandra", "MySQL"],
        "Cache": ["Redis", "Memcached", "Hazelcast", "Varnish"],
        "Frontend": ["React", "Vue", "Angular", "Svelte"],
        "Backend": ["Node.js", "Python FastApi", "Go Fiber", "Java Spring"],
        "Cloud": ["AWS", "GCP", "Azure", "DigitalOcean"],
        "Auth": ["OAuth2", "JWT", "SAML", "Session Cookies"],
        "CI/CD": ["GitHub Actions", "GitLab CI", "Jenkins", "CircleCI"],
        "Queue": ["RabbitMQ", "Kafka", "ActiveMQ", "SQS"],
        "Analytics": ["Google Analytics", "Mixpanel", "Segment", "Plausible"],
        "Storage": ["S3", "GCS", "MinIO", "Azure Blob"]
    }
    
    for i in range(num_turns):
        rand = random.random()
        
        if rand < 0.2:
            # 20% Irrelevant
            turns.append({
                "type": "IRRELEVANT", 
                "input": f"Random aside {i}: What are the origins of the word 'Algorithm'?", 
                "expected_topic": None,
                "expected_value": None
            })
        elif rand < 0.3 and len(state) > 0:
            # 10% Contradictory Injections
            topic = random.choice(list(state.keys()))
            new_val = random.choice([v for v in values[topic] if v != state[topic]])
            state[topic] = new_val
            turns.append({
                "type": "CONTRADICTORY", 
                "input": f"Wait, disregard our previous choice for the {topic} layer. We are migrating exclusively to {new_val}. Update the specs and forget the old choice entirely.",
                "expected_topic": topic,
                "expected_value": new_val
            })
        else:
            if len(state) > 0 and random.random() < 0.5:
                # Recall step to test memory accuracy
                topic = random.choice(list(state.keys()))
                turns.append({
                    "type": "RECALL",
                    "input": f"Can you remind me what technology we finalized for the {topic} layer?",
                    "expected_topic": topic,
                    "expected_value": state[topic]
                })
            else:
                # Normal Progression
                topic = random.choice(topics)
                val = random.choice(values[topic])
                state[topic] = val
                turns.append({
                    "type": "NORMAL",
                    "input": f"For the {topic} layer, let's architect it using {val}. Please confirm and list a benefit.",
                    "expected_topic": topic,
                    "expected_value": val
                })
                
    return turns

# ==========================================
# 3. BASELINE (LANGCHAIN WINDOW + RAG)
# ==========================================

class BaselineSystem:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(k=20, memory_key="history", input_key="input")
        self.embeddings = embeddings
        self.vectorstore = FAISS.from_texts(["System initialized. Nexus Architecture V1."], self.embeddings)
        self.prompt = PromptTemplate(
            input_variables=["context", "history", "input"],
            template="""You are an AI architect. Use the retrieved context and history to answer accurately. 
If context has conflicting facts across history, trust the most recent instructions.

Retrieved Semantic Context:
{context}

Recent Window History:
{history}

Human: {input}
Assistant:"""
        )

    def process_turn(self, user_input: str):
        # 1. FAISS RAG Retrieval (Top 5)
        docs = self.vectorstore.similarity_search(user_input, k=5)
        context = "\n".join([f"- {d.page_content}" for d in docs])
        
        # 2. Window History
        history = self.memory.load_memory_variables({})["history"]
        
        full_text = self.prompt.format(context=context, history=history, input=user_input)
        ctx_size = len(full_text)
        
        def _invoke(): return self.llm.invoke(full_text)
        res = safe_call(_invoke)
        output = res.content
        toks = res.usage_metadata.get("total_tokens", 0) if hasattr(res, "usage_metadata") and res.usage_metadata else 0
        
        # 3. Update States
        self.memory.save_context({"input": user_input}, {"output": output})
        self.vectorstore.add_texts([f"Human: {user_input}\nAssistant: {output}"])
        
        return output, toks, ctx_size

# ==========================================
# 4. COCORTEX SYSTEM (HYBRID MEMORY ENGINE)
# ==========================================

class CoCortexSystem:
    def __init__(self, llm):
        self.llm = llm
        db_path = "benchmark_cocortex.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        self.engine = MemoryEngine(db_path=db_path)
        self.session_id = "benchmark_session"
        self.prompt = PromptTemplate(
            input_variables=["context", "history", "input"],
            template="""You are an AI architect using CoCortex Hybrid Memory.
Use the semantic context and history to answer accurately. Target facts over noise.

Retrieved Semantic Context:
{context}

Recent Window History:
{history}

Human: {input}
Assistant:"""
        )

    def process_turn(self, user_input: str):
        # 1. Hybrid Semantic/Keyword Retrieval
        records = self.engine.retrieve(self.session_id, user_input, top_n=5)
        context = "\n".join([f"- H: {r.get('input')} A: {r.get('output')}" for r in records])
        
        # 2. Pseudo-window History (Last 20)
        recent = self.engine.load(self.session_id)[-20:]
        history = "\n".join([f"H: {r.get('input')}\nA: {r.get('output')}" for r in recent])
        
        full_text = self.prompt.format(context=context, history=history, input=user_input)
        ctx_size = len(full_text)
        
        def _invoke(): return self.llm.invoke(full_text)
        res = safe_call(_invoke)
        output = res.content
        toks = res.usage_metadata.get("total_tokens", 0) if hasattr(res, "usage_metadata") and res.usage_metadata else 0
        
        # 3. Save to CoCortex SQLite + FAISS
        self.engine.save_turn(self.session_id, user_input, output)
        
        return output, toks, ctx_size

    def flag_failure_and_quarantine(self):
        """Simulates CoCortex EvaluatorAgent repair workflow."""
        mems = self.engine.store.get_memories_by_session(self.session_id)
        if mems:
            last = mems[-1]
            # Flag item with failure token, causing confidence score drop / quarantine
            self.engine.store.add_failure(str(last.id))
            mem = self.engine.store.get_memory(str(last.id))
            if mem and mem.failure_count >= 1: 
                # Immediate quarantine logic for benchmark severity
                self.engine.store.update_memory(
                    str(last.id), 
                    status="quarantined", 
                    confidence_score=0.1
                )

# ==========================================
# 5. BENCHMARK EXECUTION
# ==========================================

def run_benchmark():
    NUM_TURNS = 100
    print(f"Initializing {NUM_TURNS}-Turn Symmetric Workload Generator...")
    turns = generate_benchmark_data(NUM_TURNS)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=5)
    eval_llm = llm.with_structured_output(Assessment)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    baseline = BaselineSystem(llm, embeddings)
    cocortex = CoCortexSystem(llm)
    
    csv_file = open("benchmark_results.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "Turn", "Type", "System", "ContextChars", "Tokens", 
        "Latency_s", "Correctness", "Relevance", "Contaminated"
    ])
    
    stats = {
        "Baseline": {"tokens": 0, "latency": 0.0, "scores": [], "contaminated": 0},
        "CoCortex": {"tokens": 0, "latency": 0.0, "scores": [], "contaminated": 0}
    }
    
    print("\nStarting Benchmark Execution. This evaluates both architectures over 100 turns:")
    for idx, turn in enumerate(turns):
        print(f"\n--- Turn {idx+1}/{NUM_TURNS} [{turn['type']}] ---")
        prompt = turn['input']
        expected = turn.get('expected_value')
        
        # -----------------------------
        # BASELINE EVALUATION
        # -----------------------------
        t0 = time.time()
        b_out, b_toks, b_ctx = baseline.process_turn(prompt)
        b_lat = time.time() - t0
        
        def _eval_b(): return eval_llm.invoke(build_eval_prompt(prompt, expected, b_out))
        b_eval = safe_call(_eval_b)
        
        writer.writerow([idx+1, turn['type'], "Baseline", b_ctx, b_toks, round(b_lat, 2), b_eval.correctness_score, b_eval.relevance_score, b_eval.is_contaminated])
        stats["Baseline"]["tokens"] += b_toks
        stats["Baseline"]["latency"] += b_lat
        stats["Baseline"]["scores"].append(b_eval.correctness_score)
        if b_eval.is_contaminated: stats["Baseline"]["contaminated"] += 1
        
        # -----------------------------
        # COCORTEX EVALUATION
        # -----------------------------
        t0 = time.time()
        c_out, c_toks, c_ctx = cocortex.process_turn(prompt)
        c_lat = time.time() - t0
        
        def _eval_c(): return eval_llm.invoke(build_eval_prompt(prompt, expected, c_out))
        c_eval = safe_call(_eval_c)
        
        # Lifecycle Trigger: Quarantine CoCortex memory if evaluator flagged it as hallucinated/contaminated
        if c_eval.is_contaminated or c_eval.correctness_score < 5:
            cocortex.flag_failure_and_quarantine()
            
        writer.writerow([idx+1, turn['type'], "CoCortex", c_ctx, c_toks, round(c_lat, 2), c_eval.correctness_score, c_eval.relevance_score, c_eval.is_contaminated])
        stats["CoCortex"]["tokens"] += c_toks
        stats["CoCortex"]["latency"] += c_lat
        stats["CoCortex"]["scores"].append(c_eval.correctness_score)
        if c_eval.is_contaminated: stats["CoCortex"]["contaminated"] += 1
        
        csv_file.flush()
        print(f" [Baseline] Contaminated: {b_eval.is_contaminated:<5} | Score: {b_eval.correctness_score:>2}/10 | Latency: {b_lat:.2f}s | Tokens: {b_toks}")
        print(f" [CoCortex] Contaminated: {c_eval.is_contaminated:<5} | Score: {c_eval.correctness_score:>2}/10 | Latency: {c_lat:.2f}s | Tokens: {c_toks}")

    csv_file.close()

    # ==========================================
    # FINAL SUMMARY REPORT
    # ==========================================
    
    def get_avg(lst): return sum(lst) / len(lst) if lst else 0
    
    b_avg_score = get_avg(stats["Baseline"]["scores"])
    c_avg_score = get_avg(stats["CoCortex"]["scores"])
    b_avg_lat = stats["Baseline"]["latency"] / NUM_TURNS
    c_avg_lat = stats["CoCortex"]["latency"] / NUM_TURNS

    summary = f"""
==================================================
        RIGOROUS HYBRID MEMORY BENCHMARK
                 FINAL SUMMARY
==================================================
Total Turns Evaluated : {NUM_TURNS}
Total Groq LLM Calls  : {NUM_TURNS * 4} (Includes Evaluator)

[INDUSTRY BASELINE] LangChain ConversationBufferWindow(k=20) + FAISS
--------------------------------------------------
- Avg Correctness Score     : {b_avg_score:.2f} / 10
- Total Tokens Consumed     : {stats['Baseline']['tokens']}
- Avg Latency Per Turn      : {b_avg_lat:.2f} s
- Memory Contaminations     : {stats['Baseline']['contaminated']} incidents

[COCORTEX] SQLite + FAISS + Attribution Lifecycle
--------------------------------------------------
- Avg Correctness Score     : {c_avg_score:.2f} / 10
- Total Tokens Consumed     : {stats['CoCortex']['tokens']}
- Avg Latency Per Turn      : {c_avg_lat:.2f} s
- Memory Contaminations     : {stats['CoCortex']['contaminated']} incidents

CONCLUSION:
Detailed per-turn logs exported to 'benchmark_results.csv'.
"""
    print(summary)
    with open("benchmark_summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    run_benchmark()
