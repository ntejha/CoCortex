# Step 06 - Causal Traceback & Memory Repair

## Deliverables

- Failure Signal from Evaluator
    - Evaluator clearly outputs PASS or FAIL
    - On FAIL, it provides:
        - Failure reason
        - Failed decision ID
- Causal Traceback Mechanism
    - We need a function takes a failed_decision_id and returns suspect_memory_ids.
    - Which scan memory store and gives the outputs 
- Memory Re-evaluation Signals
    - Each suspect memory must be evaluated using structured rules, not free-from LLM judgement.
- Determinstic Repair Policy
    - We must implement a rule-based mapping from evaluation signals to actions.
- Memory State update
- Post-Repair Behaviour Change : Same task is run again and check output
- Demo Script

## Tech Stack

- Same tech stack

## Documentation

This is for me to trace back : 