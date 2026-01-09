# Step 09 - Memory Provenance and Explainability

Key Idea here is, we already have memory, decisions, failures and repair. We just have to connects them into a narrative.

- Provenance = history
- Explainability = readable story of that history

We will add a Provenance Engine that reads existing data and explains it. This is a read-only intelligence layer. This module will reside in `memory/provenance.py`. It will analyze already-stored metadata.
- `engine.explain_memory(memory_id)`
    - This function will answer "What is this memory, where did it come from, and how trustworthy is it?"
- `engine.trace_failure(task_id)`
    - This function will answer "This task failed - which memories caused it ?"


Documentation : 
- We have edited store.py (add the cloumns in the DB) and schemas.py(add the varibales)
- We have added a new module provenance.py

Outputs : 

```
(venv) ntejha@fedora:~/Major_Project/CoCortex$ python -m experiments.step09_demo
[INFO] Memory store cleared

MEMORY EXPLANATION
----------------------------------------
Memory ID        : 1a2d2bc7-3078-4f2f-a2b8-d25b8f844800
Content          : Photosynthesis occurs only at night.
Created By       : worker
Lifecycle State  : deprecated
Reliability      : 0.45
Usage Count      : 0
Failure Count    : 3

Influenced Decisions:
- None

Associated Tasks:
- task_photosynthesis_001

Repair History:
- 2026-01-09T04:34:57.783070 - Deprecated after repeated factual failures

FAILURE TRACE REPORT
----------------------------------------
Task ID: task_photosynthesis_001

Root Cause Candidate:
- Memory ID   : 1a2d2bc7-3078-4f2f-a2b8-d25b8f844800
- Content     : Photosynthesis occurs only at night.
- Created By  : worker
- Lifecycle   : deprecated
- Failures    : 3
- Reliability : 0.45

[INFO] Memory store cleaned
(venv) ntejha@fedora:~/Major_Project/CoCortex$ 
```