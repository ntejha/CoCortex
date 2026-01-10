# Step 10 - Baseline vs CoCortex Experiments

The goal of today is to validate CoCortex in a realistic setting, not just a toy example. Instead of testing with an obviously wrong fact, we simulate knowledge drift across multiple real-world tasks, where :
- A memory is paritially misleading
- It gets reused across different tasks
- Failures accumulate gradually
- A robust system should detect, limit and repair the damage

This experiment demonstrates that CoCortex reduces error propogation and recovers faster, which directly proves the usefulness of the proposed architecture.

Output : 

```
(venv) ntejha@fedora:~/Major_Project/CoCortex$ python -m experiments.step10_demo

REALISTIC BASELINE vs COCORTEX EVALUATION
============================================================
System     | Failures | Tasks Contaminated | Recovery
------------------------------------------------------------
Baseline   | 5        | 5                  | False
CoCortex   | 3        | 3                  | 4
```