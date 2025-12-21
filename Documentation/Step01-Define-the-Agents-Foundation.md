# Step 01 : Define the Agents (Foundation)

## What we have to complete

We will start by defining 4 agents, each with a clear role.

### **Agents**
- Planner Agent
    - Breaks tasks into steps
    - Uses high-level memory summaries
- Worker Agent
    - Executes steps
    - Produces outputs, too calls, reasoning traces
- Evaluator Agent
    - Checks correctness of outputs
    - Detects failures or inconsistencies
- Memory Manager Agent
    - Controls memory writes, promotion, repair

### Tech Stack

For this,
- Python
- Groq API (primary)
- Gemini API (optional)

### Documentation

This notes is for me to trace back : 

- First we created virtual enviroment, the we created .env, .gitignore, requirements.txt and some code required files.

```
ntejha@fedora:~/Major_Project/CoCortex$ python -m venv venv
ntejha@fedora:~/Major_Project/CoCortex$ source venv/bin/activate
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls
Documentation  Papers  README.md  venv
(venv) ntejha@fedora:~/Major_Project/CoCortex$ mkdir agents
(venv) ntejha@fedora:~/Major_Project/CoCortex$ mkdir core
(venv) ntejha@fedora:~/Major_Project/CoCortex$ mkdir experiments
(venv) ntejha@fedora:~/Major_Project/CoCortex$ touch agents/__init__.py agents/planner.py agents/worker.py agents/evaluator.py agents/memory_manager.py
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls
agents  core  Documentation  experiments  Papers  README.md  venv
(venv) ntejha@fedora:~/Major_Project/CoCortex$ agents ls
bash: agents: command not found...
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls agents/
evaluator.py  __init__.py  memory_manager.py  planner.py  worker.py

(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls
agents  core  Documentation  experiments  Papers  README.md  venv
(venv) ntejha@fedora:~/Major_Project/CoCortex$ touch core/__init__.py core/llm_client.py
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls core/
__init__.py  llm_client.py
(venv) ntejha@fedora:~/Major_Project/CoCortex$ touch experiments/__init__.py experiments/step01_demo.py
(venv) ntejha@fedora:~/Major_Project/CoCortex$ experiments/ ls
bash: experiments/: Is a directory
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls experiments/
__init__.py  step01_demo.py
(venv) ntejha@fedora:~/Major_Project/CoCortex$ touch .env .gitignore
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls
agents  core  Documentation  experiments  Papers  README.md  venv
(venv) ntejha@fedora:~/Major_Project/CoCortex$ touch requirements.txt
(venv) ntejha@fedora:~/Major_Project/CoCortex$ ls
agents  core  Documentation  experiments  Papers  README.md  requirements.txt  venv
(venv) ntejha@fedora:~/Major_Project/CoCortex$ pip install -r requirements.txt 
```

- For running this step and check if it is working : 
    - `python -m environments.step01_demo`

