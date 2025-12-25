from uuid import uuid4

def generate_decision_id(agent_name: str) -> str:
    return f"{agent_name}_{uuid4().hex}"
