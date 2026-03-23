# agent-supervisor

A supervisor system for managing AI agents. Monitors execution, detects failures, handles timeouts and automatically restarts agents with exponential backoff.

## How It Works
```
Supervisor
    ↓
Task Queue
    ↓
Agent Worker ──→ success ──→ Task DONE
    ↓
  failure
    ↓
Retry with backoff (up to max_restarts)
    ↓
Max restarts reached ──→ Task FAILED
```

## Features

- Task queue with async workers
- Per-agent failure tracking and restart logic
- Configurable timeout per task
- Automatic retry with delay on failure
- Circuit breaker — stops agent after max restarts
- Full task lifecycle — pending, running, done, failed, timeout
- Stats — queue size, tasks by status, total restarts

## Usage
```python
from supervisor import Supervisor, Task

supervisor = Supervisor(max_restarts=3, task_timeout_s=30.0)

async def my_agent(task: Task) -> str:
    result = await do_something(task.payload)
    return result

agent_id = supervisor.register("my-agent", my_agent)

task = supervisor.submit("process", {"data": "..."})

await supervisor.run_until_empty([agent_id])

print(task.status)  # done
print(task.result)  # output from agent
```

## Run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python agent.py
```

## Test
```bash
pytest tests/ -v
```

## Project Structure
```
agent-supervisor/
├── supervisor.py         # Supervisor, Task, AgentRecord
├── agent.py              # Example agents — summarize, sentiment, keywords
├── requirements.txt
├── README.md
└── tests/
    └── test_supervisor.py
```

## Author

**Szymon Wypler**
