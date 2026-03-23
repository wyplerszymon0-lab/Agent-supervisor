import asyncio
import pytest
from supervisor import (
    Supervisor, Task, TaskStatus, AgentStatus,
    AgentTimeoutError, MaxRestartsError,
)


@pytest.fixture
def supervisor():
    return Supervisor(max_restarts=2, task_timeout_s=2.0, retry_delay_s=0.01)


async def success_agent(task: Task) -> str:
    return f"done:{task.name}"


async def failing_agent(task: Task) -> str:
    raise ValueError("intentional failure")


async def slow_agent(task: Task) -> str:
    await asyncio.sleep(10)
    return "too late"


async def flaky_agent(call_count: list) -> callable:
    async def fn(task: Task) -> str:
        call_count.append(1)
        if len(call_count) < 3:
            raise ValueError("not ready yet")
        return "recovered"
    return fn


def test_register_agent(supervisor):
    agent_id = supervisor.register("test", success_agent)
    assert agent_id is not None
    assert supervisor.get_agent(agent_id) is not None
    assert supervisor.get_agent(agent_id).name == "test"


def test_unregister_agent(supervisor):
    agent_id = supervisor.register("test", success_agent)
    result   = supervisor.unregister(agent_id)
    assert result is True
    assert supervisor.get_agent(agent_id) is None


def test_unregister_nonexistent_returns_false(supervisor):
    result = supervisor.unregister("nonexistent-id")
    assert result is False


def test_submit_task(supervisor):
    task = supervisor.submit("my-task", {"key": "value"})
    assert task.id is not None
    assert task.name == "my-task"
    assert task.payload == {"key": "value"}
    assert task.status == TaskStatus.PENDING


def test_get_task(supervisor):
    task  = supervisor.submit("test", {})
    found = supervisor.get_task(task.id)
    assert found is task


def test_get_nonexistent_task_returns_none(supervisor):
    assert supervisor.get_task("nonexistent") is None


@pytest.mark.asyncio
async def test_successful_task_execution(supervisor):
    agent_id = supervisor.register("success", success_agent)
    task     = supervisor.submit("test-task", {})

    await supervisor.run_until_empty([agent_id])

    assert task.status == TaskStatus.DONE
    assert task.result == "done:test-task"
    assert task.error  is None
    assert task.started_at  is not None
    assert task.finished_at is not None


@pytest.mark.asyncio
async def test_failed_task_retries(supervisor):
    agent_id = supervisor.register("failing", failing_agent)
    task     = supervisor.submit("fail-task", {})

    with pytest.raises(MaxRestartsError):
        await supervisor.run_until_empty([agent_id])

    assert task.status == TaskStatus.FAILED
    assert task.error  is not None
    assert task.attempts > 1


@pytest.mark.asyncio
async def test_task_timeout(supervisor):
    sup      = Supervisor(max_restarts=0, task_timeout_s=0.05, retry_delay_s=0.01)
    agent_id = sup.register("slow", slow_agent)
    task     = sup.submit("slow-task", {})

    with pytest.raises(MaxRestartsError):
        await sup.run_until_empty([agent_id])

    assert task.status == TaskStatus.TIMEOUT
    assert "timed out" in task.error


@pytest.mark.asyncio
async def test_multiple_agents_run_independently():
    sup = Supervisor(max_restarts=2, task_timeout_s=5.0, retry_delay_s=0.01)

    id1 = sup.register("agent1", success_agent)
    id2 = sup.register("agent2", success_agent)

    t1 = sup.submit("task1", {})
    t2 = sup.submit("task2", {})

    await sup.run_until_empty([id1, id2])

    assert t1.status == TaskStatus.DONE
    assert t2.status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_stats_after_execution(supervisor):
    agent_id = supervisor.register("success", success_agent)
    supervisor.submit("t1", {})
    supervisor.submit("t2", {})

    await supervisor.run_until_empty([agent_id])

    stats = supervisor.stats()
    assert stats["tasks_total"] == 2
    assert stats["tasks_done"]  == 2
    assert stats["tasks_failed"] == 0
    assert stats["queue_size"]  == 0


@pytest.mark.asyncio
async def test_agent_record_updated_after_success(supervisor):
    agent_id = supervisor.register("success", success_agent)
    supervisor.submit("t1", {})

    await supervisor.run_until_empty([agent_id])

    record = supervisor.get_agent(agent_id)
    assert record.tasks_done   == 1
    assert record.tasks_failed == 0
    assert record.restarts     == 0


@pytest.mark.asyncio
async def test_task_duration_calculated(supervisor):
    agent_id = supervisor.register("success", success_agent)
    task     = supervisor.submit("timed", {})

    await supervisor.run_until_empty([agent_id])

    assert task.duration() is not None
    assert task.duration() >= 0


def test_stats_empty_supervisor(supervisor):
    stats = supervisor.stats()
    assert stats["agents"]       == 0
    assert stats["tasks_total"]  == 0
    assert stats["queue_size"]   == 0
```

---

**`requirements.txt`**
```
openai==1.30.0
pytest==8.2.0
pytest-asyncio==0.23.0
