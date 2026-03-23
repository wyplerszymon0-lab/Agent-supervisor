import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


class AgentStatus(str, Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    RESTARTED = "restarted"


class TaskStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    TIMEOUT  = "timeout"


@dataclass
class Task:
    id:          str
    name:        str
    payload:     Any
    status:      TaskStatus = TaskStatus.PENDING
    result:      Any        = None
    error:       str | None = None
    attempts:    int        = 0
    created_at:  float      = field(default_factory=time.time)
    started_at:  float | None = None
    finished_at: float | None = None

    def duration(self) -> float | None:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


@dataclass
class AgentRecord:
    id:           str
    name:         str
    status:       AgentStatus = AgentStatus.IDLE
    tasks_done:   int         = 0
    tasks_failed: int         = 0
    restarts:     int         = 0
    last_error:   str | None  = None
    started_at:   float       = field(default_factory=time.time)


AgentFn = Callable[[Task], Coroutine[Any, Any, Any]]


class AgentTimeoutError(Exception):
    pass


class MaxRestartsError(Exception):
    pass


class Supervisor:
    def __init__(
        self,
        max_restarts:    int   = 3,
        task_timeout_s:  float = 30.0,
        retry_delay_s:   float = 1.0,
    ):
        self.max_restarts   = max_restarts
        self.task_timeout   = task_timeout_s
        self.retry_delay    = retry_delay_s
        self._agents:  dict[str, tuple[AgentRecord, AgentFn]] = {}
        self._tasks:   dict[str, Task]                        = {}
        self._queue:   asyncio.Queue[Task]                    = asyncio.Queue()
        self._running: bool                                   = False

    def register(self, name: str, fn: AgentFn) -> str:
        agent_id = str(uuid.uuid4())
        record   = AgentRecord(id=agent_id, name=name)
        self._agents[agent_id] = (record, fn)
        return agent_id

    def unregister(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def submit(self, name: str, payload: Any = None) -> Task:
        task = Task(id=str(uuid.uuid4()), name=name, payload=payload)
        self._tasks[task.id] = task
        self._queue.put_nowait(task)
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        rec = self._agents.get(agent_id)
        return rec[0] if rec else None

    def stats(self) -> dict:
        tasks  = list(self._tasks.values())
        agents = [r for r, _ in self._agents.values()]
        return {
            "agents":        len(agents),
            "tasks_total":   len(tasks),
            "tasks_pending":  sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "tasks_running":  sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
            "tasks_done":     sum(1 for t in tasks if t.status == TaskStatus.DONE),
            "tasks_failed":   sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "queue_size":     self._queue.qsize(),
            "total_restarts": sum(a.restarts for a in agents),
        }

    async def _run_agent(self, agent_id: str, task: Task) -> None:
        record, fn = self._agents[agent_id]
        record.status    = AgentStatus.RUNNING
        task.status      = TaskStatus.RUNNING
        task.started_at  = time.time()
        task.attempts   += 1

        try:
            result      = await asyncio.wait_for(fn(task), timeout=self.task_timeout)
            task.result      = result
            task.status      = TaskStatus.DONE
            task.finished_at = time.time()
            record.status      = AgentStatus.DONE
            record.tasks_done += 1

        except asyncio.TimeoutError:
            task.error       = f"Task timed out after {self.task_timeout}s"
            task.status      = TaskStatus.TIMEOUT
            task.finished_at = time.time()
            record.last_error  = task.error
            record.tasks_failed += 1
            await self._handle_failure(agent_id, task)

        except Exception as exc:
            task.error       = str(exc)
            task.status      = TaskStatus.FAILED
            task.finished_at = time.time()
            record.last_error  = task.error
            record.tasks_failed += 1
            await self._handle_failure(agent_id, task)

    async def _handle_failure(self, agent_id: str, task: Task) -> None:
        record, _ = self._agents[agent_id]

        if record.restarts >= self.max_restarts:
            record.status = AgentStatus.FAILED
            raise MaxRestartsError(
                f"Agent {record.name} reached max restarts ({self.max_restarts})"
            )

        record.restarts += 1
        record.status    = AgentStatus.RESTARTED
        await asyncio.sleep(self.retry_delay)

        task.status      = TaskStatus.PENDING
        task.error       = None
        task.finished_at = None
        self._queue.put_nowait(task)

    async def run(self, agent_ids: list[str] | None = None) -> None:
        ids = agent_ids or list(self._agents.keys())
        self._running = True
        workers = [self._worker_loop(aid) for aid in ids]
        await asyncio.gather(*workers)

    async def run_until_empty(self, agent_ids: list[str] | None = None) -> None:
        ids = agent_ids or list(self._agents.keys())
        self._running = True
        workers = [self._drain_worker(aid) for aid in ids]
        await asyncio.gather(*workers)

    async def _worker_loop(self, agent_id: str) -> None:
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._run_agent(agent_id, task)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except MaxRestartsError:
                break

    async def _drain_worker(self, agent_id: str) -> None:
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                await self._run_agent(agent_id, task)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
            except MaxRestartsError:
                break

    def stop(self) -> None:
        self._running = False
