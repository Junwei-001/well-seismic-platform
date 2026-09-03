"""Task-scoped cancellation and managed subprocess lifecycle support.

The API keeps task state in its control plane, while this module owns the
runtime entities that must actually become quiescent before regenerable caches
can be deleted.  A context variable binds worker threads to their task without
changing prediction-runner plugin signatures.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, TypeVar


LOGGER = logging.getLogger(__name__)

_ResultT = TypeVar("_ResultT")
_PROCESS_WAIT_SECONDS = 2.0
_POSIX_TERM_GRACE_SECONDS = 0.25


class TaskCancellationRequested(BaseException):
    """Abort one worker without letting ordinary failure handlers revive it."""

    def __init__(self, task_id: str | None = None) -> None:
        self.task_id = task_id
        suffix = f": {task_id}" if task_id else ""
        super().__init__(f"task cancellation requested{suffix}")


@dataclass(frozen=True, slots=True)
class StopReport:
    """Result of requesting runtime cancellation for a group of tasks."""

    quiescent: bool
    still_running_task_ids: tuple[str, ...]
    futures_cancelled: int
    processes_terminated: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "quiescent": self.quiescent,
            "still_running_task_ids": list(self.still_running_task_ids),
            "futures_cancelled": self.futures_cancelled,
            "processes_terminated": self.processes_terminated,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class _TaskContext:
    registry: "TaskRuntimeRegistry"
    task_id: str


@dataclass(eq=False, slots=True)
class _ProcessRecord:
    task_id: str | None
    process: subprocess.Popen[Any]
    process_group_id: int
    termination_lock: threading.Lock = field(default_factory=threading.Lock)
    termination_counted: bool = False


@dataclass(slots=True)
class _TaskState:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    futures: set[Future[Any]] = field(default_factory=set)
    processes: set[_ProcessRecord] = field(default_factory=set)


_CURRENT_TASK: ContextVar[_TaskContext | None] = ContextVar(
    "well_seismic_current_runtime_task", default=None
)


def _remaining_seconds(deadline: float | None, default: float) -> float:
    if deadline is None:
        return default
    return max(0.0, min(default, deadline - monotonic()))


def _wait_for_process(
    process: subprocess.Popen[Any], *, deadline: float | None, default: float
) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=_remaining_seconds(deadline, default))
    except subprocess.TimeoutExpired:
        return process.poll() is not None
    return True


def _terminate_windows_process_tree(
    record: _ProcessRecord, *, deadline: float | None
) -> list[str]:
    process = record.process
    errors: list[str] = []
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_remaining_seconds(deadline, _PROCESS_WAIT_SECONDS),
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"taskkill({process.pid}): {type(exc).__name__}: {exc}")
    else:
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            errors.append(
                f"taskkill({process.pid}) exited {completed.returncode}"
                + (f": {detail}" if detail else "")
            )

    if process.poll() is None:
        try:
            process.kill()
        except OSError as exc:
            if process.poll() is None:
                errors.append(f"kill({process.pid}): {type(exc).__name__}: {exc}")
    if not _wait_for_process(
        process, deadline=deadline, default=_PROCESS_WAIT_SECONDS
    ):
        errors.append(f"process tree {process.pid} did not exit before timeout")
    return errors


def _terminate_posix_process_tree(
    record: _ProcessRecord, *, deadline: float | None
) -> list[str]:
    process = record.process
    process_group_id = record.process_group_id
    errors: list[str] = []
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return errors
    except OSError as exc:
        errors.append(
            f"killpg(SIGTERM, {process_group_id}): {type(exc).__name__}: {exc}"
        )

    _wait_for_process(
        process, deadline=deadline, default=_POSIX_TERM_GRACE_SECONDS
    )
    # Kill the group even when its leader already exited: descendants can
    # outlive the direct child while retaining the process-group identifier.
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(
            f"killpg(SIGKILL, {process_group_id}): {type(exc).__name__}: {exc}"
        )
    if not _wait_for_process(
        process, deadline=deadline, default=_PROCESS_WAIT_SECONDS
    ):
        errors.append(f"process group {process_group_id} did not exit before timeout")
    return errors


def _terminate_unregistered_process(record: _ProcessRecord) -> tuple[bool, list[str]]:
    """Best-effort cleanup for a process launched outside a task context."""

    with record.termination_lock:
        if record.process.poll() is not None:
            return False, []
        errors = (
            _terminate_windows_process_tree(record, deadline=None)
            if os.name == "nt"
            else _terminate_posix_process_tree(record, deadline=None)
        )
        return True, errors


class TaskRuntimeRegistry:
    """Track futures and subprocess trees by durable API task identifier."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._states: dict[str, _TaskState] = {}
        self._cancelled_task_ids: set[str] = set()
        self._unresolved_task_ids: set[str] = set()
        self._cancel_waiters: dict[str, int] = {}
        self._termination_counts: dict[str, int] = {}
        self._cleanup_errors: dict[str, list[str]] = {}

    def submit(
        self,
        executor: Executor,
        task_id: str,
        fn: Callable[..., _ResultT],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[_ResultT]:
        """Register and submit one worker atomically with cancellation.

        ``fn`` retains the platform worker signature and receives ``task_id``
        as its first positional argument.
        """

        normalized_task_id = str(task_id).strip()
        if not normalized_task_id:
            raise ValueError("task_id must not be empty")

        with self._condition:
            if normalized_task_id in self._cancelled_task_ids:
                cancelled: Future[_ResultT] = Future()
                cancelled.cancel()
                return cancelled
            state = self._states.setdefault(normalized_task_id, _TaskState())
            try:
                future = executor.submit(
                    self._run_task,
                    normalized_task_id,
                    fn,
                    args,
                    kwargs,
                )
            except BaseException:
                self._remove_state_if_idle_locked(normalized_task_id, state)
                raise
            state.futures.add(future)
            future.add_done_callback(
                lambda completed, current_task_id=normalized_task_id: self._future_done(
                    current_task_id, completed
                )
            )
            self._condition.notify_all()
            return future

    def _run_task(
        self,
        task_id: str,
        fn: Callable[..., _ResultT],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _ResultT:
        token = _CURRENT_TASK.set(_TaskContext(self, task_id))
        try:
            self.check_cancelled(task_id)
            return fn(task_id, *args, **kwargs)
        finally:
            _CURRENT_TASK.reset(token)

    def _future_done(self, task_id: str, future: Future[Any]) -> None:
        with self._condition:
            state = self._states.get(task_id)
            if state is not None:
                state.futures.discard(future)
                self._remove_state_if_idle_locked(task_id, state)
            self._condition.notify_all()

    def check_cancelled(self, task_id: str | None = None) -> None:
        """Raise when the explicit or context-bound task was cancelled."""

        resolved_task_id = str(task_id).strip() if task_id is not None else ""
        if not resolved_task_id:
            context = _CURRENT_TASK.get()
            if context is None or context.registry is not self:
                return
            resolved_task_id = context.task_id
        with self._condition:
            state = self._states.get(resolved_task_id)
            cancelled = resolved_task_id in self._cancelled_task_ids or bool(
                state and state.cancel_event.is_set()
            )
        if cancelled:
            raise TaskCancellationRequested(resolved_task_id)

    def active_task_ids(self) -> tuple[str, ...]:
        """Return tasks that still own a live future or process."""

        with self._condition:
            return tuple(
                sorted(
                    {
                        task_id
                        for task_id, state in self._states.items()
                        if self._state_is_active_locked(state)
                    }
                    | self._unresolved_task_ids
                )
            )

    def cancel_and_wait(
        self, task_ids: Iterable[str], timeout_seconds: float
    ) -> StopReport:
        """Signal, dequeue, terminate and wait for the selected task runtimes."""

        timeout = float(timeout_seconds)
        if timeout < 0:
            raise ValueError("timeout_seconds must be non-negative")
        task_id_values = (task_ids,) if isinstance(task_ids, str) else task_ids
        selected = tuple(
            sorted(
                {
                    str(task_id).strip()
                    for task_id in task_id_values
                    if str(task_id).strip()
                }
            )
        )
        deadline = monotonic() + timeout
        if not selected:
            return StopReport(True, (), 0, 0, ())

        with self._condition:
            initial_termination_count = sum(
                self._termination_counts.get(task_id, 0) for task_id in selected
            )
            initial_error_lengths = {
                task_id: len(self._cleanup_errors.get(task_id, ()))
                for task_id in selected
            }
            futures: list[Future[Any]] = []
            processes: list[_ProcessRecord] = []
            for task_id in selected:
                self._cancelled_task_ids.add(task_id)
                self._cancel_waiters[task_id] = self._cancel_waiters.get(task_id, 0) + 1
                state = self._states.get(task_id)
                if state is None:
                    continue
                state.cancel_event.set()
                futures.extend(state.futures)
                processes.extend(state.processes)
            self._condition.notify_all()

        futures_cancelled = sum(1 for future in futures if future.cancel())
        for record in processes:
            if monotonic() >= deadline:
                break
            errors = self._terminate_registered_process(record, deadline=deadline)
            if errors:
                self._record_errors(record.task_id, errors)

        with self._condition:
            while True:
                still_running = self._active_selected_locked(selected)
                if not still_running:
                    break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            still_running = self._active_selected_locked(selected)
            final_termination_count = sum(
                self._termination_counts.get(task_id, 0) for task_id in selected
            )
            errors = tuple(
                error
                for task_id in selected
                for error in self._cleanup_errors.get(task_id, ())
            )
            # API task ids are unique, but tests and embedding applications may
            # deliberately reuse an id.  Keep the tombstone for every runtime
            # that is still active (including a cancel/register race), then
            # release it only after the task is fully quiescent.
            still_running_set = set(still_running) | {
                task_id
                for task_id in selected
                if len(self._cleanup_errors.get(task_id, ()))
                > initial_error_lengths[task_id]
            }
            still_running = tuple(sorted(still_running_set))
            for task_id in selected:
                remaining_waiters = self._cancel_waiters.get(task_id, 1) - 1
                if remaining_waiters > 0:
                    self._cancel_waiters[task_id] = remaining_waiters
                else:
                    self._cancel_waiters.pop(task_id, None)
                if task_id in still_running_set or remaining_waiters > 0:
                    continue
                self._cancelled_task_ids.discard(task_id)
                self._termination_counts.pop(task_id, None)
                self._cleanup_errors.pop(task_id, None)

        return StopReport(
            quiescent=not still_running,
            still_running_task_ids=still_running,
            futures_cancelled=futures_cancelled,
            processes_terminated=max(
                0, final_termination_count - initial_termination_count
            ),
            errors=errors,
        )

    def _active_selected_locked(self, task_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            task_id
            for task_id in task_ids
            if task_id in self._unresolved_task_ids
            or (
                (state := self._states.get(task_id)) is not None
                and self._state_is_active_locked(state)
            )
        )

    @staticmethod
    def _state_is_active_locked(state: _TaskState) -> bool:
        return any(not future.done() for future in state.futures) or any(
            record.process.poll() is None for record in state.processes
        )

    def _remove_state_if_idle_locked(self, task_id: str, state: _TaskState) -> None:
        if not state.futures and not state.processes:
            self._states.pop(task_id, None)
            if task_id not in self._unresolved_task_ids and not self._cancel_waiters.get(
                task_id, 0
            ):
                self._cancelled_task_ids.discard(task_id)
                self._termination_counts.pop(task_id, None)
                self._cleanup_errors.pop(task_id, None)

    def _register_process(
        self, task_id: str, process: subprocess.Popen[Any]
    ) -> tuple[_ProcessRecord, bool]:
        record = _ProcessRecord(task_id, process, process.pid)
        with self._condition:
            state = self._states.setdefault(task_id, _TaskState())
            state.processes.add(record)
            cancelled = task_id in self._cancelled_task_ids or state.cancel_event.is_set()
            self._condition.notify_all()
        return record, cancelled

    def _unregister_process(self, record: _ProcessRecord) -> None:
        if record.task_id is None:
            return
        with self._condition:
            state = self._states.get(record.task_id)
            if state is not None:
                state.processes.discard(record)
                self._remove_state_if_idle_locked(record.task_id, state)
            self._condition.notify_all()

    def _terminate_registered_process(
        self, record: _ProcessRecord, *, deadline: float | None
    ) -> list[str]:
        with record.termination_lock:
            if record.process.poll() is not None:
                return []
            errors = (
                _terminate_windows_process_tree(record, deadline=deadline)
                if os.name == "nt"
                else _terminate_posix_process_tree(record, deadline=deadline)
            )
            if record.process.poll() is not None and not record.termination_counted:
                record.termination_counted = True
                if record.task_id is not None:
                    with self._condition:
                        self._termination_counts[record.task_id] = (
                            self._termination_counts.get(record.task_id, 0) + 1
                        )
                        self._condition.notify_all()
            return errors

    def _record_errors(self, task_id: str | None, errors: Iterable[str]) -> None:
        materialized = [str(error) for error in errors if str(error)]
        if not materialized:
            return
        if task_id is None:
            for error in materialized:
                LOGGER.warning("managed subprocess cleanup failed: %s", error)
            return
        with self._condition:
            self._cleanup_errors.setdefault(task_id, []).extend(materialized)
            # A failed tree cleanup cannot be called quiescent merely because
            # the direct Popen handle exited.  Keep the task visible to cache
            # reset retries and fail closed rather than deleting live inputs.
            self._unresolved_task_ids.add(task_id)
            self._condition.notify_all()

    def _process_record(
        self, task_id: str, process: subprocess.Popen[Any]
    ) -> _ProcessRecord | None:
        with self._condition:
            state = self._states.get(task_id)
            if state is None:
                return None
            return next(
                (record for record in state.processes if record.process is process),
                None,
            )

    def _is_cancelled(self, task_id: str) -> bool:
        with self._condition:
            state = self._states.get(task_id)
            return task_id in self._cancelled_task_ids or bool(
                state and state.cancel_event.is_set()
            )


TASK_RUNTIME = TaskRuntimeRegistry()


def _managed_popen_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(kwargs)
    if os.name == "nt":
        prepared["creationflags"] = int(prepared.get("creationflags", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        prepared["start_new_session"] = True
    return prepared


@contextmanager
def managed_popen(*args: Any, **kwargs: Any) -> Iterator[subprocess.Popen[Any]]:
    """Launch a process group and bind its lifetime to the current task."""

    context = _CURRENT_TASK.get()
    if context is None:
        # Direct runner/unit-test calls have no task to cancel.  Preserve Popen
        # semantics exactly; API workers always enter through registry.submit.
        yield subprocess.Popen(*args, **kwargs)
        return
    context.registry.check_cancelled(context.task_id)
    process = subprocess.Popen(*args, **_managed_popen_kwargs(kwargs))
    record = _ProcessRecord(None, process, process.pid)
    registered = False
    try:
        if context is not None:
            record, cancelled = context.registry._register_process(
                context.task_id, process
            )
            registered = True
            if cancelled:
                errors = context.registry._terminate_registered_process(
                    record, deadline=None
                )
                context.registry._record_errors(context.task_id, errors)
                raise TaskCancellationRequested(context.task_id)
        try:
            yield process
            if context is not None:
                context.registry.check_cancelled(context.task_id)
        except BaseException as exc:
            if process.poll() is None:
                if context is not None:
                    errors = context.registry._terminate_registered_process(
                        record, deadline=None
                    )
                    context.registry._record_errors(context.task_id, errors)
                else:
                    _, errors = _terminate_unregistered_process(record)
                    for error in errors:
                        LOGGER.warning("managed subprocess cleanup failed: %s", error)
            if (
                context is not None
                and context.registry._is_cancelled(context.task_id)
                and not isinstance(exc, TaskCancellationRequested)
            ):
                raise TaskCancellationRequested(context.task_id) from exc
            raise
    finally:
        if process.poll() is None:
            if context is not None:
                errors = context.registry._terminate_registered_process(
                    record, deadline=None
                )
                context.registry._record_errors(context.task_id, errors)
            else:
                _, errors = _terminate_unregistered_process(record)
                for error in errors:
                    LOGGER.warning("managed subprocess cleanup failed: %s", error)
        if registered and context is not None:
            context.registry._unregister_process(record)


def managed_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """A cancellation-aware subset-compatible replacement for ``run``."""

    if _CURRENT_TASK.get() is None:
        # Besides being backwards compatible, this keeps direct library calls
        # free of a process-group policy when no registry can cancel them.
        return subprocess.run(*args, **kwargs)

    input_value = kwargs.pop("input", None)
    capture_output = bool(kwargs.pop("capture_output", False))
    timeout = kwargs.pop("timeout", None)
    check = bool(kwargs.pop("check", False))
    if input_value is not None and kwargs.get("stdin") is not None:
        raise ValueError("stdin and input arguments may not both be used")
    if input_value is not None:
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    with managed_popen(*args, **kwargs) as process:
        context = _CURRENT_TASK.get()
        try:
            stdout, stderr = process.communicate(input=input_value, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            # Match subprocess.run's useful timeout contract while terminating
            # the whole managed tree instead of only the direct child.
            if context is not None:
                record = context.registry._process_record(
                    context.task_id, process
                )
                if record is not None:
                    errors = context.registry._terminate_registered_process(
                        record, deadline=None
                    )
                    context.registry._record_errors(context.task_id, errors)
            try:
                final_stdout, final_stderr = process.communicate()
            except (OSError, ValueError):
                final_stdout, final_stderr = exc.output, exc.stderr
            exc.stdout = final_stdout
            exc.stderr = final_stderr
            raise
        if context is not None:
            context.registry.check_cancelled(context.task_id)
        completed = subprocess.CompletedProcess(
            process.args,
            process.returncode,
            stdout,
            stderr,
        )
        if check:
            completed.check_returncode()
        return completed


__all__ = [
    "StopReport",
    "TASK_RUNTIME",
    "TaskCancellationRequested",
    "TaskRuntimeRegistry",
    "managed_popen",
    "managed_run",
]
