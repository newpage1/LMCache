# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project.
"""Per-resource/per-layer/per-step state index for layerwise transfers.

This module owns the engine-side bookkeeping required by the layerwise MP
connector (design doc section 4.5). It is deliberately free of torch/vllm
imports so the concurrency and fail-closed semantics can be unit tested on
CPU. All state transitions happen under one :class:`threading.RLock`; no I/O
runs while the lock is held. Completion callbacks are validated against the
key they claim (operation id + registration epoch) before any state is
written, so a late callback from a superseded epoch can never rewrite newer
state.

Fail-closed rules enforced here:

- A layer step is admitted only when *every* required resource transfer of
  that step reached terminal ``ok``; any ``miss``/``invalid``/``failed``
  rejects the step and wakes waiters with an error.
- Duplicate completion, completion of an unknown operation id, or completion
  carrying a stale registration epoch is ignored (returns ``False``) and
  never rewrites an already-terminal state.
- ``get_finished`` reports each request at most once per direction.
- A snapshot is published only when every listed operation reached terminal
  ``ok`` in the current epoch; leases release exactly once; an invalidated
  snapshot falls back to the previous complete boundary of the same key.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import Iterable

# Third Party
import msgspec

from lmcache.v1.multiprocess.layerwise_protocol import (
    LayerwiseSnapshotCommit,
    ResourceTransferResult,
    ResourceTransferSpec,
)

TERMINAL_STATUSES = ("ok", "miss", "invalid", "failed", "cancelled")
"""Terminal statuses of a transfer; no later callback may rewrite them."""

EPOCH_SUPERSEDED = "epoch_superseded"
"""Error code assigned to in-flight ops when the registration epoch moves."""


@dataclass(frozen=True)
class ResourceOpKey:
    """Identity of one transfer operation (per resource per step)."""

    registration_epoch: int
    request_id: str
    step_id: int
    resource_id: str
    direction: str


@dataclass(frozen=True)
class LayerStepKey:
    """Identity of one layer's admission state within a request step."""

    registration_epoch: int
    request_id: str
    step_id: int
    execution_layer_name: str


@dataclass
class ResourceTransferState:
    """Mutable per-operation bookkeeping; guarded by the index lock."""

    spec: ResourceTransferSpec
    future: Future
    ready_event: Event
    save_event: Event
    remaining_consumers: int
    status: str | None = None
    error_code: str = ""
    snapshot_refs: int = 0


@dataclass
class LayerStepState:
    """Mutable per-layer admission state; guarded by the index lock."""

    required_resource_ids: frozenset[str]
    pending_resource_ids: set[str]
    admitted: Event = field(default_factory=Event)
    status: str = "pending"
    error_code: str = ""


@dataclass
class RequestRecord:
    """Aggregation state of one request within one epoch; lock-guarded.

    ``*_pending`` counts submitted-but-not-terminal transfers per direction;
    the ``*_done`` event fires only when the count reaches zero, so a request
    is never reported finished while any of its transfers is still in flight.
    """

    retrieve_done: Event = field(default_factory=Event)
    store_done: Event = field(default_factory=Event)
    retrieve_pending: int = 0
    store_pending: int = 0
    retrieve_reported: bool = False
    store_reported: bool = False


@dataclass
class SnapshotRecord:
    """One published snapshot; guarded by the store lock."""

    commit: LayerwiseSnapshotCommit
    operation_ids: tuple[str, ...]
    lease_id: str | None = None
    lease_released: bool = False


class LayerwiseTransferIndex:
    """Thread-safe index of layerwise transfer and admission state."""

    def __init__(self, registration_epoch: int = 0) -> None:
        """Create an index bound to one registration epoch.

        Args:
            registration_epoch: Epoch of the registration this index serves;
                completions carrying any other epoch are ignored.
        """
        self._lock = RLock()
        self._registration_epoch = registration_epoch
        self._ops: dict[ResourceOpKey, ResourceTransferState] = {}
        self._ops_by_id: dict[str, ResourceTransferState] = {}
        self._layer_steps: dict[LayerStepKey, LayerStepState] = {}
        self._requests: dict[tuple[int, str], RequestRecord] = {}

    @property
    def registration_epoch(self) -> int:
        return self._registration_epoch

    # ------------------------------------------------------------------
    # Transfer submission / completion
    # ------------------------------------------------------------------

    def submit_transfer(self, spec: ResourceTransferSpec) -> Future:
        """Register one transfer spec and return its shared future.

        Args:
            spec: Fully validated transfer spec.

        Returns:
            A :class:`concurrent.futures.Future` that completes exactly once
            when the transfer reaches a terminal status. Re-submitting the
            identical spec returns the *same* future (idempotent submit).

        Raises:
            ValueError: If the spec's epoch differs from the index epoch, an
                operation id is reused for a different key, or the same key
                is re-submitted with a conflicting spec.
        """
        if spec.registration_epoch != self._registration_epoch:
            raise ValueError(
                f"spec epoch {spec.registration_epoch} != index epoch {self._registration_epoch}"
            )
        key = ResourceOpKey(
            spec.registration_epoch,
            spec.request_id,
            spec.step_id,
            spec.resource_id,
            _direction_of(spec),
        )
        with self._lock:
            existing = self._ops.get(key)
            if existing is not None:
                if existing.spec != spec:
                    raise ValueError(
                        f"conflicting re-submission for {key}: specs differ"
                    )
                return existing.future
            clash = self._ops_by_id.get(spec.operation_id)
            if clash is not None:
                raise ValueError(
                    f"operation id {spec.operation_id} already bound to another transfer"
                )
            state = ResourceTransferState(
                spec=spec,
                future=Future(),
                ready_event=Event(),
                save_event=Event(),
                remaining_consumers=0,
            )
            self._ops[key] = state
            self._ops_by_id[spec.operation_id] = state
            record = self._requests.setdefault(
                (self._registration_epoch, spec.request_id), RequestRecord()
            )
            if key.direction == "retrieve":
                record.retrieve_pending += 1
            else:
                record.store_pending += 1
            return state.future

    def complete_transfer(self, result: ResourceTransferResult) -> bool:
        """Apply one terminal transfer result.

        Args:
            result: Server-side result carrying the operation id and the
                registration epoch it belongs to.

        Returns:
            ``True`` if the result was applied; ``False`` if it was ignored
            (unknown operation id, stale epoch, or duplicate completion).
            Ignored results never modify any state.
        """
        with self._lock:
            state = self._ops_by_id.get(result.operation_id)
            if state is None:
                return False
            if result.registration_epoch != self._registration_epoch:
                return False
            if state.status is not None:
                return False
            state.status = result.status
            state.error_code = result.error_code
            if result.status == "ok":
                state.future.set_result(result)
                state.ready_event.set()
                state.save_event.set()
            else:
                state.future.set_exception(
                    RuntimeError(
                        f"transfer {result.operation_id} ended {result.status}"
                        + (f" ({result.error_code})" if result.error_code else "")
                    )
                )
            self._resolve_layer_steps_locked(state)
            self._resolve_request_locked(state)
            return True

    # ------------------------------------------------------------------
    # Layer admission
    # ------------------------------------------------------------------

    def begin_layer_step(
        self,
        request_id: str,
        step_id: int,
        execution_layer_name: str,
        required_resource_ids: Iterable[str],
        *,
        retrieve_specs: Iterable[ResourceTransferSpec],
    ) -> None:
        """Open one layer's admission window for a request step.

        Args:
            request_id: Owning request.
            step_id: Scheduler step index.
            execution_layer_name: Plan execution-layer ID (``...self_attn.attn``).
            required_resource_ids: Resources that must all reach terminal
                ``ok`` before the layer is admitted.
            retrieve_specs: The already-submitted retrieve specs for those
                resources; admission resolves as their results arrive.

        Raises:
            ValueError: If a spec does not match the (request, step, layer)
                window, references a resource outside the required set, does
                not cover *every* required resource, or the layer step
                already exists.
        """
        required = frozenset(required_resource_ids)
        specs = tuple(retrieve_specs)
        for spec in specs:
            if (
                spec.request_id != request_id
                or spec.step_id != step_id
                or spec.resource_id not in required
            ):
                raise ValueError(
                    f"spec {spec.operation_id} does not match the admission window"
                )
        missing = required - {spec.resource_id for spec in specs}
        if missing:
            # A window that does not track every required resource could be
            # admitted while one of them never resolves; refuse it instead.
            raise ValueError(
                f"admission window for request {request_id} step {step_id} "
                f"layer {execution_layer_name} is missing required "
                f"resources: {sorted(missing)}"
            )
        key = LayerStepKey(
            self._registration_epoch, request_id, step_id, execution_layer_name
        )
        with self._lock:
            if key in self._layer_steps:
                raise ValueError(f"layer step {key} already exists")
            self._requests.setdefault(
                (self._registration_epoch, request_id), RequestRecord()
            )
            states = tuple(self._ops_by_id[spec.operation_id] for spec in specs)
            pending = {
                state.spec.resource_id
                for state in states
                if state.status is None
            }
            failed = {
                state.spec.resource_id
                for state in states
                if state.status is not None and state.status != "ok"
            }
            self._layer_steps[key] = LayerStepState(
                required_resource_ids=required,
                pending_resource_ids=set(pending),
            )
            for state in states:
                state.remaining_consumers += 1
            if failed:
                step = self._layer_steps[key]
                step.status = "rejected"
                step.error_code = "required_resource_failed"
                step.admitted.set()
            elif not pending:
                # An empty window (nothing to fetch — a cold/no-hit
                # request opens one) admits immediately: no transfer
                # completion will ever arrive to resolve it (T56-03).
                step = self._layer_steps[key]
                step.status = "admitted"
                step.admitted.set()

    def wait_for_admission(
        self,
        request_id: str,
        step_id: int,
        execution_layer_name: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """Block until the layer step is admitted.

        Args:
            request_id: Owning request.
            step_id: Scheduler step index.
            execution_layer_name: Plan execution-layer ID.
            timeout: Seconds to wait; ``None`` waits forever.

        Raises:
            KeyError: If the layer step was never opened (fail-closed:
                no admission recorded for this layer).
            TimeoutError: If the admission window did not resolve in time.
            RuntimeError: If the step was rejected (a required resource
                missed/failed) or its epoch was superseded.
        """
        key = LayerStepKey(
            self._registration_epoch, request_id, step_id, execution_layer_name
        )
        with self._lock:
            step = self._layer_steps.get(key)
        if step is None:
            raise KeyError(f"no admission state for {key}")
        if not step.admitted.wait(timeout):
            raise TimeoutError(f"admission not resolved for {key}")
        if step.status != "admitted":
            raise RuntimeError(
                f"layer step {key} rejected: {step.error_code or step.status}"
            )

    # ------------------------------------------------------------------
    # Cancellation / restart
    # ------------------------------------------------------------------

    def cancel_request(self, request_id: str) -> None:
        """Cancel every still-pending op of one request in this epoch.

        Terminal states are left untouched; already-admitted steps stay
        admitted so late server results are simply ignored.
        """
        with self._lock:
            for key, state in self._ops.items():
                if key.request_id != request_id:
                    continue
                if state.status is None:
                    state.status = "cancelled"
                    state.error_code = "request_cancelled"
                    state.future.set_exception(RuntimeError(f"{key} cancelled"))
                    self._resolve_request_locked(state)
                state.ready_event.set()
                state.save_event.set()
            for step_key, step in self._layer_steps.items():
                if step_key.request_id != request_id or step.status != "pending":
                    continue
                step.status = "cancelled"
                step.error_code = "request_cancelled"
                step.admitted.set()

    def rebind_epoch(self, new_epoch: int) -> None:
        """Server restart: move the index to a new registration epoch.

        Every non-terminal op fails with :data:`EPOCH_SUPERSEDED` and every
        pending layer step is rejected, so waiters wake and fail closed.
        Late results from the old epoch are ignored afterwards because the
        index epoch no longer matches them. The index itself moves to the
        new epoch so fresh submissions for it are accepted.

        Args:
            new_epoch: The new registration epoch; must differ from current.
        """
        if new_epoch == self._registration_epoch:
            raise ValueError("new epoch must differ from the current epoch")
        with self._lock:
            for state in self._ops.values():
                if state.status is None:
                    state.status = "failed"
                    state.error_code = EPOCH_SUPERSEDED
                    state.future.set_exception(
                        RuntimeError(f"{state.spec.operation_id} {EPOCH_SUPERSEDED}")
                    )
                    self._resolve_request_locked(state)
                state.ready_event.set()
                state.save_event.set()
            for step in self._layer_steps.values():
                if step.status != "pending":
                    continue
                step.status = "rejected"
                step.error_code = EPOCH_SUPERSEDED
                step.admitted.set()
            self._registration_epoch = new_epoch

    # ------------------------------------------------------------------
    # Aggregation / release
    # ------------------------------------------------------------------

    def pending_request_ids(self, direction: str) -> tuple[str, ...]:
        """Request ids that still have non-terminal ops in ``direction``.

        Args:
            direction: ``"retrieve"`` or ``"store"``.

        Returns:
            Sorted request ids with at least one submitted-but-unsettled
            transfer in that direction; empty when the direction is idle.
        """
        with self._lock:
            return tuple(
                sorted(
                    {
                        key.request_id
                        for key, state in self._ops.items()
                        if key.direction == direction and state.status is None
                    }
                )
            )

    def get_finished(
        self, request_ids: Iterable[str]
    ) -> tuple[set[str], set[str]]:
        """Aggregate finished requests, reporting each request once.

        Args:
            request_ids: Candidate request ids to settle this call.

        Returns:
            ``(finished_receiving, finished_sending)`` restricted to requests
            whose step outcomes have not been reported yet; every request
            appears at most once across the whole lifetime of the index.
        """
        finished_receiving: set[str] = set()
        finished_sending: set[str] = set()
        with self._lock:
            for request_id in request_ids:
                record = self._requests.get(
                    (self._registration_epoch, request_id)
                )
                if record is None:
                    continue
                if (
                    record.retrieve_done.is_set()
                    and not record.retrieve_reported
                ):
                    record.retrieve_reported = True
                    finished_receiving.add(request_id)
                if record.store_done.is_set() and not record.store_reported:
                    record.store_reported = True
                    finished_sending.add(request_id)
        return finished_receiving, finished_sending

    def maybe_release_resource(
        self,
        request_id: str,
        resource_id: str,
        *,
        leases_released: bool = True,
    ) -> bool:
        """Decide whether a resource's state can truly be freed now.

        ``released_after`` in the plan only encodes the *computational*
        dependency; physical release additionally requires every consumer to
        be done, the op events to be set, no snapshot to still reference the
        ops, and the snapshot lease to be handled.

        Args:
            request_id: Owning request.
            resource_id: Resource whose ops are candidates for release.
            leases_released: Whether the snapshot lease covering this
                resource (if any) has been released.

        Returns:
            ``True`` if the resource has no pending consumer, its events are
            set, no snapshot references remain, and leases are handled.
        """
        with self._lock:
            consumers_pending = any(
                key.request_id == request_id
                and step.status == "pending"
                and resource_id in step.required_resource_ids
                for key, step in self._layer_steps.items()
            )
            referenced = False
            events_set = True
            for key, state in self._ops.items():
                if key.request_id != request_id or key.resource_id != resource_id:
                    continue
                if state.status is None:
                    events_set = False
                if not state.ready_event.is_set() or not state.save_event.is_set():
                    events_set = False
                if state.snapshot_refs > 0:
                    referenced = True
            return (
                not consumers_pending
                and events_set
                and not referenced
                and leases_released
            )

    def acquire_snapshot_ref(self, operation_id: str) -> None:
        """Pin one op for a published snapshot; blocks its release."""
        with self._lock:
            state = self._ops_by_id[operation_id]
            state.snapshot_refs += 1

    def release_snapshot_ref(self, operation_id: str) -> None:
        """Drop one snapshot pin previously taken by ``acquire_snapshot_ref``."""
        with self._lock:
            state = self._ops_by_id[operation_id]
            if state.snapshot_refs <= 0:
                raise ValueError(f"operation {operation_id} has no snapshot refs")
            state.snapshot_refs -= 1

    # ------------------------------------------------------------------
    # Internals (lock already held by callers)
    # ------------------------------------------------------------------

    def _resolve_layer_steps_locked(self, state: ResourceTransferState) -> None:
        for key, step in self._layer_steps.items():
            if key.request_id != state.spec.request_id:
                continue
            if state.spec.resource_id not in step.pending_resource_ids:
                continue
            step.pending_resource_ids.discard(state.spec.resource_id)
            if state.status != "ok" and step.status == "pending":
                step.status = "rejected"
                step.error_code = f"resource_{state.status}"
                step.admitted.set()
            elif not step.pending_resource_ids and step.status == "pending":
                step.status = "admitted"
                step.admitted.set()

    def _resolve_request_locked(self, state: ResourceTransferState) -> None:
        """Settle one op's request record; called once per terminal op.

        The direction's pending count reaches zero only when every submitted
        transfer of that request reached a terminal status — a multi-resource
        request is never reported finished after its first completion.
        """
        record = self._requests.setdefault(
            (self._registration_epoch, state.spec.request_id), RequestRecord()
        )
        if _direction_of(state.spec) == "retrieve":
            record.retrieve_pending -= 1
            if record.retrieve_pending <= 0:
                record.retrieve_done.set()
        else:
            record.store_pending -= 1
            if record.store_pending <= 0:
                record.store_done.set()


class LayerwiseSnapshotStore:
    """All-or-nothing snapshot publication with exactly-once lease release."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[str, SnapshotRecord] = {}
        self._leases: dict[str, SnapshotRecord] = {}
        self._history: dict[str, list[str]] = {}

    def commit_snapshot(
        self, commit: LayerwiseSnapshotCommit, index: LayerwiseTransferIndex
    ) -> str:
        """Publish one snapshot after verifying every operation succeeded.

        Args:
            commit: The all-or-nothing commit message.
            index: Transfer index holding the terminal results of
                ``commit.operation_ids``.

        Returns:
            ``"published"`` on first success, ``"duplicate"`` when the exact
            same commit was already published.

        Raises:
            ValueError: If any operation is unknown, not terminal ``ok``,
                or belongs to a different epoch; or a different commit
                reuses an existing snapshot id.
        """
        with index._lock:  # noqa: SLF001 - store and index share one lifecycle
            for operation_id in commit.operation_ids:
                state = index._ops_by_id.get(operation_id)  # noqa: SLF001
                if state is None:
                    raise ValueError(
                        f"commit {commit.snapshot_id} references unknown op {operation_id}"
                    )
                if state.spec.registration_epoch != commit.registration_epoch:
                    raise ValueError(
                        f"commit {commit.snapshot_id} epoch mismatch on {operation_id}"
                    )
                if state.status != "ok":
                    raise ValueError(
                        f"commit {commit.snapshot_id} requires op {operation_id} "
                        f"to be ok, got {state.status}"
                    )
                if state.spec.plan_hash != commit.plan_hash:
                    raise ValueError(
                        f"commit {commit.snapshot_id} plan hash mismatch on {operation_id}"
                    )
        with self._lock:
            existing = self._snapshots.get(commit.snapshot_id)
            if existing is not None:
                if (
                    existing.commit != commit
                ):
                    raise ValueError(
                        f"snapshot id {commit.snapshot_id} reused with different content"
                    )
                return "duplicate"
            self._snapshots[commit.snapshot_id] = SnapshotRecord(
                commit=commit,
                operation_ids=tuple(commit.operation_ids),
            )
            self._history.setdefault(commit.key, []).append(commit.snapshot_id)
            return "published"

    def acquire_lease(self, snapshot_id: str) -> str:
        """Grant the single active lease of one published snapshot.

        Raises:
            KeyError: If the snapshot is unknown or invalidated.
            ValueError: If a lease is already active for the snapshot.
        """
        with self._lock:
            record = self._snapshots[snapshot_id]
            if record.lease_id is not None and not record.lease_released:
                raise ValueError(f"snapshot {snapshot_id} already leased")
            lease_id = f"{snapshot_id}-lease"
            record.lease_id = lease_id
            record.lease_released = False
            self._leases[lease_id] = record
            return lease_id

    def release_lease(self, lease_id: str) -> bool:
        """Release a lease exactly once.

        Returns:
            ``True`` on the first release, ``False`` for every later call
            (idempotent no-op, never an error).
        """
        with self._lock:
            record = self._leases.get(lease_id)
            if record is None or record.lease_released:
                return False
            record.lease_released = True
            return True

    def invalidate_snapshot(self, snapshot_id: str) -> None:
        """Drop one snapshot; lookups fall back to older boundaries."""
        with self._lock:
            record = self._snapshots.pop(snapshot_id, None)
            if record is None:
                return
            self._history[record.commit.key] = [
                snap
                for snap in self._history.get(record.commit.key, [])
                if snap != snapshot_id
            ]

    def lookup(
        self,
        key: str,
        *,
        is_valid: "callable[[LayerwiseSnapshotCommit], bool] | None" = None,
    ) -> LayerwiseSnapshotCommit | None:
        """Return the newest valid snapshot of ``key`` after fallback.

        Args:
            key: The layerwise prefix key.
            is_valid: Optional predicate applied to each candidate, newest
                first; snapshots failing the predicate (TTL, generation,
                ring overwritten, ...) are skipped as invalidated.

        Returns:
            The newest snapshot whose predicate passes, or ``None`` when no
            complete boundary survives (caller reports a full miss).
        """
        with self._lock:
            history = list(self._history.get(key, []))
        # Newest last in history; walk backwards for the fallback policy.
        for snapshot_id in reversed(history):
            record = self._snapshots.get(snapshot_id)
            if record is None:
                continue
            if is_valid is not None and not is_valid(record.commit):
                continue
            return record.commit
        return None


def plan_digest(plan: msgspec.Struct) -> str:
    """Stable digest of one wire plan for registration/commit hashing."""
    return msgspec.json.encode(plan).hex()


def _direction_of(spec: ResourceTransferSpec) -> str:
    """Direction is not a wire field; adapters mark it via the op key.

    The store direction is derived from the op id prefix assigned by the
    adapter (``retrieve:``/``store:``), keeping the wire struct unchanged.
    """
    if spec.operation_id.startswith("store:"):
        return "store"
    return "retrieve"
