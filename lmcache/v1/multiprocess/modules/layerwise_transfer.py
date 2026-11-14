# SPDX-License-Identifier: Apache-2.0
"""Server-side host storage for DSV4.1 layerwise resource transfers.

This module is the ``layerwise_transfer`` multiprocess server module: it owns
the worker-facing wire state of the layerwise protocol (registration acks,
per-resource host payloads, all-or-nothing snapshot commits, prefix lookups
and the single snapshot lease).

Storage model (S0 -> S1 milestones):

* Payloads are plain host bytes staged by ``store_resource`` and published
  atomically by ``commit_layerwise_snapshot``; nothing is served before the
  commit succeeds.
* Since S1 the engine-side C2 boundary codec exists: commits may carry real
  :class:`~lmcache.v1.multiprocess.layerwise_protocol.BoundaryStateDescriptor`
  entries, and boundary-complete snapshots are lookup-valid and restorable.
  Boundary-less commits remain legal but stay cold-only storage: the lookup
  validity predicate rejects them so a hit is never half-restored, and the
  capability tuple advertises what the server can serve.

The module is engine agnostic: it never imports torch/vllm and keeps all
state in plain Python structures guarded by one lock, so building it is
cheap for deployments that never use the layerwise path.
"""

# Standard
from dataclasses import dataclass, field
import itertools
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.layerwise_protocol import (
    LAYERWISE_PROTOCOL_VERSION,
    LayerwiseLookupRequest,
    LayerwiseLookupResult,
    LayerwiseRegistration,
    LayerwiseRegistrationAck,
    LayerwiseSnapshotCommit,
    ResourceTransferResult,
    ResourceTransferSpec,
)
from lmcache.v1.multiprocess import dsv41_sampling
from lmcache.v1.multiprocess.layerwise_runtime import (
    LayerwiseSnapshotStore,
    LayerwiseTransferIndex,
)
from lmcache.v1.multiprocess.request_handler import request_handler

logger = init_logger(__name__)

#: Capabilities advertised to the worker in registration acks. Since S1 the
#: engine-side C2 codec can produce boundary-complete snapshots, and this
#: module serves lookups and retrievals against them.
LAYERWISE_CAPABILITIES = ("token_range_host_storage", "boundary_state_codec")


def _snapshot_boundaries_present(commit: LayerwiseSnapshotCommit) -> bool:
    """Validity predicate: only boundary-complete snapshots may hit.

    Args:
        commit: Published snapshot commit to validate.

    Returns:
        ``True`` only when the commit carries at least one boundary
        descriptor; boundary-less commits are cold-only storage and must
        never be restored into a serving prefix.
    """
    return len(commit.boundaries) > 0


def _matched_end_from_boundaries(commit: LayerwiseSnapshotCommit) -> int:
    """Derive the restorable token position from a commit's boundaries.

    Args:
        commit: Boundary-complete snapshot commit whose boundaries the
            engine codec produced.

    Returns:
        The largest engine-visible boundary token position expressed as an
        ``int`` (a commit may describe several scratch resources whose
        boundaries sit at the same restorable prefix end).

    Raises:
        ValueError: If any boundary token is not a decimal integer. The
            C2 codec writes numeric token positions; anything else is a
            protocol violation and must fail closed rather than serve a
            wrong ``matched_end``.
    """
    ends: list[int] = []
    for boundary in commit.boundaries:
        try:
            ends.append(int(boundary.boundary_token))
        except ValueError as error:
            raise ValueError(
                f"boundary {boundary.resource_id} of snapshot "
                f"{commit.snapshot_id} carries non-numeric boundary token "
                f"{boundary.boundary_token!r}"
            ) from error
    return max(ends) if ends else 0


@dataclass
class _RegistrationState:
    """All server-side state of one worker registration identity.

    Args:
        registration: The accepted registration message.
        server_generation: Generation assigned to this registration.
    """

    registration: LayerwiseRegistration
    server_generation: int
    index: LayerwiseTransferIndex = field(
        default_factory=lambda: LayerwiseTransferIndex()
    )
    snapshots: LayerwiseSnapshotStore = field(default_factory=LayerwiseSnapshotStore)
    staged: dict[tuple[str, str], bytes] = field(default_factory=dict)
    op_resources: dict[str, str] = field(default_factory=dict)

    def matches(self, spec: ResourceTransferSpec) -> bool:
        """Return whether ``spec`` belongs to this registration."""
        return (
            spec.plan_hash == self.registration.plan_hash
            and spec.layout_hash == self.registration.layout_hash
            and spec.registration_epoch == self.registration.registration_epoch
        )


class LayerwiseTransferModule(DiscoverableModule):
    """Layerwise transfer host storage and registration server module.

    Handlers map one-to-one onto the ``RequestClient`` layerwise RPC
    contract; the module is discovered and built by the standard module
    lifecycle (see ``lmcache.v1.multiprocess.server``) and closes by
    dropping every staged payload.
    """

    module_name = "layerwise_transfer"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build unconditionally: the module owns no device resources."""
        return cls()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registrations: dict[tuple[str, str], _RegistrationState] = {}
        self._generation_counter = itertools.count(1)
        self._committed_snapshots = 0

    # ------------------------------------------------------------------
    # Wire handlers
    # ------------------------------------------------------------------
    @request_handler()
    def register_layerwise_cache(
        self, registration: LayerwiseRegistration
    ) -> LayerwiseRegistrationAck:
        """Accept (or idempotently re-accept) one layerwise registration.

        Args:
            registration: Worker registration carrying the validated plan,
                identity hashes and epoch.

        Returns:
            The registration ack, including the server generation the
            worker must stamp onto every subsequent transfer spec.

        Raises:
            ValueError: If the registration reuses an identity with a
                lower epoch (a superseded worker replaying old state).
        """
        key = (registration.plan_hash, registration.layout_hash)
        with self._lock:
            state = self._registrations.get(key)
            if state is not None:
                if registration.registration_epoch < state.registration.registration_epoch:
                    raise ValueError(
                        "layerwise registration epoch "
                        f"{registration.registration_epoch} supersedes live epoch "
                        f"{state.registration.registration_epoch}"
                    )
                if registration == state.registration:
                    dsv41_sampling.emit(
                        "server",
                        "server_register",
                        plan_hash=registration.plan_hash,
                        layout_hash=registration.layout_hash,
                        epoch=registration.registration_epoch,
                        generation=state.server_generation,
                        reaccepted=True,
                    )
                    return LayerwiseRegistrationAck(
                        protocol_version=LAYERWISE_PROTOCOL_VERSION,
                        server_generation=state.server_generation,
                        registration_epoch=state.registration.registration_epoch,
                        plan_hash=state.registration.plan_hash,
                        layout_hash=state.registration.layout_hash,
                        capabilities=LAYERWISE_CAPABILITIES,
                    )
            generation = next(self._generation_counter)
            self._registrations[key] = _RegistrationState(
                registration=registration,
                server_generation=generation,
                index=LayerwiseTransferIndex(
                    registration_epoch=registration.registration_epoch
                ),
            )
            dsv41_sampling.emit(
                "server",
                "server_register",
                plan_hash=registration.plan_hash,
                layout_hash=registration.layout_hash,
                epoch=registration.registration_epoch,
                generation=generation,
                reaccepted=False,
            )
            logger.info(
                "Layerwise registration accepted: plan=%s layout=%s epoch=%d "
                "generation=%d",
                registration.plan_hash[:12],
                registration.layout_hash[:12],
                registration.registration_epoch,
                generation,
            )
            return LayerwiseRegistrationAck(
                protocol_version=LAYERWISE_PROTOCOL_VERSION,
                server_generation=generation,
                registration_epoch=registration.registration_epoch,
                plan_hash=registration.plan_hash,
                layout_hash=registration.layout_hash,
                capabilities=LAYERWISE_CAPABILITIES,
            )

    def _state_for(
        self, spec: ResourceTransferSpec
    ) -> _RegistrationState | None:
        """Return the registration state owning ``spec``, or ``None``."""
        state = self._registrations.get((spec.plan_hash, spec.layout_hash))
        if state is None or not state.matches(spec):
            return None
        return state

    @request_handler()
    def store_resource(
        self, spec: ResourceTransferSpec, payload: bytes
    ) -> ResourceTransferResult:
        """Stage one resource payload as host bytes.

        The payload becomes visible to retrievals only after the enclosing
        snapshot commit succeeds.

        Args:
            spec: Transfer spec identifying resource, request and epoch.
            payload: Raw resource bytes extracted from the worker tensors.

        Returns:
            ``ok`` once the payload is staged and the transfer recorded as
            terminal in the registration index; ``invalid`` with an
            ``error_code`` otherwise.
        """
        with self._lock:
            state = self._state_for(spec)
            if state is None:
                dsv41_sampling.emit(
                    "server",
                    "server_store",
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    key=spec.key,
                    wire_bytes=len(payload),
                    status="invalid",
                    error_code="UNKNOWN_REGISTRATION",
                )
                return ResourceTransferResult(
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    registration_epoch=spec.registration_epoch,
                    server_generation=spec.server_generation,
                    status="invalid",
                    error_code="UNKNOWN_REGISTRATION",
                )
            try:
                state.index.submit_transfer(spec)
            except ValueError:
                return ResourceTransferResult(
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    registration_epoch=spec.registration_epoch,
                    server_generation=spec.server_generation,
                    status="invalid",
                    error_code="DUPLICATE_OPERATION",
                )
            state.staged[(spec.key, spec.resource_id)] = bytes(payload)
            state.op_resources[spec.operation_id] = spec.resource_id
            dsv41_sampling.emit(
                "server",
                "server_store",
                operation_id=spec.operation_id,
                resource_id=spec.resource_id,
                key=spec.key,
                wire_bytes=len(payload),
                status="ok",
            )
            result = ResourceTransferResult(
                operation_id=spec.operation_id,
                resource_id=spec.resource_id,
                registration_epoch=spec.registration_epoch,
                server_generation=state.server_generation,
                status="ok",
            )
            state.index.complete_transfer(result)
            return result

    @request_handler()
    def retrieve_resource(
        self, spec: ResourceTransferSpec
    ) -> tuple[ResourceTransferResult, bytes]:
        """Serve one committed resource payload back to the worker.

        Args:
            spec: Transfer spec identifying the resource to restore.

        Returns:
            A ``(result, payload)`` pair; on the msgpack wire the tuple is
            decoded as a two-item list. ``status`` is ``miss`` with an
            empty payload whenever no boundary-complete snapshot owns the
            resource (cold-only commits are never restorable).
        """
        miss = ResourceTransferResult(
            operation_id=spec.operation_id,
            resource_id=spec.resource_id,
            registration_epoch=spec.registration_epoch,
            server_generation=spec.server_generation,
            status="miss",
            error_code="NOT_COMMITTED",
        )
        with self._lock:
            state = self._state_for(spec)
            if state is None:
                dsv41_sampling.emit(
                    "server",
                    "server_retrieve",
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    key=spec.key,
                    wire_bytes=0,
                    status="miss",
                    error_code="UNKNOWN_REGISTRATION",
                )
                return miss, b""
            commit = state.snapshots.lookup(
                spec.key, is_valid=_snapshot_boundaries_present
            )
            if commit is None:
                dsv41_sampling.emit(
                    "server",
                    "server_retrieve",
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    key=spec.key,
                    wire_bytes=0,
                    status="miss",
                    error_code="NOT_COMMITTED",
                )
                return miss, b""
            committed_resources = {
                state.op_resources.get(operation_id)
                for operation_id in commit.operation_ids
            }
            payload = state.staged.get((spec.key, spec.resource_id))
            if spec.resource_id not in committed_resources or payload is None:
                dsv41_sampling.emit(
                    "server",
                    "server_retrieve",
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    key=spec.key,
                    wire_bytes=0,
                    status="miss",
                    error_code="NOT_IN_SNAPSHOT",
                    snapshot_id=commit.snapshot_id,
                )
                return miss, b""
            dsv41_sampling.emit(
                "server",
                "server_retrieve",
                operation_id=spec.operation_id,
                resource_id=spec.resource_id,
                key=spec.key,
                wire_bytes=len(payload),
                status="ok",
                snapshot_id=commit.snapshot_id,
            )
            return (
                ResourceTransferResult(
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    registration_epoch=spec.registration_epoch,
                    server_generation=state.server_generation,
                    status="ok",
                ),
                payload,
            )

    @request_handler()
    def commit_layerwise_snapshot(
        self, commit: LayerwiseSnapshotCommit
    ) -> bool:
        """Publish one all-or-nothing snapshot over staged operations.

        Args:
            commit: Snapshot commit listing every staged operation of the
                snapshot. Empty boundaries are accepted and published as
                cold-only storage (never restorable).

        Returns:
            ``True`` on publication or idempotent re-commit.

        Raises:
            ValueError: If the commit references unknown/failed operations
                or belongs to an unregistered identity.
        """
        with self._lock:
            state = self._registrations.get((commit.plan_hash, commit.layout_hash))
            if (
                state is None
                or state.registration.registration_epoch != commit.registration_epoch
            ):
                raise ValueError(
                    "layerwise snapshot commit for unregistered identity "
                    f"plan={commit.plan_hash[:12]} layout={commit.layout_hash[:12]}"
                )
            outcome = state.snapshots.commit_snapshot(commit, state.index)
            if outcome == "published":
                self._committed_snapshots += 1
            dsv41_sampling.emit(
                "server",
                "server_commit",
                snapshot_id=commit.snapshot_id,
                key=commit.key,
                operation_count=len(commit.operation_ids),
                boundary_count=len(commit.boundaries),
                outcome=outcome,
            )
            return True

    @request_handler()
    def lookup_layerwise_prefix(
        self, request: LayerwiseLookupRequest
    ) -> LayerwiseLookupResult | None:
        """Look up the newest boundary-complete snapshot of one prefix key.

        Args:
            request: Lookup request carrying identity and prefix key.

        Returns:
            The lookup result with a granted lease, or ``None`` when no
            boundary-complete snapshot exists. ``LayerwiseLookupResult``
            itself cannot represent a miss (its wire contract requires a
            real snapshot/lease), so ``None`` *is* the miss: a miss is
            never fabricated into a partial hit.
        """
        with self._lock:
            state = self._registrations.get((request.plan_hash, request.layout_hash))
            if (
                state is None
                or state.registration.registration_epoch != request.registration_epoch
            ):
                dsv41_sampling.emit(
                    "server",
                    "server_lookup",
                    key=request.key,
                    hit=False,
                    reason="unregistered_identity",
                )
                return None
            commit = state.snapshots.lookup(
                request.key, is_valid=_snapshot_boundaries_present
            )
            if commit is None:
                dsv41_sampling.emit(
                    "server",
                    "server_lookup",
                    key=request.key,
                    hit=False,
                    reason="no_boundary_complete_snapshot",
                )
                return None
            lease_id = state.snapshots.acquire_lease(commit.snapshot_id)
            resources = tuple(
                sorted(
                    {
                        state.op_resources[operation_id]
                        for operation_id in commit.operation_ids
                        if operation_id in state.op_resources
                    }
                )
            )
            dsv41_sampling.emit(
                "server",
                "server_lookup",
                key=request.key,
                hit=True,
                matched_end=_matched_end_from_boundaries(commit),
                snapshot_id=commit.snapshot_id,
                lease_id=lease_id,
                resources=list(resources),
                boundary_count=len(commit.boundaries),
            )
            return LayerwiseLookupResult(
                matched_end=_matched_end_from_boundaries(commit),
                snapshot_id=commit.snapshot_id,
                resource_ids=resources,
                lease_id=lease_id,
                plan_hash=request.plan_hash,
                layout_hash=request.layout_hash,
                server_generation=state.server_generation,
            )

    @request_handler()
    def release_layerwise_lease(self, lease_id: str) -> bool:
        """Release the single lease of one published snapshot (idempotent).

        Args:
            lease_id: Lease returned by a successful lookup.

        Returns:
            ``True`` on the first release of a live lease, ``False``
            otherwise (unknown or already released).
        """
        with self._lock:
            for state in self._registrations.values():
                if state.snapshots.release_lease(lease_id):
                    dsv41_sampling.emit(
                        "server",
                        "server_release",
                        lease_id=lease_id,
                        released=True,
                    )
                    return True
            dsv41_sampling.emit(
                "server",
                "server_release",
                lease_id=lease_id,
                released=False,
            )
            return False

    # ------------------------------------------------------------------
    # Module lifecycle
    # ------------------------------------------------------------------
    def report_status(self) -> dict:
        """Return handler counts and storage volume for observability."""
        with self._lock:
            return {
                "registrations": len(self._registrations),
                "committed_snapshots": self._committed_snapshots,
                "staged_bytes": sum(
                    len(payload)
                    for state in self._registrations.values()
                    for payload in state.staged.values()
                ),
                "boundary_codec": "c2-v1",
            }

    def close(self) -> None:
        """Drop every staged payload and registration state."""
        with self._lock:
            self._registrations.clear()
