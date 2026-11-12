# SPDX-License-Identifier: Apache-2.0
"""Server-side host storage for DSV4.1 layerwise resource transfers.

This module is the ``layerwise_transfer`` multiprocess server module: it owns
the worker-facing wire state of the layerwise protocol (registration acks,
per-resource host payloads, all-or-nothing snapshot commits, prefix lookups
and the single snapshot lease).

Storage model (S0 milestone):

* Payloads are plain host bytes staged by ``store_resource`` and published
  atomically by ``commit_layerwise_snapshot``; nothing is served before the
  commit succeeds.
* The boundary-state codec (C2) is not available yet, so every published
  snapshot carries empty boundaries and the lookup validity predicate
  rejects boundary-less commits. Lookups therefore report an honest full
  miss (``matched_end == 0``) instead of a half-restored hit, and the
  capability tuple advertises this explicitly.

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
from lmcache.v1.multiprocess.layerwise_runtime import (
    LayerwiseSnapshotStore,
    LayerwiseTransferIndex,
)
from lmcache.v1.multiprocess.request_handler import request_handler

logger = init_logger(__name__)

#: Capabilities advertised to the worker in registration acks. The boundary
#: codec is deliberately absent: lookups cannot hit until C2 lands (S1).
LAYERWISE_CAPABILITIES = ("token_range_host_storage",)


def _snapshot_boundaries_present(commit: LayerwiseSnapshotCommit) -> bool:
    """Validity predicate for S0: only boundary-complete snapshots may hit.

    Args:
        commit: Published snapshot commit to validate.

    Returns:
        ``True`` only when the commit carries at least one boundary
        descriptor; boundary-less commits are cold-only storage and must
        never be restored into a serving prefix.
    """
    return len(commit.boundaries) > 0


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
            resource (always, until the C2 codec lands).
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
                return miss, b""
            commit = state.snapshots.lookup(
                spec.key, is_valid=_snapshot_boundaries_present
            )
            if commit is None:
                return miss, b""
            committed_resources = {
                state.op_resources.get(operation_id)
                for operation_id in commit.operation_ids
            }
            payload = state.staged.get((spec.key, spec.resource_id))
            if spec.resource_id not in committed_resources or payload is None:
                return miss, b""
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
            real snapshot/lease), so ``None`` *is* the miss: always, until
            the C2 boundary codec exists.
        """
        with self._lock:
            state = self._registrations.get((request.plan_hash, request.layout_hash))
            if (
                state is None
                or state.registration.registration_epoch != request.registration_epoch
            ):
                return None
            commit = state.snapshots.lookup(
                request.key, is_valid=_snapshot_boundaries_present
            )
            if commit is None:
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
            return LayerwiseLookupResult(
                matched_end=commit.boundaries[-1].boundary_token
                if commit.boundaries
                else 0,
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
                    return True
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
                "boundary_codec": "unavailable-s0",
            }

    def close(self) -> None:
        """Drop every staged payload and registration state."""
        with self._lock:
            self._registrations.clear()
