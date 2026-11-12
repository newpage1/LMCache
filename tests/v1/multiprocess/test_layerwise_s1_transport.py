# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the LMCache project.
"""S1 transport integration tests over a real ZMQ request server.

These tests drive ``LayerwiseTransferModule`` through a genuine
``MessageQueueServer``/``MessageQueueClient`` pair on a loopback TCP port:
registration acks, boundary-complete snapshot publication (the C2 codec's
server-side contract), prefix lookups with real hit/miss semantics, the
single snapshot lease, and fail-closed rejection of stale identities.

The engine-side codec that produces the payloads lives with the engine
integration (LMCache-Ascend); here the payloads are opaque bytes because
the server never interprets resource contents.
"""

# Standard
import socket

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)
from lmcache.v1.multiprocess.layerwise_protocol import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    LAYERWISE_PROTOCOL_VERSION,
    BoundaryStateDescriptor,
    LayerwiseLookupRequest,
    LayerwiseRegistration,
    LayerwiseSnapshotCommit,
    ResourceTransferSpec,
    compute_boundary_checksum,
)
from lmcache.v1.multiprocess.modules.layerwise_transfer import (
    LAYERWISE_CAPABILITIES,
    LayerwiseTransferModule,
)
from tests.v1.multiprocess.transport_test_utils import (
    start_lookup_request_server,
)


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _plan() -> LayerwiseTransferPlan:
    """One token-range resource plus one boundary-state scratch resource."""
    return LayerwiseTransferPlan(
        layer_order=("exec-0", "exec-1"),
        resources=(
            CacheTransferResource(
                resource_id="res-kv",
                tensor_indices=(0,),
                engine_group_id=0,
                producer_layer="exec-0",
                consumer_layers=("exec-1",),
            ),
            CacheTransferResource(
                resource_id="res-scratch",
                tensor_indices=(1,),
                engine_group_id=None,
                producer_layer="exec-0",
                consumer_layers=("exec-1",),
                storage_kind="boundary_state",
            ),
        ),
    )


def _plan_hash(plan: LayerwiseTransferPlan) -> str:
    import hashlib

    import msgspec

    return "sha256:" + hashlib.sha256(
        msgspec.json.encode(plan)
    ).hexdigest()


def _registration(plan: LayerwiseTransferPlan, epoch: int = 0) -> LayerwiseRegistration:
    return LayerwiseRegistration(
        protocol_version=LAYERWISE_PROTOCOL_VERSION,
        plan_hash=_plan_hash(plan),
        layout_hash="sha256:" + "0" * 64,
        registration_epoch=epoch,
        plan=plan,
        cache_names=("cache-0", "cache-1"),
        vllm_rank=0,
        kv_worker_id="worker-0",
        topology_hash="sha256:" + "1" * 64,
    )


def _identity(registration: LayerwiseRegistration) -> dict:
    return {
        "registration_epoch": registration.registration_epoch,
        "server_generation": 1,
        "plan_hash": registration.plan_hash,
        "layout_hash": registration.layout_hash,
    }


def _token_spec(
    registration: LayerwiseRegistration,
    *,
    operation_id: str,
    request_id: str,
    step_id: int,
    resource_id: str,
    key: str,
) -> ResourceTransferSpec:
    return ResourceTransferSpec(
        operation_id=operation_id,
        request_id=request_id,
        step_id=step_id,
        **_identity(registration),
        resource_id=resource_id,
        key=key,
        engine_group_id=0,
        block_ids=(1, 2),
        start_token=0,
        end_token=256,
        skip_first_n_tokens=0,
        boundary=None,
    )


def _boundary_spec(
    registration: LayerwiseRegistration,
    *,
    operation_id: str,
    request_id: str,
    step_id: int,
    resource_id: str,
    key: str,
    boundary: BoundaryStateDescriptor,
) -> ResourceTransferSpec:
    return ResourceTransferSpec(
        operation_id=operation_id,
        request_id=request_id,
        step_id=step_id,
        **_identity(registration),
        resource_id=resource_id,
        key=key,
        engine_group_id=None,
        block_ids=(),
        start_token=0,
        end_token=int(boundary.boundary_token),
        skip_first_n_tokens=0,
        boundary=boundary,
    )


def _boundary(
    snapshot_id: str, resource_id: str = "res-scratch", token: str = "256"
) -> BoundaryStateDescriptor:
    rows = (3, 7, 11)
    return BoundaryStateDescriptor(
        snapshot_id=snapshot_id,
        resource_id=resource_id,
        boundary_token=token,
        ring_cursor=9,
        compression_phase=0,
        row_indices=rows,
        schema_version=BOUNDARY_STATE_SCHEMA_VERSION,
        checksum=compute_boundary_checksum(
            snapshot_id,
            resource_id,
            token,
            9,
            0,
            rows,
            BOUNDARY_STATE_SCHEMA_VERSION,
        ),
    )


def _call(client, operation: str, *payloads):
    future = client.submit_request(operation, list(payloads))
    return future.result(timeout=10)


class _ServerHandle:
    """Bound URL + started server + one real ZMQ client."""

    def __init__(self) -> None:
        from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueClient

        self.module = LayerwiseTransferModule()
        port = _unused_tcp_port()
        self.url = f"tcp://127.0.0.1:{port}"
        self.server = start_lookup_request_server("zmq", self.url, self.module)
        self.client = MessageQueueClient(self.url, zmq.Context.instance())

    def close(self) -> None:
        self.client.close()
        self.server.close()

    def __enter__(self) -> "_ServerHandle":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def test_s1_boundary_complete_snapshot_hits_over_real_zmq() -> None:
    """Store + boundary commit + lookup hit/miss + lease over real MQ."""
    plan = _plan()
    registration = _registration(plan)
    request_id = "req-hit"
    key = f"{request_id}:layerwise"

    with _ServerHandle() as handle:
        ack = _call(
            handle.client, "register_layerwise_cache", registration
        )
        assert ack.server_generation >= 1
        assert "boundary_state_codec" in ack.capabilities
        assert ack.capabilities == LAYERWISE_CAPABILITIES

        # Idempotent re-registration echoes the same generation.
        again = _call(handle.client, "register_layerwise_cache", registration)
        assert again.server_generation == ack.server_generation

        snapshot_id = "snap-s1-hit"
        kv_spec = _token_spec(
            registration,
            operation_id="store:1",
            request_id=request_id,
            step_id=0,
            resource_id="res-kv",
            key=key,
        )
        kv_payload = b"\x01\x02\x03\x04"
        result = _call(handle.client, "store_resource", kv_spec, kv_payload)
        assert result.status == "ok"

        boundary = _boundary(snapshot_id)
        boundary_spec = _boundary_spec(
            registration,
            operation_id="store:2",
            request_id=request_id,
            step_id=0,
            resource_id="res-scratch",
            key=key,
            boundary=boundary,
        )
        scratch_payload = b"\x09" * 12
        result = _call(
            handle.client, "store_resource", boundary_spec, scratch_payload
        )
        assert result.status == "ok"

        commit = LayerwiseSnapshotCommit(
            snapshot_id=snapshot_id,
            key=key,
            registration_epoch=registration.registration_epoch,
            server_generation=ack.server_generation,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            operation_ids=("store:1", "store:2"),
            boundaries=(boundary,),
        )
        assert _call(handle.client, "commit_layerwise_snapshot", commit) is True
        # Idempotent re-commit of the identical commit.
        assert _call(handle.client, "commit_layerwise_snapshot", commit) is True

        lookup = LayerwiseLookupRequest(
            protocol_version=LAYERWISE_PROTOCOL_VERSION,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            registration_epoch=registration.registration_epoch,
            key=key,
        )
        hit = _call(handle.client, "lookup_layerwise_prefix", lookup)
        # The pre-S1 server crashed with TypeError here (str boundary token
        # in an int field); the honest result is an int matched_end.
        assert hit is not None
        assert isinstance(hit.matched_end, int)
        assert hit.matched_end == 256
        assert hit.snapshot_id == snapshot_id
        assert set(hit.resource_ids) == {"res-kv", "res-scratch"}
        assert hit.lease_id

        retrieved = _call(
            handle.client,
            "retrieve_resource",
            _boundary_spec(
                registration,
                operation_id="retrieve:1",
                request_id=request_id,
                step_id=0,
                resource_id="res-scratch",
                key=key,
                boundary=boundary,
            ),
        )
        result, payload = retrieved
        assert result.status == "ok"
        assert bytes(payload) == scratch_payload

        # A different prefix key is an honest miss (None, never fabricated).
        other = LayerwiseLookupRequest(
            protocol_version=LAYERWISE_PROTOCOL_VERSION,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            registration_epoch=registration.registration_epoch,
            key="req-other:layerwise",
        )
        assert _call(handle.client, "lookup_layerwise_prefix", other) is None

        # Lease lifecycle: release is exactly-once per lease id.
        first_release = _call(
            handle.client, "release_layerwise_lease", hit.lease_id
        )
        second_release = _call(
            handle.client, "release_layerwise_lease", hit.lease_id
        )
        assert first_release is True
        assert second_release is False

        # After release the snapshot can serve a fresh lease (the lease id
        # is deterministic per snapshot, so it matches the first grant).
        rehit = _call(handle.client, "lookup_layerwise_prefix", lookup)
        assert rehit is not None
        assert rehit.snapshot_id == snapshot_id
        assert rehit.lease_id == hit.lease_id
        assert _call(
            handle.client, "release_layerwise_lease", rehit.lease_id
        ) is True


def test_s1_boundaryless_commit_stays_cold_only() -> None:
    """A commit without boundaries is stored but never lookup-valid."""
    plan = _plan()
    registration = _registration(plan)
    request_id = "req-cold"
    key = f"{request_id}:layerwise"

    with _ServerHandle() as handle:
        ack = _call(handle.client, "register_layerwise_cache", registration)
        spec = _token_spec(
            registration,
            operation_id="store:1",
            request_id=request_id,
            step_id=0,
            resource_id="res-kv",
            key=key,
        )
        assert (
            _call(handle.client, "store_resource", spec, b"cold").status == "ok"
        )
        commit = LayerwiseSnapshotCommit(
            snapshot_id="snap-s1-cold",
            key=key,
            registration_epoch=registration.registration_epoch,
            server_generation=ack.server_generation,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            operation_ids=("store:1",),
            boundaries=(),
        )
        assert _call(handle.client, "commit_layerwise_snapshot", commit) is True

        lookup = LayerwiseLookupRequest(
            protocol_version=LAYERWISE_PROTOCOL_VERSION,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            registration_epoch=registration.registration_epoch,
            key=key,
        )
        assert _call(handle.client, "lookup_layerwise_prefix", lookup) is None
        retrieved = _call(
            handle.client,
            "retrieve_resource",
            _token_spec(
                registration,
                operation_id="retrieve:1",
                request_id=request_id,
                step_id=0,
                resource_id="res-kv",
                key=key,
            ),
        )
        result, payload = retrieved
        assert result.status == "miss"
        assert result.error_code == "NOT_COMMITTED"
        assert payload == b""


def test_s1_stale_identity_and_duplicate_store_fail_closed() -> None:
    """Lookups from a stale epoch miss; duplicate stores are rejected."""
    plan = _plan()
    registration = _registration(plan)
    request_id = "req-guard"
    key = f"{request_id}:layerwise"

    with _ServerHandle() as handle:
        ack = _call(handle.client, "register_layerwise_cache", registration)

        stale_lookup = LayerwiseLookupRequest(
            protocol_version=LAYERWISE_PROTOCOL_VERSION,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            registration_epoch=registration.registration_epoch + 5,
            key=key,
        )
        assert _call(handle.client, "lookup_layerwise_prefix", stale_lookup) is None

        foreign_spec = ResourceTransferSpec(
            operation_id="store:foreign",
            request_id=request_id,
            step_id=0,
            registration_epoch=registration.registration_epoch,
            server_generation=ack.server_generation,
            plan_hash="sha256:" + "f" * 64,
            layout_hash=registration.layout_hash,
            resource_id="res-kv",
            key=key,
            engine_group_id=0,
            block_ids=(1,),
            start_token=0,
            end_token=8,
            skip_first_n_tokens=0,
            boundary=None,
        )
        result = _call(handle.client, "store_resource", foreign_spec, b"x")
        assert result.status == "invalid"
        assert result.error_code == "UNKNOWN_REGISTRATION"

        spec = _token_spec(
            registration,
            operation_id="store:1",
            request_id=request_id,
            step_id=0,
            resource_id="res-kv",
            key=key,
        )
        assert _call(handle.client, "store_resource", spec, b"a").status == "ok"
        # The identical spec replays idempotently; a *different* operation
        # claiming the same (request, step, resource, direction) key is a
        # conflicting re-submission and must be rejected fail-closed.
        assert _call(handle.client, "store_resource", spec, b"a").status == "ok"
        conflicting = _token_spec(
            registration,
            operation_id="store:2",
            request_id=request_id,
            step_id=0,
            resource_id="res-kv",
            key=key,
        )
        duplicate = _call(handle.client, "store_resource", conflicting, b"b")
        assert duplicate.status == "invalid"
        assert duplicate.error_code == "DUPLICATE_OPERATION"
