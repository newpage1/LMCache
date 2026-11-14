# SPDX-License-Identifier: Apache-2.0
"""Duplicate resource-block writes must be idempotent, not IO failures.

Several v2 workers can share one stock identity: MLA replicates KV across the
TP ranks, so all of them derive the same stock key for a resource block and
race to store it.  Only one ``reserve_write`` can own the buffer; the losers
must verify the resident bytes and still record completeness for their own
registration, otherwise no registration state is ever complete and the stock
lookup reports 0.

These tests use a stock-storage stand-in that behaves like the L1 manager for
duplicate keys (a resident key refuses a new reservation) and fake token
hashing; no ZMQ, vLLM or accelerator is involved.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.resource_block import (
    ResourceBlockModule,
    StockResourceBlockStore,
)
from lmcache.v1.multiprocess.resource_block_protocol import (
    ERROR_ALREADY_PRESENT,
    STATUS_OK,
    ResourceBlockRegistration,
    ResourceBlockSpec,
)

# Reuse the resource-block fixtures so keys, layouts and payloads agree with
# the server-side unit tests.
from tests.v1.multiprocess.test_resource_block_visibility import (
    BOUNDARIES,
    CHUNK,
    RESOURCES,
    FakeContext,
    _boundary_payload,
    _key,
    _lookup,
    _registration,
    _token_payload,
    _tokens,
)

REQUESTED_TOKENS = 2 * CHUNK


class _Object:
    """Minimal ``MemoryObj``: a byte tensor plus its L1 owner id."""

    def __init__(self, size: int) -> None:
        self.tensor = torch.zeros(size, dtype=torch.uint8)

    def get_l1_manager(self) -> int:
        """Return the fake L1 owner used by ``finish_write``."""
        return 0

    def payload(self) -> bytes:
        """Return the tensor contents as bytes."""
        return self.tensor.numpy().tobytes()


class _Reader:
    """Read side of the fake storage manager."""

    def __init__(self, store: "_ResidentStore") -> None:
        self._store = store

    def reserve_read(self, keys):
        result = {}
        with self._store.lock:
            for key in keys:
                payload = self._store.data.get(key)
                if payload is None:
                    result[key] = (object(), None)
                else:
                    obj = _Object(len(payload))
                    obj.tensor.copy_(
                        torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                    )
                    result[key] = (None, obj)
        return result

    def finish_read(self, keys):
        with self._store.lock:
            self._store.finish_read_calls += 1
        return {key: None for key in keys}


class _ResidentStore:
    """Stock-storage stand-in where a resident key refuses a new reservation.

    Args:
        fail_first: Number of initial ``reserve_write`` calls that return no
            buffer even though the key is absent (models a transient miss).
        always_fail: When True, every reservation fails.
    """

    def __init__(self, *, fail_first: int = 0, always_fail: bool = False) -> None:
        self.data: dict[object, bytes] = {}
        self.reserved: dict[object, _Object] = {}
        self.lock = threading.Lock()
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.reserve_attempts = 0
        self.finish_read_calls = 0

    def reserve_write(self, keys, layout_desc):
        with self.lock:
            self.reserve_attempts += 1
            if self.always_fail or self.fail_first > 0:
                if self.fail_first > 0:
                    self.fail_first -= 1
                return {}
            result = {}
            for key in keys:
                if key in self.data or key in self.reserved:
                    continue
                obj = _Object(int(layout_desc.shapes[0][0]))
                self.reserved[key] = obj
                result[key] = obj
            return result

    def finish_write(self, keys, owners):
        with self.lock:
            for key in keys:
                self.data[key] = self.reserved.pop(key).payload()

    def l1_managers(self):
        return [_Reader(self)]


def _store(
    manager: _ResidentStore,
    *,
    retries: int = 5,
    delay: float = 0.0,
) -> StockResourceBlockStore:
    return StockResourceBlockStore(
        manager, write_retries=retries, write_retry_delay=delay
    )


def test_sequential_duplicate_write_of_same_payload_succeeds() -> None:
    """A second write of identical bytes reports success, not IO error."""
    manager = _ResidentStore()
    store = _store(manager)
    key = "block-key"

    assert store.write(key, b"payload") is True
    assert manager.data[key] == b"payload"
    assert store.write(key, b"payload") is True
    assert manager.data[key] == b"payload"
    assert manager.reserve_attempts == 2


def test_different_payload_is_rejected_and_not_overwritten() -> None:
    """A resident payload with different bytes is never replaced."""
    manager = _ResidentStore()
    store = _store(manager)
    key = "block-key"

    assert store.write(key, b"first") is True
    assert store.write(key, b"second") is False
    assert manager.data[key] == b"first"


def test_reserve_failure_retries_until_storage_is_available() -> None:
    """A transient reservation miss is retried, then the write succeeds."""
    manager = _ResidentStore(fail_first=3)
    store = _store(manager, retries=5)

    assert store.write("block-key", b"payload") is True
    assert manager.data["block-key"] == b"payload"
    assert manager.reserve_attempts == 4


def test_reserve_failure_is_bounded() -> None:
    """An unavailable reservation eventually reports failure, never blocks."""
    manager = _ResidentStore(always_fail=True)
    store = _store(manager, retries=3)

    assert store.write("block-key", b"payload") is False
    assert manager.reserve_attempts == 4


def test_concurrent_duplicate_writes_all_succeed() -> None:
    """Eight racing writers of one payload all see success."""
    manager = _ResidentStore()
    store = _store(manager, retries=50, delay=0.001)
    key = "block-key"
    payload = b"racing payload"
    barrier = threading.Barrier(8)

    def write_once() -> bool:
        barrier.wait()
        return store.write(key, payload)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: write_once(), range(8)))

    assert results == [True] * 8
    assert manager.data[key] == payload
    assert manager.finish_read_calls >= 1


def _registration_for(instance_id: str) -> ResourceBlockRegistration:
    base = _registration(world=1)
    return ResourceBlockRegistration(
        protocol_version=base.protocol_version,
        instance_id=instance_id,
        model_name=base.model_name,
        world_size=base.world_size,
        worker_id=base.worker_id,
        plan_hash=base.plan_hash,
        layout_hash=base.layout_hash,
        registration_epoch=base.registration_epoch,
        resources=base.resources,
        layouts=base.layouts,
    )


def _store_spec(
    *,
    instance_id: str,
    generation: int,
    key: IPCCacheServerKey,
    resource_id: str,
    start: int,
    end: int,
    prefix_end: int,
    boundary: bool,
    block_ids: tuple[int, ...] = (0,),
) -> ResourceBlockSpec:
    return ResourceBlockSpec(
        operation_id=f"op-{resource_id}-{start}",
        direction="store",
        instance_id=instance_id,
        request_id=key.request_id,
        step_id=0,
        registration_epoch=1,
        server_generation=generation,
        plan_hash="plan-1",
        layout_hash="layout-1",
        stock_key=key,
        resource_id=resource_id,
        engine_group_id=None if boundary else 0,
        block_ids=() if boundary else block_ids,
        start_token=start,
        end_token=end,
        prefix_end=prefix_end,
        skip_first_n_tokens=0,
        boundary=(
            {"schema": 2, "resource_id": resource_id, "prefix_end": prefix_end}
            if boundary
            else None
        ),
    )


def _make_module(manager: _ResidentStore) -> ResourceBlockModule:
    ctx = FakeContext()
    ctx.storage_manager = manager
    module = ResourceBlockModule(ctx, block_store=_store(manager))
    ctx.set_resource_block_visibility(module)
    return module


def _store_from_registration(
    module: ResourceBlockModule,
    *,
    instance_id: str,
    generation: int,
    key: IPCCacheServerKey,
    chunks: int,
) -> None:
    for chunk in range(chunks):
        start, end = chunk * CHUNK, (chunk + 1) * CHUNK
        for resource_id in RESOURCES:
            result = module.store_resource_block(
                _store_spec(
                    instance_id=instance_id,
                    generation=generation,
                    key=key,
                    resource_id=resource_id,
                    start=start,
                    end=end,
                    prefix_end=end,
                    boundary=False,
                    block_ids=(chunk,),
                ),
                _token_payload(resource_id, (chunk,)),
            )
            assert (result.status, result.error_code) == (STATUS_OK, ""), result
    for resource_id in BOUNDARIES:
        result = module.store_resource_block(
            _store_spec(
                instance_id=instance_id,
                generation=generation,
                key=key,
                resource_id=resource_id,
                start=chunks * CHUNK,
                end=chunks * CHUNK,
                prefix_end=chunks * CHUNK,
                boundary=True,
            ),
            _boundary_payload(resource_id, chunks * CHUNK),
        )
        assert (result.status, result.error_code) == (STATUS_OK, ""), result


def test_eight_registrations_record_completeness_for_the_same_keys() -> None:
    """MLA: eight registrations share one stock identity and all become complete."""
    manager = _ResidentStore()
    module = _make_module(manager)
    key = _key(_tokens(8, 0), world=1, worker=0)
    try:
        for index in range(8):
            ack = module.register_resource_blocks(
                _registration_for(f"worker-{index}")
            )
            _store_from_registration(
                module,
                instance_id=f"worker-{index}",
                generation=ack.server_generation,
                key=key,
                chunks=2,
            )

        result = _lookup(
            module, _key(_tokens(8, 0), world=1, worker=None), REQUESTED_TOKENS
        )
        assert result.prefix_end == REQUESTED_TOKENS
        assert result.token_prefix_end == REQUESTED_TOKENS
        assert result.boundary_prefix_end == REQUESTED_TOKENS
        expected = 8 * (2 * len(RESOURCES) + len(BOUNDARIES))
        assert module.report_status()["blocks_stored"] == expected
    finally:
        module.close()


def test_interleaved_worker_races_leave_every_registration_complete() -> None:
    """Eight MLA workers race per key: every registration still becomes complete.

    The t21 B01 failure wrote the same keys from eight registration states in
    interleaved order, so each key was reserved by a different worker.  The
    losers must verify the resident payload and record completeness, otherwise
    no single state owns all markers and the stock lookup stays at 0.
    """
    manager = _ResidentStore()
    module = _make_module(manager)
    key = _key(_tokens(8, 0), world=1, worker=0)
    instances = [f"worker-{index}" for index in range(8)]
    try:
        generations = {
            instance: module.register_resource_blocks(
                _registration_for(instance)
            ).server_generation
            for instance in instances
        }
        work: list[tuple[str, str, int]] = [
            ("token", resource_id, chunk)
            for chunk in range(2)
            for resource_id in RESOURCES
        ] + [("boundary", resource_id, 1) for resource_id in BOUNDARIES]
        for index, (kind, resource_id, chunk) in enumerate(work):
            for offset in range(8):
                instance = instances[(index + offset) % 8]
                result = module.store_resource_block(
                    _store_spec(
                        instance_id=instance,
                        generation=generations[instance],
                        key=key,
                        resource_id=resource_id,
                        start=chunk * CHUNK,
                        end=(chunk + 1) * CHUNK,
                        prefix_end=(chunk + 1) * CHUNK,
                        boundary=kind == "boundary",
                        block_ids=(chunk,),
                    ),
                    (
                        _boundary_payload(resource_id, (chunk + 1) * CHUNK)
                        if kind == "boundary"
                        else _token_payload(resource_id, (chunk,))
                    ),
                )
                assert (result.status, result.error_code) == (STATUS_OK, ""), result

        result = _lookup(
            module, _key(_tokens(8, 0), world=1, worker=None), REQUESTED_TOKENS
        )
        assert result.prefix_end == REQUESTED_TOKENS
        expected = 8 * (2 * len(RESOURCES) + len(BOUNDARIES))
        assert module.report_status()["blocks_stored"] == expected
    finally:
        module.close()


def test_repeat_store_within_one_registration_is_already_present() -> None:
    """The same registration repeats a block: success without a second write."""
    manager = _ResidentStore()
    module = _make_module(manager)
    key = _key(_tokens(8, 0), world=1, worker=0)
    try:
        ack = module.register_resource_blocks(_registration_for("worker-0"))
        _store_from_registration(
            module,
            instance_id="worker-0",
            generation=ack.server_generation,
            key=key,
            chunks=2,
        )
        result = module.store_resource_block(
            _store_spec(
                instance_id="worker-0",
                generation=ack.server_generation,
                key=key,
                resource_id=RESOURCES[0],
                start=0,
                end=CHUNK,
                prefix_end=CHUNK,
                boundary=False,
            ),
            _token_payload(RESOURCES[0], (0,)),
        )
        assert (result.status, result.error_code) == (STATUS_OK, ERROR_ALREADY_PRESENT)
        assert module.report_status()["blocks_stored"] == (
            2 * len(RESOURCES) + len(BOUNDARIES)
        )
    finally:
        module.close()


def test_store_error_is_reported_when_no_reservation_and_no_payload() -> None:
    """A store that cannot reserve and cannot verify a payload is an IO error."""
    manager = _ResidentStore(always_fail=True)
    module = _make_module(manager)
    key = _key(_tokens(8, 0), world=1, worker=0)
    try:
        ack = module.register_resource_blocks(_registration_for("worker-0"))
        result = module.store_resource_block(
            _store_spec(
                instance_id="worker-0",
                generation=ack.server_generation,
                key=key,
                resource_id=RESOURCES[0],
                start=0,
                end=CHUNK,
                prefix_end=CHUNK,
                boundary=False,
            ),
            _token_payload(RESOURCES[0], (0,)),
        )
        assert result.status == "failed"
        assert result.error_code == "IO_ERROR"
        assert module.report_status()["blocks_stored"] == 0
    finally:
        module.close()
