# SPDX-License-Identifier: Apache-2.0
"""Regression tests for resource-block eviction ownership.

The stock token object key is content-addressed and shared by every
registration that stored the same chunk (MLA folds the TP ranks onto one
stock identity).  Evicting that key must therefore retire the completeness
marker of *every* registration that still advertises it; retiring only the
last writer leaves the other registrations reporting a prefix the storage no
longer holds, which turns a warm hit into a load failure inside the engine.

These tests exercise the module through its public RPC methods and the L1
deletion listener only; no ZMQ, vLLM or accelerator is involved.
"""

# Standard
from dataclasses import replace
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.modules.resource_block import ResourceBlockModule
from lmcache.v1.multiprocess.resource_block_protocol import (
    ERROR_ALREADY_PRESENT,
    STATUS_OK,
    resource_block_object_keys,
)
from tests.v1.multiprocess.test_resource_block_read_locks import (
    _RealStorageManager,
    _real_l1,
)
from tests.v1.multiprocess.test_resource_block_visibility import (
    BOUNDARIES,
    FakeContext,
    CHUNK,
    FakeTokenHasher,
    RESOURCES,
    _boundary_payload,
    _key,
    _module,
    _registration,
    _spec,
    _store_boundary,
    _store_chunk,
    _token_payload,
    _tokens,
)

PLAN_HASH = "plan-1"
LAYOUT_HASH = "layout-1"


def _registration_for(index: int, *, epoch: int = 1):
    """Return a registration that mimics TP rank ``index``."""
    return replace(
        _registration(epoch=epoch),
        instance_id=f"worker-{index}",
    )


def _spec_for(
    key,
    resource_id: str,
    *,
    instance_id: str,
    generation: int,
    direction: str = "store",
    start: int = 0,
    end: int = CHUNK,
    prefix_end: int | None = None,
    boundary: bool = False,
):
    """Build one spec addressed to a specific registration."""
    return replace(
        _spec(
            key,
            resource_id,
            direction=direction,
            start=start,
            end=end,
            prefix_end=prefix_end,
            boundary=boundary,
        ),
        instance_id=instance_id,
        server_generation=generation,
    )


def _store_chunk_for(
    module, key, chunk: int, *, instance_id: str, generation: int
) -> None:
    for resource_id in RESOURCES:
        payload = _token_payload(resource_id, (chunk,)) + bytes([chunk + 1])
        result = module.store_resource_block(
            _spec_for(
                key,
                resource_id,
                instance_id=instance_id,
                generation=generation,
                start=chunk * CHUNK,
                end=(chunk + 1) * CHUNK,
            ),
            payload,
        )
        assert result.status == STATUS_OK, result


def _store_boundary_for(
    module, key, endpoint: int, *, instance_id: str, generation: int
) -> None:
    for resource_id in BOUNDARIES:
        result = module.store_resource_block(
            _spec_for(
                key,
                resource_id,
                instance_id=instance_id,
                generation=generation,
                prefix_end=endpoint,
                boundary=True,
                start=endpoint,
                end=endpoint,
            ),
            _boundary_payload(resource_id, endpoint),
        )
        assert result.status == STATUS_OK, result


def _store_prefix_for(
    module, key, *, instance_id: str, generation: int, chunks: int = 2
) -> None:
    for chunk in range(chunks):
        _store_chunk_for(
            module, key, chunk, instance_id=instance_id, generation=generation
        )
    for chunk in range(1, chunks + 1):
        _store_boundary_for(
            module,
            key,
            chunk * CHUNK,
            instance_id=instance_id,
            generation=generation,
        )


def _kv_ranks(key) -> list:
    """Return the stock kv ranks a worker-specific key addresses."""
    return [
        ObjectKey.ComputeKVRank(
            world_size=key.world_size,
            global_rank=key.worker_id,
            local_world_size=key.world_size,
            local_rank=key.worker_id,
        )
    ]


def _chunk_hashes(key) -> list[bytes]:
    hasher = FakeTokenHasher()
    return hasher.compute_packed_chunk_hashes(key.token_bytes, end=key.end)


def _token_key(key, chunk: int, resource_id: str):
    chunk_hash = _chunk_hashes(key)[chunk]
    keys = resource_block_object_keys(
        key,
        chunk_hash,
        plan_hash=PLAN_HASH,
        layout_hash=LAYOUT_HASH,
        resource_id=resource_id,
        ranks=_kv_ranks(key),
    )
    return keys[0]


def _boundary_key(key, resource_id: str, endpoint: int, instance_id: str):
    chunk_hash = _chunk_hashes(key)[endpoint // CHUNK - 1]
    keys = resource_block_object_keys(
        key,
        chunk_hash,
        plan_hash=PLAN_HASH,
        layout_hash=LAYOUT_HASH,
        resource_id=f"{resource_id}@{endpoint}@{instance_id}",
        ranks=_kv_ranks(key),
    )
    return keys[0]


def test_evicting_a_shared_token_key_retires_every_registration():
    """TP8 shape: one content-addressed token key backs two registrations."""
    module, ctx = _module()
    second = module.register_resource_blocks(_registration_for(1))
    key = _key(_tokens(8))

    _store_prefix_for(module, key, instance_id="worker-0", generation=1)
    _store_prefix_for(
        module, key, instance_id="worker-1", generation=second.server_generation
    )
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    victim = _token_key(key, 0, RESOURCES[0])
    ctx.storage_manager.delete_keys([victim])

    # Before the fix only the last writer was retired and the surviving
    # registration kept advertising the evicted chunk.
    assert module.stock_visible_chunks(key, 2, CHUNK) == 0

    # Re-storing the chunk from both registrations makes the prefix whole.
    _store_chunk_for(module, key, 0, instance_id="worker-0", generation=1)
    _store_chunk_for(
        module, key, 0, instance_id="worker-1", generation=second.server_generation
    )
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2


def test_eviction_retires_all_owner_rows_not_only_the_last_writer():
    """The delete must retire both tables, in either store order."""
    module, ctx = _module()
    second = module.register_resource_blocks(_registration_for(1))
    key = _key(_tokens(8))

    first_key = _token_key(key, 0, RESOURCES[0])
    _store_chunk_for(
        module, key, 0, instance_id="worker-1", generation=second.server_generation
    )
    _store_chunk_for(
        module, key, 1, instance_id="worker-1", generation=second.server_generation
    )
    _store_boundary_for(
        module, key, CHUNK, instance_id="worker-1", generation=second.server_generation
    )
    _store_boundary_for(
        module,
        key,
        2 * CHUNK,
        instance_id="worker-1",
        generation=second.server_generation,
    )
    _store_prefix_for(module, key, instance_id="worker-0", generation=1)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    ctx.storage_manager.delete_keys([first_key])
    assert module.stock_visible_chunks(key, 2, CHUNK) == 0


def test_boundary_keys_are_scoped_to_the_capturing_registration():
    """A boundary object key of one rank must not retire the neighbour's."""
    module, ctx = _module()
    second = module.register_resource_blocks(_registration_for(1))
    key = _key(_tokens(8))

    _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=1)
    _store_prefix_for(
        module,
        key,
        instance_id="worker-1",
        generation=second.server_generation,
        chunks=1,
    )
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1

    ctx.storage_manager.delete_keys(
        [_boundary_key(key, BOUNDARIES[0], CHUNK, "worker-0")]
    )
    # worker-1 still has its own boundary payload.
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1

    ctx.storage_manager.delete_keys(
        [_boundary_key(key, BOUNDARIES[0], CHUNK, "worker-1")]
    )
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0


def test_duplicate_store_keeps_the_eviction_owner():
    """ALREADY_PRESENT must not drop the reverse-index entry."""
    module, ctx = _module()
    key = _key(_tokens(8))
    _store_chunk(module, key, 0)
    _store_boundary(module, key, CHUNK)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1

    duplicate = module.store_resource_block(
        _spec(key, RESOURCES[0], start=0, end=CHUNK),
        _token_payload(RESOURCES[0]),
    )
    assert duplicate.status == STATUS_OK
    assert duplicate.error_code == ERROR_ALREADY_PRESENT

    ctx.storage_manager.delete_keys([_token_key(key, 0, RESOURCES[0])])
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0


def test_reregistration_drops_old_owners_and_keeps_neighbours():
    """A new epoch must filter only the re-registered instance's owners."""
    module, ctx = _module()
    second = module.register_resource_blocks(_registration_for(1))
    key = _key(_tokens(8))
    _store_prefix_for(module, key, instance_id="worker-0", generation=1)
    _store_prefix_for(
        module, key, instance_id="worker-1", generation=second.server_generation
    )
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    third = module.register_resource_blocks(_registration_for(0, epoch=2))
    assert third.server_generation != 1
    # worker-0 lost its markers, worker-1 still backs the whole prefix.
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    ctx.storage_manager.delete_keys([_token_key(key, 0, RESOURCES[0])])
    assert module.stock_visible_chunks(key, 2, CHUNK) == 0


def test_single_registration_clear_and_restore():
    """Single-rank CLEAR/eviction/re-store still behaves as before."""
    module, ctx = _module()
    key = _key(_tokens(8))
    _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=2)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    ctx.storage_manager.clear(force=True)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 0

    _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=1)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1

    ctx.storage_manager.delete_keys(
        [_boundary_key(key, BOUNDARIES[0], CHUNK, "worker-0")]
    )
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0

    _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=1)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1


def test_duplicate_keys_in_one_notification_are_idempotent():
    """The listener may report the same key twice; retirement is idempotent."""
    module, ctx = _module()
    key = _key(_tokens(8))
    _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=1)
    victim = _token_key(key, 0, RESOURCES[0])

    module.on_l1_keys_deleted([victim, victim, victim])

    assert module.stock_visible_chunks(key, 1, CHUNK) == 0


def test_concurrent_store_and_eviction_leave_no_stale_owners():
    """Store/evict churn must not deadlock or keep advertising dead keys."""
    module, ctx = _module()
    second = module.register_resource_blocks(_registration_for(1))
    key = _key(_tokens(8))
    _store_prefix_for(module, key, instance_id="worker-0", generation=1)
    _store_prefix_for(
        module, key, instance_id="worker-1", generation=second.server_generation
    )
    victim = _token_key(key, 0, RESOURCES[0])

    workers = 4
    barrier = threading.Barrier(workers)
    store_guard = threading.Lock()
    failures: list[BaseException] = []

    def churn(instance_id: str, generation: int) -> None:
        try:
            barrier.wait(timeout=30)
            for _ in range(20):
                with store_guard:
                    _store_chunk_for(
                        module, key, 0, instance_id=instance_id, generation=generation
                    )
                ctx.storage_manager.delete_keys([victim])
        except BaseException as error:  # noqa: BLE001 - surfaced after join
            failures.append(error)

    threads = [
        threading.Thread(target=churn, args=("worker-0", 1)),
        threading.Thread(target=churn, args=("worker-1", second.server_generation)),
        threading.Thread(target=churn, args=("worker-0", 1)),
        threading.Thread(target=churn, args=("worker-1", second.server_generation)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "store/eviction churn deadlocked"

    assert failures == []
    assert module.report_status()["blocks_invalidated"] > 0

    _store_prefix_for(module, key, instance_id="worker-0", generation=1)
    _store_prefix_for(
        module, key, instance_id="worker-1", generation=second.server_generation
    )
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2


def test_real_l1_delete_and_clear_notify_every_registration():
    """Real L1 delete/clear paths retire the shared token marker for all ranks."""
    with _real_l1() as manager:
        ctx = FakeContext()
        ctx.storage_manager = _RealStorageManager(manager)
        module = ResourceBlockModule(ctx)
        ctx.set_resource_block_visibility(module)
        module.register_resource_blocks(_registration())
        second = module.register_resource_blocks(_registration_for(1))
        key = _key(_tokens(8))

        _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=1)
        _store_prefix_for(
            module,
            key,
            instance_id="worker-1",
            generation=second.server_generation,
            chunks=1,
        )
        assert module.stock_visible_chunks(key, 1, CHUNK) == 1

        victim = _token_key(key, 0, RESOURCES[0])
        deleted = manager.delete([victim], force=True)
        assert deleted[victim] == L1Error.SUCCESS
        assert module.stock_visible_chunks(key, 1, CHUNK) == 0

        _store_prefix_for(module, key, instance_id="worker-0", generation=1, chunks=1)
        _store_prefix_for(
            module,
            key,
            instance_id="worker-1",
            generation=second.server_generation,
            chunks=1,
        )
        assert module.stock_visible_chunks(key, 1, CHUNK) == 1

        manager.clear(force=True)
        assert module.stock_visible_chunks(key, 1, CHUNK) == 0
