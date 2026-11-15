# SPDX-License-Identifier: Apache-2.0
"""Race tests for resource-block stores against L1 eviction/clear.

``store_resource_block`` writes the payload through the stock L1 and only then
publishes the completeness marker that makes the chunk visible to the stock
prefix lookup.  An eviction/clear delivered between ``finish_write`` and that
publication used to leave a stale marker behind: the payload was gone but the
module still advertised it, so a warm hit failed with ``RESOURCE_MISSING``
inside the engine.

The module now opens an in-flight key window before the write and the deletion
listener records deletions for keys in that window; the store then rolls its
markers back instead of publishing them.  These tests force that interleaving
through a store wrapper whose ``write`` notifies the module after the payload
became visible, i.e. exactly in the vulnerable window.
"""

# Standard
from dataclasses import replace
import threading

# First Party
from lmcache.v1.multiprocess.modules.resource_block import (
    ResourceBlockModule,
    StockResourceBlockStore,
)
from lmcache.v1.multiprocess.resource_block_protocol import (
    ERROR_ALREADY_PRESENT,
    ERROR_IO_ERROR,
    STATUS_FAILED,
    STATUS_OK,
)
from tests.v1.multiprocess.test_resource_block_visibility import (
    BOUNDARIES,
    CHUNK,
    RESOURCES,
    TOKENS,
    FakeContext,
    _boundary_payload,
    _key,
    _registration,
    _spec,
    _store_boundary,
    _tokens,
    _token_payload,
)


class _RaceStore:
    """Stock store wrapper that fires a callback after every visible write."""

    def __init__(self, inner: StockResourceBlockStore) -> None:
        self._inner = inner
        self.on_written = None
        self.fail_writes = False

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def write(self, key, payload: bytes) -> bool:
        if self.fail_writes:
            return False
        written = self._inner.write(key, payload)
        if written and self.on_written is not None:
            self.on_written(key)
        return written


def _race_module(*, world: int = 1):
    """Build a module whose store hook can inject deletions after finish_write."""
    ctx = FakeContext()
    store = _RaceStore(StockResourceBlockStore(ctx.storage_manager))
    module = ResourceBlockModule(ctx, block_store=store)
    ctx.set_resource_block_visibility(module)
    module.register_resource_blocks(_registration(world=world))
    return module, store, ctx


def _store_chunk_quiet(module, key, chunk: int) -> list:
    """Store every token resource of one chunk, returning the results."""
    results = []
    for resource_id in RESOURCES:
        results.append(
            module.store_resource_block(
                _spec(
                    key,
                    resource_id,
                    start=chunk * CHUNK,
                    end=(chunk + 1) * CHUNK,
                ),
                _token_payload(resource_id, (chunk,)),
            )
        )
    return results


def _store_prefix(module, key, chunks: int = 1) -> None:
    """Store a clean prefix (token chunks plus the terminal boundary)."""
    for chunk in range(chunks):
        results = _store_chunk_quiet(module, key, chunk)
        assert all(result.status == STATUS_OK for result in results), results
    _store_boundary(module, key, chunks * CHUNK)


def _delete_all(ctx) -> None:
    """Notify the module about every object the fake L1 currently holds."""
    module = ctx.resource_block_visibility
    module.on_l1_keys_deleted(list(ctx.storage_manager.data.keys()))


def test_eviction_after_finish_write_rolls_back_the_marker():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))

    # A clean token prefix is visible once its boundary arrives.
    _store_prefix(module, key)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1

    _delete_all(ctx)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0
    assert module.report_status()["blocks_invalidated"] == 11
    # The lone boundary write opened no window of its own: only token keys and
    # the boundary key were retired through the reverse index.

    # Re-store chunk 1 with an injected eviction between finish_write and the
    # marker publication: the store must fail and publish nothing.
    store.on_written = lambda written: module.on_l1_keys_deleted([written])
    stored_before = module.report_status()["blocks_stored"]
    results = _store_chunk_quiet(module, key, 1)
    assert all(result.status == STATUS_FAILED for result in results)
    assert all(result.error_code == ERROR_IO_ERROR for result in results)
    assert module.report_status()["blocks_stored"] == stored_before
    assert module.stock_visible_chunks(key, 2, CHUNK) == 0

    # Closing the window must leave no tombstone behind: a clean re-store of
    # the full prefix publishes its markers again.
    store.on_written = None
    _store_prefix(module, key, chunks=2)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2


def test_boundary_store_race_rolls_back_the_endpoint_marker():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))
    _store_chunk_quiet(module, key, 0)
    store.on_written = lambda written: module.on_l1_keys_deleted([written])
    for resource_id in BOUNDARIES:
        result = module.store_resource_block(
            _spec(key, resource_id, prefix_end=CHUNK, boundary=True),
            _boundary_payload(resource_id, CHUNK),
        )
        assert result.status == STATUS_FAILED
        assert result.error_code == ERROR_IO_ERROR
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0

    store.on_written = None
    _store_boundary(module, key, CHUNK)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1


def test_delete_of_one_written_key_suppresses_the_whole_fanout_store():
    module, store, ctx = _race_module(world=2)
    key = _key(_tokens(TOKENS), world=2)

    deleted: set = set()

    def delete_first_key(written) -> None:
        deleted.add(written)
        module.on_l1_keys_deleted([written])

    store.on_written = delete_first_key
    result = module.store_resource_block(
        _spec(key, RESOURCES[0]),
        _token_payload(RESOURCES[0], (0,)),
    )
    # Two rank keys were written; deleting the first one is enough to roll the
    # whole store back because visibility requires every rank to be complete.
    assert len(deleted) == 1
    assert result.status == STATUS_FAILED
    assert result.error_code == ERROR_IO_ERROR
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0

    store.on_written = None
    result = module.store_resource_block(
        _spec(key, RESOURCES[0]),
        _token_payload(RESOURCES[0], (0,)),
    )
    assert result.status == STATUS_OK


def test_failed_write_does_not_publish_markers_or_leak_the_window():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))
    store.fail_writes = True
    for resource_id in RESOURCES:
        result = module.store_resource_block(
            _spec(key, resource_id), _token_payload(resource_id, (0,))
        )
        assert result.status == STATUS_FAILED
        assert result.error_code == ERROR_IO_ERROR
    assert module.report_status()["blocks_stored"] == 0
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0

    # The in-flight window must be closed by the failure path: a later clean
    # store publishes its markers.
    store.fail_writes = False
    _store_prefix(module, key)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1


def test_already_present_store_never_opens_a_write_window():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))
    _store_prefix(module, key)

    store.on_written = lambda written: module.on_l1_keys_deleted([written])
    for resource_id in RESOURCES:
        result = module.store_resource_block(
            _spec(key, resource_id), _token_payload(resource_id, (0,))
        )
        assert result.status == STATUS_OK
        assert result.error_code == ERROR_ALREADY_PRESENT
    # No write happened for the duplicates, so no injected deletion fired and
    # the prefix is still visible.
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1

    store.on_written = None
    _delete_all(ctx)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0


def test_race_counters_report_hits_and_rollbacks():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))
    store.on_written = lambda written: module.on_l1_keys_deleted([written])
    result = module.store_resource_block(
        _spec(key, RESOURCES[0]), _token_payload(RESOURCES[0], (0,))
    )
    assert result.status == STATUS_FAILED
    status = module.report_status()
    assert status["store_race_hits"] == 1
    assert status["store_race_rollbacks"] == 1
    assert status["blocks_stored"] == 0


def test_concurrent_clean_and_racing_stores_leave_no_stale_markers():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def racing_worker() -> None:
        ident = threading.get_ident()
        store.on_written = lambda written: (
            module.on_l1_keys_deleted([written])
            if threading.get_ident() == ident
            else None
        )
        barrier.wait()
        try:
            for _ in range(10):
                _store_chunk_quiet(module, key, 0)
        except BaseException as error:  # noqa: BLE001 - re-raised in the test
            errors.append(error)

    def clean_worker() -> None:
        barrier.wait()
        try:
            for _ in range(10):
                _store_chunk_quiet(module, key, 0)
        except BaseException as error:  # noqa: BLE001 - re-raised in the test
            errors.append(error)

    threads = [
        threading.Thread(target=racing_worker),
        threading.Thread(target=clean_worker),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)

    # Whatever interleaving happened, deleting every object must leave zero
    # visible chunks: a stale marker would keep advertising a gone payload.
    _delete_all(ctx)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0
    # And the module must still be able to publish a fresh prefix afterwards.
    _store_prefix(module, key)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1


def test_reregistration_during_the_race_window_does_not_publish_stale_markers():
    module, store, ctx = _race_module()
    key = _key(_tokens(TOKENS))

    def reregister_and_delete(written) -> None:
        module.register_resource_blocks(replace(_registration(epoch=2)))
        module.on_l1_keys_deleted([written])

    store.on_written = reregister_and_delete
    result = module.store_resource_block(
        _spec(key, RESOURCES[0]), _token_payload(RESOURCES[0], (0,))
    )
    assert result.status == STATUS_FAILED
    assert result.error_code == ERROR_IO_ERROR
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0
    assert module.report_status()["registered"] == 2

    store.on_written = None
    ack = module.register_resource_blocks(replace(_registration(epoch=2)))
    generation = ack.server_generation
    for resource_id in RESOURCES:
        result = module.store_resource_block(
            replace(
                _spec(key, resource_id),
                registration_epoch=2,
                server_generation=generation,
            ),
            _token_payload(resource_id, (0,)),
        )
        assert result.status == STATUS_OK, result
    for resource_id in BOUNDARIES:
        boundary = module.store_resource_block(
            replace(
                _spec(
                    key,
                    resource_id,
                    direction="store",
                    prefix_end=CHUNK,
                    boundary=True,
                ),
                registration_epoch=2,
                server_generation=generation,
            ),
            _boundary_payload(resource_id, CHUNK),
        )
        assert boundary.status == STATUS_OK, boundary
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1
