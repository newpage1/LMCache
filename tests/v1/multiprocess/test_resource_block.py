# SPDX-License-Identifier: Apache-2.0
"""Batch resource-block retrieve contract and server behaviour.

The batch API is an aggregation of the single-resource retrieve: it validates
one registration identity and one stock key, derives the chunk hashes once and
reads all accepted blocks through one batched L1 reservation.  These tests pin
that shape and the explicit partial/error outcomes.
"""

# Standard
import struct

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.modules import resource_block as resource_block_module
from lmcache.v1.multiprocess.modules.resource_block import (
    ResourceBlockModule,
    StockResourceBlockStore,
)
from lmcache.v1.multiprocess.resource_block_protocol import (
    ERROR_CANCELLED,
    ERROR_INVALID_SPEC,
    ERROR_PLAN_MISMATCH,
    ERROR_STALE_GENERATION,
    RESOURCE_BLOCK_BATCH_MAX_CHUNKS,
    RESOURCE_BLOCK_CAPABILITIES,
    RESOURCE_BLOCK_BATCH_MAX_TOTAL_BYTES,
    STATUS_FAILED,
    STATUS_INVALID,
    STATUS_MISS,
    STATUS_OK,
    STATUS_PARTIAL,
    ResourceBlockBatchItem,
    ResourceBlockBatchRequest,
    ResourceBlockBatchResult,
    ResourceBlockResult,
    ResourceBlockSpec,
)
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.rpc import get_rpc_specs
from lmcache.v1.multiprocess.transport.zmq_impl.client import ZmqMultiprocessClient
from tests.v1.multiprocess.test_resource_block_read_locks import (
    _FakeStorageManager,
    _LockingManager,
)
from tests.v1.multiprocess.test_resource_block_visibility import (
    BOUNDARIES,
    CHUNK,
    TOKENS,
    FakeContext,
    _key,
    _module,
    _registration,
    _spec,
    _store_boundary,
    _store_chunk,
    _tokens,
)

LONG_KV = "model.layers.2.self_attn.long_kv_cache"
INDEX_K = "model.layers.2.self_attn.indexer.k_cache"
BOUNDARY = BOUNDARIES[0]


def _item(
    resource_id: str,
    *,
    operation_id: str | None = None,
    chunk: int = 0,
    boundary_end: int | None = None,
    skip_first_n_tokens: int = 0,
) -> ResourceBlockBatchItem:
    if boundary_end is None:
        start = chunk * CHUNK
        end = start + CHUNK
        prefix_end = end
        engine_group_id: int | None = 0
        block_ids = (chunk,)
    else:
        start = 0
        end = CHUNK
        prefix_end = boundary_end
        engine_group_id = None
        block_ids = ()
    return ResourceBlockBatchItem(
        operation_id=operation_id or f"{resource_id}-{prefix_end}-{start}",
        resource_id=resource_id,
        engine_group_id=engine_group_id,
        block_ids=block_ids,
        start_token=start,
        end_token=end,
        prefix_end=prefix_end,
        skip_first_n_tokens=skip_first_n_tokens,
    )


def _request(
    items,
    *,
    batch_id: str = "batch-1",
    generation: int = 1,
    plan_hash: str = "plan-1",
    max_total_bytes: int = RESOURCE_BLOCK_BATCH_MAX_TOTAL_BYTES,
) -> ResourceBlockBatchRequest:
    key = _key(_tokens(TOKENS))
    return ResourceBlockBatchRequest(
        batch_id=batch_id,
        instance_id="worker-0",
        request_id=key.request_id,
        step_id=1,
        registration_epoch=1,
        server_generation=generation,
        plan_hash=plan_hash,
        layout_hash="layout-1",
        stock_key=key,
        items=tuple(items),
        max_total_bytes=max_total_bytes,
    )


def _single_payload(module, item: ResourceBlockBatchItem) -> bytes:
    key = _key(_tokens(TOKENS))
    boundary = item.engine_group_id is None
    result, payload = module.retrieve_resource_block(
        _spec(
            key,
            item.resource_id,
            direction="load",
            start=item.start_token,
            end=item.end_token,
            prefix_end=item.prefix_end,
            boundary=boundary,
        )
    )
    assert result.status == STATUS_OK, result
    return payload


def test_batch_retrieve_returns_multiple_blocks_in_one_call() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_chunk(module, key, 1)
    _store_boundary(module, key, 2 * CHUNK)
    before = module.report_status()["blocks_retrieved"]

    items = [
        _item(LONG_KV, chunk=0),
        _item(INDEX_K, chunk=0),
        _item("model.layers.8.self_attn.long_kv_cache", chunk=1),
        _item(BOUNDARY, boundary_end=2 * CHUNK),
    ]
    result, stream = module.retrieve_resource_blocks(_request(items))

    assert result.status == STATUS_OK, result
    assert result.error_code == ""
    assert len(result.items) == len(items)
    assert result.total_bytes == len(stream)
    assert module.report_status()["blocks_retrieved"] == before + len(items)
    offset = 0
    for expected_item, item_result in zip(items, result.items, strict=True):
        assert item_result.status == STATUS_OK
        assert item_result.error_code == ""
        assert item_result.offset == offset
        payload = stream[item_result.offset : item_result.offset + item_result.length]
        assert payload == _single_payload(module, expected_item)
        offset += item_result.length
    assert offset == len(stream)


def test_batch_retrieve_reports_partial_when_an_item_is_missing() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)

    items = [_item(LONG_KV, chunk=0), _item(LONG_KV, chunk=1)]
    result, stream = module.retrieve_resource_blocks(_request(items))

    assert result.status == STATUS_PARTIAL
    assert [item.status for item in result.items] == [STATUS_OK, STATUS_MISS]
    assert result.items[1].error_code == "RESOURCE_MISSING"
    assert result.items[0].length == len(stream)
    assert result.items[1].length == 0


def test_batch_retrieve_accepts_one_resource_for_distinct_chunks() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_chunk(module, key, 1)

    result, stream = module.retrieve_resource_blocks(
        _request([_item(LONG_KV, chunk=0), _item(LONG_KV, chunk=1)])
    )

    assert result.status == STATUS_OK
    assert len(result.items) == 2
    assert result.total_bytes == len(stream)


def test_batch_retrieve_rejects_duplicate_operation_and_range() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)

    duplicate_operation = [
        _item(LONG_KV, chunk=0, operation_id="same"),
        _item(INDEX_K, chunk=0, operation_id="same"),
    ]
    result, stream = module.retrieve_resource_blocks(_request(duplicate_operation))
    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )
    assert result.items == ()
    assert stream == b""

    duplicate_range = [
        _item(LONG_KV, chunk=0, operation_id="a"),
        _item(LONG_KV, chunk=0, operation_id="b"),
    ]
    result, _ = module.retrieve_resource_blocks(_request(duplicate_range))
    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )


def test_batch_retrieve_rejects_empty_batch() -> None:
    module, _ = _module()
    result, stream = module.retrieve_resource_blocks(_request([]))

    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )
    assert result.items == ()
    assert stream == b""


def test_batch_retrieve_rejects_batches_over_the_limits() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)

    oversized_items = [
        _item(LONG_KV, operation_id=f"op-{index}", chunk=index // 2)
        for index in range(65)
    ]
    result, _ = module.retrieve_resource_blocks(_request(oversized_items))
    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )

    over_budget = _request(
        [_item(LONG_KV, chunk=0)],
        max_total_bytes=RESOURCE_BLOCK_BATCH_MAX_TOTAL_BYTES + 1,
    )
    result, _ = module.retrieve_resource_blocks(over_budget)
    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )

    tiny_budget = _request([_item(LONG_KV, chunk=0)], max_total_bytes=1)
    result, _ = module.retrieve_resource_blocks(tiny_budget)
    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )


def test_batch_retrieve_rejects_oversized_stock_key_before_hashing() -> None:
    module, ctx = _module()
    calls = {"count": 0}
    original = ctx.token_hasher.compute_packed_chunk_hashes

    def counting_hashes(token_bytes, end=None):
        calls["count"] += 1
        return original(token_bytes, end=end)

    ctx.token_hasher.compute_packed_chunk_hashes = counting_hashes

    huge_tokens = struct.pack(">I", 7) * (RESOURCE_BLOCK_BATCH_MAX_CHUNKS * CHUNK + 1)
    huge_key = _key(huge_tokens)
    request = _request([_item(LONG_KV, chunk=0)])
    request = ResourceBlockBatchRequest(
        batch_id=request.batch_id,
        instance_id=request.instance_id,
        request_id=request.request_id,
        step_id=request.step_id,
        registration_epoch=request.registration_epoch,
        server_generation=request.server_generation,
        plan_hash=request.plan_hash,
        layout_hash=request.layout_hash,
        stock_key=huge_key,
        items=request.items,
    )
    result, _ = module.retrieve_resource_blocks(request)

    assert (result.status, result.error_code) == (
        STATUS_INVALID,
        ERROR_INVALID_SPEC,
    )
    assert calls["count"] == 0


def test_batch_retrieve_computes_chunk_hashes_once() -> None:
    module, ctx = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_chunk(module, key, 1)
    _store_boundary(module, key, 2 * CHUNK)
    calls = {"count": 0}
    original = ctx.token_hasher.compute_packed_chunk_hashes

    def counting_hashes(token_bytes, end=None):
        calls["count"] += 1
        return original(token_bytes, end=end)

    ctx.token_hasher.compute_packed_chunk_hashes = counting_hashes

    result, _ = module.retrieve_resource_blocks(
        _request(
            [
                _item(LONG_KV, chunk=0),
                _item(INDEX_K, chunk=0),
                _item(LONG_KV, chunk=1),
                _item(BOUNDARY, boundary_end=2 * CHUNK),
            ]
        )
    )

    assert result.status == STATUS_OK, result
    assert calls["count"] == 1


def test_batch_retrieve_reports_stale_identity() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)

    result, _ = module.retrieve_resource_blocks(
        _request([_item(LONG_KV, chunk=0)], generation=2)
    )
    assert (result.status, result.error_code) == (
        STATUS_FAILED,
        ERROR_STALE_GENERATION,
    )

    result, _ = module.retrieve_resource_blocks(
        _request([_item(LONG_KV, chunk=0)], plan_hash="other-plan")
    )
    assert (result.status, result.error_code) == (
        STATUS_FAILED,
        ERROR_PLAN_MISMATCH,
    )


def test_batch_read_releases_read_reservations_on_cancel() -> None:
    key = object()
    manager = _LockingManager(objects={key: b"payload"})
    store = StockResourceBlockStore(_FakeStorageManager([manager]))

    with pytest.raises(RuntimeError):
        store.read_many([key], cancelled=lambda: True)

    assert manager.locked_count() == 0
    assert manager.finish_read_calls == 1


def test_batch_retrieve_reports_cancelled_read(monkeypatch) -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    cancelled_error = resource_block_module._BatchCancelled

    def cancelled_read_many(keys):
        raise cancelled_error()

    monkeypatch.setattr(module._store, "read_many", cancelled_read_many)
    result, stream = module.retrieve_resource_blocks(
        _request([_item(LONG_KV, chunk=0), _item(INDEX_K, chunk=0)])
    )

    assert (result.status, result.error_code) == (
        STATUS_FAILED,
        ERROR_CANCELLED,
    )
    assert [item.status for item in result.items] == [
        STATUS_FAILED,
        STATUS_FAILED,
    ]
    assert [item.error_code for item in result.items] == [
        ERROR_CANCELLED,
        ERROR_CANCELLED,
    ]
    assert stream == b""


def test_batch_retrieve_echoes_the_batch_identity() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    result, _ = module.retrieve_resource_blocks(
        _request([_item(LONG_KV, chunk=0)], batch_id="batch-echo")
    )

    assert result.batch_id == "batch-echo"
    assert result.instance_id == "worker-0"
    assert result.registration_epoch == 1
    assert result.server_generation == 1
    assert result.plan_hash == "plan-1"
    assert result.layout_hash == "layout-1"


def test_single_resource_retrieve_stays_backward_compatible() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    missing, missing_payload = module.retrieve_resource_block(
        _spec(key, LONG_KV, direction="load")
    )
    assert missing.status == STATUS_MISS
    assert missing_payload == b""

    _store_chunk(module, key, 0)
    result, payload = module.retrieve_resource_block(
        _spec(key, LONG_KV, direction="load")
    )
    assert result.status == STATUS_OK
    batch_result, stream = module.retrieve_resource_blocks(
        _request([_item(LONG_KV, chunk=0)])
    )
    assert batch_result.status == STATUS_OK
    assert stream == payload


def test_batch_rpc_contract_is_discoverable_on_server_and_client() -> None:
    specs = get_rpc_specs()
    assert "retrieve_resource_blocks" in specs
    spec = specs["retrieve_resource_blocks"]
    assert spec.payload_types == (ResourceBlockBatchRequest,)
    assert spec.response_type == tuple[ResourceBlockBatchResult, bytes]
    assert not spec.streaming
    assert callable(getattr(ZmqMultiprocessClient, "retrieve_resource_blocks", None))
    operations = {
        handler.operation for handler in iter_request_handlers(ResourceBlockModule)
    }
    assert "retrieve_resource_blocks" in operations

    legacy = specs["retrieve_resource_block"]
    assert legacy.payload_types == (ResourceBlockSpec,)
    assert legacy.response_type == tuple[ResourceBlockResult, bytes]
    assert "retrieve_resource_block" in operations


def test_batch_wire_records_round_trip_through_msgspec() -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    request = _request([_item(LONG_KV, chunk=0)])
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(request), type=ResourceBlockBatchRequest
    )
    assert decoded == request
    result, stream = module.retrieve_resource_blocks(decoded)
    assert result.status == STATUS_OK
    encoded = msgspec.msgpack.encode((result, stream))
    decoded_result, decoded_stream = msgspec.msgpack.decode(
        encoded, type=tuple[ResourceBlockBatchResult, bytes]
    )
    assert decoded_result == result
    assert decoded_stream == stream


def test_batch_capability_is_advertised_in_the_registration_ack() -> None:
    assert "resource_blocks_batch_v1" in RESOURCE_BLOCK_CAPABILITIES
    module = ResourceBlockModule(FakeContext())
    ack = module.register_resource_blocks(_registration())
    assert "resource_blocks_batch_v1" in ack.capabilities
    assert "resource_blocks_v1" in ack.capabilities
