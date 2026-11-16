# SPDX-License-Identifier: Apache-2.0
"""Default-off W1a spans for the v3 batch retrieve path.

The sink must be invisible when disabled (no clock read, no file, unchanged
payloads) and must allow a B01 trace to join client transport spans with the
server validate/hash/read/result spans by ``batch_id``/``operation_id``.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import queue
import threading

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess import lw_v3_trace
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.resource_block_protocol import (
    STATUS_OK,
    ResourceBlockBatchResult,
    ResourceBlockLookupRequest,
)
from lmcache.v1.multiprocess.rpc import get_rpc_spec
from lmcache.v1.multiprocess.transport.zmq_impl import mq
from tests.v1.multiprocess.test_resource_block import (
    BOUNDARY,
    INDEX_K,
    LONG_KV,
    _item,
    _request,
)
from tests.v1.multiprocess.test_resource_block_visibility import (
    CHUNK,
    TOKENS,
    _key,
    _module,
    _spec,
    _store_boundary,
    _store_chunk,
    _tokens,
)

_ENV_ENABLED = "DSV41_LW_V2_MASK_TRACE"
_ENV_DIR = "DSV41_LW_V2_MASK_TRACE_DIR"


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    """Enable tracing into a temp dir and reset the cached sink state."""
    monkeypatch.setenv(_ENV_ENABLED, "1")
    monkeypatch.setenv(_ENV_DIR, str(tmp_path))
    lw_v3_trace.reset_for_tests()
    yield tmp_path
    lw_v3_trace.reset_for_tests()


def _records(directory) -> list[dict]:
    path = directory / f"mask-trace-{os.getpid()}.jsonl"
    assert path.exists(), f"missing trace file {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _by_event(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["event"], []).append(record)
    return grouped


def test_disabled_sink_never_reads_the_clock_or_opens_a_file(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_ENABLED, raising=False)
    lw_v3_trace.reset_for_tests()
    calls: list[int] = []
    monkeypatch.setattr(lw_v3_trace.time, "monotonic_ns", lambda: calls.append(1) or 0)
    assert lw_v3_trace.enabled() is False
    assert lw_v3_trace.start() is None
    lw_v3_trace.emit("server", "server_hash", chunks=1)
    assert calls == []
    assert lw_v3_trace.output_path() is None


def test_enabled_sink_writes_one_jsonl_record_per_emit(trace_dir) -> None:
    start = lw_v3_trace.start()
    assert start is not None
    lw_v3_trace.emit(
        "server",
        "server_hash",
        elapsed_ns=lw_v3_trace.elapsed_since(start),
        purpose="retrieval",
        chunks=3,
        request_id="r1",
        group_id=None,
        field_status={"group_id": "unavailable"},
    )
    lw_v3_trace.emit("client", "client_rpc_wait", bytes=8)

    records = _records(trace_dir)
    assert [record["event"] for record in records] == [
        "server_hash",
        "client_rpc_wait",
    ]
    first = records[0]
    assert first["trace_source"] == "server"
    assert first["purpose"] == "retrieval"
    assert first["chunks"] == 3
    assert first["group_id"] is None
    assert first["field_status"]["group_id"] == "unavailable"
    assert first["seq"] == 1
    assert "elapsed_ns" in first
    assert first["pid"] == os.getpid()
    assert isinstance(first["tid"], int)


def test_enabled_sink_serializes_concurrent_emits(trace_dir) -> None:
    barrier = threading.Barrier(4)

    def worker(index: int) -> None:
        barrier.wait()
        for step in range(5):
            lw_v3_trace.emit("server", "server_read", batch_id=f"b{index}-{step}")

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    records = _records(trace_dir)
    assert len(records) == 20
    assert len({record["seq"] for record in records}) == 20


def test_hash_spans_split_store_retrieval_and_visibility(trace_dir) -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_boundary(module, key, CHUNK)
    result, payload = module.retrieve_resource_block(
        _spec(key, LONG_KV, direction="load")
    )
    assert result.status == STATUS_OK, result
    assert payload
    module.lookup_resource_blocks(
        ResourceBlockLookupRequest(registration_epoch=0, stock_key=key, limit=CHUNK)
    )

    hashes = _by_event(_records(trace_dir))["server_hash"]
    assert {record["purpose"] for record in hashes} == {
        "store",
        "retrieval",
        "visibility",
    }
    for record in hashes:
        assert record["chunks"] >= 1
        assert record["prefix_end"] == TOKENS
        assert record["trace_source"] == "server"
        # no prompt text or payload bytes are ever recorded
        assert "payload" not in record
        assert "tokens" not in record


def test_batch_handler_emits_validate_read_and_result_spans(trace_dir) -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    request = _request([_item(LONG_KV, chunk=0)])

    result, stream = module.retrieve_resource_blocks(request)

    assert result.status == STATUS_OK, result
    assert result.batch_id == "batch-1"
    events = _by_event(_records(trace_dir))
    validate = events["server_validate_batch"][-1]
    assert validate["batch_id"] == "batch-1"
    assert validate["items"] == 1
    assert validate["status"] == STATUS_OK
    assert validate["rank"] == 0
    read = events["server_read"][-1]
    assert read["batch_id"] == "batch-1"
    assert read["keys"] == 1
    assert read["bytes"] == len(stream)
    final = events["server_batch_result"][-1]
    assert final["batch_id"] == "batch-1"
    assert final["status"] == STATUS_OK
    assert final["bytes"] == len(stream)
    assert final["field_status"]["group_id"] == "unavailable"
    assert final["field_status"]["operation_id"] == "unavailable"
    assert final["operation_ids"] == request.items[0].operation_id


def test_single_retrieve_emits_the_read_span(trace_dir) -> None:
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    spec = _spec(key, LONG_KV, direction="load")

    result, payload = module.retrieve_resource_block(spec)

    assert result.status == STATUS_OK, result
    read = _by_event(_records(trace_dir))["server_read"][-1]
    assert read["operation_id"] == spec.operation_id
    assert read["resource_id"] == LONG_KV
    assert read["bytes"] == len(payload)
    assert read["keys"] == 1


def test_client_emits_wait_and_decode_spans(trace_dir) -> None:
    client = object.__new__(mq.MessageQueueClient)
    client._trace_submit_ns = {}
    client.pending_futures = {}
    client.pending_streams = {}
    uid = 7
    rpc_spec = get_rpc_spec("retrieve_resource_blocks")
    response = (
        ResourceBlockBatchResult(
            batch_id="batch-7",
            instance_id="worker-0",
            registration_epoch=1,
            server_generation=1,
            plan_hash="plan-1",
            layout_hash="layout-1",
            status=STATUS_OK,
        ),
        b"",
    )
    future = MessagingFuture()
    client.pending_futures[uid] = (future, rpc_spec)
    client.__dict__["_trace_submit_ns"] = {uid: lw_v3_trace.start()}
    frames = [
        mq.encode_request_uid(uid),
        mq.encode_operation("retrieve_resource_blocks"),
        msgspec.msgpack.encode(response),
    ]

    class _Socket:
        def recv_multipart(self) -> list[bytes]:
            return frames

    client.socket = _Socket()
    client.process_inbound()

    assert future.result() == response
    events = _by_event(_records(trace_dir))
    wait = events["client_rpc_wait"][-1]
    assert wait["rpc_request_uid"] == uid
    assert wait["operation"] == "retrieve_resource_blocks"
    assert wait["field_status"]["batch_id"] == "unavailable"
    decode = events["client_response_decode"][-1]
    assert decode["batch_id"] == "batch-7"
    assert decode["status"] == STATUS_OK
    assert decode["bytes"] > 0


def test_pool_and_handler_spans_wrap_the_real_handler(trace_dir) -> None:
    seen: list[bytes] = []

    def handler(payload: bytes) -> bytes:
        seen.append(payload)
        return b"response"

    entry = mq.BlockingRequestHandler([bytes], bytes, handler)
    entry.executor = ThreadPoolExecutor(max_workers=1)
    entry.__dict__["_lw_v3_trace"] = {
        "submit_ns": lw_v3_trace.start(),
        "request_uid": 11,
        "operation": "op-11",
    }
    try:
        future = entry([msgspec.msgpack.encode(b"payload")])
        assert future.result(timeout=10) == b"response"
    finally:
        entry.executor.shutdown(wait=True)

    assert seen == [b"payload"]
    events = _by_event(_records(trace_dir))
    pool = events["server_pool_start"][-1]
    assert pool["rpc_request_uid"] == 11
    assert pool["operation"] == "op-11"
    assert pool["field_status"]["request_id"] == "unavailable"
    done = events["server_handler"][-1]
    assert done["rpc_request_uid"] == 11
    assert "elapsed_ns" in done


def test_server_pack_span_measures_response_encoding(trace_dir) -> None:
    class _Notifier:
        def __init__(self) -> None:
            self.notified = 0

        def notify(self) -> None:
            self.notified += 1

    class _Entry:
        def __call__(self, payloads, affinity_key: int = 0) -> Future:
            future: Future = Future()
            future.set_result(b"ok")
            return future

        @staticmethod
        def get_response_class():
            return bytes

    server = object.__new__(mq.MessageQueueServer)
    server.output_queue = queue.Queue()
    server.__dict__["_output_efd"] = _Notifier()
    entry = _Entry()

    method = server.__class__.__dict__["_call_blocking_handler"].__get__(server)
    frames = [b"identity", mq.encode_request_uid(3), mq.encode_operation("op")]
    method(entry, [], frames)

    events = _by_event(_records(trace_dir))
    pack = events["server_pack"][-1]
    assert pack["rpc_request_uid"] == 3
    assert pack["operation"] == "op"
    assert pack["bytes"] == len(msgspec.msgpack.encode(b"ok"))
    assert pack["field_status"]["batch_id"] == "unavailable"


def test_batch_result_is_identical_with_tracing_on(trace_dir) -> None:
    """Tracing must not alter the payload stream or the item results."""
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_chunk(module, key, 1)
    _store_boundary(module, key, TOKENS)
    request = _request([_item(INDEX_K, chunk=0), _item(BOUNDARY, boundary_end=TOKENS)])

    result, stream = module.retrieve_resource_blocks(request)

    assert result.status == STATUS_OK, result
    assert [item.resource_id for item in result.items] == [INDEX_K, BOUNDARY]
    assert result.total_bytes == len(stream)
    assert result.total_bytes == sum(item.length for item in result.items)


def test_disabled_batch_path_does_not_create_a_trace_file(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(_ENV_ENABLED, raising=False)
    monkeypatch.setenv(_ENV_DIR, str(tmp_path))
    lw_v3_trace.reset_for_tests()
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)

    result, _ = module.retrieve_resource_blocks(_request([_item(LONG_KV, chunk=0)]))

    assert result.status == STATUS_OK, result
    assert list(tmp_path.iterdir()) == []
