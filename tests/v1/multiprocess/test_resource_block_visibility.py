# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the engine-neutral resource-block server contract.

These tests exercise the module through its public RPC methods with fake
storage, token hashing and context objects; no ZMQ, vLLM or accelerator is
involved.  The stock lookup side is covered by
``FakeContext.resource_block_visibility`` plus the provider cap.
"""

# Standard
from dataclasses import dataclass, replace
import hashlib
import json
import struct

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.layerwise import CacheTransferResource
from lmcache.v1.multiprocess.modules.resource_block import ResourceBlockModule
from lmcache.v1.multiprocess.resource_block_protocol import (
    BOUNDARY_PAYLOAD_MAGIC,
    ERROR_PAYLOAD_CORRUPT,
    ERROR_PLAN_MISMATCH,
    ERROR_STALE_GENERATION,
    ERROR_UNSUPPORTED_BOUNDARY_SCHEMA,
    STATUS_FAILED,
    STATUS_MISS,
    STATUS_OK,
    TOKEN_PAYLOAD_MAGIC,
    ResourceBlockLayout,
    ResourceBlockLookupRequest,
    ResourceBlockRegistration,
    ResourceBlockSpec,
)

MODEL = "dsv41-test"
CHUNK = 4
TOKENS = 8
RESOURCES = (
    "model.layers.2.self_attn.long_kv_cache",
    "model.layers.2.self_attn.indexer.k_cache",
    "model.layers.8.self_attn.long_kv_cache",
    "model.layers.8.self_attn.indexer.k_cache",
    "model.layers.14.self_attn.long_kv_cache",
    "model.layers.14.self_attn.indexer.k_cache",
    "model.layers.20.self_attn.long_kv_cache",
    "model.layers.20.self_attn.indexer.k_cache",
)
BOUNDARIES = (
    "model.layers.2.self_attn.compressor.state_cache",
    "model.layers.8.self_attn.compressor.state_cache",
    "model.layers.14.self_attn.compressor.state_cache",
)


class FakeTokenHasher:
    """Cumulative-hash stand-in for ``TokenHasher`` (chunk = ``CHUNK`` tokens)."""

    chunk_size = CHUNK

    def compute_packed_chunk_hashes(self, token_bytes: bytes, end=None):
        stride = 4
        total = len(token_bytes) // stride
        limit = total if end is None else min(total, int(end))
        return [
            hashlib.sha256(token_bytes[: (i + 1) * CHUNK * stride]).digest()
            for i in range(limit // CHUNK)
        ]


class FakeMemoryObj:
    def __init__(self, size: int) -> None:
        self.tensor = torch.zeros(size, dtype=torch.uint8)
        self._l1_manager = 0

    def get_l1_manager(self) -> int:
        """Return the fake L1 owner id used by ``finish_write``."""
        return self._l1_manager


class FakeL1:
    def __init__(self, storage) -> None:
        self._storage = storage
        self._listeners = []

    def register_listener(self, listener) -> None:
        self._listeners.append(listener)

    def _notify_deleted(self, keys) -> None:
        for listener in self._listeners:
            listener.on_l1_keys_deleted_by_manager(keys)

    def clear(self, force: bool = False) -> None:
        keys = list(self._storage)
        self._storage.clear()
        self._notify_deleted(keys)

    def delete_keys(self, keys) -> None:
        deleted = [key for key in keys if self._storage.pop(key, None) is not None]
        self._notify_deleted(deleted)

    def reserve_read(self, keys):
        result = {}
        for key in keys:
            payload = self._storage.get(key)
            if payload is None:
                result[key] = (L1Error.KEY_NOT_EXIST, None)
            else:
                obj = FakeMemoryObj(len(payload))
                obj.tensor.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
                result[key] = (L1Error.SUCCESS, obj)
        return result

    def finish_read(self, keys):
        return {key: None for key in keys}


class FakeStorageManager:
    def __init__(self) -> None:
        self.data: dict[object, bytes] = {}
        self.reserved: dict[object, FakeMemoryObj] = {}
        self._l1 = FakeL1(self.data)

    def reserve_write(self, keys, layout_desc):
        result = {}
        for key in keys:
            size = int(layout_desc.shapes[0][0])
            obj = FakeMemoryObj(size)
            self.reserved[key] = obj
            result[key] = obj
        return result

    def finish_write(self, keys, owners):
        for key in keys:
            self.data[key] = self.reserved.pop(key).tensor.numpy().tobytes()

    def l1_managers(self):
        return [self._l1]

    def clear(self, force: bool = False) -> None:
        self._l1.clear(force=force)

    def delete_keys(self, keys) -> None:
        self._l1.delete_keys(keys)


@dataclass
class FakeContext:
    chunk_size: int = CHUNK

    def __post_init__(self):
        self.token_hasher = FakeTokenHasher()
        self.storage_manager = FakeStorageManager()
        self.visibility = None
        self.engine_context = self

    def set_resource_block_visibility(self, provider):
        self.visibility = provider

    @property
    def resource_block_visibility(self):
        return self.visibility


def _tokens(count: int, suffix: int = 0) -> bytes:
    values = []
    for index in range(count):
        token = 1000 + index if index < TOKENS else 5000 + suffix + index
        values.append(struct.pack(">I", token))
    return b"".join(values)


def _key(tokens: bytes, *, salt: str = "", request_id: str = "r1", start=0, end=None,
         world=1, worker=0) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=world,
        worker_id=worker,
        token_bytes=tokens,
        start=start,
        end=len(tokens) // 4 if end is None else end,
        request_id=request_id,
        cache_salt=salt,
        num_kv_readers=1,
    )


def _layout(resource_id: str, kind: str) -> ResourceBlockLayout:
    return ResourceBlockLayout(
        resource_id=resource_id,
        cache_name=resource_id,
        dtype="float32",
        shape=(32, 2, 3),
        strides=(6, 3, 1),
        max_payload_bytes=1 << 20,
        encoding="boundary_schema2" if kind == "boundary_state" else "token_range_v1",
    )


def _registration(epoch: int = 1, *, world=1) -> ResourceBlockRegistration:
    resources = tuple(
        CacheTransferResource(
            resource_id=resource_id,
            tensor_indices=(index,),
            engine_group_id=0 if kind == "token_range" else None,
            producer_layer="model.layers.2.self_attn.attn",
            consumer_layers=("model.layers.2.self_attn.attn",),
            storage_kind=kind,
        )
        for index, (resource_id, kind) in enumerate(
            [(rid, "token_range") for rid in RESOURCES]
            + [(rid, "boundary_state") for rid in BOUNDARIES]
        )
    )
    layouts = tuple(
        _layout(resource_id, kind)
        for resource_id, kind in [(rid, "token_range") for rid in RESOURCES]
        + [(rid, "boundary_state") for rid in BOUNDARIES]
    )
    return ResourceBlockRegistration(
        protocol_version=1,
        instance_id="worker-0",
        model_name=MODEL,
        world_size=world,
        worker_id=0,
        plan_hash="plan-1",
        layout_hash="layout-1",
        registration_epoch=epoch,
        resources=resources,
        layouts=layouts,
    )


def _envelope(magic: bytes, header: dict, payload: bytes = b"") -> bytes:
    encoded = json.dumps(header).encode()
    return magic + struct.pack(">I", len(encoded)) + encoded + payload


def _token_payload(resource_id: str, blocks=(0,)) -> bytes:
    return _envelope(
        TOKEN_PAYLOAD_MAGIC,
        {"schema": "token_range_v1", "resource_id": resource_id, "block_ids": list(blocks)},
        b"\x01\x02\x03",
    )


def _boundary_payload(resource_id: str, endpoint: int, *, schema: int = 2) -> bytes:
    return _envelope(
        BOUNDARY_PAYLOAD_MAGIC,
        {
            "schema": schema,
            "resource_id": resource_id,
            "boundary_end": endpoint,
            "ring_length": 4,
            "compression_phase": 1,
        },
        b"\x00" * 16,
    )


def _module() -> tuple[ResourceBlockModule, FakeContext]:
    ctx = FakeContext()
    module = ResourceBlockModule(ctx)
    ctx.set_resource_block_visibility(module)
    module.register_resource_blocks(_registration())
    return module, ctx


def _spec(
    key: IPCCacheServerKey,
    resource_id: str,
    *,
    direction="store",
    start=0,
    end=CHUNK,
    prefix_end=None,
    generation=1,
    boundary=False,
) -> ResourceBlockSpec:
    blocks = () if boundary else tuple(range(start // CHUNK, end // CHUNK))
    return ResourceBlockSpec(
        operation_id=f"{direction}-{resource_id}-{start}",
        direction=direction,
        instance_id="worker-0",
        request_id=key.request_id,
        step_id=1,
        registration_epoch=1,
        server_generation=generation,
        plan_hash="plan-1",
        layout_hash="layout-1",
        stock_key=key,
        resource_id=resource_id,
        engine_group_id=None if boundary else 0,
        block_ids=blocks,
        start_token=start,
        end_token=end,
        prefix_end=CHUNK if prefix_end is None else prefix_end,
        skip_first_n_tokens=0,
        boundary={"schema": 2} if boundary else None,
    )


def _store_chunk(module, key, chunk: int, *, value: int = 1) -> None:
    for resource_id in RESOURCES:
        payload = _token_payload(resource_id, (chunk,)) + bytes([value])
        result = module.store_resource_block(
            _spec(key, resource_id, start=chunk * CHUNK, end=(chunk + 1) * CHUNK),
            payload,
        )
        assert result.status == STATUS_OK, result


def _store_boundary(module, key, endpoint: int) -> None:
    for resource_id in BOUNDARIES:
        result = module.store_resource_block(
            _spec(
                key,
                resource_id,
                prefix_end=endpoint,
                boundary=True,
                start=endpoint,
                end=endpoint,
            ),
            _boundary_payload(resource_id, endpoint),
        )
        assert result.status == STATUS_OK, result


def _lookup(module, key, limit: int):
    return module.lookup_resource_blocks(
        ResourceBlockLookupRequest(registration_epoch=1, stock_key=key, limit=limit)
    )


def test_registration_ack_echoes_identity_and_capabilities():
    module, _ = _module()
    ack = module.register_resource_blocks(_registration())
    assert ack.instance_id == "worker-0"
    assert ack.registration_epoch == 1
    assert ack.plan_hash == "plan-1"
    assert ack.layout_hash == "layout-1"
    assert "resource_blocks_v1" in ack.capabilities
    assert "stock_prefix_visibility_v1" in ack.capabilities
    assert "boundary_schema2" in ack.capabilities


def test_lookup_requires_token_and_endpoint_boundary_completeness():
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    # Token blocks of chunk 0 only: no boundary yet, so no restorable prefix.
    assert _lookup(module, key, TOKENS).prefix_end == 0
    _store_chunk(module, key, 1)
    assert _lookup(module, key, TOKENS).prefix_end == 0
    _store_boundary(module, key, CHUNK)
    assert _lookup(module, key, TOKENS).prefix_end == CHUNK
    _store_boundary(module, key, 2 * CHUNK)
    assert _lookup(module, key, TOKENS).prefix_end == 2 * CHUNK
    # The intermediate chunk has no boundary; the terminal one is complete.
    assert _lookup(module, key, TOKENS).boundary_prefix_end == 2 * CHUNK


def test_missing_one_rank_or_resource_never_hits():
    module, _ = _module()
    key = _key(_tokens(TOKENS), world=2)
    registration = _registration(world=2)
    module.register_resource_blocks(registration)
    _store_boundary(module, key, CHUNK)
    for resource_id in RESOURCES[:-1]:
        module.store_resource_block(
            _spec(key, resource_id, start=0, end=CHUNK), _token_payload(resource_id)
        )
    assert _lookup(module, key, TOKENS).prefix_end == 0


def test_store_is_visible_to_stock_lookup_provider_and_retrievable():
    module, ctx = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_boundary(module, key, CHUNK)
    assert ctx.resource_block_visibility is module
    assert module.visible_prefix(key, 2, CHUNK) == 1
    result, payload = module.retrieve_resource_block(
        _spec(key, RESOURCES[0], direction="load", start=0, end=CHUNK)
    )
    assert result.status == STATUS_OK
    assert payload.startswith(TOKEN_PAYLOAD_MAGIC)
    assert payload.endswith(b"\x01")


def test_shared_prefix_different_suffix_and_request_hits():
    module, _ = _module()
    first = _key(_tokens(TOKENS), request_id="r1")
    _store_chunk(module, first, 0)
    _store_boundary(module, first, CHUNK)
    second = _key(_tokens(TOKENS, suffix=1), request_id="r2")
    # Same stock prefix chunks, different request id and suffix.
    assert _lookup(module, second, TOKENS).prefix_end == CHUNK
    result, _ = module.retrieve_resource_block(
        _spec(second, RESOURCES[0], direction="load", start=0, end=CHUNK)
    )
    assert result.status == STATUS_OK


def test_salt_and_model_isolate_blocks():
    module, _ = _module()
    key = _key(_tokens(TOKENS), salt="tenant-a")
    _store_chunk(module, key, 0)
    _store_boundary(module, key, CHUNK)
    other_salt = _key(_tokens(TOKENS), salt="tenant-b")
    assert _lookup(module, other_salt, TOKENS).prefix_end == 0
    model_mismatch = IPCCacheServerKey(
        model_name="other-model",
        world_size=1,
        worker_id=0,
        token_bytes=key.token_bytes,
        start=0,
        end=TOKENS,
        request_id="r1",
        cache_salt="tenant-a",
        num_kv_readers=1,
    )
    assert _lookup(module, model_mismatch, TOKENS).prefix_end == 0


def test_schema1_and_tampered_boundary_are_rejected():
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    result = module.store_resource_block(
        _spec(key, BOUNDARIES[0], boundary=True, prefix_end=CHUNK, start=CHUNK, end=CHUNK),
        _boundary_payload(BOUNDARIES[0], CHUNK, schema=1),
    )
    assert result.status == STATUS_FAILED
    assert result.error_code == ERROR_UNSUPPORTED_BOUNDARY_SCHEMA
    result = module.store_resource_block(
        _spec(key, RESOURCES[0], start=0, end=CHUNK),
        TOKEN_PAYLOAD_MAGIC + b"\x00\x00\x00\x7f{}",
    )
    assert result.status == STATUS_FAILED
    assert result.error_code == ERROR_PAYLOAD_CORRUPT


def test_stale_generation_store_is_rejected_and_miss_reports():
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    stale = module.store_resource_block(
        _spec(key, RESOURCES[0], start=0, end=CHUNK, generation=99),
        _token_payload(RESOURCES[0]),
    )
    assert stale.status == STATUS_FAILED
    assert stale.error_code == ERROR_STALE_GENERATION
    miss, payload = module.retrieve_resource_block(
        _spec(key, RESOURCES[0], direction="load", start=0, end=CHUNK)
    )
    assert miss.status == STATUS_MISS
    assert payload == b""


def test_plan_hash_mismatch_is_plan_mismatch():
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    spec = _spec(key, RESOURCES[0], start=0, end=CHUNK)
    wrong_plan = replace(spec, plan_hash="sha256:not-this-plan")

    stored = module.store_resource_block(
        wrong_plan, _token_payload(RESOURCES[0])
    )
    assert stored.status == STATUS_FAILED
    assert stored.error_code == ERROR_PLAN_MISMATCH

    miss, payload = module.retrieve_resource_block(
        replace(wrong_plan, direction="load")
    )
    assert miss.status == STATUS_FAILED
    assert miss.error_code == ERROR_PLAN_MISMATCH
    assert payload == b""


def test_duplicate_store_is_idempotent():
    module, _ = _module()
    key = _key(_tokens(TOKENS))
    spec = _spec(key, RESOURCES[0], start=0, end=CHUNK)
    payload = _token_payload(RESOURCES[0])
    assert module.store_resource_block(spec, payload).status == STATUS_OK
    again = module.store_resource_block(spec, payload)
    assert again.status == STATUS_OK
    assert again.error_code == "ALREADY_PRESENT"


def test_rpc_payloads_roundtrip_through_msgspec():
    """The new RPC dataclasses must survive the msgspec wire format."""
    # Third Party
    import msgspec

    registration = _registration()
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(registration), type=ResourceBlockRegistration
    )
    assert decoded == registration

    key = _key(_tokens(TOKENS))
    spec = _spec(key, RESOURCES[0], start=0, end=CHUNK)
    decoded_spec = msgspec.msgpack.decode(
        msgspec.msgpack.encode(spec), type=ResourceBlockSpec
    )
    assert decoded_spec.stock_key == key
    assert decoded_spec.block_ids == (0,)


def _distinct_tokens(count: int, base: int = 7000) -> bytes:
    return b"".join(struct.pack(">I", base + index) for index in range(count))


def test_same_chunk_index_for_different_tokens_is_written():
    """t37: completeness is content-addressed, not chunk-index addressed."""
    module, _ = _module()
    first = _key(_tokens(TOKENS))
    first_spec = _spec(first, RESOURCES[0], start=0, end=CHUNK)
    assert module.store_resource_block(first_spec, _token_payload(RESOURCES[0])).status == STATUS_OK

    other = _key(_distinct_tokens(TOKENS), request_id="r2")
    other_spec = _spec(other, RESOURCES[0], start=0, end=CHUNK)
    result = module.store_resource_block(other_spec, _token_payload(RESOURCES[0]))
    assert result.status == STATUS_OK
    assert result.error_code == ""
    assert module._counters["blocks_stored"] == 2


def test_visible_prefix_is_scoped_to_the_request_prefix():
    """t37: a stored prefix must not advertise a different token prefix."""
    module, _ = _module()
    stored = _key(_tokens(TOKENS))
    for resource_id in RESOURCES:
        module.store_resource_block(
            _spec(stored, resource_id, start=0, end=CHUNK),
            _token_payload(resource_id),
        )
    _store_boundary(module, stored, CHUNK)
    assert _lookup(module, stored, TOKENS).prefix_end == CHUNK

    unseen = _key(_distinct_tokens(TOKENS), request_id="r2")
    assert _lookup(module, unseen, TOKENS).prefix_end == 0


def test_clear_retires_completeness_markers():
    """t56: CLEAR/reset_external must not leave a visible but empty prefix."""
    module, ctx = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_chunk(module, key, 1)
    _store_boundary(module, key, 2 * CHUNK)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    ctx.storage_manager.clear(force=False)

    assert module.stock_visible_chunks(key, 2, CHUNK) == 0
    result, payload = module.retrieve_resource_block(
        _spec(key, RESOURCES[0], direction="load", start=0, end=CHUNK)
    )
    assert result.status == STATUS_MISS
    assert payload == b""
    assert module.report_status()["blocks_invalidated"] > 0

    # A re-store after the clear becomes visible again from scratch.
    _store_chunk(module, key, 0)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 0
    _store_boundary(module, key, CHUNK)
    assert module.stock_visible_chunks(key, 1, CHUNK) == 1


def test_evicting_one_token_key_drops_only_its_marker():
    """t56: LRU eviction invalidates exactly the deleted object's marker."""
    module, ctx = _module()
    key = _key(_tokens(TOKENS))
    _store_chunk(module, key, 0)
    _store_chunk(module, key, 1)
    _store_boundary(module, key, 2 * CHUNK)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 2

    victim = next(iter(ctx.storage_manager.data))
    ctx.storage_manager.delete_keys([victim])

    assert module.stock_visible_chunks(key, 2, CHUNK) < 2

    ctx.storage_manager.clear(force=True)
    assert module.stock_visible_chunks(key, 2, CHUNK) == 0

