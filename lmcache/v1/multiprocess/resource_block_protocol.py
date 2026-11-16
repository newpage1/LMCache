# SPDX-License-Identifier: Apache-2.0
"""Wire contract for engine-neutral resource-block transfers.

The layerwise v2 connector transfers the DSV4.1 source groups as individual
resource blocks instead of an all-or-nothing snapshot.  This module is the
server-facing half of that contract: it names the operations, the request and
result records, and the capability tuple a worker must see before it starts
serving.

It is deliberately engine and model agnostic: no torch/vllm imports, no
knowledge of DSV4.1 layer names beyond the resource ids carried in a
registration.  The concrete cache layout (dtype/shape) is described by
:class:`ResourceBlockLayout`, and payload bytes stay opaque to the transport.
"""

# Standard
from dataclasses import dataclass
import hashlib
import struct
from typing import Literal

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.layerwise import CacheTransferResource

RESOURCE_BLOCK_PROTOCOL_VERSION = 1

#: Worker-visible capabilities.  A worker refuses to start when the ack omits
#: any of these: ``resource_blocks_v1`` is the transfer contract,
#: ``stock_prefix_visibility_v1`` promises the block is visible to the stock
#: prefix lookup, and ``boundary_schema2`` is the only boundary payload schema
#: the v2 codec may put on the wire.
RESOURCE_BLOCK_CAPABILITIES = (
    "resource_blocks_v1",
    "stock_prefix_visibility_v1",
    "boundary_schema2",
    "resource_blocks_batch_v1",
)

RESOURCE_BLOCK_ENCODINGS = ("token_range_v1", "boundary_schema2")

#: Hard limits for one batch retrieve round trip.  The item cap and the byte
#: budget bound the response size; the chunk cap bounds how many token hashes
#: the server is willing to derive from one ``stock_key`` (a caller-controlled
#: token span must not turn into unbounded hashing work).
RESOURCE_BLOCK_BATCH_MAX_ITEMS = 64
RESOURCE_BLOCK_BATCH_MAX_TOTAL_BYTES = 64 * 1024 * 1024
RESOURCE_BLOCK_BATCH_MAX_CHUNKS = 1 << 16

STATUS_OK = "ok"
STATUS_MISS = "miss"
STATUS_INVALID = "invalid"
STATUS_FAILED = "failed"
#: Mixed batch outcome: some items succeeded and some did not.
STATUS_PARTIAL = "partial"

ERROR_ALREADY_PRESENT = "ALREADY_PRESENT"
ERROR_RESOURCE_MISSING = "RESOURCE_MISSING"
ERROR_PREFIX_INCOMPLETE = "PREFIX_INCOMPLETE"
ERROR_INVALID_SPEC = "INVALID_SPEC"
ERROR_STALE_EPOCH = "STALE_EPOCH"
ERROR_STALE_GENERATION = "STALE_GENERATION"
ERROR_LAYOUT_MISMATCH = "LAYOUT_MISMATCH"
ERROR_PLAN_MISMATCH = "PLAN_MISMATCH"
ERROR_UNSUPPORTED_BOUNDARY_SCHEMA = "UNSUPPORTED_BOUNDARY_SCHEMA"
ERROR_PAYLOAD_CORRUPT = "PAYLOAD_CORRUPT"
ERROR_IO_ERROR = "IO_ERROR"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_CANCELLED = "CANCELLED"

TOKEN_PAYLOAD_MAGIC = b"DSV2TR1\n"
BOUNDARY_PAYLOAD_MAGIC = b"DSV2BC1\n"
_TOKEN_HEADER = struct.Struct(">I")


@dataclass(frozen=True)
class ResourceBlockLayout:
    """Registered geometry of one transferable resource.

    Attributes:
        resource_id: Stable resource identity (the registered cache name for
            token resources, the compressor state name for boundaries).
        cache_name: Engine cache name the block belongs to.
        dtype: Torch dtype spelling, e.g. ``"float32"`` or ``"bfloat16"``.
        shape: Full cache shape (block/row major), without a rank dimension.
        strides: Element strides of the full cache.
        max_payload_bytes: Hard upper bound on one encoded payload.
        encoding: ``"token_range_v1"`` or ``"boundary_schema2"``.
    """

    resource_id: str
    cache_name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    max_payload_bytes: int
    encoding: str


@dataclass(frozen=True)
class ResourceBlockRegistration:
    """Worker registration of every transferable resource and its layout."""

    protocol_version: int
    instance_id: str
    model_name: str
    world_size: int
    worker_id: int | None
    plan_hash: str
    layout_hash: str
    registration_epoch: int
    resources: tuple[CacheTransferResource, ...]
    layouts: tuple[ResourceBlockLayout, ...]


@dataclass(frozen=True)
class ResourceBlockAck:
    """Server answer to a registration; the worker validates every echo."""

    instance_id: str
    registration_epoch: int
    plan_hash: str
    layout_hash: str
    server_generation: int
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class ResourceBlockSpec:
    """One store or retrieve operation on a single resource block.

    Token-range blocks carry ``block_ids`` and a chunk-aligned
    ``[start_token, end_token)`` range.  Boundary blocks carry no block ids and
    use ``prefix_end`` as the logical endpoint they complete.
    """

    operation_id: str
    direction: Literal["load", "store"]
    instance_id: str
    request_id: str
    step_id: int
    registration_epoch: int
    server_generation: int
    plan_hash: str
    layout_hash: str
    stock_key: IPCCacheServerKey
    resource_id: str
    engine_group_id: int | None
    block_ids: tuple[int, ...]
    start_token: int
    end_token: int
    prefix_end: int
    skip_first_n_tokens: int
    boundary: dict[str, object] | None = None


@dataclass(frozen=True)
class ResourceBlockResult:
    """Per-operation outcome; the worker matches it against its pending op."""

    operation_id: str
    direction: str
    resource_id: str
    registration_epoch: int
    server_generation: int
    status: str
    error_code: str = ""

    @property
    def ok(self) -> bool:
        """Whether the operation produced a usable block."""
        return self.status == STATUS_OK


@dataclass(frozen=True)
class ResourceBlockLookupRequest:
    """Prefix lookup over committed resource blocks.

    The lookup is answered from the same stock key identity the stock prefix
    lookup uses; ``limit`` caps the returned prefix (chunk aligned).
    """

    registration_epoch: int
    stock_key: IPCCacheServerKey
    limit: int


@dataclass(frozen=True)
class ResourceBlockLookupResult:
    """Longest endpoint whose token blocks and boundary blocks are complete."""

    prefix_end: int
    token_prefix_end: int
    boundary_prefix_end: int


@dataclass(frozen=True)
class ResourceBlockBatchItem:
    """One block inside a batch retrieve.

    The batch carries the shared registration/spec identity once, so an item
    only names the resource and the chunk it wants.  ``skip_first_n_tokens``
    and ``boundary`` are part of the wire contract for adapter symmetry; the
    retrieve path does not consume them and callers pass the defaults
    (``0`` / ``None``).
    """

    operation_id: str
    resource_id: str
    engine_group_id: int | None
    block_ids: tuple[int, ...]
    start_token: int
    end_token: int
    prefix_end: int
    skip_first_n_tokens: int = 0
    boundary: dict[str, object] | None = None


@dataclass(frozen=True)
class ResourceBlockBatchRequest:
    """Retrieve several resource blocks with one stock prefix hash.

    Attributes:
        batch_id: Client-chosen identity echoed back by the server.
        instance_id: Registration the batch is validated against.
        request_id: Transfer request the batch belongs to.
        step_id: Transfer step the batch belongs to.
        registration_epoch: Expected registration epoch.
        server_generation: Expected registration generation.
        plan_hash: Expected transfer plan namespace.
        layout_hash: Expected cache layout namespace.
        stock_key: Shared stock cache key (token identity + salt/model).
        items: Ordered blocks; a resource may appear again for another chunk.
        max_total_bytes: Response budget for the concatenated payload bytes.
    """

    batch_id: str
    instance_id: str
    request_id: str
    step_id: int
    registration_epoch: int
    server_generation: int
    plan_hash: str
    layout_hash: str
    stock_key: IPCCacheServerKey
    items: tuple[ResourceBlockBatchItem, ...]
    max_total_bytes: int = RESOURCE_BLOCK_BATCH_MAX_TOTAL_BYTES


@dataclass(frozen=True)
class ResourceBlockBatchItemResult:
    """Per-item outcome with the byte slice inside the response payload."""

    operation_id: str
    resource_id: str
    engine_group_id: int | None
    start_token: int
    end_token: int
    prefix_end: int
    status: str
    error_code: str = ""
    offset: int = 0
    length: int = 0


@dataclass(frozen=True)
class ResourceBlockBatchResult:
    """Batch outcome plus the concatenated payload stream.

    ``status`` is ``"ok"`` only when every item succeeded, ``"partial"`` when
    some did, and a single miss/invalid/failed code otherwise.  ``error_code``
    is set for batch-level rejections (empty batch, stale identity, budget).
    """

    batch_id: str
    instance_id: str
    registration_epoch: int
    server_generation: int
    plan_hash: str
    layout_hash: str
    status: str
    error_code: str = ""
    total_bytes: int = 0
    items: tuple[ResourceBlockBatchItemResult, ...] = ()


def resource_block_chunk_hash(
    chunk_hash: bytes,
    *,
    plan_hash: str,
    layout_hash: str,
    resource_id: str,
) -> bytes:
    """Namespace a stock chunk hash by plan/layout/resource.

    The resource sub-key must inherit the stock object/chunk identity: same
    model, rank, salt and prefix chunk hash, plus the plan/layout namespace and
    resource id.  Hashing (instead of extending ``ObjectKey``) keeps the
    identity inside the stock hash and therefore isolates plan/layout
    generations that share a token prefix.
    """
    digest = hashlib.sha256()
    digest.update(b"\x00dsv41-lw-v2-resource-block\x00")
    digest.update(plan_hash.encode())
    digest.update(b"\x00")
    digest.update(layout_hash.encode())
    digest.update(b"\x00")
    digest.update(resource_id.encode())
    digest.update(b"\x00")
    digest.update(chunk_hash)
    return digest.digest()


def resource_block_object_keys(
    stock_key: IPCCacheServerKey,
    chunk_hash: bytes,
    *,
    plan_hash: str,
    layout_hash: str,
    resource_id: str,
    ranks: list[int],
) -> list[ObjectKey]:
    """Build one stock ``ObjectKey`` per kv rank for a resource block."""
    namespaced = resource_block_chunk_hash(
        chunk_hash,
        plan_hash=plan_hash,
        layout_hash=layout_hash,
        resource_id=resource_id,
    )
    return [
        ObjectKey(
            chunk_hash=namespaced,
            model_name=stock_key.model_name,
            kv_rank=rank,
            object_group_id=0,
            cache_salt=stock_key.cache_salt,
        )
        for rank in ranks
    ]


def parse_payload_envelope(payload: bytes) -> tuple[bytes, dict[str, object]]:
    """Split a v2 payload into its magic and decoded JSON header.

    Raises:
        ValueError: The payload is truncated, has a foreign magic, or its
            header is not a JSON object.
    """
    # Standard
    import json

    if len(payload) < len(TOKEN_PAYLOAD_MAGIC) + _TOKEN_HEADER.size:
        raise ValueError("payload is shorter than the v2 envelope prefix")
    magic = payload[: len(TOKEN_PAYLOAD_MAGIC)]
    offset = len(magic)
    (header_len,) = _TOKEN_HEADER.unpack_from(payload, offset)
    offset += _TOKEN_HEADER.size
    if header_len > 1 << 20 or offset + header_len > len(payload):
        raise ValueError("payload header length is out of range")
    try:
        header = json.loads(payload[offset : offset + header_len].decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"payload header is not valid JSON: {error}") from error
    if not isinstance(header, dict):
        raise ValueError("payload header is not a JSON object")
    return magic, header
