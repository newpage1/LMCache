# SPDX-License-Identifier: Apache-2.0
"""Server-side resource-block transfer module for DSV4.1 layerwise v2.

Unlike the legacy ``layerwise_transfer`` module, which stages payloads and
publishes them only through an all-or-nothing snapshot commit, this module
writes every block into the stock L1 storage manager at its final key and
derives prefix completeness from those writes.  A completed store is therefore
immediately visible to the stock prefix lookup through
:meth:`ResourceBlockModule.visible_prefix`; there is no snapshot commit and no
lease.

The module keeps a lightweight completeness index because the stock
``ObjectKey`` hash is resource-namespaced and not enumerated by the attention
layout walk in ``LookupModule._all_object_keys``.  The index is rebuilt from
successful stores and only ever narrows a stock hit (see the lookup cap), so a
lost index after a restart degrades to a miss, never to a wrong hit.
"""

# Standard
from dataclasses import dataclass, field
import threading
import time
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, _lookup_kv_ranks
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.layerwise import CacheTransferResource
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.resource_block_protocol import (
    BOUNDARY_PAYLOAD_MAGIC,
    ERROR_ALREADY_PRESENT,
    ERROR_INVALID_SPEC,
    ERROR_IO_ERROR,
    ERROR_LAYOUT_MISMATCH,
    ERROR_PAYLOAD_CORRUPT,
    ERROR_PLAN_MISMATCH,
    ERROR_RESOURCE_MISSING,
    ERROR_STALE_EPOCH,
    ERROR_STALE_GENERATION,
    ERROR_UNSUPPORTED_BOUNDARY_SCHEMA,
    RESOURCE_BLOCK_CAPABILITIES,
    RESOURCE_BLOCK_PROTOCOL_VERSION,
    STATUS_FAILED,
    STATUS_MISS,
    STATUS_OK,
    TOKEN_PAYLOAD_MAGIC,
    ResourceBlockAck,
    ResourceBlockLayout,
    ResourceBlockLookupRequest,
    ResourceBlockLookupResult,
    ResourceBlockRegistration,
    ResourceBlockResult,
    ResourceBlockSpec,
    parse_payload_envelope,
    resource_block_object_keys,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

logger = init_logger(__name__)


def _l1_owners_of(objects: dict) -> dict:
    """Return the L1 manager that reserved each object (local helper)."""
    owners = {}
    for key, obj in objects.items():
        owner = obj.get_l1_manager()
        if owner is None:
            raise ValueError(f"Memory object for key {key} has no L1 owner")
        owners[key] = owner
    return owners


class StockResourceBlockStore:
    """Write/read opaque block payloads through the stock L1 manager.

    The store uses the same ``StorageManager`` path as the stock transfer
    modules: a write reserves the exact key, copies the bytes and calls
    ``finish_write``, which is the point at which the block becomes visible.
    Reads take a read lock, copy the bytes out and release it.

    Writes are idempotent for identical bytes.  Several v2 workers can share
    one stock identity (MLA replicates KV across the TP ranks), so they race
    to store the same resource block: only one ``reserve_write`` can own the
    buffer and the losers must not turn it into an IO error.  A loser verifies
    the resident payload and reports success when the bytes match; a resident
    payload that differs is never overwritten and reports failure.
    """

    #: Bounded retry budget while another writer is still committing the key.
    DEFAULT_WRITE_RETRIES = 10
    #: Delay in seconds between retries.
    DEFAULT_WRITE_RETRY_DELAY = 0.01

    def __init__(
        self,
        storage_manager: Any,
        *,
        write_retries: int = DEFAULT_WRITE_RETRIES,
        write_retry_delay: float = DEFAULT_WRITE_RETRY_DELAY,
    ) -> None:
        if write_retries < 0:
            raise ValueError("write_retries must be non-negative")
        if write_retry_delay < 0:
            raise ValueError("write_retry_delay must be non-negative")
        self._storage_manager = storage_manager
        self._write_retries = int(write_retries)
        self._write_retry_delay = float(write_retry_delay)

    def write(self, key: Any, payload: bytes) -> bool:
        """Store one payload under ``key``.

        Args:
            key: Final stock object key of the block.
            payload: Opaque block bytes.  An identical resident payload makes
                the write a successful no-op.

        Returns:
            True when ``key`` holds exactly ``payload`` after the call,
            including when another writer has just stored it; False when no
            reservation was possible and no identical resident payload exists,
            or when a different payload already occupies the key.
        """
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([len(payload)])], dtypes=[torch.uint8]
        )
        for attempt in range(self._write_retries + 1):
            reserved = self._storage_manager.reserve_write([key], layout)
            memory_obj = reserved.get(key)
            if memory_obj is not None and memory_obj.tensor is not None:
                source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                memory_obj.tensor.view(torch.uint8).copy_(source)
                self._storage_manager.finish_write([key], _l1_owners_of(reserved))
                return True
            # The key is already resident (or another thread is committing
            # it): verify the bytes instead of treating this as an IO error.
            existing = self.read(key)
            if existing is not None:
                return existing == payload
            if attempt < self._write_retries:
                time.sleep(self._write_retry_delay)
        return False

    def read(self, key: Any) -> bytes | None:
        """Return the stored bytes, or ``None`` when the key is not resident.

        ``reserve_read`` reports a granted read lock as
        ``(L1Error.SUCCESS, memory_obj)`` and an absent key as
        ``(L1Error.KEY_NOT_EXIST, None)``; the error slot is never ``None``.
        Every granted reservation is released in ``finally`` so repeated
        read-backs (for example the duplicate-write verification path) cannot
        accumulate ``read_locked`` entries.  Absent keys never took a lock and
        are simply skipped.
        """
        for manager in self._storage_manager.l1_managers():
            reserved = manager.reserve_read([key])
            error, memory_obj = reserved.get(
                key, (L1Error.KEY_NOT_EXIST, None)
            )
            if error != L1Error.SUCCESS:
                continue
            try:
                if memory_obj is None or memory_obj.tensor is None:
                    continue
                return memory_obj.tensor.view(torch.uint8).numpy().tobytes()
            finally:
                manager.finish_read([key])
        return None


@dataclass
class _Registration:
    registration: ResourceBlockRegistration
    server_generation: int
    resources: dict[str, CacheTransferResource]
    layouts: dict[str, ResourceBlockLayout]


@dataclass
class _Completeness:
    """Committed blocks of one registration identity.

    ``tokens[(resource_id, rank)]`` holds the content hash of every committed
    chunk and ``boundaries[(resource_id, rank)]`` holds ``(endpoint, hash)``
    pairs, where the hash is the chunk hash covering the endpoint.  Both are
    content-addressed: two requests that happen to write the same chunk index
    for different tokens must not make each other look present.
    """

    tokens: dict[tuple[str, int], set[bytes]] = field(default_factory=dict)
    boundaries: dict[tuple[str, int], set[tuple[int, bytes]]] = field(
        default_factory=dict
    )


class _CompletenessListener(L1ManagerListener):
    """Drop completeness markers when their stock objects leave L1.

    ``StorageManager.clear`` (the ``reset_external`` path) and LRU eviction
    delete objects behind this module's back.  The completeness index must
    follow those deletions, otherwise ``visible_prefix`` keeps advertising a
    prefix whose payloads are gone and the next load fails inside the model
    forward, where vLLM cannot recover a synchronous load.
    """

    def __init__(self, module: "ResourceBlockModule") -> None:
        self._module = module

    def on_l1_keys_reserved_read(self, keys) -> None:
        """No-op: read reservations do not change completeness."""

    def on_l1_keys_read_finished(self, keys) -> None:
        """No-op: reads do not change completeness."""

    def on_l1_keys_reserved_write(self, keys) -> None:
        """No-op: writes become complete at ``finish_write``."""

    def on_l1_keys_write_finished(self, keys) -> None:
        """No-op: stores record completeness explicitly."""

    def on_l1_keys_finish_write_and_reserve_read(self, keys) -> None:
        """No-op: stores record completeness explicitly."""

    def on_l1_keys_deleted_by_manager(self, keys) -> None:
        """Invalidate every marker whose object was deleted."""
        self._module.on_l1_keys_deleted(keys)

    def on_l1_keys_accessed(self, keys) -> None:
        """No-op: access does not change completeness."""


class ResourceBlockModule(DiscoverableModule):
    """Serve ``register_resource_blocks`` and the per-block RPC operations."""

    module_name = "resource_block"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> "ResourceBlockModule | None":
        """Build the module and register its stock-lookup visibility cap."""
        module = cls(build_ctx)
        build_ctx.engine_context.set_resource_block_visibility(module)
        return module

    def __init__(
        self,
        build_ctx: ModuleBuildContext,
        *,
        block_store: Any | None = None,
    ) -> None:
        self._ctx = build_ctx.engine_context
        self._chunk_size = int(self._ctx.chunk_size)
        self._store = block_store or StockResourceBlockStore(
            self._ctx.storage_manager
        )
        self._lock = threading.RLock()
        self._registrations: dict[str, _Registration] = {}
        self._completeness: dict[tuple, _Completeness] = {}
        # Reverse index from a stock object key to every completeness marker it
        # backs.  L1 eviction/clear deletes objects without calling this
        # module, so the deletion listener uses this index to retire markers.
        # Token object keys are content-addressed and shared by every
        # registration that stored the same chunk (MLA folds the TP ranks onto
        # one stock identity), so one key can back markers in several
        # completeness tables; the index must therefore hold an owner *set*.
        # A last-writer slot would let one eviction leave the other
        # registrations advertising a payload the storage no longer has.
        self._key_owners: dict[Any, set[tuple]] = {}
        # In-flight store windows.  A store registers the object keys it is
        # about to write *before* the payload can become visible, and closes
        # the window after the write.  The deletion listener runs under the
        # same lock, so an eviction that happens while the write is in flight
        # is recorded in ``_pending_deletions`` and the store abandons its
        # completeness markers instead of publishing them after the payload is
        # already gone.
        self._pending_store_counts: dict[Any, int] = {}
        self._pending_deletions: set[Any] = set()
        self._next_generation = 1
        self._closed = False
        self._counters = {
            "registered": 0,
            "blocks_stored": 0,
            "blocks_already_present": 0,
            "blocks_retrieved": 0,
            "blocks_invalidated": 0,
            "lookups": 0,
            "store_race_hits": 0,
            "store_race_rollbacks": 0,
        }
        self._register_l1_listeners()

    def _register_l1_listeners(self) -> None:
        """Observe L1 deletions so completeness cannot outlive its payloads."""
        managers = getattr(self._ctx.storage_manager, "l1_managers", None)
        if not callable(managers):
            return
        for manager in managers():
            register = getattr(manager, "register_listener", None)
            if not callable(register):
                continue
            try:
                register(_CompletenessListener(self))
            except Exception:  # noqa: BLE001 - visibility is best effort
                logger.warning(
                    "resource-block module could not observe L1 deletions on %r",
                    manager,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # EngineModule
    # ------------------------------------------------------------------

    @property
    def context(self) -> Any:
        """The shared server context this module was built from."""
        return self._ctx

    def report_status(self) -> dict[str, object]:
        """Expose lightweight module counters for status reporting."""
        with self._lock:
            return dict(self._counters)

    def close(self) -> None:
        """Mark the module closed; stored blocks are owned by the stock L1."""
        with self._lock:
            self._closed = True

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    @request_handler(HandlerType.BLOCKING)
    def register_resource_blocks(
        self, registration: ResourceBlockRegistration
    ) -> ResourceBlockAck:
        """Validate and accept one worker registration.

        Raises:
            ValueError: Capabilities, protocol version, resources or layout
                coverage cannot be served; the worker must fail closed.
        """
        if registration.protocol_version != RESOURCE_BLOCK_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported resource-block protocol {registration.protocol_version}"
            )
        if not registration.resources:
            raise ValueError("registration carries no transferable resources")
        layouts = {layout.resource_id: layout for layout in registration.layouts}
        if len(layouts) != len(registration.layouts):
            raise ValueError("registration repeats a resource layout")
        for resource in registration.resources:
            layout = layouts.get(resource.resource_id)
            if layout is None:
                raise ValueError(
                    f"resource {resource.resource_id} has no registered layout"
                )
            if layout.encoding not in ("token_range_v1", "boundary_schema2"):
                raise ValueError(
                    f"resource {resource.resource_id} has unsupported encoding "
                    f"{layout.encoding!r}"
                )
            if layout.max_payload_bytes <= 0:
                raise ValueError(
                    f"resource {resource.resource_id} has a non-positive payload bound"
                )
        with self._lock:
            if self._closed:
                raise ValueError("resource-block module is closed")
            previous = self._registrations.get(registration.instance_id)
            if previous is not None and (
                previous.registration.registration_epoch
                == registration.registration_epoch
            ):
                if (
                    previous.registration.plan_hash != registration.plan_hash
                    or previous.registration.layout_hash != registration.layout_hash
                ):
                    raise ValueError(
                        "registration identity changed without a new epoch"
                    )
                return self._ack(previous)
            generation = self._next_generation
            self._next_generation += 1
            state = _Registration(
                registration=registration,
                server_generation=generation,
                resources={
                    resource.resource_id: resource
                    for resource in registration.resources
                },
                layouts=layouts,
            )
            self._registrations[registration.instance_id] = state
            self._completeness = {
                key: value
                for key, value in self._completeness.items()
                if key[0] != registration.instance_id
            }
            self._key_owners = self._filter_key_owners()
            self._counters["registered"] += 1
            return self._ack(state)

    @request_handler(HandlerType.BLOCKING)
    def store_resource_block(
        self, spec: ResourceBlockSpec, payload: bytes
    ) -> ResourceBlockResult:
        """Store one opaque block and make it visible to the stock lookup.

        The write opens an in-flight key window that the L1 deletion listener
        observes: if the object is evicted or cleared before the write returns,
        the completeness markers are rolled back and the store reports an IO
        failure instead of advertising a dead payload.
        """
        try:
            state = self._validate_spec(spec, direction="store")
        except _SpecError as error:
            return self._result(spec, STATUS_FAILED, error.code)
        resource = state.resources[spec.resource_id]
        layout = state.layouts[spec.resource_id]
        if resource.storage_kind == "boundary_state":
            if spec.block_ids:
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
            expected_schema = 2
        else:
            if not spec.block_ids:
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
            expected_schema = None
        try:
            magic, header = parse_payload_envelope(payload)
        except ValueError:
            return self._result(spec, STATUS_FAILED, ERROR_PAYLOAD_CORRUPT)
        if len(payload) > layout.max_payload_bytes:
            return self._result(spec, STATUS_FAILED, ERROR_PAYLOAD_CORRUPT)
        if magic == BOUNDARY_PAYLOAD_MAGIC:
            if expected_schema is None:
                return self._result(spec, STATUS_FAILED, ERROR_UNSUPPORTED_BOUNDARY_SCHEMA)
            if header.get("schema") != 2:
                return self._result(
                    spec, STATUS_FAILED, ERROR_UNSUPPORTED_BOUNDARY_SCHEMA
                )
            if header.get("resource_id") != spec.resource_id:
                return self._result(spec, STATUS_FAILED, ERROR_PAYLOAD_CORRUPT)
            if header.get("boundary_end") != spec.prefix_end:
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
        elif magic == TOKEN_PAYLOAD_MAGIC:
            if expected_schema is not None:
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
            if header.get("resource_id") != spec.resource_id:
                return self._result(spec, STATUS_FAILED, ERROR_PAYLOAD_CORRUPT)
            if tuple(header.get("block_ids", ())) != tuple(spec.block_ids):
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
        else:
            return self._result(spec, STATUS_FAILED, ERROR_PAYLOAD_CORRUPT)

        if resource.storage_kind == "boundary_state":
            chunk_start, chunk_end = spec.prefix_end, spec.prefix_end
            if spec.prefix_end <= 0:
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
        else:
            if (
                spec.start_token % self._chunk_size
                or spec.end_token - spec.start_token != self._chunk_size
            ):
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
            chunk_start, chunk_end = spec.start_token, spec.end_token

        chunk_index = chunk_start // self._chunk_size
        ranks = _lookup_kv_ranks(spec.stock_key)
        chunk_hashes = self._chunk_hashes(spec.stock_key)
        if resource.storage_kind == "boundary_state":
            # A boundary completes the chunk ending at ``prefix_end``; use that
            # chunk's content hash so two requests with different prefixes do
            # not share an endpoint marker.
            marker_hash_index = chunk_index - 1
            if marker_hash_index < 0 or marker_hash_index >= len(chunk_hashes):
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
        else:
            marker_hash_index = chunk_index
            if marker_hash_index >= len(chunk_hashes):
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC)
        if resource.storage_kind == "boundary_state":
            try:
                keys = self._boundary_keys(spec, ranks)
            except _SpecError as error:
                return self._result(spec, STATUS_FAILED, error.code)
        else:
            keys = resource_block_object_keys(
                spec.stock_key,
                chunk_hashes[chunk_index],
                plan_hash=spec.plan_hash,
                layout_hash=spec.layout_hash,
                resource_id=spec.resource_id,
                ranks=ranks,
            )
        with self._lock:
            completeness = self._completeness_for(state, spec.stock_key)
            if resource.storage_kind == "boundary_state":
                table = completeness.boundaries
                marker = (spec.prefix_end, chunk_hashes[marker_hash_index])
            else:
                table = completeness.tokens
                marker = chunk_hashes[marker_hash_index]
            # A fanout stock key writes the same payload to every rank key, so
            # the block is only a duplicate when *every* rank already has it.
            if all(
                marker in table.get((spec.resource_id, rank), set())
                for rank in ranks
            ):
                # A duplicate store still proves this registration owns the
                # object keys; keep the eviction index complete for it.
                self._remember_key_owners(
                    keys,
                    ranks,
                    self._completeness_key(state, spec.stock_key),
                    spec.resource_id,
                    marker,
                )
                return self._result(spec, STATUS_OK, ERROR_ALREADY_PRESENT)
            completeness_key = self._completeness_key(state, spec.stock_key)
            self._open_store_window(keys)
        write_failed = False
        try:
            for key in keys:
                if not self._store.write(key, payload):
                    write_failed = True
                    break
        except Exception:  # noqa: BLE001 - surfaced as an IO failure result
            logger.exception("resource-block store failed for %s", spec.resource_id)
            write_failed = True
        finally:
            with self._lock:
                raced_keys = self._close_store_window(keys)
        if write_failed:
            return self._result(spec, STATUS_FAILED, ERROR_IO_ERROR)
        if raced_keys:
            # L1 dropped at least one written object while the store was in
            # flight; publishing completeness now would advertise a payload the
            # storage no longer holds (the fatal warm-load path).
            self._counters["store_race_hits"] += 1
            self._counters["store_race_rollbacks"] += len(raced_keys)
            logger.debug(
                "resource-block store for %s raced with L1 deletion of %d key(s);"
                " completeness markers rolled back",
                spec.resource_id,
                len(raced_keys),
            )
            return self._result(spec, STATUS_FAILED, ERROR_IO_ERROR)
        with self._lock:
            completeness = self._completeness_for(state, spec.stock_key)
            if resource.storage_kind == "boundary_state":
                table = completeness.boundaries
                marker = (spec.prefix_end, chunk_hashes[marker_hash_index])
            else:
                table = completeness.tokens
                marker = chunk_hashes[marker_hash_index]
            # Record completeness for every rank the write covered: the stock
            # lookup key has no worker id and therefore requires all ranks.
            for rank in ranks:
                table.setdefault((spec.resource_id, rank), set()).add(marker)
            # Keep the reverse index so an L1 eviction/clear can retire the
            # marker together with the object that backs it.  Token keys are
            # shared by every registration that stored the same chunk, so the
            # index keeps each of their owners instead of the last writer.
            self._remember_key_owners(
                keys,
                ranks,
                completeness_key,
                spec.resource_id,
                marker,
            )
            self._counters["blocks_stored"] += 1
        return self._result(spec, STATUS_OK, "")

    @request_handler(HandlerType.BLOCKING)
    def retrieve_resource_block(
        self, spec: ResourceBlockSpec
    ) -> tuple[ResourceBlockResult, bytes]:
        """Return one stored block payload or a typed miss."""
        try:
            state = self._validate_spec(spec, direction="load")
        except _SpecError as error:
            return self._result(spec, STATUS_FAILED, error.code), b""
        resource = state.resources[spec.resource_id]
        ranks = _lookup_kv_ranks(spec.stock_key)
        if len(ranks) != 1:
            return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC), b""
        if resource.storage_kind == "boundary_state":
            try:
                keys = self._boundary_keys(spec, ranks)
            except _SpecError as error:
                return self._result(spec, STATUS_FAILED, error.code), b""
        else:
            if (
                spec.start_token % self._chunk_size
                or spec.end_token - spec.start_token != self._chunk_size
            ):
                return self._result(spec, STATUS_FAILED, ERROR_INVALID_SPEC), b""
            chunk_hashes = self._chunk_hashes(spec.stock_key)
            chunk_index = spec.start_token // self._chunk_size
            if chunk_index >= len(chunk_hashes):
                return self._result(spec, STATUS_MISS, ERROR_RESOURCE_MISSING), b""
            keys = resource_block_object_keys(
                spec.stock_key,
                chunk_hashes[chunk_index],
                plan_hash=spec.plan_hash,
                layout_hash=spec.layout_hash,
                resource_id=spec.resource_id,
                ranks=ranks,
            )
        try:
            payload = self._store.read(keys[0])
        except Exception:  # noqa: BLE001 - surfaced as an IO failure result
            logger.exception(
                "resource-block retrieve failed for %s", spec.resource_id
            )
            return self._result(spec, STATUS_FAILED, ERROR_IO_ERROR), b""
        if payload is None:
            return self._result(spec, STATUS_MISS, ERROR_RESOURCE_MISSING), b""
        self._counters["blocks_retrieved"] += 1
        return self._result(spec, STATUS_OK, ""), payload

    @request_handler(HandlerType.BLOCKING)
    def lookup_resource_blocks(
        self, request: ResourceBlockLookupRequest
    ) -> ResourceBlockLookupResult:
        """Return the longest prefix backed by complete blocks."""
        with self._lock:
            self._counters["lookups"] += 1
            candidates = self._candidate_states(request)
            best = 0
            for state in candidates:
                best = max(best, self._visible_prefix(state, request))
            return ResourceBlockLookupResult(
                prefix_end=best,
                token_prefix_end=best,
                boundary_prefix_end=best,
            )

    def visible_prefix(
        self,
        stock_key: "IPCCacheServerKey",
        found_chunks: int,
        chunk_size: int,
    ) -> int:
        """Cap a stock hit (in chunks) to the complete resource-block prefix.

        This is the raw completeness query: it returns 0 when no resource
        block is complete, including when this server has no registration
        for ``stock_key``. Lookup code that must distinguish "v2 deployment"
        from "legacy plain-key deployment" uses :meth:`stock_visible_chunks`
        instead.

        Args:
            stock_key: Stock cache key of the request being looked up.
            found_chunks: Chunk count the caller is willing to accept.
            chunk_size: Tokens per stock chunk.

        Returns:
            The longest verified prefix in chunks, capped by ``found_chunks``.
        """
        request = ResourceBlockLookupRequest(
            registration_epoch=0,
            stock_key=stock_key,
            limit=int(found_chunks) * int(chunk_size),
        )
        with self._lock:
            best = 0
            for state in self._candidate_states(request):
                best = max(best, self._visible_prefix(state, request))
        if chunk_size <= 0:
            return 0
        return min(int(found_chunks), best // int(chunk_size))

    def stock_visible_chunks(
        self,
        stock_key: "IPCCacheServerKey",
        found_chunks: int,
        chunk_size: int,
    ) -> int | None:
        """Return the resource-block verified prefix for a stock lookup.

        Unlike :meth:`visible_prefix`, this distinguishes an inactive v2
        deployment from a v2 deployment with an incomplete prefix:

        * ``None`` -- no registration matches ``stock_key``'s model/world, so
          this server has no v2 client for the request. The caller must keep
          the stock plain-key lookup untouched.
        * ``0`` -- a v2 registration exists but no endpoint of the requested
          prefix is complete; the request must miss rather than fall back to
          plain chunk keys the v2 client never wrote.
        * ``n`` -- the longest complete resource-block endpoint, in chunks.

        The suffix namespace (plan/layout/salt) and the "token resources
        complete on every kv rank, endpoint boundary committed" rule are
        enforced by :meth:`_visible_prefix`, which also allows intermediate
        chunks without a boundary capture.

        Args:
            stock_key: Stock cache key of the request being looked up.
            found_chunks: Chunk count the caller is willing to accept.
            chunk_size: Tokens per stock chunk.

        Returns:
            The verified prefix in chunks, or ``None`` when the server has
            no v2 registration covering the key.
        """
        if chunk_size <= 0:
            return 0
        request = ResourceBlockLookupRequest(
            registration_epoch=0,
            stock_key=stock_key,
            limit=int(found_chunks) * int(chunk_size),
        )
        with self._lock:
            candidates = self._candidate_states(request)
            if not candidates:
                return None
            best = 0
            for state in candidates:
                best = max(best, self._visible_prefix(state, request))
        return min(int(found_chunks), best // int(chunk_size))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ack(self, state: _Registration) -> ResourceBlockAck:
        registration = state.registration
        return ResourceBlockAck(
            instance_id=registration.instance_id,
            registration_epoch=registration.registration_epoch,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            server_generation=state.server_generation,
            capabilities=RESOURCE_BLOCK_CAPABILITIES,
        )

    def _result(
        self, spec: ResourceBlockSpec, status: str, error_code: str
    ) -> ResourceBlockResult:
        return ResourceBlockResult(
            operation_id=spec.operation_id,
            direction=spec.direction,
            resource_id=spec.resource_id,
            registration_epoch=spec.registration_epoch,
            server_generation=spec.server_generation,
            status=status,
            error_code=error_code,
        )

    def _validate_spec(
        self, spec: ResourceBlockSpec, *, direction: str
    ) -> _Registration:
        if spec.direction != direction:
            raise _SpecError(ERROR_INVALID_SPEC)
        with self._lock:
            state = self._registrations.get(spec.instance_id)
        if state is None:
            raise _SpecError(ERROR_INVALID_SPEC)
        registration = state.registration
        if spec.registration_epoch != registration.registration_epoch:
            raise _SpecError(ERROR_STALE_EPOCH)
        if spec.server_generation != state.server_generation:
            raise _SpecError(ERROR_STALE_GENERATION)
        if spec.plan_hash != registration.plan_hash:
            # A stale/wrong plan is a distinct client bug from an invalid
            # payload shape; surface it so the worker can fail closed with a
            # precise error instead of a generic invalid spec.
            raise _SpecError(ERROR_PLAN_MISMATCH)
        if spec.layout_hash != registration.layout_hash:
            raise _SpecError(ERROR_LAYOUT_MISMATCH)
        resource = state.resources.get(spec.resource_id)
        if resource is None or resource.storage_kind == "recompute":
            raise _SpecError(ERROR_INVALID_SPEC)
        if spec.stock_key.model_name != registration.model_name:
            raise _SpecError(ERROR_INVALID_SPEC)
        if spec.engine_group_id != resource.engine_group_id:
            raise _SpecError(ERROR_INVALID_SPEC)
        return state

    def _open_store_window(self, keys: list[Any]) -> None:
        """Register in-flight object keys before their payload becomes visible.

        The caller holds ``self._lock``.  A deletion notification delivered
        while the write is in flight marks the key in
        ``_pending_deletions``; ``_close_store_window`` reports it back so the
        store does not publish completeness for an object the L1 already
        dropped.
        """
        for key in keys:
            self._pending_store_counts[key] = (
                self._pending_store_counts.get(key, 0) + 1
            )

    def _close_store_window(self, keys: list[Any]) -> set[Any]:
        """Close the in-flight window and return keys deleted during it.

        The caller holds ``self._lock``.  Tombstones are cleared once the last
        in-flight store for the key finishes, so a later store of a new object
        under the same key can publish its markers again.
        """
        deleted: set[Any] = set()
        for key in dict.fromkeys(keys):
            remaining = self._pending_store_counts.get(key, 0) - 1
            if remaining > 0:
                self._pending_store_counts[key] = remaining
            else:
                self._pending_store_counts.pop(key, None)
            if key in self._pending_deletions:
                deleted.add(key)
                if key not in self._pending_store_counts:
                    self._pending_deletions.discard(key)
        return deleted

    def _filter_key_owners(self) -> dict[Any, set[tuple]]:
        """Drop key owners whose completeness row was replaced or removed."""
        owners_by_key: dict[Any, set[tuple]] = {}
        for key, owners in self._key_owners.items():
            kept = {owner for owner in owners if owner[0] in self._completeness}
            if kept:
                owners_by_key[key] = kept
        return owners_by_key

    def _remember_key_owners(
        self,
        keys: list[Any],
        ranks: list[int],
        completeness_key: tuple,
        resource_id: str,
        marker: object,
    ) -> None:
        """Associate every written object key with this completeness marker.

        The caller holds ``self._lock``.  Keys are appended to an owner set
        instead of replacing the previous writer: token keys are shared by
        every registration that stored the same chunk, and each of them must
        be able to retire its own marker when the object is deleted.
        """
        for key, rank in zip(keys, ranks, strict=True):
            self._key_owners.setdefault(key, set()).add(
                (completeness_key, resource_id, rank, marker)
            )

    def _completeness_key(
        self, state: _Registration, stock_key: "IPCCacheServerKey"
    ) -> tuple:
        registration = state.registration
        return (
            registration.instance_id,
            registration.plan_hash,
            registration.layout_hash,
            stock_key.model_name,
            int(stock_key.world_size),
            stock_key.cache_salt,
        )

    def _completeness_for(
        self, state: _Registration, stock_key: "IPCCacheServerKey"
    ) -> _Completeness:
        key = self._completeness_key(state, stock_key)
        completeness = self._completeness.get(key)
        if completeness is None:
            completeness = _Completeness()
            self._completeness[key] = completeness
        return completeness

    def on_l1_keys_deleted(self, keys: list[Any]) -> None:
        """Retire completeness markers whose L1 objects were deleted.

        Called by the L1 manager listener on eviction, explicit deletion and
        CLEAR (including ``reset_external``).  Without this hook the module
        keeps advertising payloads the storage no longer holds, which turns a
        prefix hit into a load failure inside the model forward.  A token
        object key is shared by every registration that stored the same chunk,
        so the deletion must retire *all* of its owners before the key leaves
        the reverse index; retiring only the last writer would leave the other
        registrations advertising the evicted payload.
        """
        with self._lock:
            for key in keys:
                if key in self._pending_store_counts:
                    # The store has not published its markers yet.  Record
                    # the deletion so it can roll them back when it returns
                    # instead of advertising a payload that is already gone.
                    self._pending_deletions.add(key)
                owners = self._key_owners.pop(key, None)
                if not owners:
                    continue
                retired = False
                for completeness_key, resource_id, rank, marker in owners:
                    completeness = self._completeness.get(completeness_key)
                    if completeness is None:
                        continue
                    table = (
                        completeness.boundaries
                        if isinstance(marker, tuple)
                        else completeness.tokens
                    )
                    entries = table.get((resource_id, rank))
                    if entries is None:
                        continue
                    entries.discard(marker)
                    if not entries:
                        table.pop((resource_id, rank), None)
                    if not completeness.tokens and not completeness.boundaries:
                        self._completeness.pop(completeness_key, None)
                    retired = True
                if retired:
                    self._counters["blocks_invalidated"] += 1

    def _candidate_states(
        self, request: ResourceBlockLookupRequest
    ) -> list[_Registration]:
        key = request.stock_key
        candidates: list[_Registration] = []
        for state in self._registrations.values():
            registration = state.registration
            if registration.model_name != key.model_name:
                continue
            if registration.world_size != key.world_size:
                continue
            candidates.append(state)
        return candidates

    def _visible_prefix(
        self, state: _Registration, request: ResourceBlockLookupRequest
    ) -> int:
        key = request.stock_key
        completeness = self._completeness_for(state, key)
        chunk_hashes = self._chunk_hashes(key)
        chunks = min(len(chunk_hashes), request.limit // self._chunk_size)
        if chunks <= 0:
            return 0
        token_resources = [
            resource
            for resource in state.resources.values()
            if resource.storage_kind == "token_range"
        ]
        boundary_resources = [
            resource
            for resource in state.resources.values()
            if resource.storage_kind == "boundary_state"
        ]
        if not token_resources:
            return 0
        ranks = _lookup_kv_ranks(key)
        best = 0
        for chunk in range(chunks):
            endpoint = (chunk + 1) * self._chunk_size
            chunk_hash = chunk_hashes[chunk]
            if not all(
                chunk_hash
                in completeness.tokens.get((resource.resource_id, rank), set())
                for resource in token_resources
                for rank in ranks
            ):
                break
            if boundary_resources and not all(
                (endpoint, chunk_hash)
                in completeness.boundaries.get(
                    (resource.resource_id, rank), set()
                )
                for resource in boundary_resources
                for rank in ranks
            ):
                continue
            best = endpoint
        return min(best, request.limit)

    def _chunk_hashes(self, key: "IPCCacheServerKey") -> list[bytes]:
        return self._ctx.token_hasher.compute_packed_chunk_hashes(
            key.token_bytes, end=key.end
        )

    def _boundary_keys(
        self, spec: ResourceBlockSpec, ranks: list[int]
    ) -> list[Any]:
        if spec.prefix_end <= 0 or spec.prefix_end % self._chunk_size:
            raise _SpecError(ERROR_INVALID_SPEC)
        chunk_hashes = self._chunk_hashes(spec.stock_key)
        chunk_index = spec.prefix_end // self._chunk_size - 1
        if chunk_index < 0 or chunk_index >= len(chunk_hashes):
            raise _SpecError(ERROR_INVALID_SPEC)
        # Compressor boundary state is rank-local: every TP rank keeps its
        # own ring page and capture rows, so the payload bytes differ between
        # ranks even though MLA folds all of them onto one stock identity.
        # Namespace the object key by the writing registration so neighbours
        # cannot overwrite or fail each other's capture; the retrieve path uses
        # the same registration id and therefore reads back its own bytes.
        return resource_block_object_keys(
            spec.stock_key,
            chunk_hashes[chunk_index],
            plan_hash=spec.plan_hash,
            layout_hash=spec.layout_hash,
            resource_id=(
                f"{spec.resource_id}@{spec.prefix_end}@{spec.instance_id}"
            ),
            ranks=ranks,
        )


class _SpecError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
