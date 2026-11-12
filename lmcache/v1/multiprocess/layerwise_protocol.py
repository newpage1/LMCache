# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project.
"""Wire contracts for layerwise KV transfer over the MP message queue.

Every struct is frozen and rejects unknown fields so a protocol mismatch
between an engine build and the LMCache server fails loudly instead of
silently dropping fields. This module is the single source of truth for the
wire types (design doc section 4.3/4.4): engine-side codecs must import the
descriptor types from here rather than redeclaring them.

The module stays importable without torch/vllm: it only depends on msgspec
and hashlib so CPU-side protocol tests can run anywhere.
"""

# Standard
import hashlib

# Third Party
import msgspec

from lmcache.v1.multiprocess.layerwise import LayerwiseTransferPlan

LAYERWISE_PROTOCOL_VERSION = 1
"""Bump on any breaking wire change; both ends must agree on it."""

BOUNDARY_STATE_SCHEMA_VERSION = 1
"""Legacy schema version of :class:`BoundaryStateDescriptor` payloads."""

BOUNDARY_STATE_SCHEMA_V2 = 2
"""Full-page boundary schema (T85-02).

A v2 descriptor additionally carries the ring geometry the captured
page came from (``ring_length`` / ``page_offset``) plus the page-local
``written_rows`` the engine actually wrote, so a restore can verify
ring integrity end-to-end before rebasing onto a new request page.
"""

#: Compression phases a boundary snapshot may record. ``0`` is the row
#: payload written at the boundary token; ``1`` is the residual carry that
#: the next ring window consumes.
BOUNDARY_COMPRESSION_PHASES = (0, 1)


def compute_boundary_checksum(
    snapshot_id: str,
    resource_id: str,
    boundary_token: str,
    ring_cursor: int,
    compression_phase: int,
    row_indices: tuple[int, ...],
    schema_version: int,
) -> str:
    """Deterministic checksum over every boundary descriptor field.

    Args:
        snapshot_id: Owning snapshot identity.
        resource_id: The scratch resource the boundary state belongs to.
        boundary_token: Engine-visible token position of the boundary.
        ring_cursor: Cursor of the source ring buffer at snapshot time.
        compression_phase: See :data:`BOUNDARY_COMPRESSION_PHASES`.
        row_indices: Physical scratch rows captured for this boundary.
        schema_version: :data:`BOUNDARY_STATE_SCHEMA_VERSION` of the writer.

    Returns:
        Hex SHA-256 over the canonical ``|``-joined field order.
    """
    canonical = "|".join(
        (
            snapshot_id,
            resource_id,
            boundary_token,
            str(ring_cursor),
            str(compression_phase),
            ",".join(str(row) for row in row_indices),
            str(schema_version),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LayerwiseRegistration(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Worker -> server registration of a layerwise transfer plan.

    Carries the full validated plan plus the registration identity the
    runtime uses to reject late callbacks from a superseded epoch.
    """

    protocol_version: int
    plan_hash: str
    layout_hash: str
    registration_epoch: int
    plan: LayerwiseTransferPlan
    cache_names: tuple[str, ...]
    vllm_rank: int
    kv_worker_id: str
    topology_hash: str

    def __post_init__(self) -> None:
        if self.protocol_version != LAYERWISE_PROTOCOL_VERSION:
            raise ValueError(
                f"layerwise protocol version {self.protocol_version} != {LAYERWISE_PROTOCOL_VERSION}"
            )
        if self.registration_epoch < 0:
            raise ValueError("registration_epoch must be nonnegative")
        if self.vllm_rank < 0:
            raise ValueError("vllm_rank must be nonnegative")
        if not self.kv_worker_id:
            raise ValueError("kv_worker_id must be non-empty")
        if len(set(self.cache_names)) != len(self.cache_names):
            raise ValueError("cache_names must be unique")


class LayerwiseRegistrationAck(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Server acknowledgment; both ends must agree on every identity hash."""

    protocol_version: int
    server_generation: int
    registration_epoch: int
    plan_hash: str
    layout_hash: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.protocol_version != LAYERWISE_PROTOCOL_VERSION:
            raise ValueError(
                f"layerwise protocol version {self.protocol_version} != {LAYERWISE_PROTOCOL_VERSION}"
            )
        if self.server_generation < 0:
            raise ValueError("server_generation must be nonnegative")
        if self.registration_epoch < 0:
            raise ValueError("registration_epoch must be nonnegative")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must be unique")


class BoundaryStateDescriptor(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One scratch boundary captured for a snapshot (wire type, read-only).

    ``row_indices`` MUST be produced from real engine metadata (slot
    mapping). Values guessed from hashes or modulo arithmetic describe rows
    that were never written and are rejected by the engine-side codec.
    """

    snapshot_id: str
    resource_id: str
    boundary_token: str
    ring_cursor: int
    compression_phase: int
    row_indices: tuple[int, ...]
    schema_version: int
    checksum: str
    #: V2 ring-geometry fields (default = absent, i.e. a v1 descriptor).
    #: See :data:`BOUNDARY_STATE_SCHEMA_V2`.
    ring_length: int = -1
    page_offset: int = -1
    written_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if not self.boundary_token:
            raise ValueError("boundary_token must be non-empty")
        if self.ring_cursor < 0:
            raise ValueError("ring_cursor must be nonnegative")
        if self.compression_phase not in BOUNDARY_COMPRESSION_PHASES:
            raise ValueError(f"compression_phase must be one of {BOUNDARY_COMPRESSION_PHASES}")
        if not self.row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 for row in self.row_indices):
            raise ValueError("row_indices must be nonnegative")
        if len(set(self.row_indices)) != len(self.row_indices):
            raise ValueError("row_indices must not repeat")
        if self.schema_version == BOUNDARY_STATE_SCHEMA_V2:
            # V2 full-page identity: the rows must span exactly one ring
            # page and the written rows must be real page-local slots.
            if self.ring_length <= 0:
                raise ValueError(
                    "v2 boundary descriptor requires ring_length > 0"
                )
            if self.page_offset < 0:
                raise ValueError(
                    "v2 boundary descriptor requires page_offset >= 0"
                )
            expected_page = tuple(
                range(self.page_offset, self.page_offset + self.ring_length)
            )
            if tuple(self.row_indices) != expected_page:
                raise ValueError(
                    "v2 boundary descriptor row_indices must span exactly "
                    f"[{self.page_offset}, {self.page_offset + self.ring_length}), "
                    f"got {self.row_indices[:4]}..."
                )
            if not self.written_rows:
                raise ValueError(
                    "v2 boundary descriptor requires non-empty written_rows"
                )
            if len(set(self.written_rows)) != len(self.written_rows):
                raise ValueError("v2 written_rows must not repeat")
            if any(
                row < 0 or row >= self.ring_length
                for row in self.written_rows
            ):
                raise ValueError(
                    "v2 written_rows must be page-local indices in "
                    f"[0, {self.ring_length})"
                )
        elif (
            self.ring_length != -1
            or self.page_offset != -1
            or self.written_rows
        ):
            raise ValueError(
                "v1 boundary descriptor must not carry v2 ring fields; "
                "emit a v2 descriptor instead"
            )
        expected = compute_boundary_checksum(
            self.snapshot_id,
            self.resource_id,
            self.boundary_token,
            self.ring_cursor,
            self.compression_phase,
            self.row_indices,
            self.schema_version,
        )
        if self.checksum != expected:
            raise ValueError(
                f"boundary descriptor checksum mismatch for {self.resource_id}"
            )


class ResourceTransferSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One resource transfer request between engine and server.

    Token-range resources carry ``engine_group_id``/``block_ids`` and no
    boundary; boundary-state resources carry ``boundary`` and no block
    identity. The two shapes are mutually exclusive and enforced here.
    """

    operation_id: str
    request_id: str
    step_id: int
    registration_epoch: int
    server_generation: int
    plan_hash: str
    layout_hash: str
    resource_id: str
    key: str
    engine_group_id: int | None
    block_ids: tuple[int, ...]
    start_token: int
    end_token: int
    skip_first_n_tokens: int
    boundary: BoundaryStateDescriptor | None

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must be non-empty")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.step_id < 0:
            raise ValueError("step_id must be nonnegative")
        if self.registration_epoch < 0:
            raise ValueError("registration_epoch must be nonnegative")
        if self.server_generation < 0:
            raise ValueError("server_generation must be nonnegative")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if not self.key:
            raise ValueError("key must be non-empty")
        if self.start_token < 0 or self.end_token < self.start_token:
            raise ValueError("token range must satisfy 0 <= start <= end")
        if self.skip_first_n_tokens < 0:
            raise ValueError("skip_first_n_tokens must be nonnegative")
        if self.boundary is None:
            if self.engine_group_id is None:
                raise ValueError("token-range spec requires engine_group_id")
            if not self.block_ids:
                raise ValueError("token-range spec requires block_ids")
            if any(block < 0 for block in self.block_ids):
                raise ValueError("block_ids must be nonnegative")
        else:
            if self.engine_group_id is not None or self.block_ids:
                raise ValueError("boundary spec must not carry block identity")


class ResourceTransferResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Terminal result of one :class:`ResourceTransferSpec`."""

    operation_id: str
    resource_id: str
    registration_epoch: int
    server_generation: int
    status: str
    error_code: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("ok", "miss", "invalid", "failed"):
            raise ValueError(f"unsupported transfer status {self.status!r}")


class LayerwiseLookupRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Scheduler-side request for a layerwise prefix lookup.

    Carries the registration identity so the server can reject a lookup
    from a stale epoch/layout instead of answering with incompatible rows.
    """

    protocol_version: int
    plan_hash: str
    layout_hash: str
    registration_epoch: int
    key: str

    def __post_init__(self) -> None:
        if self.protocol_version != LAYERWISE_PROTOCOL_VERSION:
            raise ValueError(
                f"protocol version {self.protocol_version} != "
                f"{LAYERWISE_PROTOCOL_VERSION}"
            )
        if not self.plan_hash:
            raise ValueError("plan_hash must be non-empty")
        if not self.layout_hash:
            raise ValueError("layout_hash must be non-empty")
        if self.registration_epoch < 0:
            raise ValueError("registration_epoch must be nonnegative")
        if not self.key:
            raise ValueError("key must be non-empty")


class LayerwiseLookupResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Result of a layerwise prefix lookup with a granted lease."""

    matched_end: int
    snapshot_id: str
    resource_ids: tuple[str, ...]
    lease_id: str
    plan_hash: str
    layout_hash: str
    server_generation: int

    def __post_init__(self) -> None:
        if self.matched_end < 0:
            raise ValueError("matched_end must be nonnegative")
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if not self.resource_ids:
            raise ValueError("resource_ids must be non-empty")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ValueError("resource_ids must be unique")
        if not self.lease_id:
            raise ValueError("lease_id must be non-empty")
        if self.server_generation < 0:
            raise ValueError("server_generation must be nonnegative")


class LayerwiseSnapshotCommit(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """All-or-nothing publication of one layerwise snapshot.

    A snapshot becomes visible only when every resource transfer of the
    prefix chunk succeeded and every scratch boundary arrived intact; the
    receiver enforces this before publishing.
    """

    snapshot_id: str
    key: str
    registration_epoch: int
    server_generation: int
    plan_hash: str
    layout_hash: str
    operation_ids: tuple[str, ...]
    boundaries: tuple[BoundaryStateDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if not self.key:
            raise ValueError("key must be non-empty")
        if self.registration_epoch < 0:
            raise ValueError("registration_epoch must be nonnegative")
        if self.server_generation < 0:
            raise ValueError("server_generation must be nonnegative")
        if not self.operation_ids:
            raise ValueError("operation_ids must be non-empty")
        if len(set(self.operation_ids)) != len(self.operation_ids):
            raise ValueError("operation_ids must be unique")
        for boundary in self.boundaries:
            if boundary.snapshot_id != self.snapshot_id:
                raise ValueError(
                    f"boundary {boundary.resource_id} belongs to snapshot "
                    f"{boundary.snapshot_id}, not {self.snapshot_id}"
                )
