# SPDX-License-Identifier: Apache-2.0
"""Rank-local boundary payloads must not share one stock object key.

Compressor boundary state is captured per TP rank: the payload carries the
rank's own ring page rows, so two ranks that collapse onto the same stock
identity (MLA folds every worker to world_size=1/worker_id=0) still produce
different bytes and even different lengths.  Boundary object keys therefore
carry the writing registration's identity; a shared key would let one rank win
and make every other rank's store fail with ``IO_ERROR`` (B01 t40/t42).

Token resources stay on the shared key: their payloads are identical across
ranks and the idempotent write introduced for duplicate workers relies on it.

These tests use the resource-block unit-test fixtures plus the stock-storage
stand-in that refuses a reservation for a resident key (like the L1 manager);
no ZMQ, vLLM or accelerator is involved.
"""

# Standard
from dataclasses import replace

# First Party
from lmcache.v1.multiprocess.modules.resource_block import ResourceBlockModule
from lmcache.v1.multiprocess.resource_block_protocol import (
    ERROR_ALREADY_PRESENT,
    ERROR_RESOURCE_MISSING,
    STATUS_MISS,
    STATUS_OK,
    ResourceBlockRegistration,
    ResourceBlockSpec,
)

# The resident-key storage stand-in reproduces the stock L1 behaviour that
# makes a shared boundary key unusable for rank-local payloads.
from tests.v1.multiprocess.test_resource_block_idempotent import (
    _ResidentStore,
    _store,
)

# Reuse the store/lookup fixtures so both halves agree on keys and hashes.
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

ENDPOINT = 2 * CHUNK
RANKS = 8


def _registration_for(index: int) -> ResourceBlockRegistration:
    """Return a registration with a distinct worker identity per rank."""
    return replace(_registration(), instance_id=f"worker-{index}")


def _spec_for(
    ack,
    key,
    resource_id: str,
    *,
    direction: str = "store",
    prefix_end: int = ENDPOINT,
    boundary: bool = True,
    start: int = 0,
    end: int = CHUNK,
) -> ResourceBlockSpec:
    """Build a spec that rides on one registration's ack identity."""
    blocks = () if boundary else tuple(range(start // CHUNK, end // CHUNK))
    return ResourceBlockSpec(
        operation_id=f"{direction}-{resource_id}-{start}-{prefix_end}",
        direction=direction,
        instance_id=ack.instance_id,
        request_id=key.request_id,
        step_id=1,
        registration_epoch=ack.registration_epoch,
        server_generation=ack.server_generation,
        plan_hash=ack.plan_hash,
        layout_hash=ack.layout_hash,
        stock_key=key,
        resource_id=resource_id,
        engine_group_id=None if boundary else 0,
        block_ids=blocks,
        start_token=start,
        end_token=end,
        prefix_end=prefix_end,
        skip_first_n_tokens=0,
        boundary={"schema": 2} if boundary else None,
    )


def _make_module(registrations: int = RANKS):
    """Register ``registrations`` workers on one provider and return both."""
    ctx = FakeContext()
    module = ResourceBlockModule(ctx, block_store=_store(_ResidentStore()))
    ctx.set_resource_block_visibility(module)
    acks = [
        module.register_resource_blocks(_registration_for(index))
        for index in range(registrations)
    ]
    return module, acks


def _store_rank_local_boundaries(module, ack, key) -> dict[tuple[int, str], bytes]:
    """Store one different-sized boundary payload per resource for a rank."""
    payloads: dict[tuple[int, str], bytes] = {}
    index = int(ack.instance_id.rsplit("-", 1)[1])
    for offset, resource_id in enumerate(BOUNDARIES):
        payload = _boundary_payload(resource_id, ENDPOINT) + bytes([index]) * (
            index + offset + 1
        )
        payloads[(index, resource_id)] = payload
        result = module.store_resource_block(
            _spec_for(ack, key, resource_id), payload
        )
        assert result.status == STATUS_OK, result
    return payloads


def _store_tokens_for_rank(module, ack, key) -> None:
    """Store every token resource chunk so the lookup endpoint can complete."""
    for chunk in range(ENDPOINT // CHUNK):
        for resource_id in RESOURCES:
            result = module.store_resource_block(
                _spec_for(
                    ack,
                    key,
                    resource_id,
                    boundary=False,
                    start=chunk * CHUNK,
                    end=(chunk + 1) * CHUNK,
                ),
                _token_payload(resource_id, (chunk,)),
            )
            assert result.status == STATUS_OK, result


def test_rank_local_boundary_payloads_are_isolated_per_registration():
    """Each rank writes and reads back its own boundary bytes."""
    module, acks = _make_module()
    key = _key(_tokens(ENDPOINT))

    payloads: dict[tuple[int, str], bytes] = {}
    for ack in acks:
        payloads.update(_store_rank_local_boundaries(module, ack, key))

    assert module.report_status()["blocks_stored"] == RANKS * len(BOUNDARIES)

    for ack in acks:
        index = int(ack.instance_id.rsplit("-", 1)[1])
        for resource_id in BOUNDARIES:
            result, payload = module.retrieve_resource_block(
                _spec_for(ack, key, resource_id, direction="load")
            )
            assert result.status == STATUS_OK, result
            assert payload == payloads[(index, resource_id)]


def test_rank_local_boundary_store_completes_the_stock_lookup_endpoint():
    """All ranks' boundary markers together keep the endpoint visible."""
    module, acks = _make_module()
    key = _key(_tokens(ENDPOINT))
    for ack in acks:
        _store_rank_local_boundaries(module, ack, key)
        _store_tokens_for_rank(module, ack, key)

    result = _lookup(module, key, ENDPOINT)
    assert result.prefix_end == ENDPOINT
    assert result.token_prefix_end == ENDPOINT
    assert result.boundary_prefix_end == ENDPOINT


def test_registration_without_boundary_payload_reads_a_clean_miss():
    """A worker that never captured the page cannot read a neighbour's bytes."""
    module, acks = _make_module(registrations=RANKS + 1)
    key = _key(_tokens(ENDPOINT))
    for ack in acks[:RANKS]:
        _store_rank_local_boundaries(module, ack, key)

    result, payload = module.retrieve_resource_block(
        _spec_for(acks[-1], key, BOUNDARIES[0], direction="load")
    )
    assert result.status == STATUS_MISS
    assert result.error_code == ERROR_RESOURCE_MISSING
    assert payload == b""


def test_duplicate_boundary_store_of_one_rank_is_idempotent():
    """Re-storing the same rank payload is reported, not written twice."""
    module, acks = _make_module(registrations=1)
    key = _key(_tokens(ENDPOINT))
    payload = _boundary_payload(BOUNDARIES[0], ENDPOINT)

    first = module.store_resource_block(
        _spec_for(acks[0], key, BOUNDARIES[0]), payload
    )
    second = module.store_resource_block(
        _spec_for(acks[0], key, BOUNDARIES[0]), payload
    )

    assert first.status == STATUS_OK
    assert second.status == STATUS_OK
    assert second.error_code == ERROR_ALREADY_PRESENT
    assert module.report_status()["blocks_stored"] == 1

    result, stored = module.retrieve_resource_block(
        _spec_for(acks[0], key, BOUNDARIES[0], direction="load")
    )
    assert result.status == STATUS_OK
    assert stored == payload


def test_token_payloads_still_share_one_object_key_across_registrations():
    """Token resources keep the shared key and stay readable by every rank."""
    module, acks = _make_module(registrations=2)
    key = _key(_tokens(CHUNK))
    payload = _token_payload(RESOURCES[0], (0,))

    for ack in acks:
        result = module.store_resource_block(
            _spec_for(
                ack,
                key,
                RESOURCES[0],
                boundary=False,
                start=0,
                end=CHUNK,
            ),
            payload,
        )
        assert result.status == STATUS_OK, result

    for ack in acks:
        result, stored = module.retrieve_resource_block(
            _spec_for(
                ack,
                key,
                RESOURCES[0],
                direction="load",
                boundary=False,
                start=0,
                end=CHUNK,
            )
        )
        assert result.status == STATUS_OK
        assert stored == payload
