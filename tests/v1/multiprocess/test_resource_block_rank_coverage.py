# SPDX-License-Identifier: Apache-2.0
"""Completeness must cover every KV rank a resource-block store wrote.

A stored block is written under one object key per KV rank derived from the
request's stock key.  The scheduler lookup key has no worker id, so it expands
to all ranks when it asks for a verified endpoint.  The visibility index must
therefore record completeness for every rank the write covered; recording only
the first rank makes a fanout store invisible to the scheduler even though all
rank objects exist.

These tests use the resource-block unit-test fixtures (fake storage and token
hashing); no ZMQ, vLLM or accelerator is involved.
"""

# First Party
from lmcache.v1.multiprocess.modules.resource_block import ResourceBlockModule

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
    _spec,
    _store_boundary,
    _store_chunk,
    _token_payload,
    _tokens,
)

REQUESTED_TOKENS = 2 * CHUNK


def _make_module(world: int) -> ResourceBlockModule:
    """Build a provider with one registration covering ``world`` KV ranks.

    Args:
        world: KV world size recorded by the registration.

    Returns:
        The registered module, also installed on its fake engine context.
    """
    ctx = FakeContext()
    module = ResourceBlockModule(ctx)
    ctx.set_resource_block_visibility(module)
    module.register_resource_blocks(_registration(world=world))
    return module


def _hit(module: ResourceBlockModule, key, limit: int) -> int:
    """Return the verified endpoint reported by ``lookup_resource_blocks``."""
    result = _lookup(module, key, limit)
    assert result.token_prefix_end == result.prefix_end
    assert result.boundary_prefix_end == result.prefix_end
    return result.prefix_end


def test_fanout_store_is_visible_to_worker_less_lookup() -> None:
    """A store addressed with a fanout key covers all ranks, so it must hit.

    The stock scheduler builds its key without a worker id; the store payload
    was written under every rank key.  Recording completeness only for rank 0
    made this lookup return 0 at chunk 0.
    """
    world = 8
    module = _make_module(world)
    fanout = _key(_tokens(8, 0), world=world, worker=None)
    _store_chunk(module, fanout, 0)
    _store_chunk(module, fanout, 1)
    _store_boundary(module, fanout, REQUESTED_TOKENS)

    assert _hit(module, fanout, REQUESTED_TOKENS) == REQUESTED_TOKENS


def test_fanout_store_without_endpoint_boundary_never_hits() -> None:
    """Terminal completeness still gates the endpoint, even for all ranks."""
    world = 8
    module = _make_module(world)
    fanout = _key(_tokens(8, 0), world=world, worker=None)
    _store_chunk(module, fanout, 0)
    _store_chunk(module, fanout, 1)

    assert _hit(module, fanout, REQUESTED_TOKENS) == 0


def test_single_rank_store_is_not_enough_for_worker_less_lookup() -> None:
    """One rank's shard must not satisfy a lookup that requires every rank."""
    world = 8
    module = _make_module(world)
    worker_key = _key(_tokens(8, 0), world=world, worker=0)
    _store_chunk(module, worker_key, 0)
    _store_chunk(module, worker_key, 1)
    _store_boundary(module, worker_key, REQUESTED_TOKENS)

    fanout = _key(_tokens(8, 0), world=world, worker=None)
    assert _hit(module, fanout, REQUESTED_TOKENS) == 0


def test_single_rank_world_still_hits() -> None:
    """TP=1 keeps the worker key and the fanout lookup identical."""
    module = _make_module(1)
    worker_key = _key(_tokens(8, 0), world=1, worker=0)
    _store_chunk(module, worker_key, 0)
    _store_chunk(module, worker_key, 1)
    _store_boundary(module, worker_key, REQUESTED_TOKENS)

    fanout = _key(_tokens(8, 0), world=1, worker=None)
    assert _hit(module, fanout, REQUESTED_TOKENS) == REQUESTED_TOKENS


def test_duplicate_store_reports_already_present_only_for_all_ranks() -> None:
    """A repeat fanout store is idempotent and stays visible."""
    world = 8
    module = _make_module(world)
    fanout = _key(_tokens(8, 0), world=world, worker=None)
    _store_chunk(module, fanout, 0)
    _store_boundary(module, fanout, CHUNK)
    # A second store of the same chunk may report ALREADY_PRESENT; either way
    # the block must remain visible and the endpoint must stay verified.
    module.store_resource_block(
        _spec(fanout, RESOURCES[0], start=0, end=CHUNK),
        _token_payload(RESOURCES[0], (0,)),
    )
    module.store_resource_block(
        _spec(fanout, BOUNDARIES[0], prefix_end=CHUNK, boundary=True),
        _boundary_payload(BOUNDARIES[0], CHUNK),
    )
    assert _hit(module, fanout, REQUESTED_TOKENS) == CHUNK
