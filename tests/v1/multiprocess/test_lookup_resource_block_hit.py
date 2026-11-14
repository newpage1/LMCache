# SPDX-License-Identifier: Apache-2.0
"""Stock lookup driven by resource-block completeness (v2 deployments).

A v2 client writes only resource-block namespaced keys, so the stock plain
chunk keys never exist for it.  The stock lookup must therefore take its hit
from the resource-block completeness index instead of prefetching plain keys,
while plain-key deployments (no v2 registration) keep the existing behavior.
"""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    PrefetchHandle,
    PrefetchResult,
)
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.resource_block import ResourceBlockModule
from lmcache.v1.multiprocess.session import SessionManager

# Reuse the resource-block unit-test fixtures: the store side and the lookup
# side must agree on keys, chunk hashes and completeness bookkeeping.  These
# live in the same test package and are imported rather than duplicated so the
# two halves cannot drift apart.
from tests.v1.multiprocess.test_resource_block_visibility import (
    CHUNK,
    FakeContext,
    FakeTokenHasher,
    _key,
    _module,
    _store_boundary,
    _store_chunk,
    _tokens,
)

#: Two stock chunks at 4 tokens each.
REQUESTED_TOKENS = 8


def _make_lookup(
    provider: ResourceBlockModule | None,
    hasher: FakeTokenHasher,
) -> tuple[LookupModule, MagicMock, SessionManager]:
    """Build a ``LookupModule`` over a mock engine context.

    Args:
        provider: The resource-block provider the engine context exposes, or
            None to model a deployment without one.
        hasher: Token hasher shared with the provider so chunk hashes agree.

    Returns:
        The lookup module, its mock context, and the real session manager so
        assertions can read the recorded session state.
    """
    sessions = SessionManager(hasher=hasher, cleanup_interval=None)
    ctx = MagicMock()
    ctx.chunk_size = CHUNK
    ctx.token_hasher = hasher
    ctx.resource_block_visibility = provider
    ctx.session_manager = sessions
    ctx.get_read_owners.return_value = None
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req",
        total_requested_keys=1,
        submit_time=0.0,
    )
    ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
        hit_cells=[], l1_hit_cells=[], l2_hit_cells=[]
    )
    return LookupModule(ctx), ctx, sessions


def _published(ctx: MagicMock, event_type: EventType) -> list[dict]:
    """Return the metadata of published events of ``event_type``."""
    return [
        call.args[0].metadata
        for call in ctx.event_bus.publish.call_args_list
        if call.args[0].event_type == event_type
    ]


def _store_two_chunks(provider: ResourceBlockModule, store_key) -> None:
    """Store both token chunks and both endpoint boundaries."""
    _store_chunk(provider, store_key, 0)
    _store_boundary(provider, store_key, CHUNK)
    _store_chunk(provider, store_key, 1)
    _store_boundary(provider, store_key, 2 * CHUNK)


def test_stock_visible_chunks_distinguishes_registration_from_completeness():
    """None means "no v2 client"; 0 means "v2 client but incomplete prefix"."""
    provider = ResourceBlockModule(FakeContext())
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-none")

    assert provider.stock_visible_chunks(key, 2, CHUNK) is None
    # The raw completeness query cannot tell the two cases apart.
    assert provider.visible_prefix(key, 2, CHUNK) == 0


def test_stock_visible_chunks_tracks_tokens_and_endpoint_boundaries():
    """A chunk only counts once its tokens and its endpoint boundary are stored."""
    provider, fake_ctx = _module()
    store_key = _key(_tokens(REQUESTED_TOKENS), request_id="r-prov")
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-prov")

    assert provider.stock_visible_chunks(key, 2, CHUNK) == 0
    _store_chunk(provider, store_key, 0)
    # Token resources alone cannot serve the endpoint: the boundary is missing.
    assert provider.stock_visible_chunks(key, 2, CHUNK) == 0
    _store_boundary(provider, store_key, CHUNK)
    assert provider.stock_visible_chunks(key, 2, CHUNK) == 1
    _store_chunk(provider, store_key, 1)
    assert provider.stock_visible_chunks(key, 2, CHUNK) == 1
    _store_boundary(provider, store_key, 2 * CHUNK)
    assert provider.stock_visible_chunks(key, 2, CHUNK) == 2
    # The caller's candidate length still caps the endpoint.
    assert provider.stock_visible_chunks(key, 1, CHUNK) == 1
    assert fake_ctx.resource_block_visibility is provider


def test_lookup_reports_endpoint_without_plain_key_prefetch():
    """A verified resource-block endpoint is the stock hit; no plain keys exist."""
    provider, fake_ctx = _module()
    store_key = _key(_tokens(REQUESTED_TOKENS), request_id="r-hit")
    _store_two_chunks(provider, store_key)
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-hit")
    lookup, ctx, sessions = _make_lookup(provider, fake_ctx.token_hasher)

    lookup.lookup(key, tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_not_called()
    assert lookup.query_prefetch_status("r-hit") == 2
    session = sessions.get("r-hit")
    assert session is not None
    assert session.prefetch_hit_chunks == 2
    assert session.prefetch_synthetic is True

    (metadata,) = _published(ctx, EventType.MP_LOOKUP_PREFETCH_END)
    assert metadata["found_count"] == 2
    assert metadata["requested_tokens"] == REQUESTED_TOKENS
    assert metadata["hit_tokens"] == 2 * CHUNK
    assert metadata["resource_blocks_synthetic"] is True
    assert metadata["early_exit_reason"] == ""


def test_lookup_waits_for_synthetic_endpoint_without_storage_polling():
    """Waiting on a synthetic hit must not poll the storage manager."""
    provider, fake_ctx = _module()
    key = _key(_tokens(REQUESTED_TOKENS), request_id="r-wait")
    _store_two_chunks(provider, _key(_tokens(REQUESTED_TOKENS), request_id="r-wait"))
    lookup, ctx, _ = _make_lookup(provider, fake_ctx.token_hasher)

    lookup.lookup(key, tp_size=1)

    assert lookup.wait_prefetch_status("r-wait", timeout=1.0) == 2
    ctx.storage_manager.wait_prefetch_status.assert_not_called()
    ctx.storage_manager.query_prefetch_status.assert_not_called()


def test_lookup_after_external_reset_still_hits():
    """``/reset_prefix_cache(reset_external=False)`` clears no v2 blocks.

    A second request with the same prompt must still see the verified
    resource-block endpoint through the original stock lookup.
    """
    provider, fake_ctx = _module()
    tokens = _tokens(REQUESTED_TOKENS)
    _store_two_chunks(provider, _key(tokens, request_id="r-first"))
    first = _key(tokens, worker=None, request_id="r-first")
    second = _key(tokens, worker=None, request_id="r-second")
    lookup, ctx, _ = _make_lookup(provider, fake_ctx.token_hasher)

    lookup.lookup(first, tp_size=1)
    assert lookup.query_prefetch_status("r-first") == 2

    # The external reset only clears the engine's own prefix cache; the
    # resource-block index and its payloads stay committed.
    assert provider.stock_visible_chunks(second, 2, CHUNK) == 2
    lookup.lookup(second, tp_size=1)
    assert lookup.query_prefetch_status("r-second") == 2
    ctx.storage_manager.submit_prefetch_task.assert_not_called()


def test_lookup_falls_back_to_shorter_verified_endpoint():
    """An uncommitted terminal boundary yields the previous complete endpoint."""
    provider, fake_ctx = _module()
    store_key = _key(_tokens(REQUESTED_TOKENS), request_id="r-short")
    _store_chunk(provider, store_key, 0)
    _store_boundary(provider, store_key, CHUNK)
    _store_chunk(provider, store_key, 1)  # tokens past the last boundary
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-short")
    lookup, ctx, _ = _make_lookup(provider, fake_ctx.token_hasher)

    lookup.lookup(key, tp_size=1)

    # The whole prompt is 2 chunks, but only the first chunk has a committed
    # endpoint boundary: the lookup must report the shorter verified endpoint.
    assert lookup.query_prefetch_status("r-short") == 1
    ctx.storage_manager.submit_prefetch_task.assert_not_called()


def test_lookup_misses_when_no_endpoint_is_complete():
    """A registration with nothing stored is a clean miss, not a plain-key query."""
    provider, fake_ctx = _module()
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-miss")
    lookup, ctx, sessions = _make_lookup(provider, fake_ctx.token_hasher)

    lookup.lookup(key, tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_not_called()
    assert lookup.query_prefetch_status("r-miss") == 0
    session = sessions.get("r-miss")
    assert session is not None
    assert session.prefetch_hit_chunks == 0
    assert session.prefetch_synthetic is False
    (metadata,) = _published(ctx, EventType.MP_LOOKUP_PREFETCH_END)
    assert metadata["early_exit_reason"] == "resource_blocks_incomplete"


def test_lookup_without_registration_keeps_plain_key_prefetch():
    """A server without a v2 registration must keep the legacy stock path."""
    provider = ResourceBlockModule(FakeContext())
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-legacy")
    lookup, ctx, _ = _make_lookup(provider, FakeTokenHasher())

    lookup.lookup(key, tp_size=1)

    # The legacy path submits a plain-key prefetch; the synthetic path never
    # would, so this is the observable difference.
    ctx.storage_manager.submit_prefetch_task.assert_called_once()


def test_free_lookup_locks_skips_plain_release_for_synthetic_hit():
    """The synthetic hit holds no plain-key read locks to release."""
    provider, fake_ctx = _module()
    _store_two_chunks(provider, _key(_tokens(REQUESTED_TOKENS), request_id="r-free"))
    key = _key(_tokens(REQUESTED_TOKENS), worker=None, request_id="r-free")
    lookup, ctx, sessions = _make_lookup(provider, fake_ctx.token_hasher)

    lookup.lookup(key, tp_size=1)
    assert lookup.query_prefetch_status("r-free") == 2

    lookup.free_lookup_locks(key, tp_size=1)

    ctx.storage_manager.finish_read_prefetched.assert_not_called()
    session = sessions.get("r-free")
    assert session is not None
    assert session.prefetch_synthetic is True
