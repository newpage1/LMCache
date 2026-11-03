# Layerwise Resource Manifest Prototype

`lmcache.v1.multiprocess.layerwise` describes cache ownership and execution-layer
dependencies without enabling transfers. Existing registration, RPCs, object
grouping and scheduler behavior are unchanged.

## Contract

`LayerwiseTransferPlan.layer_order` lists execution-layer names in order.
`CacheTransferResource` identifies one semantic cache owner using:

- `resource_id`: stable name, independent of an allocation address.
- `tensor_indices`: indices in the original registered cache dictionary.
- `engine_group_id`: the engine block-ID space, shared by multiple resources
  when appropriate; this is not a kernel-group or object-group index.
- `producer_layer` and ordered `consumer_layers`: compute lifetime.
- `storage_kind`: `token_range`, `boundary_state` or `recompute`.

A multi-plane registered cache is one dictionary entry. Its tensor index covers
the entire tuple; callers must not flatten it and change registration indices.
Two resources can share backing storage while holding different content.
Conversely, many execution layers can depend on one resource without duplicating
the resource or its registered tensors.

Only `token_range` resources have an engine group. `boundary_state` requires an
explicit snapshot boundary and restoration contract. `recompute` is reconstructed
by the engine. Neither scratch category is silently treated as prefix KV.

```python
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

plan = LayerwiseTransferPlan(
    layer_order=("producer", "consumer"),
    resources=(
        CacheTransferResource(
            resource_id="long_context_cache",
            tensor_indices=(0,),
            engine_group_id=0,
            producer_layer="producer",
            consumer_layers=("producer", "consumer"),
        ),
    ),
)
assert plan.required_by("consumer") == plan.resources
assert plan.written_by("consumer") == ()
assert plan.released_after("producer") == ()
```

Construction and typed MessagePack decoding reject duplicate resources/tensor
ownership, invalid groups and backward or unordered consumers.
`released_after()` reports the final compute consumer only. Reusing the physical
storage additionally requires completion of device consumption and every
asynchronous transfer reading or writing that storage.

## Next Integration Boundary

The next batch must negotiate protocol capabilities, partition objects by
resource without changing engine block spaces, and expose selected-resource
STORE/RETRIEVE completion. Only after that can a connector admit a hit into a
forward and wait for its resources before cache access. This manifest alone does
not establish cache-hit completeness, snapshot correctness or copy ordering.
