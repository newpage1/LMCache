# SPDX-License-Identifier: Apache-2.0
# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)


def _plan() -> LayerwiseTransferPlan:
    return LayerwiseTransferPlan(
        layer_order=("layer20", "layer24", "layer26", "layer39"),
        resources=(
            CacheTransferResource(
                "long20",
                (0,),
                0,
                "layer20",
                ("layer20", "layer24", "layer26", "layer39"),
            ),
            CacheTransferResource(
                "index20", (1,), 0, "layer20", ("layer20", "layer24")
            ),
            CacheTransferResource("swa26", (2,), 1, "layer26", ("layer26",)),
            CacheTransferResource(
                "ring20", (3,), None, "layer20", ("layer20",), "boundary_state"
            ),
        ),
    )


def test_plan_roundtrip_preserves_resource_lifetimes_and_block_spaces() -> None:
    plan = _plan()
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(plan), type=LayerwiseTransferPlan
    )
    assert decoded == plan
    assert [resource.resource_id for resource in decoded.required_by("layer26")] == [
        "long20",
        "swa26",
    ]
    assert [resource.engine_group_id for resource in decoded.written_by("layer20")] == [
        0,
        0,
        None,
    ]
    assert [resource.resource_id for resource in decoded.released_after("layer24")] == [
        "index20"
    ]
    assert [resource.resource_id for resource in decoded.released_after("layer39")] == [
        "long20"
    ]


def test_shared_resource_is_selected_once_per_layer() -> None:
    plan = _plan()
    assert [resource.resource_id for resource in plan.required_by("layer39")] == [
        "long20"
    ]
    assert (
        len(
            [
                resource
                for resource in plan.resources
                if resource.resource_id == "long20"
            ]
        )
        == 1
    )
    assert plan.required_by("layer20")[-1].storage_kind == "boundary_state"


def test_unknown_execution_layer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown execution layer"):
        _plan().required_by("layer99")


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        (CacheTransferResource("missing", (), 0, "layer20", ("layer20",)), "tensor"),
        (
            CacheTransferResource("backwards", (4,), 0, "layer39", ("layer20",)),
            "before its producer",
        ),
        (
            CacheTransferResource("unknown", (4,), 0, "layer20", ("layer99",)),
            "Unknown execution layer",
        ),
        (
            CacheTransferResource(
                "unordered", (4,), 0, "layer20", ("layer26", "layer24")
            ),
            "execution order",
        ),
        (
            CacheTransferResource("negative", (-1,), 0, "layer20", ("layer20",)),
            "tensor",
        ),
        (
            CacheTransferResource("no_group", (4,), None, "layer20", ("layer20",)),
            "engine group",
        ),
    ],
)
def test_invalid_resource_rejected(
    resource: CacheTransferResource, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LayerwiseTransferPlan(_plan().layer_order, (resource,))


def test_tensor_cannot_be_owned_by_two_different_resources() -> None:
    original = _plan()
    duplicate = CacheTransferResource("other", (0,), 0, "layer20", ("layer20",))
    with pytest.raises(ValueError, match="tensor.*multiple resources"):
        LayerwiseTransferPlan(original.layer_order, (*original.resources, duplicate))


def test_invalid_wire_plan_is_validated_on_decode() -> None:
    payload = {
        "layer_order": ("producer", "consumer"),
        "resources": (
            {
                "resource_id": "late_write",
                "tensor_indices": (0,),
                "engine_group_id": 0,
                "producer_layer": "consumer",
                "consumer_layers": ("producer",),
            },
        ),
    }
    with pytest.raises(msgspec.ValidationError, match="before its producer"):
        msgspec.msgpack.decode(
            msgspec.msgpack.encode(payload), type=LayerwiseTransferPlan
        )


@pytest.mark.parametrize("layers", [(), ("same", "same"), ("",)])
def test_execution_order_requires_unique_nonempty_layers(
    layers: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="Execution layer"):
        LayerwiseTransferPlan(layers, ())


def test_recompute_resources_cannot_masquerade_as_token_range_cache() -> None:
    resource = CacheTransferResource(
        "replayed", (0,), 0, "producer", ("producer",), "recompute"
    )
    with pytest.raises(ValueError, match="Scratch/replay state"):
        LayerwiseTransferPlan(("producer",), (resource,))
