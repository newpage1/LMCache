# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral cache ownership for planning layerwise MP transfers.

These descriptors do not enable transfers or change the existing MP RPCs.
Completion of asynchronous copies must be fenced separately from the last
execution-layer consumer reported by a plan.
"""

# Standard
from typing import Literal

# Third Party
import msgspec


class CacheTransferResource(msgspec.Struct, frozen=True):
    """One cache owner and its ordered execution-layer consumers.

    Tensor indices address the original registration, including excluded scratch
    tensors. ``engine_group_id`` is the engine's block-ID address space, not a
    kernel-group or object-group index. Boundary state has no token-range group
    and requires a separate snapshot contract before it can be transferred.
    ``recompute`` resources also have no transfer group and must be reconstructed
    by the engine's replay path.
    """

    resource_id: str
    tensor_indices: tuple[int, ...]
    engine_group_id: int | None
    producer_layer: str
    consumer_layers: tuple[str, ...]
    storage_kind: Literal["token_range", "boundary_state", "recompute"] = "token_range"


class LayerwiseTransferPlan(msgspec.Struct, frozen=True):
    """Immutable registration manifest for a sequential layer execution order.

    Each registered tensor belongs to at most one resource. Distinct resources
    may use the same engine group or backing allocation without becoming aliases
    of the same content. Validation also runs when decoding a msgspec payload.

    Raises:
        ValueError: If layers, resources, tensor ownership, block spaces or
            producer/consumer execution order are inconsistent.
    """

    layer_order: tuple[str, ...]
    resources: tuple[CacheTransferResource, ...]

    def __post_init__(self) -> None:
        if not self.layer_order or len(set(self.layer_order)) != len(self.layer_order):
            raise ValueError("Execution layers must be nonempty and unique")
        if any(not layer for layer in self.layer_order):
            raise ValueError("Execution layer names must be nonempty")
        positions = {layer: index for index, layer in enumerate(self.layer_order)}
        resource_ids: set[str] = set()
        owned_tensors: set[int] = set()
        for resource in self.resources:
            if not resource.resource_id or resource.resource_id in resource_ids:
                raise ValueError("Resource IDs must be nonempty and unique")
            resource_ids.add(resource.resource_id)
            indices = resource.tensor_indices
            if not indices or any(index < 0 for index in indices):
                raise ValueError(
                    "Resource tensor indices must be nonempty and nonnegative"
                )
            if len(set(indices)) != len(indices) or owned_tensors.intersection(indices):
                raise ValueError("A registered tensor belongs to multiple resources")
            owned_tensors.update(indices)
            if resource.storage_kind not in (
                "token_range",
                "boundary_state",
                "recompute",
            ):
                raise ValueError(f"Unknown storage kind: {resource.storage_kind}")
            if resource.storage_kind == "token_range":
                if resource.engine_group_id is None or resource.engine_group_id < 0:
                    raise ValueError(
                        "Token-range resources need a nonnegative engine group"
                    )
            elif resource.engine_group_id is not None:
                raise ValueError(
                    "Scratch/replay state must not use a token-range engine group"
                )
            names = (resource.producer_layer, *resource.consumer_layers)
            for layer in names:
                if layer not in positions:
                    raise ValueError(f"Unknown execution layer: {layer}")
            consumers = tuple(positions[layer] for layer in resource.consumer_layers)
            if not consumers or tuple(sorted(set(consumers))) != consumers:
                raise ValueError(
                    "Consumers must be nonempty and unique in execution order"
                )
            if consumers[0] < positions[resource.producer_layer]:
                raise ValueError("A resource is consumed before its producer")

    def required_by(self, layer_name: str) -> tuple[CacheTransferResource, ...]:
        """Select resources read or written by a layer, once per resource.

        Args:
            layer_name: Execution-layer name from ``layer_order``.

        Returns:
            Resources in manifest order, including write destinations.

        Raises:
            ValueError: If the execution layer is unknown.
        """
        self._check_layer(layer_name)
        return tuple(
            resource
            for resource in self.resources
            if layer_name == resource.producer_layer
            or layer_name in resource.consumer_layers
        )

    def written_by(self, layer_name: str) -> tuple[CacheTransferResource, ...]:
        """Select a layer's owned write destinations.

        Args:
            layer_name: Execution-layer name from ``layer_order``.

        Returns:
            Resources produced by this layer in manifest order.

        Raises:
            ValueError: If the execution layer is unknown.
        """
        self._check_layer(layer_name)
        return tuple(
            resource
            for resource in self.resources
            if resource.producer_layer == layer_name
        )

    def released_after(self, layer_name: str) -> tuple[CacheTransferResource, ...]:
        """Select resources whose final compute consumer is this layer.

        This is a compute lifetime boundary, not permission to overwrite an
        in-flight copy's source or destination. The caller must fence device
        consumption and asynchronous transfers before reusing physical storage.

        Args:
            layer_name: Execution-layer name from ``layer_order``.

        Returns:
            Resources whose final consumer is this layer.

        Raises:
            ValueError: If the execution layer is unknown.
        """
        self._check_layer(layer_name)
        return tuple(
            resource
            for resource in self.resources
            if resource.consumer_layers[-1] == layer_name
        )

    def _check_layer(self, layer_name: str) -> None:
        if layer_name not in self.layer_order:
            raise ValueError(f"Unknown execution layer: {layer_name}")
