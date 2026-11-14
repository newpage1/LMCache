# SPDX-License-Identifier: Apache-2.0
"""Resource-block reads must never leak L1 read locks.

``BaseL1Manager.reserve_read`` reports a granted read lock as
``(L1Error.SUCCESS, memory_obj)`` and an absent key as
``(L1Error.KEY_NOT_EXIST, None)``.  The error slot is therefore never ``None``:
treating "error is not None" as a miss turns every successful reservation into
a leak (the object stays read-locked and the bytes are never returned).  B01
observed exactly that shape: ``read_locked_count`` grew monotonically while
resource-block lookups still reported zero.

These tests pin the released-reservation invariant on a fake manager that
counts locks and on the real ``DramL1Manager`` status counters.
"""

# Standard
from contextlib import contextmanager

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.multiprocess.modules.resource_block import StockResourceBlockStore
from tests.v1.distributed.utils import should_use_lazy_alloc

MODEL = "model"
PAYLOAD = b"12345678"
OTHER_PAYLOAD = b"87654321"


class _FakeObject:
    """Minimal ``MemoryObj`` exposing the byte tensor the store reads."""

    def __init__(self, payload: bytes) -> None:
        self.tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()


class _LockingManager:
    """Fake L1 manager that tracks granted and released read locks."""

    def __init__(
        self,
        objects: dict[object, bytes] | None = None,
        grant_without_object: tuple[object, ...] = (),
    ) -> None:
        self.objects = dict(objects or {})
        self.grant_without_object = set(grant_without_object)
        self.locks: dict[object, int] = {}
        self.reserve_read_calls = 0
        self.finish_read_calls = 0

    def reserve_read(self, keys):
        """Return the stock ``(L1Error, MemoryObj | None)`` results."""
        result = {}
        for key in keys:
            self.reserve_read_calls += 1
            payload = self.objects.get(key)
            if payload is not None:
                self.locks[key] = self.locks.get(key, 0) + 1
                result[key] = (L1Error.SUCCESS, _FakeObject(payload))
            elif key in self.grant_without_object:
                self.locks[key] = self.locks.get(key, 0) + 1
                result[key] = (L1Error.SUCCESS, None)
            else:
                result[key] = (L1Error.KEY_NOT_EXIST, None)
        return result

    def finish_read(self, keys):
        """Release the read locks taken by ``reserve_read``."""
        self.finish_read_calls += 1
        for key in keys:
            remaining = self.locks.get(key, 0) - 1
            if remaining <= 0:
                self.locks.pop(key, None)
            else:
                self.locks[key] = remaining
        return {key: L1Error.SUCCESS for key in keys}

    def locked_count(self) -> int:
        """Return how many read locks are currently held."""
        return sum(self.locks.values())


class _FakeStorageManager:
    """Storage-manager stand-in that only exposes read managers."""

    def __init__(self, managers) -> None:
        self._managers = list(managers)

    def l1_managers(self):
        """Return the fake L1 managers in lookup order."""
        return list(self._managers)


class _RealStorageManager:
    """Adapt ``StorageManager`` surface used by ``StockResourceBlockStore``."""

    def __init__(self, manager: DramL1Manager) -> None:
        self._manager = manager

    def l1_managers(self):
        """Return the single real L1 manager."""
        return [self._manager]

    def reserve_write(self, keys, layout):
        """Map the raw ``(L1Error, MemoryObj)`` results to objects."""
        result = self._manager.reserve_write(keys, [False] * len(keys), layout)
        return {
            key: memory_obj
            for key, (error, memory_obj) in result.items()
            if error == L1Error.SUCCESS and memory_obj is not None
        }

    def finish_write(self, keys, owners):
        """Admit the reserved staging objects."""
        return self._manager.finish_write(keys)


def _object_key(seed: int) -> ObjectKey:
    """Build a deterministic stock object key for the tests."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(seed), model_name=MODEL, kv_rank=0
    )


@contextmanager
def _real_l1():
    """Yield a real DRAM L1 manager, closing it afterwards."""
    manager = DramL1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
    )
    try:
        yield manager
    finally:
        manager.close()


def _read_locked(manager: DramL1Manager) -> int:
    """Return the manager's current read-locked object count."""
    return int(manager.report_status()["read_locked_count"])


def test_read_returns_payload_and_releases_the_lock():
    """A hit returns the resident bytes and releases its read lock."""
    key = _object_key(1)
    manager = _LockingManager({key: PAYLOAD})
    store = StockResourceBlockStore(_FakeStorageManager([manager]))

    assert store.read(key) == PAYLOAD
    assert manager.locked_count() == 0
    assert manager.finish_read_calls == 1


def test_absent_key_takes_no_lock():
    """An absent key is a clean miss and never holds a reservation."""
    manager = _LockingManager()
    store = StockResourceBlockStore(_FakeStorageManager([manager]))

    assert store.read(_object_key(2)) is None
    assert manager.locked_count() == 0
    assert manager.finish_read_calls == 0


def test_granted_reservation_without_object_is_released():
    """A SUCCESS result with no object still releases its reservation."""
    key = _object_key(3)
    manager = _LockingManager(grant_without_object=(key,))
    store = StockResourceBlockStore(_FakeStorageManager([manager]))

    assert store.read(key) is None
    assert manager.locked_count() == 0
    assert manager.finish_read_calls == 1


def test_first_manager_miss_then_second_manager_hit():
    """A miss on an earlier manager must not interfere with the later hit."""
    key = _object_key(4)
    first = _LockingManager()
    second = _LockingManager({key: PAYLOAD})
    store = StockResourceBlockStore(_FakeStorageManager([first, second]))

    assert store.read(key) == PAYLOAD
    assert first.locked_count() == 0
    assert second.locked_count() == 0
    assert first.finish_read_calls == 0
    assert second.finish_read_calls == 1


def test_granted_reservation_without_object_then_later_hit():
    """A defensive empty grant is released before a later manager serves."""
    key = _object_key(12)
    first = _LockingManager(grant_without_object=(key,))
    second = _LockingManager({key: PAYLOAD})
    store = StockResourceBlockStore(_FakeStorageManager([first, second]))

    assert store.read(key) == PAYLOAD
    assert first.locked_count() == 0
    assert first.finish_read_calls == 1
    assert second.locked_count() == 0


def test_repeated_hits_and_misses_keep_read_locks_at_zero():
    """Repeated read-backs cannot accumulate read locks."""
    key = _object_key(5)
    missing = _object_key(6)
    manager = _LockingManager({key: PAYLOAD})
    store = StockResourceBlockStore(_FakeStorageManager([manager]))

    for _ in range(25):
        assert store.read(key) == PAYLOAD
        assert store.read(missing) is None

    assert manager.locked_count() == 0
    assert manager.finish_read_calls == 25


def test_real_l1_read_returns_payload_and_unlocks():
    """The real L1 manager reports no read lock after a successful read."""
    key = _object_key(7)
    with _real_l1() as manager:
        store = StockResourceBlockStore(_RealStorageManager(manager))
        assert store.write(key, PAYLOAD) is True
        assert _read_locked(manager) == 0

        assert store.read(key) == PAYLOAD
        assert _read_locked(manager) == 0


def test_real_l1_missing_key_does_not_lock():
    """A missing key in the real L1 manager leaves read locks untouched."""
    with _real_l1() as manager:
        store = StockResourceBlockStore(_RealStorageManager(manager))

        assert store.read(_object_key(8)) is None
        assert _read_locked(manager) == 0


def test_real_l1_repeated_reads_stay_unlocked():
    """Hits and misses in a loop keep ``read_locked_count`` at zero."""
    key = _object_key(9)
    missing = _object_key(10)
    with _real_l1() as manager:
        store = StockResourceBlockStore(_RealStorageManager(manager))
        assert store.write(key, PAYLOAD) is True

        for _ in range(20):
            assert store.read(key) == PAYLOAD
            assert store.read(missing) is None

        assert _read_locked(manager) == 0


def test_real_l1_duplicate_write_keeps_read_locks_zero():
    """Duplicate-write verification reads must not leak read locks."""
    key = _object_key(11)
    with _real_l1() as manager:
        store = StockResourceBlockStore(_RealStorageManager(manager))

        assert store.write(key, PAYLOAD) is True
        assert store.write(key, PAYLOAD) is True
        assert store.read(key) == PAYLOAD
        assert _read_locked(manager) == 0

        assert store.write(key, OTHER_PAYLOAD) is False
        assert store.read(key) == PAYLOAD
        assert _read_locked(manager) == 0
