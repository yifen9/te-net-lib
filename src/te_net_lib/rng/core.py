from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Mapping, Sequence

import numpy as np


def _canonicalize(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, bytes):
        return x.hex()
    if isinstance(x, Mapping):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(x, (list, tuple)):
        return [_canonicalize(v) for v in x]
    if isinstance(x, set):
        return sorted(_canonicalize(v) for v in x)
    return str(x)


def _stable_repr(fields: Mapping[str, Any]) -> bytes:
    canon = _canonicalize(fields)
    parts: list[str] = []

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                walk(f"{prefix}{k}.", v)
            return
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(f"{prefix}{i}.", v)
            return
        parts.append(f"{prefix[:-1]}={obj}")

    if not isinstance(canon, Mapping):
        raise TypeError("fields must be a mapping")
    walk("", canon)
    parts.sort()
    return ("\n".join(parts)).encode("utf-8")


def _validate_blake2_person(person: bytes) -> None:
    if not isinstance(person, (bytes, bytearray)):
        raise TypeError("person must be bytes")
    if len(person) > 16:
        raise ValueError("person must be at most 16 bytes")


def seed_from_fields(fields: Mapping[str, Any], digest_size: int, person: bytes) -> int:
    if digest_size < 4 or digest_size > 16:
        raise ValueError("digest_size must be in [4, 16]")
    _validate_blake2_person(person)
    payload = _stable_repr(fields)
    h = blake2b(payload, digest_size=digest_size, person=person)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def rng(
    seed: int | np.random.SeedSequence | np.random.Generator,
) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if isinstance(seed, int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return np.random.default_rng(seed)
    raise TypeError("seed must be int, SeedSequence, or Generator")


def derive_rng(
    base: int | np.random.SeedSequence | np.random.Generator,
    labels: Sequence[str],
    digest_size: int,
    person: bytes,
    spawn_key: Sequence[int],
) -> np.random.Generator:
    if isinstance(base, np.random.Generator):
        ss = getattr(base.bit_generator, "seed_seq", None)
        if ss is None:
            raise ValueError("base Generator does not expose seed_seq")
        base_ss = ss
    elif isinstance(base, np.random.SeedSequence):
        base_ss = base
    elif isinstance(base, int):
        if base < 0:
            raise ValueError("seed must be non-negative")
        base_ss = np.random.SeedSequence(base)
    else:
        raise TypeError("base must be int, SeedSequence, or Generator")

    _validate_blake2_person(person)
    key: list[int] = []
    if labels:
        key_int = seed_from_fields({"labels": list(labels)}, digest_size, person)
        key = [key_int & 0xFFFFFFFF, (key_int >> 32) & 0xFFFFFFFF]

    sk = list(spawn_key)
    child_ss = np.random.SeedSequence(
        entropy=base_ss.entropy,
        spawn_key=tuple(base_ss.spawn_key) + tuple(key) + tuple(sk),
    )
    return np.random.default_rng(child_ss)


@dataclass(frozen=True, slots=True)
class RngScope:
    base: np.random.Generator
    digest_size: int
    person: bytes
    spawn_key: tuple[int, ...]

    @classmethod
    def from_seed(
        cls,
        seed: int | np.random.SeedSequence | np.random.Generator,
        digest_size: int,
        person: bytes,
        spawn_key: Sequence[int],
    ) -> "RngScope":
        return cls(
            base=rng(seed),
            digest_size=digest_size,
            person=person,
            spawn_key=tuple(spawn_key),
        )

    def child(self, labels: Sequence[str]) -> np.random.Generator:
        return derive_rng(
            self.base, labels, self.digest_size, self.person, self.spawn_key
        )
