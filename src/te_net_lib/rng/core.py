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
    """
    Derive a deterministic integer seed from a structured parameter mapping.

    This function canonicalizes the input mapping and hashes it using BLAKE2b,
    producing a stable seed suitable for constructing reproducible random number
    generators. The caller controls the hashing namespace via `person` and the
    output size via `digest_size`.

    Parameters
    ----------

    fields:
        A mapping containing configuration fields. Keys are stringified and the
        structure is canonicalized so that key order does not affect the result.

    digest_size:
        Number of digest bytes to use (must be in [4, 16]). Larger values reduce
        collision risk.

    person:
        BLAKE2b personalization string (bytes, length <= 16). Use this to
        namespace seeds across projects or subsystems.

    Returns
    -------

    int

        A non-negative integer seed derived from the hash digest (little-endian).

    Raises
    ------

    ValueError

        If `digest_size` is outside [4, 16] or `person` exceeds 16 bytes.

    TypeError

        If `fields` is not a mapping or `person` is not bytes-like.

    Examples
    --------
    >>> person = b"te_net_lib"
    >>> s1 = seed_from_fields({"N": 200, "T": 400, "trial": 1}, 8, person)
    >>> s2 = seed_from_fields({"trial": 1, "T": 400, "N": 200}, 8, person)
    >>> s1 == s2
    True
    >>> seed_from_fields({"x": 1}, 8, b"this_person_is_too_long")
    Traceback (most recent call last):
    ...
    ValueError: person must be at most 16 bytes
    """
    if digest_size < 4 or digest_size > 16:
        raise ValueError("digest_size must be in [4, 16]")
    _validate_blake2_person(person)
    payload = _stable_repr(fields)
    h = blake2b(payload, digest_size=digest_size, person=person)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def rng(
    seed: int | np.random.SeedSequence | np.random.Generator,
) -> np.random.Generator:
    """
    Construct a NumPy `Generator` from an explicit seed specification.

    Parameters
    ----------

    seed:

    One of:

    - `int`: a non-negative seed passed to `np.random.default_rng`
    - `np.random.SeedSequence`: converted into a `Generator`
    - `np.random.Generator`: returned as-is

    Returns
    -------

    numpy.random.Generator

        A NumPy random number generator using the modern Generator API.

    Raises
    ------

    ValueError

        If an integer seed is negative.

    TypeError

        If `seed` is not an int, SeedSequence, or Generator.

    Examples
    --------
    >>> g1 = rng(123)
    >>> g2 = rng(123)
    >>> (g1.normal(size=5) == g2.normal(size=5)).all()
    True
    """
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
    """
    Derive a child `Generator` from a base seed or generator using labeled substreams.

    This function creates a deterministic substream by combining:

    - the base SeedSequence entropy/spawn_key
    - a hashed label-derived key
    - an explicit `spawn_key` suffix

    The intent is to provide reproducible, non-overlapping substreams for distinct
    pipeline stages such as "dgp", "preprocess", and "te".

    Parameters
    ----------

    base:
        An int seed, a `np.random.SeedSequence`, or an existing `Generator`.
        If a `Generator` is provided, its underlying `bit_generator.seed_seq`
        must be available.

    labels:
        A sequence of strings defining the logical substream identity.
        Different labels produce different child streams.

    digest_size:
        Hash digest size passed to `seed_from_fields`, in bytes (must be in [4, 16]).

    person:
        BLAKE2b personalization string (bytes, length <= 16). Must match the
        namespace you use for the library or subsystem.

    spawn_key:
        Additional integer components appended to the derived SeedSequence spawn key.
        Pass an empty sequence if not needed.

    Returns
    -------

    numpy.random.Generator

        A derived Generator for the requested substream.

    Raises
    ------

    ValueError

        If constraints on inputs are violated (e.g., negative seed or missing seed_seq).

    TypeError

        If the `base` type is invalid.

    Examples
    --------
    >>> person = b"te_net_lib"
    >>> g = derive_rng(123, ["dgp"], 8, person, [])
    >>> h = derive_rng(123, ["dgp"], 8, person, [])
    >>> (g.normal(size=3) == h.normal(size=3)).all()
    True
    >>> a = derive_rng(123, ["stageA"], 8, person, [])
    >>> b = derive_rng(123, ["stageB"], 8, person, [])
    >>> (a.normal(size=3) == b.normal(size=3)).all()
    False
    """
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
    """
    A lightweight scope object for reproducible substream management.

    `RngScope` stores:

    - a base Generator
    - the hashing parameters used to derive label keys
    - an explicit spawn_key suffix

    It provides `child(labels)` to derive deterministic substreams.

    Attributes
    ----------

    base:
        The base NumPy Generator used as the root of derivations.

    digest_size:
        Hash digest size passed to `seed_from_fields`, in bytes (must be in [4, 16]).

    person:
        BLAKE2b personalization string (bytes, length <= 16).

    spawn_key:
        A tuple of integer components appended to the derived SeedSequence spawn key.

    Examples
    --------
    >>> scope = RngScope.from_seed(123, 8, b"te_net_lib", [])
    >>> g1 = scope.child(["dgp"])
    >>> g2 = scope.child(["dgp"])
    >>> (g1.normal(size=4) == g2.normal(size=4)).all()
    True
    >>> a = scope.child(["a"])
    >>> b = scope.child(["b"])
    >>> (a.normal(size=4) == b.normal(size=4)).all()
    False
    """

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
        """
        Construct an `RngScope` from a seed specification.

        Parameters
        ----------

        seed:
            An int seed, SeedSequence, or Generator.

        digest_size:
            Hash digest size in bytes (must be in [4, 16]).

        person:
            BLAKE2b personalization string (bytes, length <= 16).

        spawn_key:
            Additional integer components appended to the derived spawn key.

        Returns
        -------

        RngScope

            A scope that can derive labeled child Generators.
        """
        return cls(
            base=rng(seed),
            digest_size=digest_size,
            person=person,
            spawn_key=tuple(spawn_key),
        )

    def child(self, labels: Sequence[str]) -> np.random.Generator:
        """
        Derive a labeled child Generator from this scope.

        Parameters
        ----------

        labels:

            A sequence of label strings defining the substream identity.

        Returns
        -------

        numpy.random.Generator

            A deterministic child Generator derived from the scope configuration.
        """
        return derive_rng(
            self.base, labels, self.digest_size, self.person, self.spawn_key
        )
