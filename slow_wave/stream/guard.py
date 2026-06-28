"""Confound guard: prove labels never leak into the online view (FR1.6).

This module is the *enforcement* half of the FR1.6 confound separation (Phase 1
exit criterion #4 in ``docs/PHASE1_CONTRACT.md``). The *structural* half lives in
:mod:`slow_wave.stream.schema`: a :class:`~slow_wave.stream.schema.StreamItem`
carries no relevance label and holds no pointer to
:class:`~slow_wave.stream.schema.GroundTruth`, so by construction there is no
attribute path from an online-visible item to its ground-truth label.

The guard here makes that separation *checkable* (and falsifiable):

* :func:`assert_no_label_leak` recursively walks an arbitrary object graph and
  raises :class:`ConfoundLeakError` the moment it can reach a
  :class:`~slow_wave.stream.schema.Label` value, a
  :class:`~slow_wave.stream.schema.GroundTruth` sidecar, or any field/key whose
  name is in :data:`BANNED_FIELD_NAMES`.
* :func:`online_view` projects a stream down to its immutable
  :class:`~slow_wave.stream.schema.StreamItem` tuple, guard-checking every item
  first, so it is *impossible* to obtain an online view that leaks a label.
* :func:`assert_online_view_is_clean` is the one-call invariant used by tests and
  by online code paths; it also checks at the **class level** that
  ``StreamItem`` declares no banned field, so a future edit that bolts a
  ``label`` field onto ``StreamItem`` is caught immediately.

The guard intentionally treats a full :class:`~slow_wave.stream.schema.Stream` as
*not* online-safe: a ``Stream`` legitimately owns its ``GroundTruth`` sidecar, so
handing one to :func:`assert_no_label_leak` reaches ``GroundTruth``/``Label`` and
trips the guard. Online code must consume :func:`online_view`, never the
``Stream`` itself.

Lean by design: this module depends only on the standard library, ``pydantic``,
and :mod:`slow_wave.stream.schema`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence, Set
from typing import Any

from pydantic import BaseModel

from slow_wave.stream import schema


class ConfoundLeakError(AssertionError):
    """Raised when a ground-truth relevance label is reachable from an object.

    Subclasses :class:`AssertionError` so that the confound invariant reads like
    an assertion at call sites and is stripped by ``python -O`` only where the
    caller chooses to wrap it in a bare ``assert`` (we never rely on that — the
    guard raises explicitly).
    """


#: Field / mapping-key names that must never appear on an online-visible object.
#: Any of these names anywhere in a reachable object graph is treated as a leak,
#: independent of the value stored under it (a name like ``relevance`` signals
#: intent to expose the confound even if the value is innocuous today).
BANNED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "label",
        "labels",
        "relevance",
        "relevance_label",
        "ground_truth",
        "ground_truth_label",
        "groundtruth",
        "gt",
        "is_signal",
        "is_distractor",
        "is_noise",
        "is_relevant",
    }
)

# Value types treated as atomic leaves (no members worth traversing). ``Label``
# is deliberately NOT in here even though it subclasses ``str``: it is checked
# (and rejected) before the atomic short-circuit so a leaked label is never
# mistaken for a harmless string.
_ATOMIC_TYPES: tuple[type, ...] = (str, bytes, bytearray, bool, int, float, complex)


def assert_no_label_leak(obj: Any, *, _seen: set[int] | None = None) -> None:
    """Assert that no ground-truth label is reachable from ``obj``.

    Recursively traverses the reachable members of ``obj`` and raises
    :class:`ConfoundLeakError` if any of the following is found:

    * a value that is an instance of
      :class:`~slow_wave.stream.schema.Label`,
    * a value that is an instance of
      :class:`~slow_wave.stream.schema.GroundTruth`, or
    * a field name or mapping key contained in :data:`BANNED_FIELD_NAMES`.

    The traversal understands pydantic v2 models (via
    ``type(obj).model_fields`` and ``obj.__dict__``), dataclasses, plain objects
    exposing ``__dict__``, mappings (both keys *and* values), and
    sequences/sets/tuples. ``str``/``bytes`` are treated as atomic (their
    characters are never iterated). The walk is cycle-safe via ``_seen``, a set
    of ``id()`` values for objects already visited.

    This function is for **online-view objects only**. A full
    :class:`~slow_wave.stream.schema.Stream` legitimately contains a
    ``GroundTruth`` sidecar, so passing one here intentionally trips the guard
    (a ``Stream`` is not online-safe — use :func:`online_view`).

    Args:
        obj: The object graph to inspect (typically a ``StreamItem`` or a
            container of them).
        _seen: Internal recursion guard tracking already-visited object ids.
            Callers should not pass this.

    Returns:
        None, if no label is reachable.

    Raises:
        ConfoundLeakError: If a label value, ``GroundTruth``, or banned field
            name is reachable from ``obj``.
    """
    if _seen is None:
        _seen = set()

    # 1. Forbidden value types FIRST. ``Label`` subclasses ``str``, so this must
    #    precede the atomic short-circuit below or a leaked label would be
    #    silently accepted as an ordinary string.
    if isinstance(obj, schema.Label):
        raise ConfoundLeakError(
            f"reachable ground-truth Label value {obj!r}: labels must never be "
            f"reachable from an online-visible object (FR1.6)"
        )
    if isinstance(obj, schema.GroundTruth):
        raise ConfoundLeakError(
            "reachable GroundTruth sidecar: the offline label store is not part "
            "of the online view (FR1.6) — consume online_view(stream) instead"
        )

    # 2. Atomic leaves: nothing further to traverse.
    if obj is None or isinstance(obj, _ATOMIC_TYPES):
        return

    # 3. Never descend into classes/types themselves (their __dict__ is metadata,
    #    not data, and would create spurious, unbounded traversals).
    if isinstance(obj, type):
        return

    # 4. Cycle safety: skip objects we have already visited.
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    # 5. Mappings: check key NAMES against the ban-list, then recurse into both
    #    keys and values.
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str) and key in BANNED_FIELD_NAMES:
                raise ConfoundLeakError(
                    f"banned mapping key {key!r}: implies a path to relevance "
                    f"labels in an online object (FR1.6)"
                )
            assert_no_label_leak(key, _seen=_seen)
            assert_no_label_leak(value, _seen=_seen)
        return

    # 6. Sequences and sets (str/bytes already handled as atomic above).
    if isinstance(obj, (Sequence, Set)):
        for element in obj:
            assert_no_label_leak(element, _seen=_seen)
        return

    # 7. Structured objects with named members: pydantic models, dataclasses, or
    #    anything exposing __dict__. Check member names against the ban-list,
    #    then recurse into member values.
    members = _named_members(obj)
    if members is not None:
        for name, value in members:
            if name in BANNED_FIELD_NAMES:
                raise ConfoundLeakError(
                    f"banned field name {name!r} on {type(obj).__name__}: "
                    f"online objects must not expose relevance labels (FR1.6)"
                )
            assert_no_label_leak(value, _seen=_seen)
        return

    # 8. Anything else (e.g. a plain function or opaque object with no members)
    #    is a leaf with nothing to inspect.
    return


def _named_members(obj: Any) -> list[tuple[str, Any]] | None:
    """Return ``(name, value)`` pairs for a structured object, or ``None``.

    Covers pydantic v2 models (declared fields plus any instance attributes),
    dataclass instances, and any other object exposing ``__dict__``. Returns
    ``None`` for objects with no inspectable named members so the caller can
    treat them as leaves.

    Args:
        obj: The object whose named members to enumerate.

    Returns:
        A list of ``(name, value)`` pairs, or ``None`` if ``obj`` has no
        inspectable members.
    """
    if isinstance(obj, BaseModel):
        names: list[str] = list(type(obj).model_fields)
        # Include any instance attributes not covered by declared fields (e.g.
        # ``extra="allow"`` extras) so nothing reachable is skipped.
        for extra in vars(obj):
            if extra not in names:
                names.append(extra)
        return [(name, getattr(obj, name, None)) for name in names]

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)]

    instance_dict = getattr(obj, "__dict__", None)
    if isinstance(instance_dict, dict):
        return list(instance_dict.items())

    return None


def online_view(stream: schema.Stream) -> tuple[schema.StreamItem, ...]:
    """Return the guard-checked, immutable online view of ``stream``.

    Projects ``stream`` down to exactly the data the wake agent and retrieval
    policy are allowed to see: its ordered :class:`~slow_wave.stream.schema.StreamItem`
    tuple. Every item is passed through :func:`assert_no_label_leak` *before* the
    tuple is returned, so it is impossible to obtain an online view that leaks a
    relevance label.

    Args:
        stream: The stream to project.

    Returns:
        The stream's items as an immutable ``tuple`` of ``StreamItem``.

    Raises:
        ConfoundLeakError: If any item exposes a label-bearing field or a path to
            ``GroundTruth`` (should never happen for a well-formed stream).
    """
    items = tuple(stream.items)
    for item in items:
        assert_no_label_leak(item)
    return items


def assert_online_view_is_clean(stream: schema.Stream) -> None:
    """Assert the full FR1.6 invariant for ``stream`` (Phase 1 exit #4).

    Performs two complementary checks:

    1. **Class level.** ``StreamItem`` must declare none of
       :data:`BANNED_FIELD_NAMES` in ``StreamItem.model_fields`` — so a future
       edit that adds, say, a ``label`` field to ``StreamItem`` is caught even
       before any instance is built.
    2. **Instance level.** Every item returned by :func:`online_view` is run
       through :func:`assert_no_label_leak`.

    Args:
        stream: The stream whose online view must be label-free.

    Returns:
        None, if the online view is clean.

    Raises:
        ConfoundLeakError: If ``StreamItem`` declares a banned field, or if any
            item in the online view can reach a label.
    """
    banned_on_class = set(schema.StreamItem.model_fields) & BANNED_FIELD_NAMES
    if banned_on_class:
        raise ConfoundLeakError(
            f"StreamItem declares banned field(s) {sorted(banned_on_class)}: the "
            f"online item type must not carry relevance labels (FR1.6)"
        )

    for item in online_view(stream):
        assert_no_label_leak(item)
