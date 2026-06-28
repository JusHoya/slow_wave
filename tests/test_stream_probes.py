"""Tests for slow_wave.stream.probes (Phase 1, WS3).

These tests are self-contained: they hand-construct small, deterministic
:class:`~slow_wave.stream.schema.Stream` objects directly from the schema types
(disjoint probed subjects per task, plus distractor/noise items labeled via
:class:`~slow_wave.stream.schema.GroundTruth`) so the core suite never depends on
the WS1 generator. The decisive check is Phase 1 exit criterion #6: with
``contradiction_rate=0.0`` the trivial oracle yields a lower-triangular ones
accuracy matrix over a square ``n_tasks x n_tasks`` shape.
"""

from __future__ import annotations

import pytest

from slow_wave.stream.probes import (
    OracleAgent,
    build_probe_set,
    compute_accuracy_matrix,
)
from slow_wave.stream.schema import (
    CLScenario,
    Fact,
    GroundTruth,
    ItemKind,
    Label,
    Probe,
    ProbeSet,
    Stream,
    StreamGenConfig,
    StreamItem,
)


# --------------------------------------------------------------------------- #
# Deterministic stream builder (no RNG, no generator dependency)
# --------------------------------------------------------------------------- #
def _build_stream(
    *,
    signal_counts: list[int],
    probes_per_task: int = 6,
    scenario: CLScenario = CLScenario.TASK_INCREMENTAL,
    contradiction: bool = False,
) -> Stream:
    """Build a tiny deterministic stream from schema types.

    Each task ``t`` asserts ``signal_counts[t]`` signal facts over disjoint
    probed subjects ``subj_{t}_{k}`` (attribute ``attr_0``), one distractor fact
    in a disjoint ``dsubj_*`` namespace, and one pure-noise item. When
    ``contradiction`` is set, every signal key is re-asserted later in the same
    task with a different value so the latest value wins.
    """
    n_tasks = len(signal_counts)
    items: list[StreamItem] = []
    labels: dict[str, Label] = {}
    order = 0

    def add(label: Label, task_index: int, fact: Fact | None, content: str) -> None:
        nonlocal order
        item_id = f"i{order:06d}"
        items.append(
            StreamItem(
                item_id=item_id,
                order=order,
                kind=ItemKind.INGEST,
                task_index=task_index,
                content=content,
                fact=fact,
            )
        )
        labels[item_id] = label
        order += 1

    max_per_task = 0
    for t in range(n_tasks):
        n_here = 0
        for k in range(signal_counts[t]):
            subject = f"subj_{t}_{k}"
            value = f"val_{t}_{k}"
            add(
                Label.SIGNAL,
                t,
                Fact(subject=subject, attribute="attr_0", value=value),
                f"The attr_0 of {subject} is {value}.",
            )
            n_here += 1
            if contradiction:
                value2 = f"val_{t}_{k}_b"
                add(
                    Label.SIGNAL,
                    t,
                    Fact(subject=subject, attribute="attr_0", value=value2),
                    f"The attr_0 of {subject} is {value2}.",
                )
                n_here += 1
        # Distractor in a disjoint namespace (never probed).
        dsubj = f"dsubj_{t}"
        add(
            Label.DISTRACTOR,
            t,
            Fact(subject=dsubj, attribute="attr_0", value="dval"),
            f"The attr_0 of {dsubj} is dval.",
        )
        n_here += 1
        # Pure noise (no fact).
        add(Label.NOISE, t, None, "tok_1 tok_2 tok_3")
        n_here += 1
        max_per_task = max(max_per_task, n_here)

    config = StreamGenConfig(
        scenario=scenario,
        n_tasks=n_tasks,
        items_per_task=max(max_per_task, 1),
        probes_per_task=probes_per_task,
        contradiction_rate=0.0 if not contradiction else 0.5,
    )
    return Stream(
        stream_id=f"{scenario.value}-test",
        scenario=scenario,
        seed=0,
        n_tasks=n_tasks,
        config=config,
        items=items,
        ground_truth=GroundTruth(labels=labels),
    )


# --------------------------------------------------------------------------- #
# build_probe_set
# --------------------------------------------------------------------------- #
def test_build_probe_set_basic_shape_and_answers() -> None:
    """Probe set is non-empty, one probe per signal key, all answers known."""
    stream = _build_stream(signal_counts=[2, 2, 2], probes_per_task=6)
    ps = build_probe_set(stream)

    assert isinstance(ps, ProbeSet)
    assert ps.scenario == CLScenario.TASK_INCREMENTAL
    assert ps.n_tasks == 3
    assert len(ps.probes) == 6  # 3 tasks x 2 keys, under the cap

    keys = {(p.subject, p.attribute) for p in ps.probes}
    assert len(keys) == len(ps.probes)  # one probe per key, no duplicates

    for p in ps.probes:
        assert isinstance(p, Probe)
        assert p.answer != ""  # every probe carries a known answer
        assert p.query == f"What is the {p.attribute} of {p.subject}?"
        assert p.task_id_visible is True  # task-incremental
        assert 0 <= p.task_index < 3
        # subject namespace encodes the task it first appears in
        assert p.subject.startswith(f"subj_{p.task_index}_")


def test_build_probe_set_available_after_and_probe_ids() -> None:
    """available_after_order matches first signal order; ids are stable/ordered."""
    stream = _build_stream(signal_counts=[2, 2, 2], probes_per_task=6)
    ps = build_probe_set(stream)

    # probe_id are zero-padded and assigned in stable (task, subject, attr) order.
    assert [p.probe_id for p in ps.probes] == [f"p{i:06d}" for i in range(6)]
    assert ps.probes == sorted(
        ps.probes, key=lambda p: (p.task_index, p.subject, p.attribute)
    )

    # available_after_order points at the FIRST signal assertion for the key.
    first_order_by_key: dict[tuple[str, str], int] = {}
    for it in sorted(stream.items, key=lambda x: x.order):
        if it.fact is not None and stream.ground_truth.labels[it.item_id] is Label.SIGNAL:
            first_order_by_key.setdefault(it.fact.key(), it.order)
    assert first_order_by_key  # sanity: there were signal facts
    for p in ps.probes:
        assert p.available_after_order == first_order_by_key[(p.subject, p.attribute)]


def test_build_probe_set_answer_is_latest_value() -> None:
    """With contradictions, the probe answer is the latest signal value."""
    stream = _build_stream(signal_counts=[2, 2, 2], contradiction=True)
    ps = build_probe_set(stream)

    assert len(ps.probes) == 6
    for p in ps.probes:
        # latest assertion for subj_t_k is "val_t_k_b"
        assert p.answer.endswith("_b")


def test_build_probe_set_per_task_cap() -> None:
    """No more than probes_per_task probes per task; first-N-by-sorted-key wins."""
    stream = _build_stream(signal_counts=[4, 4, 4], probes_per_task=2)
    ps = build_probe_set(stream)

    assert len(ps.probes) == 6  # 3 tasks x cap(2)

    per_task: dict[int, list[str]] = {}
    for p in ps.probes:
        per_task.setdefault(p.task_index, []).append(p.subject)
    for t in range(3):
        assert len(per_task[t]) == 2
        # first two by sorted key: subj_t_0, subj_t_1
        assert sorted(per_task[t]) == [f"subj_{t}_0", f"subj_{t}_1"]


def test_build_probe_set_task_id_visible_non_task_incremental() -> None:
    """task_id_visible is False outside the task-incremental scenario."""
    stream = _build_stream(
        signal_counts=[2, 2], scenario=CLScenario.DOMAIN_INCREMENTAL
    )
    ps = build_probe_set(stream)
    assert ps.scenario == CLScenario.DOMAIN_INCREMENTAL
    assert ps.probes  # non-empty
    assert all(p.task_id_visible is False for p in ps.probes)


def test_build_probe_set_deterministic() -> None:
    """Building twice yields byte-identical probe sets (no RNG)."""
    stream = _build_stream(signal_counts=[3, 2, 4], probes_per_task=3)
    a = build_probe_set(stream)
    b = build_probe_set(stream)
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# OracleAgent
# --------------------------------------------------------------------------- #
def test_oracle_unseen_key_returns_empty() -> None:
    """A probe for a never-observed key yields the empty string."""
    stream = _build_stream(signal_counts=[2, 2, 2])
    ps = build_probe_set(stream)

    agent = OracleAgent()
    agent.reset()
    # Observe nothing, then ask: must be "".
    assert agent.answer(ps.probes[0]) == ""

    # Observe an unrelated fact; the probed key is still unseen.
    unrelated = next(
        it for it in stream.items if it.fact is not None and it.fact.subject.startswith("dsubj_")
    )
    agent.observe(unrelated)
    assert agent.answer(ps.probes[0]) == ""


def test_oracle_recalls_latest_observed_value() -> None:
    """The oracle returns the latest value it observed for a key."""
    agent = OracleAgent()
    probe = Probe(
        probe_id="p000000",
        task_index=0,
        subject="subj_0_0",
        attribute="attr_0",
        answer="v2",
        available_after_order=0,
        query="What is the attr_0 of subj_0_0?",
        task_id_visible=True,
    )
    agent.observe(
        StreamItem(
            item_id="i000000",
            order=0,
            kind=ItemKind.INGEST,
            task_index=0,
            content="...",
            fact=Fact(subject="subj_0_0", attribute="attr_0", value="v1"),
        )
    )
    assert agent.answer(probe) == "v1"
    agent.observe(
        StreamItem(
            item_id="i000001",
            order=1,
            kind=ItemKind.INGEST,
            task_index=0,
            content="...",
            fact=Fact(subject="subj_0_0", attribute="attr_0", value="v2"),
        )
    )
    assert agent.answer(probe) == "v2"
    agent.reset()
    assert agent.answer(probe) == ""


# --------------------------------------------------------------------------- #
# compute_accuracy_matrix (exit #6)
# --------------------------------------------------------------------------- #
def test_accuracy_matrix_shape_and_range() -> None:
    """R is square n_tasks x n_tasks with every entry in [0, 1]."""
    stream = _build_stream(signal_counts=[2, 2, 2])
    ps = build_probe_set(stream)
    R = compute_accuracy_matrix(stream, ps)

    assert R.n_tasks == 3
    assert R.scenario == CLScenario.TASK_INCREMENTAL
    assert len(R.R) == 3
    assert all(len(row) == 3 for row in R.R)
    assert all(0.0 <= v <= 1.0 for row in R.R for v in row)


def test_accuracy_matrix_lower_triangular_ones_no_contradiction() -> None:
    """Exit #6: zero-contradiction stream -> lower-triangular ones matrix."""
    stream = _build_stream(signal_counts=[2, 2, 2], probes_per_task=6)
    ps = build_probe_set(stream)
    R = compute_accuracy_matrix(stream, ps)

    for i in range(3):
        for j in range(3):
            expected = 1.0 if j <= i else 0.0
            assert R.R[i][j] == expected, f"R[{i}][{j}]={R.R[i][j]} != {expected}"


def test_accuracy_matrix_default_agent_constructed() -> None:
    """compute_accuracy_matrix builds an OracleAgent when agent is None."""
    stream = _build_stream(signal_counts=[2, 2, 2])
    ps = build_probe_set(stream)
    R_default = compute_accuracy_matrix(stream, ps, agent=None)
    R_explicit = compute_accuracy_matrix(stream, ps, agent=OracleAgent())
    assert R_default.R == R_explicit.R


def test_accuracy_matrix_empty_task_column_is_zero() -> None:
    """A task with no probes contributes an all-zero column."""
    stream = _build_stream(signal_counts=[2, 0, 2])  # task 1 has no signals
    ps = build_probe_set(stream)

    # No probe is attributed to task 1.
    assert all(p.task_index != 1 for p in ps.probes)

    R = compute_accuracy_matrix(stream, ps)
    assert len(R.R) == 3
    for i in range(3):
        assert R.R[i][1] == 0.0  # empty task column
    # Tasks 0 and 2 still behave lower-triangularly.
    assert R.R[0][0] == 1.0
    assert R.R[2][0] == 1.0
    assert R.R[2][2] == 1.0
    assert R.R[0][2] == 0.0


# --------------------------------------------------------------------------- #
# Optional: integration with the WS1 generator (skips if unavailable)
# --------------------------------------------------------------------------- #
def test_with_generator_if_available() -> None:
    """Smoke-test against a real generated stream (skips if WS1 absent)."""
    pytest.importorskip("slow_wave.stream.generator")
    from slow_wave.stream.generator import generate_stream  # type: ignore

    config = StreamGenConfig(
        scenario=CLScenario.TASK_INCREMENTAL,
        n_tasks=3,
        items_per_task=30,
        probes_per_task=4,
        contradiction_rate=0.0,
    )
    stream = generate_stream(config, seed=7)
    ps = build_probe_set(stream)
    assert ps.probes
    assert all(p.answer != "" for p in ps.probes)

    R = compute_accuracy_matrix(stream, ps)
    assert R.n_tasks == stream.n_tasks
    assert all(0.0 <= v <= 1.0 for row in R.R for v in row)
    # With no contradictions the oracle is lower-triangular ones.
    for i in range(stream.n_tasks):
        for j in range(stream.n_tasks):
            if any(p.task_index == j for p in ps.probes):
                assert R.R[i][j] == (1.0 if j <= i else 0.0)
