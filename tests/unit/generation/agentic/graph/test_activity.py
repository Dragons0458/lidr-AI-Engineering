"""Live per-agent activity feed helpers."""

from __future__ import annotations

from app.generation.agentic.graph.activity import GraphActivityLog, describe_node


def test_describe_classifier_reads_complexity():
    lines = describe_node("classifier_agent", {"complexity": "high"})
    assert lines == [
        {"node": "classifier", "label": "Classifier", "message": "Complejidad: high"}
    ]


def test_describe_structure_counts_modules_and_tasks():
    update = {"structure": {"modules": [{"tasks": [1, 2]}, {"tasks": [3]}]}}
    (line,) = describe_node("structure_agent", update)
    assert line["message"] == "2 módulos · 3 tareas"


def test_describe_hours_fanout_multiline():
    update = [
        {"task_hours": [{"task": "A", "has_match": True, "estimated_hours": 37}]},
        {"task_hours": [{"task": "B", "has_match": False, "estimated_hours": None}]},
    ]
    lines = describe_node("estimate_task_hours", update)
    assert [line["message"] for line in lines] == ["A: 37 h", "B: SIN ANÁLOGO"]


def test_describe_interrupt_and_unknown_never_raise():
    assert describe_node("__interrupt__", None)[0]["message"].startswith("⏸")
    assert describe_node("mystery_node", {"weird": 1})[0]["node"] == "mystery_node"


def test_activity_log_in_memory_append_read_reset():
    log = GraphActivityLog(redis_client=None)
    log.append(
        "run-1", node="classifier", label="Classifier", message="Complejidad: high"
    )
    log.append("run-1", node="structure", label="Structure", message="2 módulos")
    entries = log.read("run-1")
    assert [entry["seq"] for entry in entries] == [0, 1]
    log.reset("run-1")
    assert log.read("run-1") == []
