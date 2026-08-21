"""Render a Session 16 eval report to JSON-safe dict / markdown. No I/O."""

from __future__ import annotations

from evals.production.schemas import ArmReport, CaseEvaluation, EvalReport


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _case_row(ev: CaseEvaluation) -> str:
    predicted = (
        "—" if ev.predicted_engineer_days is None else str(ev.predicted_engineer_days)
    )
    signal = ev.abstention_signal or "—"
    hit = "—" if ev.source_hit is None else ("yes" if ev.source_hit else "no")
    return (
        f"| {ev.case_id} | {ev.verdict} | {ev.expected_engineer_days} | "
        f"{predicted} | {ev.acceptable_range[0]}–{ev.acceptable_range[1]} | "
        f"{'yes' if ev.abstained else 'no'} ({signal}) | {hit} | "
        f"{ev.latency_ms:.0f} | {_fmt_num(ev.cost_usd, 4)} |"
    )


def _arm_table(report: ArmReport) -> list[str]:
    lines = [
        f"### Arm `{report.arm}`",
        "",
    ]
    if report.skipped:
        lines.append(f"_Skipped:_ {report.skip_reason or 'runtime unavailable'}")
        lines.append("")
        return lines
    lines.extend(
        [
            f"- cases: {report.n_cases} (estimation {report.n_estimation}, "
            f"abstention {report.n_abstention})",
            f"- within_range_rate: {_fmt_rate(report.within_range_rate)}",
            f"- mean_absolute_error: {_fmt_num(report.mean_absolute_error)} engineer-days",
            f"- abstention_correct: {report.abstention_correct}",
            f"- mean_latency_ms: {_fmt_num(report.mean_latency_ms, 0)}",
            f"- p95_latency_ms: {_fmt_num(report.p95_latency_ms, 0)} (n={report.p95_n})",
            f"- error_rate: {_fmt_rate(report.error_rate)}",
            f"- abstention_rate: {_fmt_rate(report.abstention_rate)}",
            f"- source_recall: {_fmt_rate(report.source_recall)}",
            f"- total_cost_usd: {_fmt_num(report.total_cost_usd, 4)}",
            f"- mean_cost_usd: {_fmt_num(report.mean_cost_usd, 4)}",
            f"- throttled: {report.n_throttled}",
            "",
            "| case | verdict | expected | predicted | range | abstained | source_hit | latency_ms | cost_usd |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for ev in report.evaluations:
        lines.append(_case_row(ev))
    lines.append("")
    return lines


def render_markdown(report: EvalReport) -> str:
    golden = report.golden_set
    lines = [
        f"# Session 16 eval `{report.run_id}`",
        "",
        f"- label: `{report.label or '—'}`",
        f"- started: {report.started_at.isoformat()}",
        f"- finished: {report.finished_at.isoformat() if report.finished_at else '—'}",
        f"- golden set: `{golden.get('path', '')}` sha256=`{golden.get('sha256', '')}` "
        f"({golden.get('cases', '?')} cases)",
        "",
        "A p95 with n=7 is not a p95. Read it next to `n`.",
        "",
        "This report never includes transcripts, generated text, or API keys.",
        "",
    ]
    env = report.environment
    if env:
        lines.append("## Environment")
        lines.append("")
        for key, value in env.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    for arm_name in ("rag", "graph"):
        arm = report.arms.get(arm_name)
        if arm is not None:
            lines.extend(_arm_table(arm))
    if report.ab:
        lines.extend(
            [
                "## A/B (rag vs graph)",
                "",
                f"- quality_delta (within_range_rate rag − graph): "
                f"`{report.ab.get('quality_delta')}`",
                f"- cost_ratio (rag / graph): `{report.ab.get('cost_ratio')}`",
                f"- latency_ratio (rag / graph): `{report.ab.get('latency_ratio')}`",
                f"- abstention_gap (case ids): `{report.ab.get('abstention_gap')}`",
                "- verdict: _not computed — write it by hand in "
                "`exercises/session-16/README.md`_",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
