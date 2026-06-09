import re

from app.domain.schemas.estimation import OutputFormat, StructureCheck

_OK_FINISH_REASONS = {"stop", "end_turn", "tool_use"}
_V1_HEADER_RE = re.compile(
    r"\|\s*Phase\s*\|\s*Tasks\s*\|\s*Hours\s*\|\s*Team\s*\|", re.IGNORECASE
)
_V2_HEADER_RE = re.compile(
    r"\|\s*Phase\s*\|\s*Scope\s*\|\s*Base\s+Hours\s*\|\s*Buffer\s+Hours\s*\|\s*Team\s*\|",
    re.IGNORECASE,
)
_TOTAL_HOURS_RE = re.compile(
    r"Total\s+(?:estimated|planned)?\s*hours\s*[:* ]+\s*([\d.,]+)\s*h?",
    re.IGNORECASE,
)
_TOTAL_COST_RE = re.compile(
    r"Total\s+cost\s*[:* ]+\s*([\d.,]+)\s*(?:EUR)?",
    re.IGNORECASE,
)
_SPANISH_TOTAL_RE = re.compile(
    r"Total\s+estimado\s*[:* ]+\s*([\d.,]+)\s*horas?", re.IGNORECASE
)
_COST_COLUMN_HEADER_RE = re.compile(r"\|\s*Cost", re.IGNORECASE)
_COST_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<task>[^|]+?)\s*\|\s*(?P<hours>[\d.,]+)\s*\|\s*(?P<cost>[\d.,\sEURer]+)\s*\|\s*$",
    re.MULTILINE,
)


def evaluate_estimation_structure(
    text: str, finish_reason: str | None, output_format: OutputFormat
) -> StructureCheck:
    """Evaluate estimation structure with regex/parsing only."""
    normalized_text = text or ""
    has_title = bool(
        re.search(r"^##\s+\S", normalized_text, re.MULTILINE)
        or re.search(r"^\s*(?:Estimacion|Estimación|Scope summary):", normalized_text)
    )
    has_breakdown_table = _has_breakdown_table(normalized_text, output_format)
    declared_total_hours = _extract_declared_total_hours(normalized_text)
    sum_row_hours = _sum_table_hours(normalized_text, output_format)
    hours_match = _hours_match(sum_row_hours, declared_total_hours)
    declared_total_cost = _extract_declared_total_cost(normalized_text)
    sum_row_cost = _sum_table_costs(normalized_text)
    cost_match = _cost_match(sum_row_cost, declared_total_cost)
    has_totals_section = declared_total_hours is not None or _has_total_table_row(
        normalized_text
    )
    has_team_section = bool(
        re.search(
            r"(Equipo\s+recomendado|Recommended\s+Team|Team(\s+composition)?|"
            r"\b(PM|Engineer|Developer|Designer|QA|Core team)\b)",
            normalized_text,
            re.IGNORECASE,
        )
    )
    has_duration_section = bool(
        re.search(
            r"(Duraci[oó]n|Estimated\s+Duration|Suggested\s+timeline|Timeline|"
            r"\bweeks?\b|\bsemanas?\b)",
            normalized_text,
            re.IGNORECASE,
        )
    )
    finish_reason_ok = finish_reason in _OK_FINISH_REASONS

    checks = _applicable_checks(
        output_format=output_format,
        has_title=has_title,
        has_breakdown_table=has_breakdown_table,
        has_totals_section=has_totals_section,
        has_team_section=has_team_section,
        has_duration_section=has_duration_section,
        hours_match=hours_match,
        finish_reason_ok=finish_reason_ok,
    )
    score = round(sum(checks) / len(checks), 3) if checks else 0.0

    issues = _build_issues(
        output_format=output_format,
        has_title=has_title,
        has_breakdown_table=has_breakdown_table,
        has_totals_section=has_totals_section,
        has_team_section=has_team_section,
        has_duration_section=has_duration_section,
        declared_total_hours=declared_total_hours,
        sum_row_hours=sum_row_hours,
        hours_match=hours_match,
        declared_total_cost=declared_total_cost,
        sum_row_cost=sum_row_cost,
        cost_match=cost_match,
        finish_reason=finish_reason,
        finish_reason_ok=finish_reason_ok,
    )

    return StructureCheck(
        has_title=has_title,
        has_breakdown_table=has_breakdown_table,
        has_totals_section=has_totals_section,
        has_team_section=has_team_section,
        has_duration_section=has_duration_section,
        declared_total_hours=declared_total_hours,
        sum_row_hours=sum_row_hours,
        hours_match=hours_match,
        declared_total_cost=declared_total_cost,
        sum_row_cost=sum_row_cost,
        cost_match=cost_match,
        finish_reason_ok=finish_reason_ok,
        score=score,
        issues=issues,
    )


def _has_breakdown_table(text: str, output_format: OutputFormat) -> bool:
    if output_format != OutputFormat.PHASES_TABLE:
        return False
    return bool(_V1_HEADER_RE.search(text) or _V2_HEADER_RE.search(text))


def _extract_declared_total_hours(text: str) -> int | None:
    for regex in (_TOTAL_HOURS_RE, _SPANISH_TOTAL_RE):
        match = regex.search(text)
        if match:
            return _to_int(match.group(1))
    return _extract_total_from_table_row(text)


def _sum_table_hours(text: str, output_format: OutputFormat) -> int | None:
    if output_format != OutputFormat.PHASES_TABLE:
        return None

    is_v2 = bool(_V2_HEADER_RE.search(text))
    running_total = 0
    row_count = 0
    for columns in _markdown_table_rows(text):
        first_column = columns[0].strip().lower()
        if first_column in {"phase", "", "---"} or "total" in first_column:
            continue
        if is_v2 and len(columns) >= 5:
            base = _to_int(columns[2])
            buffer = _to_int(columns[3])
            if base is None or buffer is None:
                continue
            running_total += base + buffer
            row_count += 1
            continue
        if len(columns) >= 4:
            hours = _to_int(columns[2])
            if hours is None:
                continue
            running_total += hours
            row_count += 1

    return running_total if row_count else None


def _extract_total_from_table_row(text: str) -> int | None:
    for columns in _markdown_table_rows(text):
        if columns and "total" in columns[0].strip().lower():
            numbers = [_to_int(column) for column in columns[1:]]
            numbers = [number for number in numbers if number is not None]
            if numbers:
                return numbers[-1]
    return None


def _has_total_table_row(text: str) -> bool:
    return any(
        columns and "total" in columns[0].strip().lower()
        for columns in _markdown_table_rows(text)
    )


def _markdown_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line.startswith("|") or not stripped_line.endswith("|"):
            continue
        if re.fullmatch(r"\|[\s:|\-]+\|", stripped_line):
            continue
        rows.append([column.strip() for column in stripped_line.strip("|").split("|")])
    return rows


def _hours_match(
    sum_row_hours: int | None, declared_total_hours: int | None
) -> bool | None:
    if sum_row_hours is None or declared_total_hours is None:
        return None
    return abs(sum_row_hours - declared_total_hours) <= 1


def _extract_declared_total_cost(text: str) -> float | None:
    match = _TOTAL_COST_RE.search(text)
    if not match:
        return None
    return _to_float(match.group(1))


def _sum_table_costs(text: str) -> float | None:
    if not _COST_COLUMN_HEADER_RE.search(text):
        return None

    running_total = 0.0
    row_count = 0
    for match in _COST_TABLE_ROW_RE.finditer(text):
        task = match.group("task").strip().lower()
        if task in {"task", ""} or re.fullmatch(r"[-: ]+", task):
            continue
        cost = _to_float(match.group("cost"))
        if cost is None:
            continue
        running_total += cost
        row_count += 1

    return running_total if row_count else None


def _cost_match(
    sum_row_cost: float | None, declared_total_cost: float | None
) -> bool | None:
    if sum_row_cost is None or declared_total_cost is None or declared_total_cost <= 0:
        return None
    return abs(sum_row_cost - declared_total_cost) / declared_total_cost <= 0.02


def _applicable_checks(
    *,
    output_format: OutputFormat,
    has_title: bool,
    has_breakdown_table: bool,
    has_totals_section: bool,
    has_team_section: bool,
    has_duration_section: bool,
    hours_match: bool | None,
    finish_reason_ok: bool,
) -> list[bool]:
    if output_format == OutputFormat.PHASES_TABLE:
        return [
            has_title,
            has_breakdown_table,
            has_totals_section,
            has_team_section,
            has_duration_section,
            bool(hours_match),
            finish_reason_ok,
        ]

    return [
        has_title,
        has_totals_section,
        has_team_section,
        has_duration_section,
        finish_reason_ok,
    ]


def _build_issues(
    *,
    output_format: OutputFormat,
    has_title: bool,
    has_breakdown_table: bool,
    has_totals_section: bool,
    has_team_section: bool,
    has_duration_section: bool,
    declared_total_hours: int | None,
    sum_row_hours: int | None,
    hours_match: bool | None,
    declared_total_cost: float | None,
    sum_row_cost: float | None,
    cost_match: bool | None,
    finish_reason: str | None,
    finish_reason_ok: bool,
) -> list[str]:
    issues: list[str] = []
    if not has_title:
        issues.append("Missing estimation title or scope summary")
    if output_format == OutputFormat.PHASES_TABLE and not has_breakdown_table:
        issues.append("Missing phases table with the expected header")
    if not has_totals_section:
        issues.append("Missing total hours section")
    if not has_team_section:
        issues.append("Missing team recommendation or responsible roles")
    if not has_duration_section:
        issues.append("Missing duration or timeline")
    if output_format == OutputFormat.PHASES_TABLE and hours_match is False:
        issues.append(
            "Total hours mismatch: "
            f"declared {declared_total_hours} vs sum of rows {sum_row_hours}"
        )
    if cost_match is False:
        issues.append(
            "Total cost mismatch: "
            f"declared {declared_total_cost} EUR vs sum of rows {sum_row_cost} EUR"
        )
    if not finish_reason_ok:
        issues.append(
            f"Response truncated or unexpected finish_reason='{finish_reason}'"
        )
    return issues


def _to_int(raw: str) -> int | None:
    cleaned = raw.strip()
    if "=" in cleaned:
        cleaned = cleaned.rsplit("=", maxsplit=1)[-1]
    digits = re.sub(r"[^\d]", "", cleaned)
    return int(digits) if digits else None


def _to_float(raw: str) -> float | None:
    cleaned = raw.strip()
    if "=" in cleaned:
        cleaned = cleaned.rsplit("=", maxsplit=1)[-1]
    normalized = cleaned.replace(",", ".")
    number_match = re.search(r"[\d.]+", normalized)
    if not number_match:
        return None
    try:
        return float(number_match.group(0))
    except ValueError:
        return None
