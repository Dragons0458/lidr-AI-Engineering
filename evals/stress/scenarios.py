from dataclasses import dataclass
from typing import Iterable


SCENARIO_LENGTHS = (1, 3, 6, 10, 20)


@dataclass(frozen=True)
class StressTurn:
    turn_index: int
    transcript: str
    fact_to_remember: str

    def __post_init__(self) -> None:
        if self.fact_to_remember.lower() in self.transcript.lower():
            return

        object.__setattr__(
            self,
            "transcript",
            f"{self.transcript} Memory fact: {self.fact_to_remember}.",
        )


@dataclass(frozen=True)
class StressScenario:
    name: str
    description: str
    turns: tuple[StressTurn, ...]

    def truncate(self, max_turns: int) -> "StressScenario":
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        return StressScenario(
            name=f"{self.name}_{max_turns}",
            description=self.description,
            turns=self.turns[:max_turns],
        )

    def facts_before(self, turn_index: int) -> tuple[str, ...]:
        return tuple(
            turn.fact_to_remember for turn in self.turns if turn.turn_index < turn_index
        )


GROWING_PROJECT = StressScenario(
    name="growing",
    description="A coherent project that accumulates scope turn by turn.",
    turns=(
        StressTurn(
            1,
            "The project name: Nimbus. Build a web SaaS for B2B customer onboarding.",
            "project name: Nimbus",
        ),
        StressTurn(
            2,
            "Nimbus needs email and password authentication with password reset.",
            "scope includes authentication",
        ),
        StressTurn(
            3,
            "Add multi-tenant organizations with role based access for admins and agents.",
            "scope includes multi-tenant organizations",
        ),
        StressTurn(
            4,
            "Add an audit log for every onboarding status change and user action.",
            "scope includes audit log",
        ),
        StressTurn(
            5,
            "Add CSV export for filtered onboarding records and monthly reports.",
            "scope includes CSV export",
        ),
        StressTurn(
            6,
            "Integrate Slack notifications when a customer reaches blocked status.",
            "scope includes Slack notifications",
        ),
        StressTurn(
            7,
            "Add a dashboard with conversion funnel metrics and SLA breach counters.",
            "scope includes funnel dashboard",
        ),
        StressTurn(
            8,
            "Support Spanish and English localization for all customer facing screens.",
            "scope includes localization",
        ),
        StressTurn(
            9,
            "Add document upload for tax forms, contracts, and identity files.",
            "scope includes document upload",
        ),
        StressTurn(
            10,
            "Add virus scanning and file type validation for uploaded documents.",
            "scope includes virus scanning",
        ),
        StressTurn(
            11,
            "Add webhook callbacks for downstream CRM systems when onboarding completes.",
            "scope includes CRM webhooks",
        ),
        StressTurn(
            12,
            "Add an internal comments thread on each customer onboarding case.",
            "scope includes internal comments",
        ),
        StressTurn(
            13,
            "Add saved filters and custom columns for operations managers.",
            "scope includes saved filters",
        ),
        StressTurn(
            14,
            "Add weekly digest emails for accounts with overdue onboarding tasks.",
            "scope includes weekly digest emails",
        ),
        StressTurn(
            15,
            "Add SSO with SAML for enterprise customers.",
            "scope includes SAML SSO",
        ),
        StressTurn(
            16,
            "Add SCIM provisioning for enterprise user lifecycle management.",
            "scope includes SCIM provisioning",
        ),
        StressTurn(
            17,
            "Add data retention controls with configurable deletion windows.",
            "scope includes data retention controls",
        ),
        StressTurn(
            18,
            "Add an admin billing page with plan limits and usage history.",
            "scope includes billing usage",
        ),
        StressTurn(
            19,
            "Add public API keys and rate limits for partner integrations.",
            "scope includes public API keys",
        ),
        StressTurn(
            20,
            "Add production observability with traces, metrics, and alerting.",
            "scope includes production observability",
        ),
    ),
)


PIVOT_PROJECT = StressScenario(
    name="pivot",
    description="A project that changes stack after initial requirements.",
    turns=(
        StressTurn(
            1,
            "The project name: Atlas Field. Build a React web app for field inspections.",
            "project name: Atlas Field",
        ),
        StressTurn(
            2,
            "Use React with a FastAPI backend and PostgreSQL for inspection data.",
            "stack includes React",
        ),
        StressTurn(
            3,
            "The app needs offline draft inspections and later synchronization.",
            "scope includes offline drafts",
        ),
        StressTurn(
            4,
            "Inspectors need photo capture, GPS coordinates, and checklist scoring.",
            "scope includes photo capture",
        ),
        StressTurn(
            5,
            "Pivot the client app from React to Flutter because inspectors need native mobile support.",
            "stack includes Flutter",
        ),
        StressTurn(
            6,
            "Keep FastAPI for the backend API after the Flutter pivot.",
            "stack keeps FastAPI",
        ),
        StressTurn(
            7,
            "Replace browser local storage with SQLite on device for offline mode.",
            "storage includes SQLite on device",
        ),
        StressTurn(
            8,
            "Add push notifications for assigned inspections and rejected reports.",
            "scope includes push notifications",
        ),
        StressTurn(
            9,
            "Add supervisor review queues in a lightweight admin web portal.",
            "scope includes supervisor review",
        ),
        StressTurn(
            10,
            "The admin portal can stay React, but the inspector app must be Flutter.",
            "admin portal stays React",
        ),
        StressTurn(
            11,
            "Add maps with clustered inspection pins and route hints.",
            "scope includes maps",
        ),
        StressTurn(
            12,
            "Add device camera compression before upload to reduce bandwidth.",
            "scope includes camera compression",
        ),
        StressTurn(
            13,
            "Add conflict resolution when two supervisors edit the same inspection.",
            "scope includes conflict resolution",
        ),
        StressTurn(
            14,
            "Add PDF report generation from completed inspections.",
            "scope includes PDF reports",
        ),
        StressTurn(
            15,
            "Add geofencing alerts when inspections happen outside the expected area.",
            "scope includes geofencing alerts",
        ),
        StressTurn(
            16,
            "Add role based permissions for inspectors, supervisors, and admins.",
            "scope includes role permissions",
        ),
        StressTurn(
            17,
            "Add an API integration with the legacy asset registry.",
            "scope includes asset registry integration",
        ),
        StressTurn(
            18,
            "Add background sync retries with exponential backoff.",
            "scope includes background sync retries",
        ),
        StressTurn(
            19,
            "Add crash reporting and mobile release monitoring.",
            "scope includes crash reporting",
        ),
        StressTurn(
            20,
            "Add phased rollout through TestFlight and Google Play internal testing.",
            "scope includes phased mobile rollout",
        ),
    ),
)


CONTRADICTION_PROJECT = StressScenario(
    name="contradiction",
    description="A project that revises facts later in the conversation.",
    turns=(
        StressTurn(
            1,
            "The project name: LedgerFlow. Build an internal finance approval tool.",
            "project name: LedgerFlow",
        ),
        StressTurn(
            2,
            "The first version should cover purchase requests and invoice approvals.",
            "scope includes invoice approvals",
        ),
        StressTurn(
            3,
            "Budget locked: 30000 EUR for the initial delivery.",
            "budget locked: 30000 EUR",
        ),
        StressTurn(
            4,
            "Use Django, PostgreSQL, and server rendered admin screens.",
            "stack includes Django",
        ),
        StressTurn(
            5,
            "Add approval chains with finance manager and CFO thresholds.",
            "scope includes approval chains",
        ),
        StressTurn(
            6,
            "Add ERP export through scheduled CSV files.",
            "scope includes ERP CSV export",
        ),
        StressTurn(
            7,
            "Add an audit trail for approvals, rejections, and comments.",
            "scope includes finance audit trail",
        ),
        StressTurn(
            8,
            "Correction: budget locked: 80000 EUR because ERP integration is now in scope.",
            "budget locked: 80000 EUR",
        ),
        StressTurn(
            9,
            "Remove the idea of server rendered screens; the team now wants a React frontend.",
            "stack includes React frontend",
        ),
        StressTurn(
            10,
            "Keep Django only for the API and business rules.",
            "stack keeps Django API",
        ),
        StressTurn(
            11,
            "Add SSO through Azure AD for all finance users.",
            "scope includes Azure AD SSO",
        ),
        StressTurn(
            12,
            "Add notifications by email for pending approvals older than 48 hours.",
            "scope includes overdue approval emails",
        ),
        StressTurn(
            13,
            "Add exception reports for approvals outside policy thresholds.",
            "scope includes exception reports",
        ),
        StressTurn(
            14,
            "Add monthly close dashboards with spend by department.",
            "scope includes monthly close dashboards",
        ),
        StressTurn(
            15,
            "Correction: invoices are out of scope for phase one; purchase requests remain.",
            "phase one excludes invoices",
        ),
        StressTurn(
            16,
            "Add attachment uploads for quotes and purchase request evidence.",
            "scope includes quote attachments",
        ),
        StressTurn(
            17,
            "Add vendor master data lookup from the ERP.",
            "scope includes vendor lookup",
        ),
        StressTurn(
            18,
            "Add permission rules by department and cost center.",
            "scope includes cost center permissions",
        ),
        StressTurn(
            19,
            "Add immutable approval records for audit compliance.",
            "scope includes immutable approval records",
        ),
        StressTurn(
            20,
            "Final constraint: go live before the Q4 close process.",
            "deadline before Q4 close",
        ),
    ),
)


SCENARIOS = {
    scenario.name: scenario
    for scenario in (GROWING_PROJECT, PIVOT_PROJECT, CONTRADICTION_PROJECT)
}


def get_scenario(name: str, *, max_turns: int | None = None) -> StressScenario:
    try:
        scenario = SCENARIOS[name]
    except KeyError as e:
        valid_names = ", ".join(sorted(SCENARIOS))
        raise ValueError(
            f"Unknown scenario {name!r}. Valid scenarios: {valid_names}"
        ) from e

    if max_turns is None:
        return scenario

    return scenario.truncate(max_turns)


def iter_scenarios(
    names: Iterable[str] | None = None,
    *,
    lengths: Iterable[int] = SCENARIO_LENGTHS,
) -> Iterable[StressScenario]:
    selected_names = tuple(names) if names is not None else tuple(SCENARIOS)

    for name in selected_names:
        scenario = get_scenario(name)
        for length in lengths:
            yield scenario.truncate(length)
