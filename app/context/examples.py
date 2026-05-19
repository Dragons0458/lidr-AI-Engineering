from dataclasses import dataclass
import json

from app.schemas.estimation import ExampleFormat


@dataclass(frozen=True)
class ExamplePhase:
    name: str
    scope: str
    hours: int
    team: str


@dataclass(frozen=True)
class CanonicalExample:
    title: str
    meeting_summary: str
    phases: tuple[ExamplePhase, ...]
    total_hours: int
    team: str
    duration_weeks: str
    estimation_markdown: str


ESTIMATION_EXAMPLES: tuple[CanonicalExample, ...] = (
    CanonicalExample(
        title="Plataforma de Gestion de Inventario",
        meeting_summary="El cliente necesita una plataforma web de gestion de inventario con roles y metricas.",
        phases=(
            ExamplePhase(
                "Discovery", "Definicion de alcance y criterios", 16, "PM + Tech Lead"
            ),
            ExamplePhase(
                "Diseno",
                "Wireframes y UI para flujos principales",
                32,
                "Product Designer",
            ),
            ExamplePhase(
                "Backend",
                "API CRUD de inventario y autenticacion",
                64,
                "Backend Engineer",
            ),
            ExamplePhase(
                "Frontend", "Dashboard, formularios y reportes", 48, "Frontend Engineer"
            ),
            ExamplePhase(
                "QA", "Pruebas funcionales y estabilizacion", 24, "QA Engineer"
            ),
        ),
        total_hours=184,
        team="2 full-stack developers + 1 product designer part-time",
        duration_weeks="6-8",
        estimation_markdown=(
            "## Estimacion: Plataforma de Gestion de Inventario\n\n"
            "| Phase | Tasks | Hours | Team |\n"
            "|---|---|---:|---|\n"
            "| Discovery | Definicion de alcance y criterios | 16 | PM + Tech Lead |\n"
            "| Diseno | Wireframes y UI para flujos principales | 32 | Product Designer |\n"
            "| Backend | API CRUD de inventario y autenticacion | 64 | Backend Engineer |\n"
            "| Frontend | Dashboard, formularios y reportes | 48 | Frontend Engineer |\n"
            "| QA | Pruebas funcionales y estabilizacion | 24 | QA Engineer |\n"
            "| Total | - | 184 | Core team |\n\n"
            "Total estimated hours: 184h\n"
            "Equipo recomendado: 2 full-stack developers + 1 product designer part-time\n"
            "Duracion estimada: 6-8 semanas"
        ),
    ),
    CanonicalExample(
        title="API de Pagos y Suscripciones",
        meeting_summary="El cliente quiere una API para pagos con Stripe, suscripciones y webhooks.",
        phases=(
            ExamplePhase(
                "Discovery", "Alineacion de flujos de pago", 12, "PM + Backend Lead"
            ),
            ExamplePhase(
                "Arquitectura",
                "Modelo de pagos, eventos y seguridad",
                20,
                "Backend Lead",
            ),
            ExamplePhase(
                "Backend",
                "Integracion Stripe y gestion de suscripciones",
                72,
                "Backend Engineer",
            ),
            ExamplePhase(
                "Webhooks",
                "Procesamiento idempotente de eventos",
                24,
                "Backend Engineer",
            ),
            ExamplePhase(
                "QA", "Pruebas de pagos, errores y regresion", 28, "QA Engineer"
            ),
        ),
        total_hours=156,
        team="1 backend senior + 1 QA",
        duration_weeks="4-6",
        estimation_markdown=(
            "## Estimacion: API de Pagos y Suscripciones\n\n"
            "1. Discovery y alineacion de flujos - 12h - PM + Backend Lead\n"
            "2. Arquitectura de pagos y seguridad - 20h - Backend Lead\n"
            "3. Integracion Stripe y suscripciones - 72h - Backend Engineer\n"
            "4. Webhooks e idempotencia - 24h - Backend Engineer\n"
            "5. QA de pagos y regresion - 28h - QA Engineer\n"
            "Total estimated hours: 156h\n"
            "Equipo recomendado: 1 backend senior + 1 QA\n"
            "Duracion estimada: 4-6 semanas"
        ),
    ),
    CanonicalExample(
        title="App Movil de Reservas Medicas",
        meeting_summary="El cliente necesita una app movil para reservas medicas, perfiles y notificaciones.",
        phases=(
            ExamplePhase(
                "Discovery", "Alcance, roles y criterios de aceptacion", 18, "PM"
            ),
            ExamplePhase("Diseno", "Flujos moviles y UI", 46, "Product Designer"),
            ExamplePhase(
                "Backend",
                "APIs de citas, usuarios y disponibilidad",
                62,
                "Backend Engineer",
            ),
            ExamplePhase(
                "Mobile", "Reservas, perfiles y notificaciones", 86, "Mobile Engineer"
            ),
            ExamplePhase(
                "QA", "Pruebas funcionales en dispositivos", 34, "QA Engineer"
            ),
        ),
        total_hours=246,
        team="1 mobile engineer + 1 backend + 1 product designer",
        duration_weeks="8-10",
        estimation_markdown=(
            "## Estimacion: App Movil de Reservas Medicas\n\n"
            "| Phase | Tasks | Hours | Team |\n"
            "|---|---|---:|---|\n"
            "| Discovery | Alcance, roles y criterios de aceptacion | 18 | PM |\n"
            "| Diseno | Flujos moviles y UI | 46 | Product Designer |\n"
            "| Backend | APIs de citas, usuarios y disponibilidad | 62 | Backend Engineer |\n"
            "| Mobile | Reservas, perfiles y notificaciones | 86 | Mobile Engineer |\n"
            "| QA | Pruebas funcionales en dispositivos | 34 | QA Engineer |\n"
            "| Total | - | 246 | Core team |\n\n"
            "Total estimated hours: 246h\n"
            "Equipo recomendado: 1 mobile engineer + 1 backend + 1 product designer\n"
            "Duracion estimada: 8-10 semanas"
        ),
    ),
    CanonicalExample(
        title="Automatizacion de Reportes en AWS",
        meeting_summary="El cliente quiere reportes diarios desde datos en S3 usando servicios AWS.",
        phases=(
            ExamplePhase(
                "Arquitectura", "Diseno serverless y permisos", 16, "Cloud Engineer"
            ),
            ExamplePhase(
                "Procesamiento",
                "Lambda/Glue para transformar datos",
                44,
                "Data Engineer",
            ),
            ExamplePhase(
                "Reportes",
                "Generacion CSV/Excel y almacenamiento",
                28,
                "Backend Engineer",
            ),
            ExamplePhase(
                "Orquestacion", "Step Functions y monitoreo", 24, "Cloud Engineer"
            ),
            ExamplePhase("QA", "Pruebas con datos historicos", 20, "QA Engineer"),
        ),
        total_hours=132,
        team="1 cloud engineer + 1 data/backend engineer",
        duration_weeks="3-5",
        estimation_markdown=(
            "## Estimacion: Automatizacion de Reportes en AWS\n\n"
            "1. Arquitectura serverless y permisos - 16h - Cloud Engineer\n"
            "2. Procesamiento Lambda/Glue - 44h - Data Engineer\n"
            "3. Generacion de reportes CSV/Excel - 28h - Backend Engineer\n"
            "4. Orquestacion y monitoreo - 24h - Cloud Engineer\n"
            "5. QA con datos historicos - 20h - QA Engineer\n"
            "Total estimated hours: 132h\n"
            "Equipo recomendado: 1 cloud engineer + 1 data/backend engineer\n"
            "Duracion estimada: 3-5 semanas"
        ),
    ),
    CanonicalExample(
        title="Implementacion OAuth2 y SSO",
        meeting_summary="El cliente necesita OAuth2 y SSO en una plataforma existente.",
        phases=(
            ExamplePhase(
                "Analisis", "Revision del sistema actual", 12, "Backend Engineer"
            ),
            ExamplePhase(
                "OAuth2", "Integracion Google y Microsoft", 32, "Backend Engineer"
            ),
            ExamplePhase(
                "SSO", "Flujos SSO y manejo de sesiones", 36, "Backend Engineer"
            ),
            ExamplePhase(
                "Seguridad",
                "Validaciones, expiracion y auditoria",
                20,
                "Security Engineer",
            ),
            ExamplePhase(
                "QA", "Pruebas de autenticacion y regresion", 16, "QA Engineer"
            ),
        ),
        total_hours=116,
        team="1 backend engineer + security review part-time",
        duration_weeks="3-4",
        estimation_markdown=(
            "## Estimacion: Implementacion OAuth2 y SSO\n\n"
            "1. Analisis del sistema actual - 12h - Backend Engineer\n"
            "2. Integracion OAuth2 - 32h - Backend Engineer\n"
            "3. SSO y manejo de sesiones - 36h - Backend Engineer\n"
            "4. Seguridad y auditoria - 20h - Security Engineer\n"
            "5. QA de autenticacion - 16h - QA Engineer\n"
            "Total estimated hours: 116h\n"
            "Equipo recomendado: 1 backend engineer + security review part-time\n"
            "Duracion estimada: 3-4 semanas"
        ),
    ),
)


def select_examples(n: int) -> tuple[CanonicalExample, ...]:
    return ESTIMATION_EXAMPLES[: max(0, min(n, 5))]


def format_example_markdown(example: CanonicalExample) -> str:
    return example.estimation_markdown


def format_example_json(example: CanonicalExample) -> str:
    payload = {
        "title": example.title,
        "meeting_summary": example.meeting_summary,
        "phases": [
            {
                "name": phase.name,
                "scope": phase.scope,
                "hours": phase.hours,
                "team": phase.team,
            }
            for phase in example.phases
        ],
        "total_hours": example.total_hours,
        "team": example.team,
        "duration_weeks": example.duration_weeks,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_example_narrative(example: CanonicalExample) -> str:
    phases = "; ".join(
        f"{phase.name}: {phase.scope} ({phase.hours}h)" for phase in example.phases
    )
    return (
        f"{example.title}. Alcance: {example.meeting_summary} "
        f"Desglose: {phases}. Total: {example.total_hours}h. "
        f"Equipo: {example.team}. Duracion: {example.duration_weeks} semanas."
    )


def format_example(example: CanonicalExample, example_format: ExampleFormat) -> str:
    if example_format == "json":
        return format_example_json(example)
    if example_format == "narrative":
        return format_example_narrative(example)
    return format_example_markdown(example)


def build_prompt_examples(
    *, use_examples: bool, num_examples: int, example_format: ExampleFormat
) -> list[dict[str, str]]:
    if not use_examples:
        return []

    return [
        {
            "id": str(index),
            "format": example_format,
            "input": example.meeting_summary,
            "output": format_example(example, example_format),
        }
        for index, example in enumerate(select_examples(num_examples), start=1)
    ]


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Keep the old helper available for callers outside the prompt loader."""
    parts: list[str] = []
    for i, example in enumerate(examples, start=1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"Meeting Summary:\n{example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}\n"
        )
    return "\n".join(parts)
