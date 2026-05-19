ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": "El cliente necesita una plataforma web de gestión de inventario...",
        "estimation": """
        ## Estimación: Plataforma de Gestión de Inventario

        ### Desglose de tareas:
        1. Diseño UI/UX: 40 horas
        2. Backend API (CRUD inventario): 60 horas
        3. Autenticación y roles: 20 horas
        4. Dashboard con métricas: 30 horas
        5. Testing y QA: 25 horas

        **Total estimado: 175 horas**
        **Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
        **Duración estimada: 6-8 semanas**
        """,
    },
    {
        "meeting_summary": "El cliente quiere una API para procesamiento de pagos con integración a Stripe y manejo de suscripciones.",
        "estimation": """
        ## Estimación: API de Pagos y Suscripciones

        ### Desglose de tareas:
        1. Diseño de arquitectura: 16 horas
        2. Integración con Stripe: 40 horas
        3. Gestión de suscripciones: 32 horas
        4. Webhooks y eventos: 24 horas
        5. Seguridad y validaciones: 20 horas
        6. Testing y QA: 24 horas

        **Total estimado: 156 horas**
        **Equipo recomendado: 1 backend senior + 1 QA**
        **Duración estimada: 4-6 semanas**
        """,
    },
    {
        "meeting_summary": "El cliente necesita una aplicación móvil básica para reservas de citas médicas con notificaciones.",
        "estimation": """
        ## Estimación: App Móvil de Reservas Médicas

        ### Desglose de tareas:
        1. Diseño UI/UX móvil: 50 horas
        2. Desarrollo frontend (React Native): 80 horas
        3. Backend API (citas y usuarios): 60 horas
        4. Notificaciones push: 20 horas
        5. Integración calendario: 16 horas
        6. Testing y QA: 30 horas

        **Total estimado: 256 horas**
        **Equipo recomendado: 1 frontend mobile + 1 backend + 1 diseñador UX**
        **Duración estimada: 8-10 semanas**
        """,
    },
    {
        "meeting_summary": "El cliente quiere automatizar reportes diarios a partir de datos en S3 usando AWS.",
        "estimation": """
        ## Estimación: Automatización de Reportes en AWS

        ### Desglose de tareas:
        1. Diseño de arquitectura serverless: 12 horas
        2. Procesamiento de datos (Lambda/Glue): 40 horas
        3. Integración con S3: 16 horas
        4. Generación de reportes (CSV/Excel): 24 horas
        5. Orquestación con Step Functions: 20 horas
        6. Testing y monitoreo: 20 horas

        **Total estimado: 132 horas**
        **Equipo recomendado: 1 backend/cloud engineer**
        **Duración estimada: 3-5 semanas**
        """,
    },
    {
        "meeting_summary": "El cliente necesita implementar autenticación con OAuth2 y SSO en su plataforma existente.",
        "estimation": """
        ## Estimación: Implementación OAuth2 y SSO

        ### Desglose de tareas:
        1. Análisis del sistema actual: 12 horas
        2. Integración OAuth2 (Google, Microsoft): 32 horas
        3. Implementación SSO: 24 horas
        4. Manejo de sesiones y tokens: 16 horas
        5. Seguridad y validaciones: 16 horas
        6. Testing y QA: 16 horas

        **Total estimado: 116 horas**
        **Equipo recomendado: 1 backend engineer**
        **Duración estimada: 3-4 semanas**
        """,
    },
]


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Format estimation examples into a string suitable for injection into a system prompt."""
    parts: list[str] = []
    for i, example in enumerate(examples, start=1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"Meeting Summary:\n{example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}\n"
        )
    return "\n".join(parts)
