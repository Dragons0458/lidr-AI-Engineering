# Estimador CAG

Aplicación FastAPI + Streamlit que genera estimaciones de esfuerzo para proyectos de software a partir de resúmenes de
reuniones usando LLMs a través de LiteLLM.

## Inicio rápido (Docker Compose)

1. Crea el archivo de entorno:

```bash
cp .env.example .env
```

2. Edita `.env` con una clave de API real (OpenAI, Anthropic o Google).

    - Para Docker Compose, `streamlit` lee `ESTIMATION_API_BASE_URL` desde el entorno.
    - El archivo `.streamlit/secrets.toml` es opcional y no es obligatorio.

3. Construye y ejecuta todo:

```bash
docker compose up --build
```

4. Abre las URLs:

- Documentación de la API: `http://localhost:8000/docs`
- Salud: `http://localhost:8000/health`
- Interfaz de Streamlit: `http://localhost:8501`

> Nota: salud se expone en `/health` (raíz), no en `/api/v1/health`.

5. Comprobación rápida opcional:

```bash
curl http://localhost:8000/health
```

Detener los servicios:

```bash
docker compose down
```

## Funcionalidades

- API REST para generar estimaciones de proyectos.
- Endpoint opcional de estimación por streaming.
- Versionado de prompts (`v1`, `v2`) con plantillas Jinja.
- Memoria de sesión con `<project_metadata>` inyectado en el system prompt.
- Esquemas estructurados de solicitud/respuesta con validación de Pydantic.
- Reporte de costo y uso de tokens basado en reglas de precios por modelo.
- Interfaz de Streamlit para pruebas interactivas y uso demostrativo.
- Pruebas de renderizado de prompts con `pytest`.

## Stack tecnológico

- Python `3.11+`
- FastAPI
- LiteLLM
- Instructor
- Jinja2
- Streamlit
- Structlog
- Pytest
- UV (se incluye el lockfile del proyecto como `uv.lock`)

## Estructura del proyecto

```text
app/
  main.py                        # Punto de entrada de la app FastAPI
  config.py                      # Configuración basada en variables de entorno
  routers/estimations.py         # Endpoints de la API
  services/estimation_service.py # Llamada al LLM + llamada por streaming
  services/sessions.py           # Estado de sesión y ProjectMetadata en memoria
  services/project_metadata_extractor.py # Extracción LLM de hechos del proyecto
  formatters/llm_formatters.py   # Mapea la salida del LLM a la respuesta de la API
  schemas/estimation.py          # Solicitud/respuesta y enums
  prompts/
    loader.py                    # Renderizado Jinja para versiones de prompts
    estimation/
      v1/
      v2/
streamlit_app.py                 # Frontend de Streamlit
tests/
  unit/
    prompts/test_loader.py
    routers/test_sessions.py
    services/test_project_metadata_extractor.py
    services/test_sessions_service.py
  integration/
    routers/test_sessions_integration.py
    evals/
      fixtures.py
      helpers.py
      test_estimation_goldens.py
      test_estimation_judge.py
```

## Requisitos

- Python `>=3.11`
- Una clave de API de proveedor según el `LLM_PROVIDER` seleccionado:
    - OpenAI -> `OPENAI_API_KEY`
    - Anthropic -> `ANTHROPIC_API_KEY`
    - Google -> `GOOGLE_API_KEY`

## Instalación

Usando `uv` (recomendado):

```bash
uv sync
```

Si necesitas dependencias de desarrollo:

```bash
uv sync --group dev
```

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
APP_ENV=development
LOG_LEVEL=DEBUG
```

Para otros proveedores:

- Define `LLM_PROVIDER=anthropic` y `ANTHROPIC_API_KEY=...`
- Define `LLM_PROVIDER=google` y `GOOGLE_API_KEY=...`

## Ejecutar la API

```bash
uv run uvicorn app.main:app --reload
```

Endpoints locales predeterminados:

- `GET /health`
- `POST /sessions`
- `POST /sessions/{session_id}/estimate`
- `POST /api/v1/estimate`
- `POST /api/v1/estimate/stream`

## Ejecutar la app de Streamlit

```bash
uv run streamlit run streamlit_app.py
```

De forma predeterminada, la app de Streamlit apunta a `http://localhost:8000/api/v1`.

Puedes sobrescribir esto mediante la clave secreta de Streamlit:

- `ESTIMATION_API_BASE_URL`

## Modelo de solicitud de la API

`POST /api/v1/estimate?prompt_version=v1|v2`

Forma del cuerpo de la solicitud:

```json
{
  "description": "Meeting summary text...",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "line_items",
  "reference_projects": [
    {
      "name": "Billing MVP",
      "summary": "Project focused on subscriptions and invoicing.",
      "estimated_hours": 280,
      "team": "2 backend, 1 frontend",
      "outcome": "Released in 8 weeks"
    }
  ]
}
```

Enums:

- `project_type`: `mobile_app`, `web_saas`, `internal_tool`, `data_pipeline`
- `detail_level`: `summary`, `medium`, `detailed`
- `output_format`: `phases_table`, `line_items`, `narrative`

## Estimación de sesión con adjuntos

`POST /sessions` crea una sesión local al proceso y devuelve:

```json
{
  "session_id": "2b5d1f4a-4cb8-4f02-ae55-0c4dbcbf2f72"
}
```

Para reutilizar memoria entre páginas, envía ese `session_id` en llamadas posteriores:

```bash
curl -X POST "http://localhost:8000/sessions/{session_id}/estimate" \
  -F "description=Project description text..." \
  -F "attachments=@scope.pdf" \
  -F "attachments=@requirements.docx"
```

Este endpoint acepta `multipart/form-data`:

- `description`: texto requerido con la descripción del proyecto o resumen de la reunión.
- `attachments`: campo de archivo repetido opcional con documentación complementaria.
  Solo se aceptan archivos `.pdf` y `.docx`; otros tipos se rechazan con `415`.

Por ahora, este endpoint de sesión mantiene pequeño el contrato multipart solicitado y
usa valores predeterminados de estimación internamente: `web_saas`, `medium` y `line_items`.

Cada estimación de sesión renderiza el system prompt con un bloque
`<project_metadata>`. En la primera llamada el bloque se envía vacío; después de
recibir la respuesta del LLM, el servicio hace una segunda llamada de extracción
para actualizar `ProjectMetadata` con hechos durables como nombre del proyecto,
tecnologías mencionadas, tecnologías excluidas, tamaño de equipo asumido y
alcance acordado. En las siguientes llamadas esos hechos se inyectan en el mismo
bloque para dar contexto acumulado a la estimación.

Se eligió la estrategia de **LLM extractor** en vez de una heurística por regex
porque los datos relevantes pueden aparecer con formulaciones muy variadas en la
transcripción, los adjuntos o la propia estimación. La llamada adicional cuesta
más tokens y latencia por turno, pero reduce reglas frágiles, permite conservar
hechos previos cuando el turno nuevo no los menciona y valida la salida contra el
modelo Pydantic `ProjectMetadata` antes de guardarla en memoria. Los prompts del
extractor también viven en plantillas Jinja y la respuesta estructurada se
obtiene con `instructor`, evitando parseo manual de JSON sobre la salida de
LiteLLM. Cuando la descripción más reciente contradice hechos anteriores, por
ejemplo "ya no usar React" o "evitar Firebase", el extractor devuelve el estado
actualizado completo para retirar esas tecnologías de las mencionadas y moverlas
a `excluded_technologies`.

La implementación usa la ruta B: los adjuntos se leen en el servicio de IA y se
extraen como texto usando `pypdf` para PDF y `python-docx` para DOCX. Las
plantillas del prompt los renderizan junto a la
transcripción con un separador claro:

```text
--- attachment: scope.pdf ---
...
```

Esta ruta se eligió porque hace explícita la memoria de sesión en nuestro propio servicio:
el texto extraído de los adjuntos se puede almacenar, inspeccionar, probar y reutilizar
en comportamientos futuros de sesión sin depender de la semántica específica de Files
API de cada proveedor.

## Stress test CAG

El stress test mide donde empieza a degradarse el CAG cuando crecen el historial
conversacional y los adjuntos. El flujo genera:

- `evals/stress/results.csv`: una fila por turno ejecutado.
- `evals/stress/REPORT.md`: lectura del CSV con tablas de latencia, coste y recall.

El runner usa los escenarios de `evals/stress/scenarios.py`, genera PDFs
sinteticos deterministas con `evals/stress/fixtures/build_pdfs.py` y evalua tres
metricas binarias:

- `LatencyBudgetMetric`: pasa si `latency_ms <= 4000` por defecto.
- `CostBudgetMetric`: pasa si `cost_usd <= 0.05` por defecto.
- `MemoryDriftMetric`: pasa si los facts esperados aparecen en `summary`,
  `anchors` o `metadata` del snapshot de sesion.

### 1. Preparar entorno

Instala dependencias:

```bash
uv sync --group dev
```

Configura el proveedor LLM en `.env`. Ejemplo con Gemini:

```env
LLM_PROVIDER=google
LLM_MODEL=gemini/gemini-2.5-flash
GOOGLE_API_KEY=your_key_here
```

### 2. Elegir modo de ejecucion

Hay dos formas equivalentes.

Modo in-process, sin levantar servidor:

```bash
uv run python -m evals.stress.run \
  --scenarios growing \
  --attachment-sizes 0 \
  --repeats 1 \
  --turn-counts 1 \
  --output evals/stress/results_smoke.csv
```

Modo HTTP, contra una API local. En una terminal levanta la API:

```bash
uv run uvicorn app.main:app --reload
```

En otra terminal ejecuta el runner:

```bash
uv run python -m evals.stress.run \
  --http http://localhost:8000 \
  --scenarios growing \
  --attachment-sizes 0 \
  --repeats 1 \
  --turn-counts 1 \
  --output evals/stress/results_smoke.csv
```

Al terminar el smoke test se espera un CSV con 2 lineas: header + 1 fila de
datos. Sirve para validar credenciales, conectividad con el LLM y escritura de
resultados antes de gastar mas tokens.

### 3. Generar una muestra reducida para reporte

Esta es la ejecucion usada para el reporte versionado en este repo:

```bash
uv run python -m evals.stress.run \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,20,100 \
  --repeats 1 \
  --turn-counts 1,6,20 \
  --output evals/stress/results.csv
```

Resultado esperado:

- `evals/stress/results.csv` existe.
- `wc -l evals/stress/results.csv` devuelve `244`.
- Hay 243 filas de datos: `3 escenarios x 3 tamanos x (1 + 6 + 20) turnos`.
- Las columnas incluyen `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`,
  `memory_drift_score`, `latency_budget_passed` y `cost_budget_passed`.

Valida el archivo:

```bash
wc -l evals/stress/results.csv
sed -n '1,5p' evals/stress/results.csv
```

### 4. Ejecutar el stress test completo del ejercicio

Usa este comando si quieres cubrir todos los tamanos y repeticiones pedidos por
la guia:

```bash
uv run python -m evals.stress.run \
  --http http://localhost:8000 \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,5,20,50,100 \
  --repeats 3 \
  --output evals/stress/results.csv
```

Resultado esperado:

- `evals/stress/results.csv` existe.
- `wc -l evals/stress/results.csv` devuelve `1801`.
- Hay 1800 filas de datos:
  `3 escenarios x 5 tamanos x 3 repeticiones x (1 + 3 + 6 + 10 + 20) turnos`.

Esta ejecucion hace muchas llamadas al LLM. Para ahorrar coste, usa primero el
smoke test y luego la muestra reducida.

### 5. Actualizar el reporte Markdown

El runner produce el CSV. El reporte final se guarda en:

```text
evals/stress/REPORT.md
```

El reporte debe contener, como minimo:

- Tabla resumen por escenario y tamano de adjunto con P50/P95 de `latency_ms`,
  coste total, hit rate de cache y recall medio.
- Curva `latency_ms` vs `tokens_in`.
- Curva de `cost_usd` acumulado vs `turn_index`.
- Curva de `MemoryDriftMetric` vs longitud de historial.
- Dos parrafos de lectura con al menos una afirmacion cuantitativa.

Despues de editar el reporte, verifica los entregables:

```bash
test -f evals/stress/results.csv
test -f evals/stress/REPORT.md
wc -l evals/stress/results.csv evals/stress/REPORT.md
```

Para validar el codigo relacionado:

```bash
uv run pytest tests/unit/stress tests/unit/routers/test_sessions.py tests/integration/routers/test_sessions_integration.py
```

Resultado esperado de la suite relevante:

```text
30 passed
```

## Modelo de respuesta

```json
{
  "estimation": "...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "timestamp": "2026-05-13T23:00:00.000000",
  "usage": {
    "tokens_used": 1234,
    "cost_estimate": 0.0009
  },
  "prompt_version": "v1"
}
```

## Versiones de prompt

- `v1`: instrucciones clásicas de estimación con salidas de planificación concisas.
- `v2`: estilo de planificación consciente de riesgos; incluye guía de horas de colchón y mayor énfasis en
  riesgos/dependencias.

## Flujo de commit con linters

Para ejecutar validaciones de calidad automáticamente en cada commit:

```bash
uv run pre-commit install
```

Hooks configurados:

- `ruff-check --fix` para linting y autofixes rápidos.
- `ruff-format` para formato consistente.
- `pre-commit-hooks` básicos (`check-yaml`, `end-of-file-fixer`, `trailing-whitespace`).
- `pytest` rápido sobre `tests/unit` y `tests/integration/routers` para evitar regresiones básicas antes del commit.

Ejecutar todos los hooks manualmente:

```bash
uv run pre-commit run --all-files
```

Para evitar costo/latencia, las evaluaciones reales del LLM (`tests/integration/evals`) no corren en el hook de commit.
