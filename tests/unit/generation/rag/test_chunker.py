from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.schemas import Budget


def test_chunker_creates_traceable_component_chunks():
    budget = Budget.model_validate(
        {
            "budget_id": "BUD-001",
            "client_metadata": {
                "name": "Demo Bank",
                "sector": "finance",
                "country": "Colombia",
            },
            "project_summary": "Banking API modernization",
            "main_technology": "FastAPI",
            "year": 2025,
            "total_estimated_hours": 100,
            "components": [
                {
                    "component_id": "auth",
                    "name": "Authorization service",
                    "description": "JWT access control for banking APIs.",
                    "tech_stack": ["FastAPI", "JWT"],
                    "estimated_hours": 60,
                    "complexity": "high",
                    "dependencies": [],
                }
            ],
        }
    )

    chunks = JSONStructuralChunker().chunk([budget])

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "BUD-001::auth"
    assert "Banking API modernization" in chunks[0].text
    assert "Authorization service" in chunks[0].text
    assert chunks[0].metadata == {
        "budget_id": "BUD-001",
        "component_id": "auth",
        "client_sector": "finance",
        "main_technology": "FastAPI",
        "year": 2025,
        "complexity": "high",
        "estimated_hours": 60,
    }
    assert chunks[0].token_count > 0
