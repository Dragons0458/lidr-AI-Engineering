run:
    uv run uvicorn app.main:app --reload

format:
    uv run ruff format .

lint:
    uv run ruff check .

test:
    uv run pytest

streamlit:
    PYTHONPATH=. uv run streamlit run streamlit_ui/home.py

enable-pre-commit:
    uv run pre-commit install
