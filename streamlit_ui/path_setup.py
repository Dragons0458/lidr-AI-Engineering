"""Add project root to sys.path so ``streamlit_ui`` and ``app`` imports work.

Streamlit adds ``streamlit_ui/`` to sys.path; entry script must be ``home.py`` (not
``app.py``) so it does not shadow the backend ``app`` package.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
