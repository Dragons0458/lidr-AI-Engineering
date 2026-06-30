"""Compatibility shims for RAGAS 0.4.3 with modern langchain-community.

RAGAS imports VertexAI classes from removed ``langchain_community`` paths at
module load time. Patch those legacy modules before ``import ragas``.
"""

from __future__ import annotations

import sys
import types


def patch_vertexai_imports() -> None:
    """Make ``import ragas`` succeed without Google VertexAI configured."""
    legacy_chat = "langchain_community.chat_models.vertexai"
    if legacy_chat not in sys.modules:
        try:
            from langchain_google_vertexai import ChatVertexAI, VertexAI
        except ImportError:
            chat_mod = types.ModuleType(legacy_chat)
            chat_mod.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[legacy_chat] = chat_mod
            _ensure_llms_vertexai(type("VertexAI", (), {}))
        else:
            chat_mod = types.ModuleType(legacy_chat)
            chat_mod.ChatVertexAI = ChatVertexAI
            sys.modules[legacy_chat] = chat_mod
            _ensure_llms_vertexai(VertexAI)


def _ensure_llms_vertexai(vertexai_cls: type) -> None:
    legacy_llms = "langchain_community.llms"
    try:
        import langchain_community.llms as llms_mod
    except ImportError:
        llms_mod = types.ModuleType(legacy_llms)
        sys.modules[legacy_llms] = llms_mod
    setattr(llms_mod, "VertexAI", vertexai_cls)
