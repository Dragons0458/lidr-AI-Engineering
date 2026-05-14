import time
import re

import httpx
import streamlit as st
DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"


def get_api_base_url() -> str:
    return st.secrets.get("ESTIMATION_API_BASE_URL", DEFAULT_API_BASE_URL)


def add_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)


def build_stream_url() -> str:
    return f"{get_api_base_url().rstrip('/')}/estimate/stream"


def extract_markdown_content(raw_text: str) -> str:
    """Extract markdown payload when model wraps output in structured text."""
    text = raw_text.strip()

    wrapped_match = re.search(r'"estimation"\s*:\s*"""(.*?)"""', text, re.DOTALL)
    if wrapped_match:
        return wrapped_match.group(1).strip()

    heading_match = re.search(r"(?m)^\s{0,3}#{1,6}\s+\S.*$", text)
    if heading_match:
        return text[heading_match.start():].strip().strip('"')

    return text.replace('"""', "").strip()


# Initialize chat and UI state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.model = "-"
    st.session_state.response_time = 0

# Render existing history
for message in st.session_state.messages:
    add_message(message["role"], message["content"])

# Accept user input
if prompt := st.chat_input("Escribe tu mensaje"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    add_message("user", prompt)

    start_time = time.perf_counter()
    try:
        with st.chat_message("assistant"):
            markdown_placeholder = st.empty()
            response_chunks = []
            url = build_stream_url()

            with httpx.stream(
                "POST",
                url,
                json={"transcript": prompt},
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                st.session_state.model = response.headers.get("X-LLM-Model", "unknown")

                for chunk in response.iter_text():
                    if chunk:
                        response_chunks.append(chunk)
                        markdown_placeholder.markdown(
                            extract_markdown_content("".join(response_chunks))
                        )

            response = extract_markdown_content("".join(response_chunks))
            markdown_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.response_time = time.perf_counter() - start_time
    except httpx.HTTPError as exc:
        error_message = f"Error consumiendo la API: {exc}"
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.session_state.response_time = time.perf_counter() - start_time

with st.sidebar:
    st.title("Estimaciones")
    if st.button("Nueva conversacion"):
        st.session_state.messages = []
        st.session_state.response_time = 0
        st.session_state.model = "-"
        st.rerun()
    st.text_input("API base URL", value=get_api_base_url(), disabled=True)
    st.metric("Model", st.session_state.model)
    st.metric("Response time", f"{st.session_state.response_time:.2f}s")
