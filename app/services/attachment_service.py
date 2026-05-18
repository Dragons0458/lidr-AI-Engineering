from io import BytesIO
from pathlib import PurePath

from docx import Document
from pypdf import PdfReader
from starlette.datastructures import UploadFile

from app.schemas.attachments import AttachmentText


class AttachmentTextExtractionError(Exception):
    """Raised when an uploaded attachment cannot be converted into prompt text."""


class UnsupportedAttachmentTypeError(Exception):
    """Raised when an uploaded attachment type is not supported."""


async def extract_attachment_texts(
    attachments: list[UploadFile] | None,
) -> list[AttachmentText]:
    """Extract text from supported uploaded documents for prompt rendering."""
    if not attachments:
        return []

    attachment_texts = []
    for attachment in attachments:
        content = await attachment.read()
        if not content:
            continue

        filename = attachment.filename or "unnamed"
        attachment_texts.append(
            AttachmentText(
                filename=filename,
                content=_extract_attachment_text(content, filename),
            )
        )

    return attachment_texts


def _extract_attachment_text(content: bytes, filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(content, filename)
    if suffix == ".docx":
        return _extract_docx_text(content, filename)

    raise UnsupportedAttachmentTypeError(
        f"Attachment {filename!r} is not supported. Only PDF and DOCX files are allowed."
    )


def _extract_pdf_text(content: bytes, filename: str) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        raise AttachmentTextExtractionError(
            f"Attachment {filename!r} could not be read as a PDF."
        ) from e

    if not text:
        raise AttachmentTextExtractionError(
            f"Attachment {filename!r} does not contain extractable PDF text."
        )

    return text


def _extract_docx_text(content: bytes, filename: str) -> str:
    try:
        document = Document(BytesIO(content))
        text = "\n".join(
            paragraph.text for paragraph in document.paragraphs if paragraph.text
        ).strip()
    except Exception as e:
        raise AttachmentTextExtractionError(
            f"Attachment {filename!r} could not be read as a DOCX file."
        ) from e

    if not text:
        raise AttachmentTextExtractionError(
            f"Attachment {filename!r} does not contain extractable DOCX text."
        )

    return text
