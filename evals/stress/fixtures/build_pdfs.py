from __future__ import annotations

import argparse
from pathlib import Path


ATTACHMENT_SIZES_KB = (5, 20, 50, 100)
DEFAULT_OUTPUT_DIR = Path(__file__).parent
BYTES_PER_KB = 1024

BASE_PARAGRAPH = (
    "Synthetic requirements packet for CAG attachment stress testing. "
    "The project name is AttachmentLoad. "
    "Attachment fact: upload workflow requires audit evidence. "
    "The system must preserve customer onboarding, reporting, permissions, "
    "and operational notes while estimating scope. "
)


def build_pdf_bytes(target_kb: int) -> bytes:
    if target_kb <= 0:
        raise ValueError("target_kb must be greater than 0")

    target_bytes = target_kb * BYTES_PER_KB
    lower_bound = 1
    upper_bound = 1

    while len(_render_pdf(_build_pages(upper_bound))) < target_bytes:
        upper_bound *= 2

    while lower_bound < upper_bound:
        midpoint = (lower_bound + upper_bound) // 2
        if len(_render_pdf(_build_pages(midpoint))) < target_bytes:
            lower_bound = midpoint + 1
        else:
            upper_bound = midpoint

    return _render_pdf(_build_pages(lower_bound))


def write_fixture_pdfs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for size_kb in ATTACHMENT_SIZES_KB:
        path = output_dir / f"attach_{size_kb}kb.pdf"
        path.write_bytes(build_pdf_bytes(size_kb))
        paths.append(path)

    return paths


def _build_pages(paragraph_count: int) -> list[list[str]]:
    paragraphs = [
        f"{BASE_PARAGRAPH}Paragraph index: {index:04d}."
        for index in range(1, paragraph_count + 1)
    ]
    return [paragraphs[index : index + 18] for index in range(0, len(paragraphs), 18)]


def _render_pdf(pages: list[list[str]]) -> bytes:
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages "
            f"/Kids [{' '.join(f'{3 + index * 2} 0 R' for index in range(len(pages)))}] "
            f"/Count {len(pages)} >>"
        ),
    ]

    for index, page_lines in enumerate(pages):
        page_object_id = 3 + index * 2
        content_object_id = page_object_id + 1
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >> >> >> "
            f"/Contents {content_object_id} 0 R >>"
        )
        content = _page_content_stream(page_lines)
        objects.append(
            f"<< /Length {len(content.encode('latin-1'))} >>\n"
            f"stream\n{content}\nendstream"
        )

    return _serialize_pdf(objects)


def _page_content_stream(lines: list[str]) -> str:
    commands = ["BT", "/F1 9 Tf", "72 740 Td", "12 TL"]

    for line in lines:
        commands.append(f"({_escape_pdf_text(line)}) Tj")
        commands.append("T*")

    commands.append("ET")
    return "\n".join(commands)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _serialize_pdf(objects: list[str]) -> bytes:
    data = "%PDF-1.4\n"
    offsets = []

    for index, pdf_object in enumerate(objects, start=1):
        offsets.append(len(data.encode("latin-1")))
        data += f"{index} 0 obj\n{pdf_object}\nendobj\n"

    xref_offset = len(data.encode("latin-1"))
    data += f"xref\n0 {len(objects) + 1}\n"
    data += "0000000000 65535 f \n"

    for offset in offsets:
        data += f"{offset:010d} 00000 n \n"

    data += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    return data.encode("latin-1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic synthetic PDF fixtures for stress tests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where attach_<size>kb.pdf files will be written.",
    )
    args = parser.parse_args()

    for path in write_fixture_pdfs(args.output_dir):
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
