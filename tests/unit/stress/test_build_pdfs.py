from io import BytesIO

from pypdf import PdfReader

from evals.stress.fixtures.build_pdfs import (
    ATTACHMENT_SIZES_KB,
    build_pdf_bytes,
    write_fixture_pdfs,
)


def test_build_pdf_bytes_creates_extractable_pdf() -> None:
    pdf_bytes = build_pdf_bytes(5)
    reader = PdfReader(BytesIO(pdf_bytes))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert len(pdf_bytes) >= 5 * 1024
    assert len(reader.pages) >= 1
    assert "Attachment fact: upload workflow requires audit evidence" in extracted_text


def test_write_fixture_pdfs_creates_expected_sizes(tmp_path) -> None:
    paths = write_fixture_pdfs(tmp_path)

    assert [path.name for path in paths] == [
        f"attach_{size_kb}kb.pdf" for size_kb in ATTACHMENT_SIZES_KB
    ]
    assert all(path.exists() for path in paths)
    assert all(
        path.stat().st_size >= size_kb * 1024
        for path, size_kb in zip(paths, ATTACHMENT_SIZES_KB, strict=True)
    )
    assert all(
        path.stat().st_size < (size_kb + 2) * 1024
        for path, size_kb in zip(paths, ATTACHMENT_SIZES_KB, strict=True)
    )
