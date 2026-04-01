from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
from docx import Document

from ..config import Settings


def _paragraphs_to_markdown(document: Document) -> str:
    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = re.sub(r"\s+", " ", paragraph.text).strip()
        if not text:
            continue
        style_name = getattr(paragraph.style, "name", "") or ""
        match = re.search(r"Heading\s*(\d+)", style_name, re.IGNORECASE)
        if match:
            level = max(1, min(6, int(match.group(1))))
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _extract_pdf_markdown(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append(f"## Page {page_number}\n\n{text}")
    return "\n\n".join(pages)


def extract_tr290_documents(settings: Settings) -> list[dict[str, Any]]:
    normalized_dir = settings.repo.normalized_dir / "tr290"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    markdown_files = sorted(settings.repo.tr290_dir.glob("*.md"))
    corpus: list[dict[str, Any]] = []
    if markdown_files:
        for markdown_path in markdown_files:
            text = markdown_path.read_text(encoding="utf-8")
            corpus.append(
                {
                    "id": f"tr290:{markdown_path.stem}",
                    "source_type": "tr290_markdown",
                    "title": markdown_path.stem,
                    "text": text,
                    "metadata": {"path": str(markdown_path)},
                }
            )
        return corpus

    docx_document = Document(settings.repo.tr290_docx)
    docx_markdown = _paragraphs_to_markdown(docx_document)
    docx_output = normalized_dir / "TR290_Intent_Common_Model_v3.0.0.docx.md"
    docx_output.write_text(docx_markdown, encoding="utf-8")

    pdf_markdown = _extract_pdf_markdown(settings.repo.tr290_pdf)
    pdf_output = normalized_dir / "TR290_Intent_Common_Model_v3.0.0.pdf.md"
    pdf_output.write_text(pdf_markdown, encoding="utf-8")

    corpus.extend(
        [
            {
                "id": "tr290:docx",
                "source_type": "tr290_docx",
                "title": settings.repo.tr290_docx.stem,
                "text": docx_markdown,
                "metadata": {"path": str(docx_output)},
            },
            {
                "id": "tr290:pdf",
                "source_type": "tr290_pdf",
                "title": settings.repo.tr290_pdf.stem,
                "text": pdf_markdown,
                "metadata": {"path": str(pdf_output)},
            },
        ]
    )
    return corpus
