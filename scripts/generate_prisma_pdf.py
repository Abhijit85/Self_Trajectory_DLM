#!/usr/bin/env python3
"""Generate a simple PDF copy of repro_kit/PRISMA_2020_checklist.md."""

from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "repro_kit" / "PRISMA_2020_checklist.md"
OUT = ROOT / "repro_kit" / "PRISMA_2020_checklist.pdf"


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_lines(text):
    lines = []
    for raw in text.splitlines():
        if raw.startswith("|---"):
            continue
        if raw.startswith("|"):
            cells = [c.strip().replace("`", "") for c in raw.strip("|").split("|")]
            if len(cells) == 3:
                raw = f"{cells[0]} - {cells[1]}: {cells[2]}"
        raw = re.sub(r"`([^`]*)`", r"\1", raw)
        raw = raw.replace("# ", "").replace("## ", "")
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=92) or [""])
    return lines


def build_pdf(lines):
    page_width, page_height = 612, 792
    margin_x, margin_y = 54, 54
    line_height = 12
    lines_per_page = int((page_height - 2 * margin_y) / line_height)
    pages = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]

    objects = []

    def add(obj):
        objects.append(obj)
        return len(objects)

    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    content_ids = []
    for page_lines in pages:
        commands = ["BT", f"/F1 9 Tf", f"{margin_x} {page_height - margin_y} Td"]
        for idx, line in enumerate(page_lines):
            if idx:
                commands.append(f"0 -{line_height} Td")
            commands.append(f"({pdf_escape(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands)
        content_id = add(f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream")
        content_ids.append(content_id)
        page_id = add(
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    fixed = []
    for obj in objects:
        fixed.append(obj.replace("/Parent 0 0 R", f"/Parent {pages_id} 0 R"))

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for i, obj in enumerate(fixed, start=1):
        offsets.append(sum(len(c) for c in chunks))
        chunks.append(f"{i} 0 obj\n{obj}\nendobj\n".encode("utf-8"))
    xref_offset = sum(len(c) for c in chunks)
    chunks.append(f"xref\n0 {len(fixed)+1}\n0000000000 65535 f \n".encode("utf-8"))
    for off in offsets[1:]:
        chunks.append(f"{off:010d} 00000 n \n".encode("utf-8"))
    chunks.append(
        f"trailer\n<< /Size {len(fixed)+1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("utf-8")
    )
    return b"".join(chunks)


def main():
    lines = markdown_to_lines(SRC.read_text())
    OUT.write_bytes(build_pdf(lines))
    print(OUT)


if __name__ == "__main__":
    main()
