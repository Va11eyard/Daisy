"""
Convert book files under data/ (PDF, FB2, TXT, RTF, DOCX) to Markdown mirrors under data/md/.

Skips: data/raw/*.json (dataset), *.jsonl, README, output dir data/md.

Usage:
  pip install -r scripts/requirements-books.txt
  python scripts/books_to_markdown.py --input-dir data --output-dir data/md
  python scripts/books_to_markdown.py --input-dir data --output-dir data/md --skip-existing

.doc (legacy Word) is not supported without extra Windows tooling; use Word "Save as docx" or omit.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _configure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles when printing paths."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _safe_name(stem: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", stem)
    return s.strip() or "unnamed"


def _read_txt(path: Path) -> str:
    import chardet

    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _fb2_to_text(path: Path) -> str:
    tree = ET.parse(path)
    root = tree.getroot()
    parts: list[str] = []
    # FictionBook 2.0: grab text from paragraph-like nodes
    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in ("p", "empty-line"):
            if tag == "empty-line":
                parts.append("")
                continue
            text = "".join(el.itertext()).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _rtf_to_text(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text

    return rtf_to_text(path.read_text(encoding="utf-8", errors="replace"))


def _docx_to_text(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


def _plain_to_markdown(text: str, title: str) -> str:
    """Wrap body; normalize excessive newlines."""
    body = re.sub(r"\n{3,}", "\n\n", text.strip())
    return f"# {title}\n\n{body}\n"


def convert_file(src: Path, out_md: Path) -> None:
    ext = src.suffix.lower()
    title = src.stem

    if ext == ".pdf":
        raw = _pdf_to_text(src)
    elif ext == ".fb2":
        raw = _fb2_to_text(src)
    elif ext == ".txt":
        raw = _read_txt(src)
    elif ext == ".rtf":
        raw = _rtf_to_text(src)
    elif ext == ".docx":
        raw = _docx_to_text(src)
    elif ext == ".xml":
        raw = _read_txt(src)
    elif ext == ".doc":
        raise RuntimeError(
            ".doc (legacy) is not supported. Save as .docx or convert with Word/LibreOffice, then re-run."
        )
    else:
        raise RuntimeError(f"unsupported extension: {ext}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = _plain_to_markdown(raw, title=title)
    out_md.write_text(md, encoding="utf-8")


def main() -> None:
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sources whose output .md already exists (resume after interrupt).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    skip_names = {"train.jsonl", "val.jsonl", "README.md"}
    exts = {".pdf", ".fb2", ".txt", ".rtf", ".docx", ".doc", ".xml"}

    files: list[Path] = []
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        if output_dir in p.parents or p.resolve() == output_dir.resolve():
            continue
        if p.name in skip_names:
            continue
        if p.suffix.lower() == ".json" and "raw" in p.parts:
            continue
        if p.suffix.lower() not in exts:
            continue
        files.append(p)

    if not files:
        print("No convertible files found under", input_dir)
        sys.exit(0)

    ok, skipped, err = 0, 0, 0
    for src in sorted(files):
        rel = src.relative_to(input_dir)
        out_name = _safe_name(rel.stem) + ".md"
        out_path = output_dir / rel.parent / out_name
        if args.skip_existing and out_path.is_file():
            print(f"SKIP (exists): {rel}")
            skipped += 1
            continue
        print(f"{rel} -> {out_path}")
        if args.dry_run:
            ok += 1
            continue
        try:
            convert_file(src, out_path)
            ok += 1
        except Exception as e:
            err += 1
            print(f"  ERROR: {e}", file=sys.stderr)

    print(f"Done: {ok} converted, {skipped} skipped, {err} errors -> {output_dir}")


if __name__ == "__main__":
    main()
