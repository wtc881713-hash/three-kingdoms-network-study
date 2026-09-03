"""Prepare a clean simplified corpus from the Project Gutenberg reference."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from opencc import OpenCC


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILE = ROOT / "data" / "reference" / "gutenberg_pg23950.txt"
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "three_kingdoms_gutenberg_simplified.txt"
)
PROVENANCE_FILE = ROOT / "data" / "metadata" / "corpus_provenance.md"
SOURCE_URL = "https://www.gutenberg.org/cache/epub/23950/pg23950.txt"
CATALOGUE_URL = "https://www.gutenberg.org/ebooks/23950"
RETRIEVAL_DATE = "2026-07-29"

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"
CHAPTER_LINE_PATTERN = re.compile(r"^第[一二三四五六七八九十百零〇○兩两]+回[：:]")


def integer_to_chinese(number: int) -> str:
    """Return the standard Chinese chapter numeral for 1 through 120."""
    if not 1 <= number <= 120:
        raise ValueError("Chapter number must be between 1 and 120.")

    digits = "零一二三四五六七八九"
    if number < 10:
        return digits[number]
    if number < 20:
        return "十" + (digits[number % 10] if number % 10 else "")
    if number < 100:
        tens, ones = divmod(number, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    if number == 100:
        return "一百"
    if number < 110:
        return "一百零" + digits[number - 100]
    if number < 120:
        return "一百一十" + (digits[number - 110] if number > 110 else "")
    return "一百二十"


def read_reference(file_path: Path) -> tuple[bytes, str]:
    """Read and decode the downloaded UTF-8 reference."""
    if not file_path.exists():
        raise FileNotFoundError(f"Reference file not found: {file_path}")
    raw_bytes = file_path.read_bytes()
    return raw_bytes, raw_bytes.decode("utf-8-sig")


def extract_ebook_body(text: str) -> str:
    """Remove the Gutenberg header and licence footer from a working copy."""
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Project Gutenberg start or end marker was not found.")

    body = text[start + len(START_MARKER) : end]
    lines = body.splitlines()
    first_chapter = next(
        (index for index, line in enumerate(lines) if CHAPTER_LINE_PATTERN.match(line.strip())),
        None,
    )
    if first_chapter is None:
        raise ValueError("The first chapter heading was not found.")
    return "\n".join(lines[first_chapter:]).strip()


def normalise_chapter_headings(text: str) -> str:
    """Rewrite the 120 headings with stable standard chapter numerals."""
    lines = text.splitlines()
    chapter_indexes = [
        index
        for index, line in enumerate(lines)
        if CHAPTER_LINE_PATTERN.match(line.strip())
    ]
    if len(chapter_indexes) != 120:
        raise ValueError(
            f"Expected 120 chapter headings in reference; found {len(chapter_indexes)}."
        )

    for chapter_number, line_index in enumerate(chapter_indexes, start=1):
        title = re.sub(
            r"^第[一二三四五六七八九十百零〇○兩两]+回[：:]\s*",
            "",
            lines[line_index].strip(),
        )
        title = title.replace("，", " ", 1)
        lines[line_index] = (
            f"第{integer_to_chinese(chapter_number)}回 {title}".strip()
        )
    return "\n".join(lines)


def unwrap_hard_wrapped_lines(text: str) -> str:
    """Join Gutenberg's hard-wrapped lines while preserving paragraphs."""
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append("".join(buffer).strip())
            buffer.clear()

    for line in text.splitlines():
        stripped = line.strip().replace("\u3000", "")
        if not stripped:
            flush()
        elif stripped.startswith("第") and "回 " in stripped[:12]:
            flush()
            paragraphs.append(stripped)
        else:
            buffer.append(stripped)
    flush()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def prepare_corpus(reference_text: str) -> str:
    """Create a simplified, chapter-normalised derived corpus."""
    body = extract_ebook_body(reference_text)
    normalised = normalise_chapter_headings(body)
    simplified = OpenCC("t2s").convert(normalised)
    return unwrap_hard_wrapped_lines(simplified).strip() + "\n"


def write_provenance(
    source_bytes: bytes,
    reference_file: Path,
    output_file: Path,
    provenance_file: Path,
) -> None:
    """Write source, licence, checksum, and transformation information."""
    checksum = hashlib.sha256(source_bytes).hexdigest()
    content = f"""# Corpus Provenance

## Reference source

- Title: *三國志演義*
- Author: Luo Guanzhong
- Project Gutenberg eBook number: 23950
- Catalogue: {CATALOGUE_URL}
- Download URL: {SOURCE_URL}
- Retrieved: {RETRIEVAL_DATE}
- Downloaded file: `{reference_file.as_posix()}`
- SHA-256: `{checksum}`
- Project Gutenberg catalogue status: Public domain in the USA

The downloaded reference file contains the Project Gutenberg licence notice.
Users outside the United States should check the law that applies to them and
their institution's requirements.

## Derived corpus

- Output: `{output_file.as_posix()}`
- The Project Gutenberg header and footer were excluded from the working text.
- Traditional Chinese was converted to Simplified Chinese with OpenCC `t2s`.
- Hard-wrapped lines were joined into paragraphs.
- Chapter headings were normalised and checked for exactly 120 chapters.
- The original raw corpus was not overwritten.

The derived corpus must be cited and described as a transformed Project
Gutenberg text. It must not be described as identical to the earlier raw file
or to a named modern publisher's edition.
"""
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    provenance_file.write_text(content, encoding="utf-8")


def main() -> None:
    """Prepare the derived corpus and write provenance documentation."""
    source_bytes, reference_text = read_reference(REFERENCE_FILE)
    corpus = prepare_corpus(reference_text)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(corpus, encoding="utf-8")
    write_provenance(
        source_bytes,
        REFERENCE_FILE,
        OUTPUT_FILE,
        PROVENANCE_FILE,
    )

    print(f"Reference file: {REFERENCE_FILE}")
    print(f"Derived corpus: {OUTPUT_FILE}")
    print(f"Derived characters: {len(corpus):,}")
    print(f"Provenance file: {PROVENANCE_FILE}")


if __name__ == "__main__":
    main()
