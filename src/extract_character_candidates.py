"""Build reviewable character candidates from a classical Chinese novel."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "raw" / "three_kingdoms.txt"
METADATA_DIR = ROOT / "data" / "metadata"
CANDIDATE_FILE = METADATA_DIR / "character_candidates.csv"
ALIAS_FILE = METADATA_DIR / "extracted_alias_pairs.csv"
INTEGRITY_FILE = METADATA_DIR / "text_integrity_report.txt"

CHINESE_NUMBER = "一二三四五六七八九十百零〇○两0-9"
CHAPTER_PATTERN = re.compile(
    rf"^第[{CHINESE_NUMBER}]+回[^\r\n]*$", re.MULTILINE
)
CHINESE_NAME_PATTERN = r"[\u4e00-\u9fff]{1,4}"

CANDIDATE_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "frequency",
    "speech_frequency",
    "introduction_frequency",
    "chapter_title_frequency",
    "evidence_types",
    "sample_evidence",
    "canonical_name",
    "is_character",
    "alias_type",
    "is_ambiguous",
    "review_status",
    "notes",
]

ALIAS_COLUMNS = [
    "canonical_name",
    "alias",
    "alias_type",
    "evidence_snippet",
    "source_method",
    "review_status",
    "notes",
]

STOPWORDS = {
    "话说",
    "却说",
    "且说",
    "后人",
    "后人有诗",
    "古人有诗",
    "众人",
    "军士",
    "左右",
    "一人",
    "二人",
    "三人",
    "此人",
    "其人",
    "有人",
    "使者",
    "探子",
    "小校",
    "百姓",
    "众官",
    "群臣",
    "诸侯",
    "众将",
    "将士",
    "太守",
    "丞相",
    "主公",
    "使君",
    "军师",
    "将军",
    "天子",
    "皇帝",
    "帝",
    "后",
    "公",
    "乃",
    "问",
    "答",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "大",
}

AMBIGUOUS_CANDIDATES = {
    "操",
    "瑜",
    "肃",
    "权",
    "懿",
    "布",
    "维",
    "飞",
    "云",
    "绍",
    "先主",
    "丞相",
    "主公",
    "使君",
    "军师",
    "将军",
    "皇叔",
}

SUSPICIOUS_SEQUENCES = (
    "锛",
    "銆",
    "鈥",
    "绗",
    "鍥炴",
    "闁",
    "馃",
)


@dataclass(frozen=True)
class CandidateEvent:
    """One rule-based extraction event with auditable evidence."""

    candidate_name: str
    evidence_type: str
    snippet: str
    position: int


@dataclass(frozen=True)
class AliasPair:
    """A possible canonical-name and alias relationship."""

    canonical_name: str
    alias: str
    alias_type: str
    evidence_snippet: str
    source_method: str = "formal_introduction"


@dataclass
class IntegrityResult:
    """Summary of input-text integrity checks."""

    status: str
    character_count: int
    line_count: int
    chapter_count: int
    replacement_count: int
    null_count: int
    control_count: int
    suspicious_count: int
    examples: list[str]
    warnings: list[str]


def read_text(file_path: Path) -> str:
    """Read a required UTF-8 text file and return its content."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UnicodeError(
            f"Input is not valid UTF-8 at byte position {error.start}: {file_path}"
        ) from error
    except PermissionError as error:
        raise PermissionError(f"Permission denied while reading: {file_path}") from error

    if not text:
        raise ValueError(f"Input file is empty: {file_path}")
    return text


def detect_chapters(text: str) -> list[tuple[str, int]]:
    """Return chapter titles and their starting positions."""
    return [(match.group(0).strip(), match.start()) for match in CHAPTER_PATTERN.finditer(text)]


def _context_snippet(text: str, position: int, width: int = 32) -> str:
    """Return a short, single-line context snippet around a position."""
    start = max(0, position - width)
    end = min(len(text), position + width)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet[:120]


def check_text_integrity(text: str, file_path: Path = INPUT_FILE) -> IntegrityResult:
    """Inspect the text without silently changing possible corruption."""
    chapters = detect_chapters(text)
    replacement_positions = [match.start() for match in re.finditer("\ufffd", text)]
    null_positions = [match.start() for match in re.finditer("\x00", text)]
    control_positions = [
        index
        for index, character in enumerate(text)
        if ord(character) < 32 and character not in "\n\r\t"
    ]

    suspicious_hits: list[tuple[int, str]] = []
    for sequence in SUSPICIOUS_SEQUENCES:
        suspicious_hits.extend(
            (match.start(), sequence) for match in re.finditer(re.escape(sequence), text)
        )
    # The source is a Chinese literary text and normally uses full-width
    # question marks. Frequent ASCII question marks are strong evidence of
    # lossy decoding in this corpus.
    suspicious_hits.extend(
        (match.start(), "?") for match in re.finditer(re.escape("?"), text)
    )
    suspicious_hits.sort()

    warnings: list[str] = []
    if len(text) < 10_000:
        warnings.append("Input is unexpectedly short.")
    if replacement_positions:
        warnings.append("Unicode replacement characters were detected.")
    if null_positions:
        warnings.append("Null characters were detected.")
    if control_positions:
        warnings.append("Unexpected control characters were detected.")
    if suspicious_hits:
        warnings.append("Possible mojibake-like sequences were detected.")
    if len(chapters) != 120:
        warnings.append(f"Expected 120 chapters but detected {len(chapters)}.")

    titles = [title for title, _ in chapters]
    if not titles or not titles[0].startswith("第一回"):
        warnings.append("The first chapter was not detected.")
    if not titles or not titles[-1].startswith("第一百二十回"):
        warnings.append("The final chapter was not detected.")

    serious_problem = (
        not text.strip()
        or bool(null_positions)
        or len(chapters) == 0
        or len(replacement_positions) > 10
    )
    status = "FAIL" if serious_problem else ("WARNING" if warnings else "PASS")

    example_positions = (
        replacement_positions[:3]
        + null_positions[:3]
        + control_positions[:3]
        + [position for position, _ in suspicious_hits[:5]]
    )
    examples = [
        f"Position {position}: {_context_snippet(text, position)}"
        for position in sorted(set(example_positions))
    ]

    return IntegrityResult(
        status=status,
        character_count=len(text),
        line_count=len(text.splitlines()),
        chapter_count=len(chapters),
        replacement_count=len(replacement_positions),
        null_count=len(null_positions),
        control_count=len(control_positions),
        suspicious_count=len(suspicious_hits),
        examples=examples,
        warnings=warnings,
    )


def write_integrity_report(
    result: IntegrityResult, input_file: Path, output_file: Path
) -> None:
    """Write a plain-text integrity report."""
    lines = [
        "Text Integrity Report",
        "=====================",
        f"Input file: {input_file.resolve()}",
        f"Character count: {result.character_count}",
        f"Line count: {result.line_count}",
        f"Detected chapter count: {result.chapter_count}",
        f"Replacement character count: {result.replacement_count}",
        f"Null character count: {result.null_count}",
        f"Unexpected control character count: {result.control_count}",
        f"Suspicious sequence count: {result.suspicious_count}",
        f"Overall status: {result.status}",
        "",
        "Warnings:",
    ]
    lines.extend(f"- {warning}" for warning in result.warnings)
    if not result.warnings:
        lines.append("- None")
    lines.extend(["", "Example suspicious passages:"])
    lines.extend(f"- {example}" for example in result.examples)
    if not result.examples:
        lines.append("- None")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_speech_candidates(text: str) -> list[CandidateEvent]:
    """Extract conservative speaker candidates from reporting structures."""
    reporting_verbs = (
        "大叫曰",
        "大笑曰",
        "大喝曰",
        "大怒曰",
        "问曰",
        "答曰",
        "笑曰",
        "喝曰",
        "叫曰",
        "言曰",
        "叹曰",
        "哭曰",
        "骂曰",
        "奏曰",
        "禀曰",
        "告曰",
        "大叫",
        "曰",
    )
    verb_pattern = "|".join(map(re.escape, reporting_verbs))

    # A lazy name group lets the longer reporting verb win. For example,
    # "孔明笑曰" becomes name="孔明", verb="笑曰", not name="孔明笑".
    direct_pattern = re.compile(
        rf"(?<![\u4e00-\u9fff])(?P<name>{CHINESE_NAME_PATTERN}?)(?P<verb>{verb_pattern})"
    )
    address_pattern = re.compile(
        rf"(?<![\u4e00-\u9fff])(?P<name>{CHINESE_NAME_PATTERN}?)"
        rf"谓(?:众将|左右|众人|诸将|其人|之)?(?:曰)?"
    )

    events: list[CandidateEvent] = []
    for pattern in (direct_pattern, address_pattern):
        for match in pattern.finditer(text):
            name = match.group("name")
            # Do not turn an unattributed reporting phrase into a speaker.
            # For example, "大笑曰" must not produce the candidate "大".
            if match.group(0) in reporting_verbs:
                continue
            if (
                name in STOPWORDS
                or name.endswith(("一人", "二人", "三人", "有人", "众人"))
                or name.startswith(("忽见", "只见", "却有", "又见"))
            ):
                continue
            events.append(
                CandidateEvent(
                    candidate_name=name,
                    evidence_type="speech",
                    snippet=_context_snippet(text, match.start()),
                    position=match.start(),
                )
            )
    return events


def extract_formal_introductions(
    text: str,
) -> tuple[list[CandidateEvent], list[AliasPair]]:
    """Extract full names and courtesy names from formal introductions."""
    pattern = re.compile(
        r"姓(?P<surname>[\u4e00-\u9fff]{1,2})"
        r"\s*[，,]?\s*名(?P<given>[\u4e00-\u9fff]{1,2})"
        r"\s*[，,]?\s*字(?P<courtesy>[\u4e00-\u9fff]{1,3})"
        r"(?:\s*[，,]?\s*后改(?P<later>[\u4e00-\u9fff]{1,3}))?"
    )
    explicit_pattern = re.compile(
        r"(?<![\u4e00-\u9fff])"
        r"(?P<full>(?:诸葛|司马|公孙|夏侯|太史|皇甫|"
        r"[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
        r"金魏陶姜戚谢邹喻柏水窦章云苏潘葛范彭郎鲁韦昌马苗凤花方俞"
        r"任袁柳鲍史唐费廉岑薛雷贺倪汤罗毕郝安常乐于时傅皮卞齐康伍"
        r"余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴宋茅"
        r"庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林"
        r"刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯管卢莫房裘缪解应宗丁宣"
        r"邓郁单杭洪包诸左石崔吉龚程嵇邢裴陆荣翁荀羊甄麴])"
        r"[\u4e00-\u9fff]{1,2})"
        r"\s*[，,]?\s*字(?P<courtesy>[\u4e00-\u9fff]{1,3})"
    )

    events: list[CandidateEvent] = []
    aliases: list[AliasPair] = []
    occupied_ranges: list[tuple[int, int]] = []

    for match in pattern.finditer(text):
        full_name = match.group("surname") + match.group("given")
        courtesy_name = match.group("courtesy").removesuffix("者")
        later_name = match.group("later")
        snippet = _context_snippet(text, match.start())
        occupied_ranges.append(match.span())

        for name in (full_name, courtesy_name):
            events.append(
                CandidateEvent(name, "introduction", snippet, match.start())
            )
        aliases.append(
            AliasPair(full_name, courtesy_name, "courtesy_name", snippet)
        )

        if later_name:
            events.append(
                CandidateEvent(later_name, "introduction", snippet, match.start())
            )
            aliases.append(
                AliasPair(full_name, later_name, "later_courtesy_name", snippet)
            )

    for match in explicit_pattern.finditer(text):
        if any(start <= match.start() < end for start, end in occupied_ranges):
            continue
        full_name = match.group("full")
        courtesy_name = match.group("courtesy")
        snippet = _context_snippet(text, match.start())
        events.extend(
            [
                CandidateEvent(full_name, "introduction", snippet, match.start()),
                CandidateEvent(courtesy_name, "introduction", snippet, match.start()),
            ]
        )
        aliases.append(
            AliasPair(full_name, courtesy_name, "courtesy_name", snippet)
        )

    return events, aliases


def extract_chapter_title_candidates(
    text: str, known_candidates: Iterable[str]
) -> list[CandidateEvent]:
    """Record title evidence only for candidates found by stronger rules."""
    chapters = detect_chapters(text)
    candidates = sorted(set(known_candidates), key=lambda value: (-len(value), value))
    events: list[CandidateEvent] = []

    for title, position in chapters:
        occupied: set[int] = set()
        for candidate in candidates:
            for match in re.finditer(re.escape(candidate), title):
                match_positions = set(range(match.start(), match.end()))
                if occupied.intersection(match_positions):
                    continue
                occupied.update(match_positions)
                events.append(
                    CandidateEvent(
                        candidate_name=candidate,
                        evidence_type="chapter_title",
                        snippet=title,
                        position=position + match.start(),
                    )
                )
    return events


def clean_candidate(candidate: str) -> str | None:
    """Return a conservative cleaned candidate or None."""
    candidate = re.sub(r"\s+", "", candidate)
    if not candidate or candidate in STOPWORDS:
        return None
    if not re.fullmatch(CHINESE_NAME_PATTERN, candidate):
        return None
    if "第" in candidate or "回" in candidate:
        return None
    return candidate


def collect_candidate_events(text: str) -> tuple[list[CandidateEvent], list[AliasPair]]:
    """Run all extraction methods and return cleaned events and alias pairs."""
    speech_events = extract_speech_candidates(text)
    introduction_events, aliases = extract_formal_introductions(text)

    cleaned_primary_events: list[CandidateEvent] = []
    for event in speech_events + introduction_events:
        cleaned_name = clean_candidate(event.candidate_name)
        if cleaned_name:
            cleaned_primary_events.append(
                CandidateEvent(
                    cleaned_name,
                    event.evidence_type,
                    event.snippet,
                    event.position,
                )
            )

    title_events = extract_chapter_title_candidates(
        text, (event.candidate_name for event in cleaned_primary_events)
    )
    return cleaned_primary_events + title_events, deduplicate_alias_pairs(aliases)


def build_candidate_dataframe(events: Sequence[CandidateEvent]) -> pd.DataFrame:
    """Aggregate extraction events into the required review table."""
    if not events:
        raise ValueError("No character candidates were extracted.")

    grouped: dict[str, list[CandidateEvent]] = defaultdict(list)
    for event in events:
        grouped[event.candidate_name].append(event)

    rows: list[dict[str, object]] = []
    for name, name_events in grouped.items():
        type_counts = Counter(event.evidence_type for event in name_events)
        evidence_types = sorted(type_counts)
        rows.append(
            {
                "candidate_name": name,
                "frequency": len(name_events),
                "speech_frequency": type_counts["speech"],
                "introduction_frequency": type_counts["introduction"],
                "chapter_title_frequency": type_counts["chapter_title"],
                "evidence_types": ";".join(evidence_types),
                "sample_evidence": name_events[0].snippet,
                "canonical_name": "",
                "is_character": "",
                "alias_type": "",
                "is_ambiguous": "yes" if name in AMBIGUOUS_CANDIDATES else "no",
                "review_status": "pending",
                "notes": (
                    "Context-dependent shortened name or title."
                    if name in AMBIGUOUS_CANDIDATES
                    else ""
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values(
        ["frequency", "candidate_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    dataframe.insert(
        0,
        "candidate_id",
        [f"CAND{index:04d}" for index in range(1, len(dataframe) + 1)],
    )
    return dataframe[CANDIDATE_COLUMNS]


def deduplicate_alias_pairs(alias_pairs: Iterable[AliasPair]) -> list[AliasPair]:
    """Remove exact canonical-name and alias duplicates while preserving order."""
    unique: list[AliasPair] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in alias_pairs:
        key = (pair.canonical_name, pair.alias, pair.alias_type)
        if key not in seen and pair.canonical_name != pair.alias:
            seen.add(key)
            unique.append(pair)
    return unique


def build_alias_dataframe(alias_pairs: Sequence[AliasPair]) -> pd.DataFrame:
    """Build a pending-review alias dataframe."""
    rows = [
        {
            "canonical_name": pair.canonical_name,
            "alias": pair.alias,
            "alias_type": pair.alias_type,
            "evidence_snippet": pair.evidence_snippet,
            "source_method": pair.source_method,
            "review_status": "pending",
            "notes": "",
        }
        for pair in alias_pairs
    ]
    return pd.DataFrame(rows, columns=ALIAS_COLUMNS)


def save_outputs(
    candidates: pd.DataFrame,
    aliases: pd.DataFrame,
    candidate_file: Path = CANDIDATE_FILE,
    alias_file: Path = ALIAS_FILE,
) -> None:
    """Save CSV outputs with a UTF-8 byte order mark for Excel."""
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        candidates.to_csv(candidate_file, index=False, encoding="utf-8-sig")
        aliases.to_csv(alias_file, index=False, encoding="utf-8-sig")
    except PermissionError as error:
        raise PermissionError(
            "Could not write metadata CSV files. Close them in Excel and retry."
        ) from error


def parse_args() -> argparse.Namespace:
    """Parse optional corpus and output paths."""
    parser = argparse.ArgumentParser(
        description="Extract reviewable character candidates from the novel."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="UTF-8 novel file (default: data/raw/three_kingdoms.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METADATA_DIR,
        help="Directory for candidate, alias, and integrity outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run integrity checks and create candidate and alias metadata."""
    args = parse_args()
    input_file = args.input.resolve()
    output_dir = args.output_dir.resolve()
    candidate_file = output_dir / CANDIDATE_FILE.name
    alias_file = output_dir / ALIAS_FILE.name
    integrity_file = output_dir / INTEGRITY_FILE.name

    try:
        text = read_text(input_file)
        integrity = check_text_integrity(text, input_file)
        write_integrity_report(integrity, input_file, integrity_file)

        if integrity.status == "FAIL":
            raise ValueError(
                "Serious text-integrity problems were detected. "
                f"Review {integrity_file} before continuing."
            )
        if integrity.status == "WARNING":
            print(
                f"WARNING: Text integrity status is WARNING. Review {integrity_file}."
            )

        events, alias_pairs = collect_candidate_events(text)
        candidate_dataframe = build_candidate_dataframe(events)
        alias_dataframe = build_alias_dataframe(alias_pairs)
        save_outputs(
            candidate_dataframe,
            alias_dataframe,
            candidate_file,
            alias_file,
        )

        print("=" * 68)
        print("Character candidate extraction completed")
        print("=" * 68)
        print(f"Integrity status: {integrity.status}")
        print(f"Chapters detected: {integrity.chapter_count}")
        print(f"Candidate extraction events: {len(events)}")
        print(f"Unique candidates: {len(candidate_dataframe)}")
        print(f"Unique alias pairs: {len(alias_dataframe)}")
        print(f"Candidate output: {candidate_file}")
        print(f"Alias output: {alias_file}")
        print(f"Integrity report: {integrity_file}")
        print("\nTop 20 candidates:\n")
        print(
            candidate_dataframe[
                [
                    "candidate_name",
                    "frequency",
                    "speech_frequency",
                    "introduction_frequency",
                    "chapter_title_frequency",
                    "is_ambiguous",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )
    except (FileNotFoundError, PermissionError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
