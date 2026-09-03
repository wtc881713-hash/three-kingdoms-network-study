"""Prepare an auditable provisional review of high-frequency candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from src.extract_character_candidates import collect_candidate_events, read_text
except ModuleNotFoundError:
    from extract_character_candidates import collect_candidate_events, read_text


ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = (
    ROOT / "data" / "processed" / "three_kingdoms_gutenberg_simplified.txt"
)
FILTERED_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "character_candidates_frequency_ge_10.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "gutenberg"
    / "candidate_review_frequency_ge_10.csv"
)

# These mappings are provisional research data, not final ground truth.
# Only aliases that are stable enough for a first review pass are included.
STABLE_ALIASES = {
    "孔明": "诸葛亮",
    "操": "曹操",
    "玄德": "刘备",
    "瑜": "周瑜",
    "懿": "司马懿",
    "权": "孙权",
    "布": "吕布",
    "关公": "关羽",
    "云长": "关羽",
    "维": "姜维",
    "飞": "张飞",
    "云": "赵云",
    "先主": "刘备",
    "绍": "袁绍",
    "卓": "董卓",
    "统": "庞统",
    "表": "刘表",
    "允": "王允",
    "松": "张松",
    "辽": "张辽",
    "宫": "陈宫",
    "逊": "陆逊",
    "丕": "曹丕",
    "策": "孙策",
    "获": "孟获",
    "忠": "黄忠",
    "叡": "曹叡",
    "泽": "阚泽",
    "艾": "邓艾",
    "慈": "太史慈",
    "超": "马超",
    "会": "钟会",
    "后主": "刘禅",
    "师": "司马师",
    "辂": "管辂",
    "修": "杨修",
    "惇": "夏侯惇",
    "真": "曹真",
    "诩": "贾诩",
    "水镜": "司马徽",
    "庶": "徐庶",
    "瑾": "诸葛瑾",
    "璋": "刘璋",
    "袆": "费祎",
    "郃": "张郃",
    "佗": "华佗",
    "延": "魏延",
    "登": "陈登",
    "达": "孟达",
    "丰": "田丰",
    "坚": "孙坚",
    "芝": "邓芝",
    "蒙": "吕蒙",
    "顾": "顾雍",
    "儒": "李儒",
    "孙干": "孙乾",
    "瑁": "蔡瑁",
    "盖": "黄盖",
    "国太": "吴国太",
    "汜": "郭汜",
    "瓒": "公孙瓒",
    "融": "孔融",
    "霸": "夏侯霸",
    "晃": "徐晃",
    "腾": "马腾",
    "衡": "祢衡",
    "谡": "马谡",
    "嘉": "郭嘉",
    "歆": "华歆",
    "洪": "曹洪",
    "淮": "郭淮",
    "温": "张温",
    "琦": "刘琦",
}

KNOWN_FALSE_POSITIVES = {
    "名",
    "遂",
    "书略",
    "长",
    "夫人",
    "刘",
    "厉声",
    "大呼",
    "书",
    "诗",
    "因",
    "诸将",
    "玄",
}

CONTEXT_DEPENDENT = {
    "肃": "May refer to Lu Su or Li Su.",
    "昭": "May refer to Zhang Zhao or Sima Zhao.",
    "承": "Short form needs contextual confirmation.",
    "平": "May refer to Wang Ping, Guan Ping, or another character.",
    "干": "May refer to Sun Qian or another character.",
    "张": "Surname-only reference cannot identify one character.",
    "德": "May be part of Xuande or another name.",
    "亮": "May refer to Zhuge Liang or Sun Liang.",
    "攸": "May refer to Xun You or Xu You.",
    "进": "May refer to He Jin, Yue Jin, or another character.",
    "兴": "Short form needs contextual confirmation.",
    "定": "May refer to Gao Ding, Guan Ding, or another character.",
    "范": "Short form needs contextual confirmation.",
    "何": "Surname or interrogative word; evidence requires review.",
}


def classify_candidate(name: str) -> tuple[str, str, str]:
    """Return provisional decision, canonical name, and decision basis."""
    if name in KNOWN_FALSE_POSITIVES:
        return "no", "", "lexical_false_positive"
    if name in CONTEXT_DEPENDENT:
        return "uncertain", "", CONTEXT_DEPENDENT[name]
    if name in STABLE_ALIASES:
        return "yes", STABLE_ALIASES[name], "stable_alias"
    return "yes", name, "explicit_or_full_name"


def collect_evidence(corpus_file: Path) -> dict[str, list[str]]:
    """Collect up to three distinct snippets for each candidate."""
    events, _ = collect_candidate_events(read_text(corpus_file))
    snippets: dict[str, list[str]] = defaultdict(list)
    for event in events:
        values = snippets[event.candidate_name]
        if event.snippet not in values and len(values) < 3:
            values.append(event.snippet)
    return snippets


def prepare_review(
    candidates: pd.DataFrame,
    evidence: dict[str, list[str]],
) -> pd.DataFrame:
    """Add provisional decisions and multiple evidence snippets."""
    rows: list[dict[str, object]] = []
    for row in candidates.to_dict(orient="records"):
        name = str(row["candidate_name"])
        decision, canonical_name, basis = classify_candidate(name)
        snippets = evidence.get(name, [])
        row.update(
            {
                "proposed_is_character": decision,
                "proposed_canonical_name": canonical_name,
                "decision_basis": basis,
                "evidence_1": snippets[0] if len(snippets) > 0 else "",
                "evidence_2": snippets[1] if len(snippets) > 1 else "",
                "evidence_3": snippets[2] if len(snippets) > 2 else "",
                "human_decision": "",
                "human_canonical_name": "",
                "human_notes": "",
                "review_status": "provisional",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Generate the provisional high-frequency review table."""
    candidates = pd.read_csv(
        FILTERED_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    evidence = collect_evidence(CORPUS_FILE)
    review = prepare_review(candidates, evidence)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    counts = review["proposed_is_character"].value_counts().to_dict()
    decided = counts.get("yes", 0) + counts.get("no", 0)
    provisional_precision = counts.get("yes", 0) / decided if decided else 0.0

    print(f"Reviewed candidates: {len(review)}")
    print(f"Provisional character: {counts.get('yes', 0)}")
    print(f"Provisional false positive: {counts.get('no', 0)}")
    print(f"Context-dependent: {counts.get('uncertain', 0)}")
    print(
        "Provisional precision among non-uncertain decisions: "
        f"{provisional_precision:.3f}"
    )
    print(f"Output file: {OUTPUT_FILE}")
    print("All decisions remain provisional until human confirmation.")


if __name__ == "__main__":
    main()
