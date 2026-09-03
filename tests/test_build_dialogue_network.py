import pandas as pd

from src.build_dialogue_network import (
    build_edges,
    extract_adjacent_turns,
    extract_explicit_targets,
    extract_speech_turns,
)


ALIASES = {"玄德": "刘备", "孔明": "诸葛亮", "曹操": "曹操"}


def test_extract_named_speech_turns() -> None:
    turns = extract_speech_turns("玄德曰：「可。」孔明笑曰：「善。」", ALIASES)

    assert [(turn.speaker, turn.alias) for turn in turns] == [
        ("刘备", "玄德"),
        ("诸葛亮", "孔明"),
    ]


def test_extract_explicit_named_target() -> None:
    events = extract_explicit_targets("玄德谓孔明曰：「先生何以教我？」", ALIASES)

    assert len(events) == 1
    assert events[0]["source"] == "刘备"
    assert events[0]["target"] == "诸葛亮"


def test_adjacent_turns_ignore_repeated_same_speaker() -> None:
    events = extract_adjacent_turns(
        "玄德曰：「一。」玄德曰：「二。」孔明答曰：「三。」", ALIASES
    )

    assert len(events) == 1
    assert (events[0]["source"], events[0]["target"]) == ("刘备", "诸葛亮")


def test_build_edges_preserves_direction_and_rule_counts() -> None:
    events = pd.DataFrame(
        [
            {
                "source": "刘备",
                "target": "诸葛亮",
                "chapter_number": 1,
                "paragraph_id": "P001-001",
                "paragraph_text": "evidence",
                "extraction_rule": "explicit_named_target",
            },
            {
                "source": "刘备",
                "target": "诸葛亮",
                "chapter_number": 2,
                "paragraph_id": "P002-001",
                "paragraph_text": "evidence 2",
                "extraction_rule": "adjacent_named_turns",
            },
        ]
    )

    edges = build_edges(events)

    assert edges.loc[0, "weight"] == 2
    assert edges.loc[0, "chapter_count"] == 2
    assert edges.loc[0, "explicit_target_events"] == 1
    assert edges.loc[0, "adjacent_turn_events"] == 1
