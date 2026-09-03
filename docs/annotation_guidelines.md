# Character Relation Annotation Guidelines

## Purpose

The task is to identify the main relationship shown between two named
characters in one passage from *Romance of the Three Kingdoms*. Base the label
mainly on the supplied passage. Do not label a relation only because it is
generally known from the novel.

## What to complete

For every row, complete the fields from `primary_relation` through
`annotation_status`. Use `reviewed` only after all required decisions are
complete. Copy the shortest useful words or sentence into `evidence_text`.

## Primary relation labels

### `cooperation`

The characters help each other, work together, form an alliance, or share an
immediate goal. Example: one character supplies troops while the other accepts
the help. Do not use it when they only appear in the same meeting.

### `hierarchy_loyalty`

The passage shows service, command, obedience, duty, or loyalty between ruler
and subordinate, commander and officer, or lord and adviser. Example: an
officer receives and follows an order. Do not assume loyalty from titles alone.

### `kinship`

The passage identifies a relation by blood, marriage, adoption, or accepted
family status. Example: a father gives instructions to his son. Do not use this
label for sworn brotherhood.

### `friendship_brotherhood`

The passage shows personal friendship, trust, sworn brotherhood, or a personal
bond outside official duty. Example: sworn brothers protect one another. Do not
use it for ordinary military cooperation.

### `hostility_conflict`

The characters attack, threaten, punish, reject, or clearly oppose each other.
Example: one character orders an attack against the other. Mere political
difference without action may be too weak.

### `deception_manipulation`

One character tricks, tests, persuades, traps, secretly controls, or uses the
other. Example: a false message is used to lead another character into an
ambush. The manipulation may be indirect.

### `affection_romance`

The passage shows romantic interest, marriage, desire, jealousy, or intimate
attachment. Do not use this label for kinship without romantic evidence.

### `no_clear_relation`

Both characters occur, but the passage gives no useful evidence of a relation
between them. Use this when the passage is clear but the pair merely co-occurs.

### `uncertain`

The passage may show a relation, but the evidence is too ambiguous to choose a
reliable label. Use this when the meaning is unclear, not when there is clearly
no relation.

## Choosing one primary label

Choose the relation that is most important in this passage. If two labels are
both supported, put the less important one in `secondary_relation`. Do not add
new labels. Explain a difficult choice in `annotator_notes`.

## Additional attributes

- `relation_direction`: `A_to_B`, `B_to_A`, `bidirectional`, or `unclear`.
- `relation_polarity`: `positive`, `negative`, `mixed`, `neutral`, or `unclear`.
- `relation_explicitness`: `explicit` when directly stated, `implicit` when
  shown through action, `inferred` when several clues are needed, or `unclear`.
- `relation_temporality`: `stable`, `temporary`, `changing`, or `unclear`.
- `annotator_confidence`: use a number from 1 (very unsure) to 5 (very sure).

Do not force an attribute when the passage does not support it. Use `unclear`.

## Aliases and titles

`character_a` and `character_b` are canonical names. `surface_a` and
`surface_b` show the form found in the passage. Treat an alias or title as the
canonical character only when the supplied mapping and passage support it.

## Changing relations and wider context

If the relation changes inside the passage, label the main change and use
`changing` for temporality. You may consult nearby narrative context only when
the supplied passage is incomplete. If you do, record this in
`annotator_notes`. Do not use later events or general knowledge during model
evaluation.

## Evidence and notes

Copy exact, short evidence into `evidence_text`; do not paraphrase it. Use
`annotator_notes` for ambiguity, wider context, overlapping relations, or a
problem with the passage. The `suggested_relation` field is only a simple rule
suggestion and may be wrong. Always make an independent decision.
