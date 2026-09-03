# Three Kingdoms Character Network Project

This project supports a Digital Humanities study of character networks in
*Romance of the Three Kingdoms*.

## Relation Annotation Round 1

Generate the reproducible 60-example Round 1 batch after the corrected Round 0
annotations pass formal validation:

```text
python src/annotation/select_round1_candidates.py
```

The script writes `data/annotation/annotation_batch_02.csv` and
`outputs/reports/round1_selection_report.md`. The editable workbook is stored at
`outputs/few_shot_round1/annotation_batch_02.xlsx`. Complete the yellow columns,
copy exact supporting words into `evidence_text`, and change
`annotation_status` to `reviewed` only after finishing each row. The
`suggested_relation` column is a rule-based sampling hint and may be wrong; it
must not be treated as a ground-truth label.

## Frozen Few-Shot Retrieval Diagnostic

Use the 20 reviewed demonstrations to create a non-training nearest-example
diagnostic:

```text
python src/model/few_shot_retrieval.py
```

This writes `outputs/few_shot_round1/annotation_batch_02_model_aided.csv` and
`outputs/reports/few_shot_retrieval_baseline.json`. The added predictions are
suggestions only. The script never fills `primary_relation` or changes
`annotation_status`, and the output must not replace researcher review.

## Exploratory Round 0 Classifier

Freeze deterministic 12/4/4 splits from the 20 reviewed examples:

```text
python -m src.model.prepare_splits
```

Train a logistic-regression head on frozen multilingual MiniLM embeddings:

```text
python -m src.model.train_frozen_classifier
```

The split manifest is stored under `data/model/round0_five_label`. Model files,
validation results, protected-test predictions, and the model card are stored
under `outputs/model/round0_five_label_frozen_minilm`. This experiment covers
only the five labels found in Round 0 and must not be presented as a complete
nine-label relation model.

## Character Candidate Extraction

The candidate extraction script creates an initial list of possible character
names from the novel. It uses repeatable rules for speech labels, formal name
introductions, courtesy names, and chapter-title evidence.

The method is semi-automatic. A rule can find useful names, but it can also
produce false positives or miss a character. The generated files must therefore
be reviewed by a person before they are used to build a network.

Alias normalisation is necessary because one character can have several names.
For example, 刘备 and 玄德 refer to the same character. Without normalisation,
the network would treat them as separate people.

Run the extraction from the project root:

```text
python src/extract_character_candidates.py
```

The script creates:

```text
data/metadata/character_candidates.csv
data/metadata/extracted_alias_pairs.csv
data/metadata/text_integrity_report.txt
```

Open `character_candidates.csv` and review high-frequency rows first. Fill in
the review columns only after checking the sample evidence in the novel. The
alias-pair file contains possible links between full names and courtesy names.
All pairs start with a `pending` status.

Titles such as 丞相, 主公, 使君, and 皇叔 can refer to different people in
different passages. They must not be assigned automatically to one character.

Run the tests from the project root:

```text
pytest
```

The first candidate list is not a complete character dictionary. It can contain
false positives, missed names, shortened names, and ambiguous titles. Human
validation remains a required part of the research method.

### Current corpus warning

The integrity report currently records possible encoding damage in the source
text. The novel still contains 120 detected chapters, but some passages contain
ASCII question marks and mojibake-like text. Review
`data/metadata/text_integrity_report.txt` before using the corpus for final
research results. The extraction script reports this issue but does not alter
the raw text.

## Prepared reference corpus

A clean reference was downloaded from Project Gutenberg eBook 23950 and is
preserved at:

```text
data/reference/gutenberg_pg23950.txt
```

Prepare a separate Simplified Chinese corpus with:

```text
python src/prepare_reference_corpus.py
```

This creates:

```text
data/processed/three_kingdoms_gutenberg_simplified.txt
data/metadata/corpus_provenance.md
```

The preparation script excludes the Gutenberg header and footer from the
working text, converts Traditional Chinese with OpenCC, joins hard-wrapped
lines, and normalises 120 chapter headings. It does not overwrite the original
raw file.

Run candidate extraction on the prepared corpus without overwriting earlier
metadata:

```text
python src/extract_character_candidates.py --input data/processed/three_kingdoms_gutenberg_simplified.txt --output-dir data/metadata/gutenberg
```

Read `data/metadata/corpus_provenance.md` before citing or sharing the derived
corpus.

## Frequency filtering

Keep only candidates with at least 10 extraction events:

```text
python src/filter_character_candidates.py
```

The default output is:

```text
data/metadata/gutenberg/character_candidates_frequency_ge_10.csv
```

The threshold is inclusive. A frequency of 10 is retained, while a frequency
of 9 is excluded. This is a candidate-level filter, not a final person-level
appearance count. Aliases still need to be reviewed and merged, and false
positive candidates still need to be removed.

## Provisional candidate review

Create a review table with three evidence snippets per high-frequency
candidate:

```text
python src/prepare_candidate_review.py
```

The output is:

```text
data/metadata/gutenberg/candidate_review_frequency_ge_10.csv
```

The table separates provisional characters, lexical false positives, and
context-dependent aliases. Stable aliases receive a proposed canonical name.
Ambiguous short forms such as 昭, 平, 亮, and 攸 are not forced into one
identity. All automatic decisions remain `provisional`; complete the
`human_decision`, `human_canonical_name`, and `human_notes` columns before
calling the data validated.

## Event-level ambiguity resolution

Resolve context-dependent short forms by chapter and evidence event:

```text
python src/resolve_ambiguous_candidates.py
```

This creates:

```text
data/metadata/gutenberg/ambiguous_candidate_events.csv
```

The same short form can map to different people. For example, 肃 can refer to
李肃, 鲁肃, or 王肃 in different chapters. The event table preserves each
decision, rule, chapter, and evidence snippet.

Build the provisional canonical table:

```text
python src/build_provisional_character_dictionary.py
```

The output is:

```text
data/metadata/gutenberg/provisional_canonical_characters_frequency_ge_10.csv
```

Aliases are merged before the inclusive threshold of 10 is applied again. The
current frequency remains a count of rule-based evidence events, not a complete
raw-text mention count. All rows remain provisional until human confirmation.

## Formal alias review and enrichment

Create a transparent review of formal names and courtesy names extracted from
the corpus:

```text
python src/review_formal_alias_pairs.py
```

Then enrich the 83-character provisional table:

```text
python src/enrich_character_dictionary_aliases.py
```

The enrichment produces:

```text
data/metadata/gutenberg/provisional_character_dictionary_with_aliases.csv
data/metadata/gutenberg/alias_conflicts.csv
```

Conflicting short aliases are retained for documentation but marked
`requires_context`. They must not be assigned globally to one character.

## Full-text mention extraction

Scan all 120 chapters with the enriched dictionary:

```text
python src/extract_character_mentions.py
```

The matcher uses the longest available alias at each position. It excludes all
one-character aliases and every alias marked as a cross-character conflict.
The outputs are:

```text
data/metadata/gutenberg/character_mention_events.csv
data/metadata/gutenberg/character_mention_summary.csv
```

`raw_mention_frequency` is an exact-match mention count across the derived
corpus. It is intentionally separate from the earlier rule-based
`extraction_event_frequency`.

## Mention validation

Create a balanced human-review sample:

```text
python src/prepare_mention_validation.py
```

The sample contains two mentions for every retained character and prefers
different aliases where possible:

```text
data/metadata/gutenberg/mention_validation_sample.csv
```

Fill `human_is_correct` with `yes`, `no`, or `uncertain`. If a match is wrong,
enter the correct person in `human_correct_canonical_name` and explain the
decision in `human_notes`. Run the script again to calculate precision without
overwriting any human-entered fields. Blank and uncertain rows are excluded
from the precision denominator.

After a complete successful review, create the validated network-input
dictionary:

```text
python src/finalise_character_dictionary.py
```

The output is `data/metadata/gutenberg/final_character_dictionary.csv`.

## Paragraph-level co-occurrence baseline

Build the first network model:

```text
python src/build_paragraph_cooccurrence_network.py
```

Two characters receive one unit of edge weight when both occur in the same
body paragraph. Repeated mentions inside that paragraph do not add more
weight. Chapter headings are excluded. Outputs are stored in:

```text
outputs/cooccurrence/paragraph/paragraph_mentions.csv
outputs/cooccurrence/paragraph/edges.csv
outputs/cooccurrence/paragraph/nodes.csv
outputs/cooccurrence/paragraph/network_summary.txt
```

This network represents textual proximity, not confirmed dialogue, friendship,
alliance, or sentiment.

## Network figures

Generate reproducible full-network and strong-tie figures:

```text
python src/visualise_cooccurrence_network.py
```

The full view contains all 83 validated characters and all 1,144 paragraph
co-occurrence edges; only the top 25 characters by weighted degree are labelled
to limit overlap. The core view retains edges with weight at least 30, meaning
that both characters occur in at least 30 of the same body paragraphs. It
contains 35 characters and 62 edges. Both figures encode weighted degree by
node size, edge weight by line width, and Louvain community by colour. PNG and
SVG outputs, reproducible coordinates, and figure notes are stored in:

```text
outputs/cooccurrence/paragraph/figures/
```

## Named-speech dialogue network

Build the conservative dialogue model:

```text
python src/build_dialogue_network.py
```

The script detects either an explicit validated speaker and named target or two
consecutive different named speakers in the same body paragraph. It preserves
speech direction and saves inspectable evidence under
`outputs/dialogue/named_speech/`. Unnamed pronouns and implied addressees are
not resolved, so this is a conservative named-speech baseline.

## Semantic-context network

Build the multilingual sentence-embedding model:

```text
python src/build_semantic_context_network.py
```

The method uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. It extracts a
local context of 90 characters on either side of a validated alias, samples at
most 100 contexts per character across the narrative, mean-pools normalised
384-dimensional vectors, and retains mutual five-nearest-neighbour pairs with
cosine similarity at least 0.45. Before character-level mean pooling, the
corpus-wide context centroid is subtracted to reduce the shared classical-style
component. Outputs are stored under
`outputs/semantic/multilingual_minilm/`. Semantic similarity describes textual
contexts, not direct social interaction.

## Multi-method validation and comparison

```text
python src/prepare_multi_method_validation.py
python src/compare_network_methods.py
```

The workbook at
`outputs/validation/multi_method/multi_method_validation_workbook.xlsx`
contains 60 dialogue events and 60 semantic edges. Complete its yellow review
fields before treating either new network as final. Common graph summaries,
rankings, and edge-overlap measures are stored under `outputs/comparison/`.

For a shorter calibration review, generate the ten most difficult dialogue
cases and ten lowest-similarity retained semantic cases:

```text
python src/prepare_hard_case_validation.py
```

Review the yellow cells in
`outputs/validation/hard_cases/hard_case_validation_for_researcher.xlsx`.
These 20 cases are a calibration subset rather than a replacement for the full
validation evidence. After the researcher returns the decisions, they can be
used to guide a transparent first-pass completion of the two 60-row sheets;
the researcher must still confirm the completed full workbook.

## Few-shot relation annotation pilot

Build the passage-level candidate pool and select the fixed 20-example Round 0
pilot:

```text
python src/annotation/build_relation_instances.py
python src/annotation/select_annotation_candidates.py
```

The editable workbook is stored at
`outputs/few_shot_pilot/annotation_batch_01.xlsx`. Complete only the yellow
review columns and change `annotation_status` to `reviewed` after finishing a
row. The pilot must be returned and validated before any model training begins.

After the researcher completes the workbook, extract its values with the
artifact-tool inspection workflow, then create and validate a versioned CSV:

```text
python src/annotation/import_reviewed_pilot.py
python src/annotation/validate_annotations.py
```

Validation writes `outputs/reports/annotation_validation_report.json`. Do not
expand the annotation dataset or train a model while its status is `BLOCKED`.

## Co-occurrence edge validation

Create a deterministic review sample:

```text
python src/prepare_edge_validation.py
```

The sample contains 20 strong, 20 medium, and 20 weak ranked edges. Complete
the three categorical review fields in either the CSV or the formatted Excel
workbook:

```text
outputs/cooccurrence/paragraph/edge_validation_sample.csv
outputs/cooccurrence/paragraph/edge_validation_workbook.xlsx
```

Run the Python script again after review to calculate sample precision and the
distribution of direct, indirect, and unclear interactions.

If review was completed in the Excel workbook, export its `Validation` sheet
to `edge_validation_results.csv`, then score that file explicitly:

```text
python src/prepare_edge_validation.py --input outputs/cooccurrence/paragraph/edge_validation_results.csv
```

For easier manual searching, generate the highlighted evidence version:

```text
python src/highlight_edge_validation_evidence.py
```

The highlighted workbook uses blue for the source character and orange for the
target character. Actual aliases in the paragraph are wrapped with
`【SOURCE:...】` and `【TARGET:...】` markers:

```text
outputs/cooccurrence/paragraph/edge_validation_workbook_highlighted.xlsx
```

## Automatic Weak-Supervision Experiment

To continue without manual Round 1 annotation, create explicitly unreviewed
weak labels, train the experimental nine-class head, and run full inference:

```text
python -m src.model.create_weak_round1_labels
python -m src.model.train_weakly_supervised_classifier
python -m src.model.predict_relation_instances
```

The full-corpus outputs are under
`outputs/predictions/weakly_supervised_nine_label`. They are automatic,
unreviewed predictions and must not be described as validated character
relations. The current model collapses to two predicted classes.
# Interactive method-comparison artefact

Run the dissertation comparison interface from the project root:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

The app keeps the three network definitions separate while applying one visual
system and one deterministic union-graph layout. It includes whole-network
metrics, edge overlap, thresholded side-by-side views, focal-character ego
networks, top-node rankings, and explicit interpretation limits. The
`Project and data` tab adds an anonymous project description, corpus
provenance, frozen method settings, nine CSV download buttons, reproducibility
guidance, and a clear statement that dialogue and semantic outputs remain
exploratory because their complete human-validation samples were not scored.

For deployment, `requirements.txt` contains only website packages. The full
research environment is preserved in `requirements-research.txt`. Run
`python tools/check_deployment.py` before publishing and follow
`docs/DEPLOYMENT.md`.
