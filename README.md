# Three Kingdoms Character Network Study

Digital Humanities research comparing paragraph co-occurrence, named dialogue,
and semantic-context networks in *Romance of the Three Kingdoms*.

## View the project

- Website: https://three-kingdoms-network-study.streamlit.app/
- This branch contains the research source code and tests.
- The separate `codex/20260817-streamlit-release` branch runs the website.

## Code guide

| Location | Purpose |
| --- | --- |
| `app.py` | Streamlit pages and downloads |
| `src/network_visualization.py` | Shared network layout and interactive views |
| `src/prepare_reference_corpus.py` | Prepare the Gutenberg working corpus |
| `src/extract_character_candidates.py` | Propose character names for review |
| `src/extract_character_mentions.py` | Match reviewed character aliases |
| `src/build_paragraph_cooccurrence_network.py` | Build paragraph links |
| `src/build_dialogue_network.py` | Extract named speech links |
| `src/build_semantic_context_network.py` | Build context-similarity links |
| `src/compare_network_methods.py` | Compare the three networks |
| `src/annotation/` | Prepare and validate annotation batches |
| `src/model/` | Exploratory few-shot and weak-supervision experiments |
| `tests/` | Automated tests |
| `tools/` | Workbook and deployment utilities |
| `config/` | Relation labels and model/training settings |
| `outputs/` | Nine saved tables used by the website |

All 37 Python source files from the research `src` directory are included.
Dissertation drafting/formatting scripts are not part of this code release.

## Run the website

Use a separate Python environment, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The website reads saved results. It does not retrain models or rebuild the
networks when a reader opens it.

## Research workflow and dependencies

See [the research workflow](docs/RESEARCH_WORKFLOW.md) for script commands and
expected input/output paths. Some sections record earlier stages rather than
the final validation state. Research development used Python 3.11.
`requirements-research.txt` is the recorded research environment, not a claim
that a fresh installation has been tested on every platform.

After installing the research dependencies, tests can be run with:

```bash
python -m pytest tests -q
```

The archive test expects a generated website ZIP. Before the full test suite,
run `python tools/build_release_archive.py` to create that local file.

The raw novel, reviewed annotation files, intermediate data, trained models,
and model caches are not included in this source-code release. Rebuilding
every result therefore requires additional inputs; this is not a self-contained
full reproduction archive. The corpus source is Project Gutenberg eBook 23950:
https://www.gutenberg.org/ebooks/23950 . Obtain the text under its applicable
terms and follow the input paths in the workflow. Do not replace recorded
human judgements with automatically generated labels.

The optional `.mjs` workbook scripts use `@oai/artifact-tool` from the original
development environment and retain original local paths. They are included
for inspection, but require that tool and path configuration to run elsewhere.
They are not required to run Streamlit.

## Interpretation and validation

Paragraph co-occurrence, dialogue, and semantic similarity define different
types of edges. Shared paragraphs do not by themselves establish direct contact.
Context similarity does not establish a meeting or conversation.

The researcher checked 166 sampled mentions and 60 co-occurrence examples.
The full dialogue and semantic review samples were not completed, so these
networks remain exploratory. The few-shot and weak-supervision classifiers
did not establish reliable generalisation and are not used as final relations
in the website. Automated software tests are not a substitute for text review.

AI tools assisted code development and documentation. Their use is disclosed
in the dissertation; the researcher is responsible for the submitted work.

## Release scope

This release excludes dissertation drafts, university forms, account credentials,
personal files, virtual environments, and model caches. No new licence for
third-party material is granted by this repository. For a stable reference,
cite the commit-specific GitHub URL rather than only a moving branch.
