# RAG Eval Harness — Implementation Plan

## Purpose

Build a local, reproducible harness that scores a configurable RAG pipeline
against the fixture corpus (`corpus/`, 18 markdown docs) and the hand-verified
question set (`questions.jsonl`, 20 questions). The harness exists to *compare
configurations* — chunk size, k, embedder, prompt, etc. — not to serve queries.
That framing drives every design decision below: reproducibility, stability of
scores across re-runs, and per-question traceability all rank above throughput
or ergonomics.

## What we're scoring, exactly

Two independent axes, kept independent in reporting:

1. **Retrieval quality** at the document level, since ground truth
   (`relevant_doc_ids`) is at the document level:
   - **any-hit@k** — at least one relevant doc appears in top-k
   - **recall@k** — fraction of relevant docs retrieved (matters for the
     multi-doc questions q013–q016)
   - **MRR** — reciprocal rank of the first relevant doc
   All three are computed only over answerable questions.

2. **Answer faithfulness**, scored two ways in parallel so we can cross-check:
   - **Rule-based** — keyword/regex matching of the reference answer's
     concrete facts against the generated answer. The reference answers in
     `questions.jsonl` are factoid-heavy ("15 days", "1.25 days per month",
     "$1,200"), which this method is well-suited to.
   - **LLM-as-judge** — Qwen 2.5 1.5B scoring with a small closed verdict set
     `{CORRECT, INCORRECT, REFUSED, UNPARSEABLE}`.

   Both are segmented by answerable / unanswerable so that "refused" is scored
   as correct behavior for the four unanswerable questions and as a failure
   for the sixteen answerable ones. LLM-as-judge is *gated on human
   validation* (see Step 10 below); if agreement with a human-labeled sample
   is too low, judge numbers are suppressed from run reports and the rule-based
   scorer stands alone.

Every run emits per-question outcomes so comparisons can be examined
question-by-question, not just at the aggregate level. See "Comparing
configurations" below for how sample-noise is handled.

## Comparing configurations (the whole point of the harness)

With 16 answerable questions, one question ≈ 6.25% of the marginal
faithfulness score. Aggregate deltas smaller than that are literal noise, and
even deltas of one or two questions can easily be sampling variation rather
than a real config difference. Naively comparing two run summaries is
therefore misleading. The harness handles this at three levels.

**Per-question outcome matrix.** The primary artifact of a config comparison
is a table (question × config → verdict). Everything else is derived from
this. A reader who ignores the aggregate and reads only the matrix should
still reach the right conclusion.

**Paired significance test.** Both configs answer the same 16 questions, so
comparing marginals throws away the pairing. The harness runs **McNemar's
exact test** on the discordant pairs (questions where config A is correct and
B is not, and vice versa) and reports the p-value. On 16 questions this is
much more powerful than comparing two independent proportions — e.g., 6 wins
for A vs 1 for B is roughly p≈0.13, but 6 vs 0 is p≈0.03.

**Bootstrap CI on the delta.** Resample question IDs with replacement (default
2,000 iterations, seed fixed), recompute the delta on each resample, report
the 95% percentile CI. This gives an interpretable range for "how much better
is A than B on this eval set," not just a p-value.

Every pairwise comparison report includes: the outcome matrix, discordant
pair counts, McNemar's p, bootstrap CI, and an explicit "noise floor: one
question = 6.25%" line so nobody over-reads a two-percent aggregate delta.
The harness never presents a config as "better" without these numbers next
to it.

---

## Threats to score validity

These are the things that would make a comparison between two runs untrustworthy
even when the code is technically correct.

### 1. Generator and judge non-determinism
Qwen at any temperature > 0 gives different outputs per run. If run-to-run noise
exceeds the delta between two configs, we can't tell the configs apart.
**Mitigation:** greedy decoding (`do_sample=False`, temperature=0) for both
generator and judge; fix torch/numpy/random seeds; record seed + torch version +
model revision in the run manifest.

### 2. Judge validity (this axis is meaningless if the judge is wrong)
We generate and judge with the same 1.5B model. Two real problems: (a)
self-preference — the judge may reward its own phrasing over the reference;
(b) capability — a 1.5B instruct model is not obviously reliable at following
"emit one word from this list." Chatty preambles ("Looking at this, I think
the answer is more CORRECT than INCORRECT because...") would parse to whatever
the naive parser sees first, and the run would look successful while producing
garbage.

If judge output doesn't correlate with a human, every faithfulness number is a
random draw. That's worse than not publishing one, because it looks
authoritative. So judge validation is a **first-class gating step**, not a
footnote (see Step 10).

**Mitigations, in order of importance:**

1. **Human-labeled validation set + Cohen's kappa gate.** Before the judge is
   trusted for scoring, its agreement with a human-labeled sample is measured.
   If Cohen's kappa < 0.6 (a common "substantial agreement" threshold), judge
   scores are suppressed from run reports and the rule-based scorer stands
   alone. Judge agreement metrics ship in every run report so a reader knows
   how much to trust the faithfulness column.
2. **Rule-based faithfulness scorer runs in parallel.** For factoid reference
   answers, keyword/regex matching is often a stronger signal than a small
   LLM judge, and the two scorers disagreeing is a useful diagnostic.
3. **Robust judge parsing.** Verdict extraction uses a structured pattern
   (e.g., look for `VERDICT: <token>` on its own line, requested explicitly
   in the prompt), and outputs that don't parse cleanly are recorded as
   `UNPARSEABLE` rather than silently coerced. Unparseable rate is tracked
   as a first-class metric — a rising rate means the judge is drifting and
   the run should not be trusted.
4. **Fixed judge inputs across runs.** The judge sees `(question, reference
   answer, retrieved context, generated answer)` in a fixed template; the
   template and its hash are part of the run manifest so a prompt change
   forces re-validation.

Self-preference specifically is unaddressed by any of these; we accept it as
a residual risk. A stronger external judge would help, but CLAUDE.md is
explicit: local-only, no API keys.

### 3. Chunking granularity vs. gold label granularity
Ground truth is per document, but retrieval works on chunks. If we forget to
collapse chunk-hits to doc-hits before computing metrics, we misreport. Also,
chunking that splits related facts (e.g., "$1,200 stipend" and "expires after
90 days" for q004) can silently drop faithfulness even when retrieval hits the
right doc.
**Mitigation:** always deduplicate by `doc_id` before computing retrieval
metrics. Default chunker is **whole-doc**: each of the 18 markdown files is
one chunk. Docs are 19–27 lines, so this is defensible and avoids fragmenting
related facts. Chunker is pluggable so we can compare alternatives later, but
whole-doc is the baseline.

### 4. Multi-doc questions
Four questions (q013–q016) have two relevant docs. `hit@1` can never exceed
0.5 in the "all relevant retrieved" sense for these, and `recall@k` behaves very
differently from `any-hit@k`. Reporting a single number will hide this.
**Mitigation:** report `any-hit@k`, `recall@k`, and `MRR` separately, at
`k ∈ {1, 3, 5}`. Compute both aggregate and per-question values.

### 5. Unanswerable questions contaminating metrics
q017–q020 have `relevant_doc_ids: []`. Including them in retrieval metrics is
meaningless (there is nothing to retrieve). Including them naively in
faithfulness scoring would treat a correct refusal as an error.
**Mitigation:** exclude unanswerable questions from all retrieval metrics.
Report faithfulness in two buckets: `answerable_correct` and
`unanswerable_correctly_refused`. Each has a clear correct answer.

### 6. Small N, large per-question swing
20 questions total, 16 answerable. Each answerable question is 6.25% of the
marginal faithfulness score, and even 2–3 question deltas can be sampling
noise on this N.
**Mitigation:** aggregate deltas are never presented alone. Every pairwise
comparison ships with (a) the per-question outcome matrix, (b) McNemar's
exact p-value on discordant pairs, and (c) a bootstrap 95% CI on the delta.
See "Comparing configurations" above for the full protocol.

### 7. Reproducibility across machines
Different torch/transformer/embedding-model versions produce different scores.
**Mitigation:** run manifest records `sentence_transformers` model name and
revision, `transformers` version, `torch` version, seed, chunker config,
top-k, prompt templates (as hashes), and a hash of the corpus. Same manifest
→ same numbers.

### 8. Embedding cache staleness
Faster iteration wants a disk cache of chunk embeddings; a stale cache would
silently mismatch a new chunker.
**Mitigation:** cache key = hash of (embedder model name, chunker config,
chunk text). Cache is content-addressed, so a change to any input invalidates.

### 9. Prompt-injection from corpus into judge
Not realistic for this fixture corpus, but a real class of bug in a RAG system.
Called out here so we don't accidentally sanitize it away in a future refactor:
we intentionally leave it in scope for later, and note it in the plan.

---

## Data flow

```
corpus/*.md
    │
    ▼
corpus loader ─► List[Document(doc_id, text)]
    │
    ▼
chunker ─► List[Chunk(chunk_id, doc_id, text)]
    │
    ▼
embedder (sentence-transformers, cached on disk) ─► np.ndarray (n_chunks × d)
    │                                                     ▲
    │                                                     │
questions.jsonl ─► List[Question]                         │
    │                                                     │
    ▼                                                     │
for each question:                                        │
    embed question ────────────────────────────────────► cosine sim
                                                          │
                                                          ▼
                                                    top-k chunks
                                                    (chunk_id, doc_id, score)
                                                          │
        ┌─────────────────────────────────────────────────┴──────────────────┐
        ▼                                                                    ▼
retrieval metrics                                              generator (Qwen greedy)
(dedupe chunks→docs;                                           prompt = system + context + question
compute any-hit@k, recall@k, MRR;                                     │
skip unanswerable)                                                    ▼
                                                              generated_answer (string)
                                                                      │
                                                                      ▼
                                                      judge (Qwen greedy, constrained verdict)
                                                      inputs: question, reference_answer (nullable),
                                                              retrieved_context, generated_answer
                                                              │
                                                              ▼
                                                      verdict ∈ {CORRECT, INCORRECT, REFUSED}

per-question record + aggregate metrics ─► runs/{timestamp}-{config_hash}.json
```

---

## Edge cases and how they're handled

- **Empty retrieval.** Cosine top-k always returns k results, so this can only
  happen if we later add a score threshold. If it does: pass an empty context
  to the generator, whose prompt already instructs it to refuse when the
  context is insufficient. Judge treats the refusal per the rubric.

- **Unanswerable questions (q017–q020).** Retrieval still runs (we want to see
  what it returns), but the results are excluded from retrieval metrics.
  Generator prompt allows "I don't know." Judge is given `reference_answer =
  None` and instructed that the correct verdict is `REFUSED`; any confident
  answer is `INCORRECT`.

- **Judge non-determinism.** Greedy decoding + fixed seed + constrained output
  vocabulary. Parse only the verdict token from the model's output; any output
  that doesn't parse to a known verdict is recorded as `UNPARSEABLE` (a bug
  signal, not a score). Add a smoke test asserting a fixed judge input yields
  the same verdict across two calls in the same process.

- **Multi-doc gold answers.** Handled by using `recall@k` (not just `any-hit@k`)
  and by reporting per-question rank of *each* relevant doc.

- **Chunk-to-doc deduplication.** When k=5 and three of five chunks come from
  the same doc, doc-level retrieval sees one doc, not three. Dedup happens
  before scoring.

- **Reference answer text longer/shorter than generated.** Judge is prompted
  to score factual alignment, not string similarity; keep the rubric explicit.

---

## Approach recommendation

**Baseline pipeline (v0):**

- **Chunker:** whole-doc. One chunk per file. Chosen because the corpus files
  are short (max 27 lines), and splitting mid-doc would fragment related facts
  the questions depend on (e.g., q004, q009, q011). Chunker interface is
  pluggable so `HeadingChunker` or `SlidingWindowChunker` can be dropped in
  later for a comparison.

- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2`. Small, fast, well
  understood, defensible baseline. Model name is a config knob so we can swap
  in a stronger encoder as a comparison run.

- **Retriever:** numpy cosine similarity over an in-memory `(n_chunks × d)`
  matrix. Top-k, `k=3` default. No approximate index — 18 chunks is trivial.

- **Generator:** Qwen 2.5 1.5B Instruct, `do_sample=False`, seed fixed,
  `max_new_tokens` capped (~200). Prompt: system message that establishes the
  RAG contract ("answer only from the context; if the context doesn't cover
  it, say you don't know"), then the retrieved chunks concatenated with doc
  headers, then the question.

- **Judge:** Qwen 2.5 1.5B Instruct, greedy, seed fixed. Prompt gives the
  rubric, the question, the reference answer (or "no reference — this
  question was marked unanswerable"), the retrieved context, and the
  generated answer, and asks for a single verdict on its own line. Parse the
  first `{CORRECT, INCORRECT, REFUSED}` occurrence.

- **Reporting:** one JSON per run in `runs/`, containing the config manifest,
  aggregate metrics, and the per-question record (retrieved doc ids and
  ranks, generated answer, verdict, notes).

**Considered and rejected:**

- *FAISS / Chroma / a real vector DB.* Overkill at 18 chunks; adds a dependency
  and version drift with no benefit to score validity.
- *Anthropic (or any hosted) LLM as judge.* Would probably give a better judge,
  but CLAUDE.md is explicit: local model, no API key.
- *Ensemble judging (multiple prompts, vote).* Doubles or triples judge cost
  and adds a tuning surface we don't need for a baseline. Revisit if the judge
  turns out to be too noisy in practice.
- *LLM-generated additional questions or "silver" labels.* Explicitly forbidden
  by CLAUDE.md and would defeat the purpose of a hand-verified set.
- *Chunking by heading or by sliding window as the default.* Rejected as the
  starting point because it risks splitting facts that the questions require
  jointly; kept as a future comparison run.
- *Judging faithfulness and correctness as separate calls.* Cleaner
  conceptually but doubles judge cost. Fold both concerns into one verdict
  for v0, split later if the single verdict proves ambiguous.
- *LLM-as-judge as the only faithfulness signal.* Rejected because with a
  1.5B judge the reliability floor is unclear. Rule-based scoring runs
  alongside from day one, and if the judge fails validation it is dropped
  from published reports entirely.

---

## Where I'm most likely wrong

If this plan doesn't survive contact with real output, the failure will most
likely be one of these:

1. **Qwen 2.5 1.5B is too weak to be a reliable judge, full stop.** Small
   instruct models routinely ignore "output only one word" and produce chatty
   preambles that break parsing in ways that look like success. My best guess
   is that judge kappa comes in below the 0.6 gate, and the LLM-as-judge
   apparatus is dropped from the final harness in favor of the rule-based
   scorer. That's why rule-based is planned in parallel, not as a fallback:
   I'd rather that path be well-tested from the start than bolted on after
   the LLM judge disappoints.

2. **Generator quality dominates the signal on the faithfulness axis.** If
   Qwen 1.5B is not good enough to exploit better retrieval, retrieval
   deltas won't show up in faithfulness deltas — you can improve k or the
   embedder and see nothing change downstream. Retrieval metrics
   (hit@k, recall@k, MRR) are unaffected by this because they don't touch
   the generator, so those remain trustworthy. But the *end-to-end*
   faithfulness comparison across configs may be flat regardless of what
   the retrieval layer does. If that happens, the harness is still useful
   (the retrieval axis works) but the story it tells is narrower than
   advertised. Contingency: honestly report the flat result, don't fabricate
   a story from noise.

3. **The single-verdict rubric is too coarse.** Real generated answers will
   be partially correct — right fact, missing quantifier ("15 days" but not
   the monthly rate). Forcing to `{CORRECT, INCORRECT, REFUSED}` throws
   away useful signal and may correlate poorly with human judgment even for
   a competent judge. If judge validation shows the coarseness is what's
   dragging kappa down (as opposed to the model being weak), add a
   `PARTIALLY_CORRECT` verdict and re-validate.

I'd bet #1 hardest. #2 second. #3 third and most fixable.

---

## Incremental steps

Each step ends in a green test and a commit. All work is done on a feature
branch (`feature/ai-eval-harness`) per CLAUDE.md; nothing goes to `main`
directly.

**Step 0. Branch + skeleton.**
Create `feature/ai-eval-harness`. Add `src/rag_eval/__init__.py`,
`tests/__init__.py`, and a `pytest.ini` (or `pyproject.toml` addition)
pointing pytest at `tests/`. Run `pytest` — should collect zero tests and
pass. Commit.

**Step 1. Corpus loader.**
`src/rag_eval/corpus.py`: `load_corpus(path) -> list[Document]`, where
`Document = (doc_id, text)` and `doc_id` is the filename stem (`doc_01`).
Loader fails loudly rather than returning a partial set: raises if the
directory is missing, if it contains zero `.md` files, or if any file is
empty / whitespace-only. An eval that runs on a partial corpus and reports
scores anyway is worse than one that crashes.

Tests:
- Loads all 18 docs; ids match filenames; text preserved verbatim.
- Missing directory raises `FileNotFoundError`.
- Directory with no `.md` files raises `ValueError`.
- Empty file and whitespace-only file each raise `ValueError`.
- Load order is deterministic across calls.
- **Cross-consistency test** (`tests/test_corpus_questions_sync.py`): every
  `doc_id` referenced by `questions.jsonl.relevant_doc_ids` exists in the
  corpus. Written independently of the future question loader (parses
  `questions.jsonl` directly) so a bug in the loader can't hide a mismatch.
  Rationale: labels reference doc_ids by filename stem; a rename or
  deletion would silently zero out recall for affected questions without
  this guard.

Commit.

**Step 2. Chunker.**
`src/rag_eval/chunker.py`: `Chunk(chunk_id, doc_id, text)` and a
`WholeDocChunker` implementation returning one chunk per document.

`chunk_id` is `"{doc_id}#{i}"` where `i` disambiguates multiple chunks from
the same doc (always `0` for whole-doc chunking). This is deliberate: doc-level
retrieval metrics require collapsing chunk hits to doc hits, and if that
collapse is buggy `hit@k` silently inflates (e.g., three chunks from doc_05
in top-3 looks like three hits but is one doc). Baking the doc_id into the
chunk_id gives future splitting chunkers a fixed contract to preserve, so the
collapse logic downstream has one invariant to check instead of trusting
whatever a new chunker happens to produce.

Contract (documented on `WholeDocChunker` and enforced by tests, applies to
every future chunker):
- `chunk.doc_id == chunk.chunk_id.split("#")[0]`
- `chunk_id` values are unique across the corpus
- Concatenating chunks with the same `doc_id` in emitted order recovers the
  source document's text
- Emission order is deterministic across calls

Tests: chunk dataclass is frozen; small-input cases (empty list → empty
output, 2-doc list → 2 chunks); `chunk_id` format and doc_id extraction; ids
unique; text preserved verbatim; **applied to the full corpus, produces
exactly one chunk per doc with `{c.doc_id for c in chunks} == {d.doc_id for
d in docs}` and no doc emitting more than one chunk** — this is the
1:1 assertion the doc-collapse logic will rely on. Commit.

**Step 3. Question loader.**
`src/rag_eval/questions.py`: `load_questions(path, corpus_doc_ids: set[str])
-> QuestionSet`. `Question` is a frozen dataclass. `QuestionSet` wraps the
loaded rows and provides `.answerable_only()` / `.unanswerable_only()`.

`questions.jsonl` is authoritative ground truth; every retrieval and
faithfulness score derives from it. Silent breakage there produces
authoritative-looking numbers computed against a broken set. The loader
therefore re-validates every invariant at load time and raises rather than
returning a partially valid set:

- File exists and is non-empty.
- Each line parses as JSON with all required fields present and correctly
  typed (id: non-empty str; question: non-empty str; answer: str | None;
  relevant_doc_ids: list[str]; answerable: bool).
- Every `id` is unique across the file.
- Three-way consistency: `answerable == (answer is not None) ==
  bool(relevant_doc_ids)`. All three must agree; any mismatch raises.
- Every `relevant_doc_id` exists in the passed `corpus_doc_ids` set.
  Corpus ids are supplied by the caller (typically the eval runner using
  `load_corpus`) rather than hardcoded, so the invariant tracks the real
  corpus rather than a stale snapshot.

Error messages include line number and question id.

Tests: happy-path load produces 20 questions, 16 answerable / 4 unanswerable,
`.answerable_only()` / `.unanswerable_only()` return the expected splits,
unanswerable rows have `answer is None` and empty `relevant_doc_ids`,
`Question` is frozen. One test per failure mode: missing file, empty file,
malformed JSON, missing field, wrong-type field, duplicate id, each of the
four ways the three-way invariant can break (answerable=True + no answer,
answerable=True + no docs, answerable=False + has answer, answerable=False
+ has docs), and unknown `relevant_doc_id` against the passed corpus set.
The corpus-sync test uses `load_corpus` to derive the set, so a rename in
`corpus/` fails the loader without any test edit. Commit.

**Step 4. Embedder + on-disk cache.**
`src/rag_eval/embedder.py`: wraps `SentenceTransformer`. `Embedder(model_name,
cache_dir=None, model=None).embed(texts) -> np.ndarray`. Model is injected
via the `model=` kwarg for tests; the real `sentence_transformers` import is
deferred to first `embed()` so unit tests never load ST.

Cache is content-addressed on `(model_name, text)`. Two properties are
non-negotiable:

- **Model name is part of the key.** Otherwise a config-comparison run
  swapping MiniLM for mpnet reads MiniLM vectors out of cache and the whole
  embedder bake-off is silently invalid. Key = `sha256(model_name || \0 ||
  text)` with a null-byte separator so `(model="ab", text="cd")` and
  `(model="a", text="bcd")` cannot collide.
- **Cached vector is bit-identical to freshly computed.** Otherwise a
  cache write/read bug could quietly change embeddings between runs. `.npy`
  format preserves dtype and shape; test asserts `np.array_equal(fresh,
  cached)`, not approximate equality.

Empty text list raises (dimensionality is unknown; returning any shape is
a guess).

Unit tests (fast, all stub-injected):
- `_cache_key` includes model name — same text under two model names yields
  different keys, and two `Embedder`s with different `model_name` embedding
  the same text produce two files on disk.
- No separator collision between `(model, text)` boundary variants.
- Cache miss computes and writes; cache hit returns without invoking the
  model (verified by a stub that records call history).
- Cached vector `np.array_equal` fresh vector.
- Mixed hit/miss in one call — only misses are computed, output stays in
  input order.
- Cache persists across `Embedder` instances: a new instance with a
  different stub model reads the prior cache and never invokes the new
  stub.
- Empty list raises.

Slow test (`@pytest.mark.slow`, opt-in via `-m slow`): loads
`sentence-transformers/all-MiniLM-L6-v2`, embeds two short strings, asserts
shape `(2, 384)`, and asserts `np.array_equal` on a second call to prove
cache determinism against the real model. Commit.

**Step 5. Retriever.**
`src/rag_eval/retriever.py`: `Retriever(chunks, embeddings)` with
`query(question_embedding, k) -> list[Retrieved(chunk_id, doc_id, score)]`.
Uses numpy cosine (chunk embeddings pre-normalized at construction). No
approximate index — 18 chunks is trivial.

Two ordering invariants, documented on the class and covered by tests:

- **Descending score.** `result[i].score >= result[i+1].score` always.
- **Deterministic tie-break by `chunk_id` ascending.** When two chunks have
  identical cosine, the one with the lexicographically smaller `chunk_id`
  comes first. Otherwise numpy's argsort would return an implementation-
  defined order for tied scores and the same query would produce different
  top-k across runs, silently making every downstream metric
  non-reproducible. Implemented via Python's stable `sorted(...,
  key=lambda i: (-score[i], chunk_id[i]))`.

`k <= 0` raises; `k > n` returns all n results.

Tests (all fast, hand-built numpy arrays):
- `Retrieved` is frozen; length mismatch at init raises; `k=0` raises;
  `k > n` returns n.
- **Hand-worked cosine.** Chunks with un-normalized embeddings `[1,0]`,
  `[0,1]`, `[1,1]`; query `[1,0]`. Expected cosines by hand: `1.0`, `0.0`,
  `1/√2`. Test asserts both order (`a, c, b`) and each score with tolerance.
  Also proves the normalization path works (chunks were not unit-length).
- **Ties broken by chunk_id ascending.** Three chunks with identical
  embeddings, ids `z#0`, `a#0`, `m#0`; result must be `[a#0, m#0, z#0]`.
- **Stability across repeated calls.** Same tie-configured retriever,
  same query, called three times — all three results identical.
- **Partial tie.** Two tied at top, one lower — top pair broken by chunk_id.
- Scores are descending across a mixed set; orthogonal query scores 0;
  `doc_id` field populated correctly. Commit.

**Step 6. Retrieval metrics.**
`src/rag_eval/metrics_retrieval.py`: `collapse_to_docs`, `any_hit_at_k`,
`recall_at_k`, `reciprocal_rank` (all at k), and
`aggregate_retrieval_metrics` producing per-question and aggregate records.

Chunk→doc collapse is the critical operation here — it's the inflation risk
the chunk_id contract was built for. `collapse_to_docs(retrieved, k)` takes
the top-k chunks in rank order, deduplicates by `doc_id`, and returns unique
doc_ids in first-seen order (dict insertion order for determinism). All
three metrics call it, so if a splitting chunker later returns two chunks of
the same doc in top-k they collapse to one doc and cannot inflate hit@k or
recall@k.

All three metrics take the same `k` so a single retrieval call at max-k can
be scored at multiple k values (k=1, 3, 5) in one pass. MRR at k means
"1 / rank of first relevant doc within collapsed top-k, else 0."

Unanswerable questions are excluded structurally, not scored as zero: the
aggregate divides by `n_answerable` only, and metric functions raise on
empty `relevant` sets. The aggregator routes unanswerable questions around
the metric calls and records `None` in their per-question row. Per-question
records still include unanswerable entries with their retrieved doc_ids so a
reader can see what retrieval returned when nothing was relevant.

Tests (all hand-crafted with known outputs):

- **collapse_to_docs:** no-dupes case, dupes-in-top-k case, k-limits-input
  case, dupe-outside-top-k excluded, k=0 raises.
- **any_hit_at_k:** hit and miss, and — the invariant that the whole
  chunk_id contract exists for — retrieved [doc_05#0, doc_05#1, doc_03#0]
  with relevant {doc_05} at k=3 is one hit, not two.
- **recall_at_k:** single-relevant hit and miss, multi-doc one-of-two vs
  both-of-two (distinguishes 0.5 from 1.0), chunk-duplication does not
  inflate: retrieved [a#0, a#1, a#2] with relevant {a, b} at k=3 is 0.5
  (not 1.5), empty relevant raises.
- **reciprocal_rank:** first position (1.0), second position (0.5), not
  found (0.0), rank-is-at-doc-level: retrieved [a#0, a#1, b#0] with
  relevant {b} is 0.5 not 1/3 (b is the 2nd distinct doc), empty raises.
- **Aggregator:** hand-crafted 3-answerable + 2-unanswerable case with
  values computed by hand. Asserts aggregate any_hit_rate, mean_recall,
  mean_reciprocal_rank divide by 3 (not 5); per_question has 5 rows with
  unanswerable ones showing `None` for hit/recall/rr but populated
  `retrieved_doc_ids`; `n_answerable=3`, `n_unanswerable=2`. Commit.

**Step 7. Retrieval-only end-to-end.**
`src/rag_eval/eval_retrieval.py`: thin wiring — loader → chunker → embedder
→ retriever → metrics → JSON under `runs/`. Returns the written `Path`.

The run JSON must record which chunker, embedder, and k produced it, or
the report is unusable for the config comparisons this harness exists to
enable. Config section is first-class:

```
{
  "config": {"chunker": "WholeDocChunker", "embedder_model": "...", "k": 3},
  "generated_at": "...",
  "aggregate": {...},
  "per_question": [...]
}
```

Filename also encodes the config so `ls runs/` is informative:
`retrieval-{chunker_slug}-{embedder_slug}-k{k}-{ts}.json`. Microsecond
timestamp so same-second re-runs don't collide.

Fast tests (in-memory corpus, stub-injected Embedder — no ST):
- Full run writes a file at the expected path and returns it.
- `config` section carries chunker name, embedder model, and k accurately;
  changing any of the three shows up in the file.
- `aggregate` section has all required fields.
- `per_question` has one entry per question with expected shape.
- Filename slugs match the config values used.

Slow test (`@pytest.mark.slow`): real MiniLM on the fixture corpus and
questions at k=3, asserts the file is valid, `per_question` has 20 entries,
`n_answerable=16`, `n_unanswerable=4`. Commit.

**Step 8. Generator.**
Two files:

- `src/rag_eval/prompts.py`: `render_generator_prompt(question,
  context_chunks) -> list[dict]` returning a Qwen-style message list
  (system + user). `context_chunks` is a list of `(doc_id, text)` tuples
  in retrieval rank order; the renderer formats them with visible headers
  (`--- {doc_id} ---\n{text}`) so chunk boundaries don't blur.
  `SYSTEM_PROMPT` establishes the RAG contract: answer only from
  context, and if the context does not cover the question respond
  *exactly* with `"I don't know."`. The "exactly" matters because the
  judge and rule-based scorer both parse refusals — a fixed refusal
  string keeps that parse simple.
- `src/rag_eval/generator.py`: `Generator(model_name, seed=42,
  max_new_tokens=200, model=None, tokenizer=None)`. `.generate(messages)
  -> str` applies the tokenizer's chat template, runs greedy decoding
  (`do_sample=False`), decodes only the newly generated tokens (not the
  echoed prompt). `torch.manual_seed` is set at first model load.
  Transformers import is deferred to first call; model + tokenizer
  injectable so unit tests don't pull ML deps.

Fast tests: prompt renderer returns 2 messages (system + user), system
message contains the "I don't know" contract, user message contains the
question, each `(doc_id, text)` tuple appears in the user message,
multi-chunk contexts get their delimiter, empty context still renders.

Slow test (`@pytest.mark.slow`): real Qwen greedy generation of a fixed
prompt twice, `assert output_1 == output_2` — this is the determinism
guarantee the config-comparison harness depends on. Commit.

**Step 9. Rule-based faithfulness scorer.**
`src/rag_eval/scorer_rules.py`: `score_rule_based(reference, generated) ->
Verdict`. Returns `{CORRECT, INCORRECT, REFUSED}` per the "did the right
thing for this question type" semantics: CORRECT means the model behaved
correctly for the question (matched reference, or refused an unanswerable);
INCORRECT means it did the wrong thing; REFUSED is reserved for answerable
questions where the model gave up.

Strategy is number-only extraction with multi-fact strictness — every
numeric fact in the reference must appear in the answer (after
normalization) for CORRECT. Missing any → INCORRECT. This is the primary
over-credit guard: a partial answer scores INCORRECT rather than CORRECT,
because over-crediting is the failure mode that would make the scorer
useless as a standalone signal.

Normalizations:
- Strip commas and currency symbols: `"1,200"` ≡ `"1200"` ≡ `"$1,200"` ≡
  `"1,200 dollars"`.
- Strip trailing decimal zeros: `"1.250"` ≡ `"1.25"`, `"100.00"` ≡ `"100"`.
- Number-words for 0-20 and tens 30-90: `"fifteen"` ≡ `"15"`, `"five"` ≡
  `"5"`. Compound number-words ("twenty-five") not handled — deferred.

Design decisions with tradeoffs, documented in the code:
- **"Just the number" matches** — `"15"` alone matches reference `"15 days"`
  because the number is the primary fact and the unit is usually implied
  by the question. Matches the user's explicit requirement.
- **Extra numbers in the answer don't penalize.** A verbose answer that
  mentions the right facts plus extras is CORRECT. Penalizing verbosity
  would depress scores on well-formed answers.
- **Reference with no extractable numbers → INCORRECT conservatively.**
  Can't verify; under-crediting is preferable to over-crediting.

Refusal detection: case-insensitive substring match on `"I don't know"`
(matches the REFUSAL_STRING constant from prompts.py). The system prompt
tells the generator to emit the exact string, so this is a tight coupling
by design — a change to REFUSAL_STRING requires updating both.

Tests are hand-crafted, one per case:
- Number normalization: `"$1,200"`, `"1200"`, and `"1,200 dollars"` all
  match `"$1,200 stipend"` reference; `"fifteen"` and `"15"` both match
  `"15 days"` reference.
- Multi-fact strictness: reference with two numbers, answer with one →
  INCORRECT (both the "missing dollar amount" and "missing second fact"
  variants of partial answer).
- Wrong number, empty answer → INCORRECT.
- Answerable + refused → REFUSED (distinct from INCORRECT so we can
  measure "give up" separately from "guess wrong").
- Unanswerable + refused (in isolation and embedded in longer text) →
  CORRECT.
- Unanswerable + confident answer → INCORRECT (hallucinated on nothing).
- Unanswerable + empty → INCORRECT (not a refusal).
- Extra numbers in answer → CORRECT (design decision documented).
- Reference without numbers → INCORRECT (conservative). Commit.

**Step 10. LLM-as-judge + judge validation.**
Two files, one commit:
- `src/rag_eval/judge.py`: `judge(question, reference, context, answer) ->
  Verdict`. Prompt requests a strict `VERDICT: <token>` line; parser looks
  only for that pattern and returns `UNPARSEABLE` if not found or if
  multiple conflicting verdicts appear. Handles `reference is None` path
  for unanswerable questions.
- `src/rag_eval/judge_validation.py`: loads a human-labeled sample from
  `judge_validation.jsonl` (created in this step by running 3–4 diverse
  configs to collect ~40 outputs and hand-labeling with the same rubric —
  human, not LLM, per CLAUDE.md), scores each with the judge, computes raw
  agreement and Cohen's kappa, and writes a `judge_validation.json` report.

Tests: verdict parsing handles whitespace / case / trailing punctuation and
correctly returns `UNPARSEABLE` on chatty preambles; agreement math
(raw, kappa, confusion matrix) is correct against hand-worked examples; a
slow test runs three canned inputs (clearly correct, clearly incorrect,
clear refusal) and asserts expected verdicts.

The judge is considered validated only if kappa ≥ 0.6. This gate is
enforced by the eval runner (Step 11): below threshold, LLM-as-judge
scores are computed and stored but suppressed from headline metrics,
and the run report carries a prominent warning.

Rationale for 0.6: it's the conventional "substantial agreement" floor
from the Landis & Koch scale. For a v0 harness with a coarse 3-verdict
rubric on ~40 labeled examples, 0.7 overclaims precision we can't
support at that N, and 0.4 is too loose to trust the faithfulness
numbers derived from it. Revisit if the labeled set grows or the rubric
gets finer. Commit.

**Step 11. Full pipeline eval.**
`src/rag_eval/eval.py`: retrieval + generation + both faithfulness scorers
(rule-based always headline; LLM-as-judge conditional on validation gate)
+ aggregation. Emits per-question detail and aggregate rates: `retrieval:
{any_hit@k, recall@k, mrr}`, `faithfulness_rule_based: {answerable_correct,
unanswerable_correctly_refused}`, and optionally `faithfulness_judge: {...,
unparseable_rate, judge_kappa}`. Writes `runs/{timestamp}-{config_hash}.json`.
Test: opt-in end-to-end smoke test producing a valid run file with all
sections populated; a second test with a stubbed sub-threshold kappa
confirms judge scores are computed but not headlined. Commit.

**Step 12. Run manifest + config.**
Introduce `configs/baseline.yaml` (chunker, embedder model, k, generator
model, seed, judge model, prompt hashes, bootstrap iterations, kappa
threshold). `src/rag_eval/manifest.py` computes a stable config hash and
captures runtime versions (`torch.__version__`, `transformers.__version__`,
`sentence_transformers` version, corpus hash, judge_validation kappa at
time of run). Test: given a fixed config file, manifest fields present;
hash is stable across two invocations. Commit.

**Step 13. Pairwise config comparison report.**
`src/rag_eval/compare.py`: takes two run JSONs and emits a comparison
report containing (a) the per-question outcome matrix for each scoring
axis, (b) discordant pair counts and McNemar's exact p-value on each
axis, (c) bootstrap 95% CI on each aggregate delta (default 2,000
resamples, seed from config), and (d) an explicit "noise floor: 1 question
= 6.25% on the 16-answerable set" line at the top of every faithfulness
comparison. Test: hand-crafted pairs of runs yield known McNemar counts and
known bootstrap CIs (seed pinned). Commit.

**Step 14. CLI entrypoint.**
`python -m rag_eval run --config configs/baseline.yaml` and
`python -m rag_eval compare --a runs/A.json --b runs/B.json`. Thin wrappers
around `eval.py` and `compare.py`. Test: `--help` runs; a `--dry-run` prints
resolved config without loading models. Commit.

At the end of Step 14 there is a runnable baseline and a principled way to
compare it against a second config. Every subsequent experiment (different
chunker, different k, different embedder) is a new config file plus a
`compare` invocation, and the comparison output tells you whether the
delta is real or noise before you draw any conclusion from it.
