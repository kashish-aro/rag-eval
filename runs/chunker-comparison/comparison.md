# Chunker comparison (retrieval-only)
- embedder: `sentence-transformers/all-MiniLM-L6-v2`
- k: 3
- corpus: fixture (`corpus/`, 18 docs)
- questions: fixture (`questions.jsonl`, 16 answerable / 4 unanswerable)

## Aggregates (over 16 answerable questions)

| Chunker | Params | any_hit@k | recall@k | MRR |
|---|---|---:|---:|---:|
| WholeDocChunker | - | 1.000 | 0.969 | 1.000 |
| FixedWindowChunker | size=400, overlap=50 | 1.000 | 0.938 | 1.000 |
| HeadingChunker | - | 1.000 | 0.938 | 1.000 |

## Per-question hit@k (answerable only)

Every answerable question is a hit for every chunker (all three at 1.000
on any_hit@3, all three at MRR 1.000). The interesting axis is recall on
the four multi-doc questions, since every single-doc question is trivially
recall=1 when any_hit=1.

## Multi-doc recall breakdown (q013–q016)

| Question | Relevant docs | WholeDoc | FixedWindow | Heading |
|---|---|---:|---:|---:|
| q013 | doc_04, doc_05 (expenses + per diem) | 1.0 | 0.5 | 0.5 |
| q014 | doc_11, doc_12 (incident + on-call)  | 1.0 | 0.5 | 1.0 |
| q015 | doc_14, doc_15 (API keys + rate limits) | 1.0 | 1.0 | 0.5 |
| q016 | doc_03, doc_16 (winter closure + deploy freeze) | 0.5 | 1.0 | 1.0 |

Each chunker misses one relevant doc on the multi-doc set — just a
different one. WholeDoc misses doc_16 on q016; FixedWindow misses doc_05
on q013 and doc_12 on q014; Heading misses doc_05 on q013 and doc_15 on
q015. Two of the three chunkers miss doc_05 for q013, suggesting the per
diem doc's phrasing does not surface easily when split.

## Files

- `retrieval-wholedoc-all-minilm-l6-v2-k3-20260722T011552804836Z.json`
- `retrieval-fixedwindow-all-minilm-l6-v2-k3-20260722T011553564294Z.json`
- `retrieval-heading-all-minilm-l6-v2-k3-20260722T011553805527Z.json`

## Analysis

**Honest read: no winner on this evidence.** WholeDoc has a
0.031-point lead in recall over the other two (3/4 multi-doc perfect vs
2/4). That corresponds to exactly one question. Our noise floor is 6.25%
per question on the 16-question set — the aggregate delta (3.1%) is
below that floor. On a 4-question subset it is trivially within noise;
you cannot distinguish 2/4 from 3/4 with any statistical confidence.

Everything else is a tie: all three at 1.000 for `any_hit@3` and MRR.
Every chunker finds at least one relevant doc for every answerable
question, and always at rank 1. The only axis on which chunkers differ
is whether the *second* relevant doc for a multi-doc question makes it
into top-3.

**Why this pattern makes sense on this corpus.** The docs are short and
topically focused. Any competent embedder can find the primary relevant
doc for a query; the split only matters when a query needs two different
docs at once. The specific failures suggest a genuine tradeoff rather
than one strategy being uniformly better:

- WholeDoc's whole-doc vectors favor queries where the topic is the
  document's main theme. It misses doc_16 on q016 because the deploy
  freeze is a small section of a doc that is broadly about deployment
  process, and the whole-doc vector doesn't emphasize the freeze angle
  enough.
- Splitting chunkers do better exactly here (both catch doc_16 for q016)
  because the freeze section becomes its own signal. They lose on q013
  and q014/q015 because splitting fragments the "per diem" and "on-call
  escalation" documents whose relevance depends on the reader assembling
  facts across the whole doc.

Whole-doc favors *broad-theme* multi-doc questions; splitting favors
*specific-section* multi-doc questions. On the four questions we have,
that trade lands one question in WholeDoc's favor. Different four
questions and it would land the other way.

**Recommendation.** Keep WholeDocChunker as the baseline. Not because
it's demonstrably better on this evidence, but because:

1. It ties or leads on every metric we measured.
2. It's the simplest (no `size`/`overlap` to tune).
3. The retrieval axis is not where the chunker's cost lies on this
   corpus — the choice matters more for the *generator's* prompt, which
   this comparison doesn't measure. WholeDoc gives Qwen ~500-1500 chars
   of context per retrieved doc; Heading gives ~100-400. That's the
   next thing to A/B, not retrieval.
4. The evidence for switching would need to come from either a larger
   question set (so a 1-question delta stops being noise) or a downstream
   faithfulness measurement where prompt length actually affects the
   generator.

For a stronger signal on chunking specifically, the useful next step is
either grow the multi-doc question set (we currently have 4; even 8–12
would let us start seeing real deltas) or add a k-sweep (k=1, 3, 5) so we
can see whether the multi-doc gap closes at larger k.
