"""Run the retrieval-only eval with three chunkers and dump a comparison table.

Uses the same embedder (MiniLM), same k, same fixture corpus and questions
across all three runs so the only axis that varies is the chunker. Writes:

  runs/chunker-comparison/retrieval-*.json    (one per chunker)
  runs/chunker-comparison/comparison.md       (table + provenance)

The comparison prose (which chunker wins, whether the delta is decisive)
is added by a human after inspecting the numbers. This script only produces
the table.
"""
from __future__ import annotations

import json
from pathlib import Path

from rag_eval.chunker import (
    FixedWindowChunker,
    HeadingChunker,
    WholeDocChunker,
)
from rag_eval.embedder import Embedder
from rag_eval.eval_retrieval import run_retrieval_eval

REPO_ROOT = Path(__file__).parent.parent
CORPUS_PATH = REPO_ROOT / "corpus"
QUESTIONS_PATH = REPO_ROOT / "questions.jsonl"
OUT_DIR = REPO_ROOT / "runs" / "chunker-comparison"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 3


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO_ROOT / ".cache" / "embeddings"

    chunkers = [
        WholeDocChunker(),
        FixedWindowChunker(size=400, overlap=50),
        HeadingChunker(),
    ]

    embedder = Embedder(EMBEDDER_MODEL, cache_dir=cache_dir)

    reports: list[dict] = []
    for chunker in chunkers:
        out_path = run_retrieval_eval(
            corpus_path=CORPUS_PATH,
            questions_path=QUESTIONS_PATH,
            chunker=chunker,
            embedder=embedder,
            k=K,
            runs_dir=OUT_DIR,
        )
        report = json.loads(out_path.read_text())
        report["_out_path"] = out_path.name
        reports.append(report)
        print(f"wrote {out_path.name}")

    _write_comparison(reports)


def _write_comparison(reports: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Chunker comparison (retrieval-only)\n")
    lines.append(f"- embedder: `{EMBEDDER_MODEL}`\n")
    lines.append(f"- k: {K}\n")
    lines.append("- corpus: fixture (`corpus/`, 18 docs)\n")
    lines.append("- questions: fixture (`questions.jsonl`, 16 answerable / 4 unanswerable)\n")
    lines.append("\n## Aggregates (over 16 answerable questions)\n\n")
    lines.append("| Chunker | Params | any_hit@k | recall@k | MRR |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for r in reports:
        cfg = r["config"]
        agg = r["aggregate"]
        params_str = (
            ", ".join(f"{k}={v}" for k, v in cfg["chunker_params"].items())
            if cfg["chunker_params"]
            else "-"
        )
        lines.append(
            f"| {cfg['chunker']} | {params_str} | "
            f"{agg['any_hit_rate']:.3f} | "
            f"{agg['mean_recall']:.3f} | "
            f"{agg['mean_reciprocal_rank']:.3f} |\n"
        )

    lines.append("\n## Per-question hit@k (answerable only)\n\n")
    ans_ids = [
        p["question_id"]
        for p in reports[0]["per_question"]
        if p["answerable"]
    ]
    header = "| Question | " + " | ".join(r["config"]["chunker"] for r in reports) + " |\n"
    sep = "|---|" + "".join("---:|" for _ in reports) + "\n"
    lines.append(header)
    lines.append(sep)
    for qid in ans_ids:
        row = [qid]
        for r in reports:
            p = next(x for x in r["per_question"] if x["question_id"] == qid)
            row.append("hit" if p["any_hit_at_k"] else ".")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Files\n\n")
    for r in reports:
        lines.append(f"- `{r['_out_path']}`\n")

    lines.append(
        "\n## Analysis\n\n"
        "_To be filled in by a human after inspecting the numbers above._\n"
    )

    (OUT_DIR / "comparison.md").write_text("".join(lines))
    print(f"wrote {OUT_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
