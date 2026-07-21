cat > CLAUDE.md << 'EOF'
# Project Overview
rag-eval is a RAG evaluation harness. It ingests a fixed corpus of markdown
documents, runs a retrieval + generation pipeline, and scores two things
against a hand-verified question set: retrieval quality (hit-rate@k, MRR)
and answer faithfulness (LLM-as-judge). Its purpose is to compare pipeline
configurations and inform decisions, not to serve queries.

# Data Contract
- corpus/ contains the document set. Documents are fixture data.
- questions.jsonl is hand-verified ground truth. Each line:
  {id, question, answer, relevant_doc_ids, answerable}
- questions.jsonl is authoritative. NEVER edit, regenerate, or "fix" it.
  If a question looks wrong, report it and stop.

# Tech Stack
- Python 3.11, venv at .venv
- Embeddings: sentence-transformers (local)
- Retrieval: numpy cosine similarity, no external vector DB
- Generation + judging: local model via transformers (Qwen/Qwen2.5-1.5B-Instruct),
  run on-device, no API key
- Testing: pytest

# Version Control
- Before ANY change, create and checkout a feature branch
- Branch naming: feature/ai-[brief-description]
- NEVER commit directly to main
- Commit when a step is complete, with a detailed message explaining what and why
- Read git log before committing to match existing style

# Planning Requirements
For any feature touching more than 2 files or introducing a new subsystem:
1. Think hard and write a plan to [feature]-plan.md first
2. Wait for approval before implementing
3. Reference the plan throughout implementation

# Quality Requirements - every time
1. Write tests for all new code
2. Run pytest before committing; fix failures before considering work done
3. Never commit code that fails tests or does not run
4. Report which tests passed and failed in your summary

# Code Design
- Small, focused files, one responsibility each
- Split anything past ~200 lines
- New features go in new files, do not pile onto existing ones
- Name files so their purpose is obvious

# Never Do
- Never hardcode API keys or secrets
- Never modify questions.jsonl or corpus/
- Never use LLM-generated content as ground truth
- Never commit directly to main
EOF