from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


def load_corpus(path: str | Path) -> list[Document]:
    """Load every markdown document under `path`.

    Fails loudly rather than returning a partial set. An eval that runs on a
    partial corpus reports scores that look valid but aren't: questions whose
    relevant docs happen to be missing will silently register as retrieval
    failures regardless of the retriever's actual behavior.
    """
    corpus_dir = Path(path)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"no markdown files found in {corpus_dir}")

    documents = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"corpus document is empty: {md_file}")
        documents.append(Document(doc_id=md_file.stem, text=text))
    return documents
