from pathlib import Path

import pytest

from rag_eval.corpus import Document, load_corpus

CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def test_loads_all_18_docs():
    docs = load_corpus(CORPUS_PATH)
    assert len(docs) == 18


def test_doc_ids_match_filenames():
    docs = load_corpus(CORPUS_PATH)
    assert {d.doc_id for d in docs} == {f"doc_{i:02d}" for i in range(1, 19)}


def test_all_docs_have_content():
    docs = load_corpus(CORPUS_PATH)
    for d in docs:
        assert d.text.strip(), f"{d.doc_id} is empty"


def test_text_preserved_verbatim():
    docs = load_corpus(CORPUS_PATH)
    doc_02 = next(d for d in docs if d.doc_id == "doc_02")
    on_disk = (CORPUS_PATH / "doc_02.md").read_text(encoding="utf-8")
    assert doc_02.text == on_disk


def test_document_is_frozen():
    doc = Document(doc_id="x", text="y")
    with pytest.raises(Exception):
        doc.doc_id = "z"  # type: ignore[misc]


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus directory not found"):
        load_corpus(tmp_path / "nonexistent")


def test_directory_without_markdown_raises(tmp_path):
    (tmp_path / "notes.txt").write_text("not markdown")
    with pytest.raises(ValueError, match="no markdown files"):
        load_corpus(tmp_path)


def test_empty_file_raises(tmp_path):
    (tmp_path / "doc_bad.md").write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_corpus(tmp_path)


def test_whitespace_only_file_raises(tmp_path):
    (tmp_path / "doc_bad.md").write_text("\n\n  \t\n")
    with pytest.raises(ValueError, match="empty"):
        load_corpus(tmp_path)


def test_load_order_is_deterministic():
    ids_first = [d.doc_id for d in load_corpus(CORPUS_PATH)]
    ids_second = [d.doc_id for d in load_corpus(CORPUS_PATH)]
    assert ids_first == ids_second
