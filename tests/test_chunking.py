from pathlib import Path

import pytest

from rag_me.chunking import (
    Chunk,
    KnowledgeBaseError,
    load_knowledge_base_chunks,
    slugify_heading,
    split_markdown_into_chunks,
)

TWO_SECTION_DOCUMENT = """\
# Arash: profile

Editor's note that is not ingested.

## Summary

Arash is a backend engineer.

## Contact

Email: me@example.com
"""


# --- split_markdown_into_chunks: the normal case --------------------------------


def test_each_level_two_section_becomes_one_chunk_in_document_order() -> None:
    chunks = split_markdown_into_chunks(TWO_SECTION_DOCUMENT, "01-profile.md")

    assert [chunk.section_heading for chunk in chunks] == ["Summary", "Contact"]
    assert [chunk.body for chunk in chunks] == [
        "Arash is a backend engineer.",
        "Email: me@example.com",
    ]


def test_chunks_carry_source_title_and_a_stable_id() -> None:
    first_chunk = split_markdown_into_chunks(TWO_SECTION_DOCUMENT, "01-profile.md")[0]

    assert first_chunk == Chunk(
        chunk_id="01-profile.md#summary",
        source_path="01-profile.md",
        document_title="Arash: profile",
        section_heading="Summary",
        body="Arash is a backend engineer.",
    )


def test_text_between_title_and_first_section_is_not_ingested() -> None:
    chunks = split_markdown_into_chunks(TWO_SECTION_DOCUMENT, "01-profile.md")

    assert all("Editor's note" not in f"{chunk.title} {chunk.body}" for chunk in chunks)


def test_title_joins_document_title_and_section_heading() -> None:
    chunk = split_markdown_into_chunks(TWO_SECTION_DOCUMENT, "01-profile.md")[0]

    assert chunk.title == "Arash: profile - Summary"


def test_deeper_headings_stay_inside_their_section() -> None:
    document = "# Title\n\n## Section\n\nIntro.\n\n### Detail\n\nMore.\n"

    chunks = split_markdown_into_chunks(document, "doc.md")

    assert len(chunks) == 1
    assert chunks[0].body == "Intro.\n\n### Detail\n\nMore."


def test_document_without_title_uses_file_name_as_title() -> None:
    chunks = split_markdown_into_chunks("## Section\n\nBody.\n", "notes/04-projects.md")

    assert chunks[0].document_title == "04-projects"


def test_body_keeps_inner_markdown_unchanged() -> None:
    body = "- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n**bold** and `code`"
    document = f"# Title\n\n## Section\n\n{body}\n"

    assert split_markdown_into_chunks(document, "doc.md")[0].body == body


def test_last_section_runs_to_end_of_file_without_trailing_newline() -> None:
    chunks = split_markdown_into_chunks("# T\n\n## A\n\nFirst.\n\n## B\n\nLast line", "doc.md")

    assert chunks[-1].body == "Last line"


# --- split_markdown_into_chunks: parsing edge cases -----------------------------


def test_heading_lookalike_inside_fenced_code_block_does_not_split() -> None:
    document = "# T\n\n## Real section\n\n```markdown\n## Not a heading\n```\n\nAfter.\n"

    chunks = split_markdown_into_chunks(document, "doc.md")

    assert [chunk.section_heading for chunk in chunks] == ["Real section"]
    assert "## Not a heading" in chunks[0].body


def test_setext_level_two_heading_is_a_section() -> None:
    document = "Title\n=====\n\nSection\n-------\n\nBody.\n"

    chunks = split_markdown_into_chunks(document, "doc.md")

    assert chunks[0].document_title == "Title"
    assert chunks[0].section_heading == "Section"
    assert chunks[0].body == "Body."


def test_heading_with_closing_hashes_and_extra_spaces_is_trimmed() -> None:
    chunks = split_markdown_into_chunks("#  Title  #\n\n##   Section   ##\n\nBody.\n", "doc.md")

    assert chunks[0].document_title == "Title"
    assert chunks[0].section_heading == "Section"


def test_windows_and_unix_line_endings_give_identical_chunks() -> None:
    unix_chunks = split_markdown_into_chunks(TWO_SECTION_DOCUMENT, "doc.md")
    windows_chunks = split_markdown_into_chunks(
        TWO_SECTION_DOCUMENT.replace("\n", "\r\n"), "doc.md"
    )

    assert windows_chunks == unix_chunks


def test_unicode_headings_and_bodies_survive() -> None:
    chunks = split_markdown_into_chunks("# Profil\n\n## Café résumé\n\nNaïve — ✓\n", "doc.md")

    assert chunks[0].chunk_id == "doc.md#café-résumé"
    assert chunks[0].body == "Naïve — ✓"


# --- split_markdown_into_chunks: malformed documents fail loudly ----------------


@pytest.mark.parametrize("document", ["", "   \n\n", "# Title only\n\nSome text.\n"])
def test_document_without_sections_is_rejected(document: str) -> None:
    with pytest.raises(KnowledgeBaseError, match=r"doc\.md: has no '##' sections"):
        split_markdown_into_chunks(document, "doc.md")


def test_more_than_one_title_is_rejected() -> None:
    with pytest.raises(KnowledgeBaseError, match="has 2 '#' headings"):
        split_markdown_into_chunks("# One\n\n## A\n\nx\n\n# Two\n\n## B\n\ny\n", "doc.md")


def test_title_after_first_section_is_rejected() -> None:
    with pytest.raises(KnowledgeBaseError, match="appears after a '##' section"):
        split_markdown_into_chunks("## A\n\nx\n\n# Late title\n\ny\n", "doc.md")


@pytest.mark.parametrize(
    "document",
    ["# T\n\n## Empty\n\n## Next\n\nx\n", "# T\n\n## A\n\nx\n\n## Empty at end\n   \n"],
)
def test_section_with_empty_body_is_rejected(document: str) -> None:
    with pytest.raises(KnowledgeBaseError, match="has no body"):
        split_markdown_into_chunks(document, "doc.md")


def test_headings_that_produce_the_same_id_are_rejected() -> None:
    document = "# T\n\n## Contact & links\n\nx\n\n## Contact links\n\ny\n"

    with pytest.raises(KnowledgeBaseError, match="produce the same ID"):
        split_markdown_into_chunks(document, "doc.md")


def test_heading_without_letters_or_digits_is_rejected() -> None:
    with pytest.raises(KnowledgeBaseError, match="no letters or digits"):
        split_markdown_into_chunks("# T\n\n## ???\n\nx\n", "doc.md")


# --- slugify_heading ------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected_slug"),
    [
        ("Summary", "summary"),
        ("Contact and links", "contact-and-links"),
        (
            "Backend engineer (Dec 2023 – Mar 2025)",  # noqa: RUF001 - en dash, as used in data/
            "backend-engineer-dec-2023-mar-2025",
        ),
        ("What is Arash's main tech stack?", "what-is-arash-s-main-tech-stack"),
        ("snake_case and `code`", "snake-case-and-code"),
        ("  --Leading and trailing--  ", "leading-and-trailing"),
        ("!!!", ""),
    ],
)
def test_slugify_heading(heading: str, expected_slug: str) -> None:
    assert slugify_heading(heading) == expected_slug


# --- load_knowledge_base_chunks -------------------------------------------------


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_loads_all_markdown_files_in_sorted_order_and_skips_readme(tmp_path: Path) -> None:
    write_file(tmp_path / "02-second.md", "# Second\n\n## B\n\nb\n")
    write_file(tmp_path / "01-first.md", "# First\n\n## A\n\na\n")
    write_file(tmp_path / "README.md", "# About this folder\n\n## Files\n\nlist\n")
    write_file(tmp_path / "notes.txt", "## Not markdown\n\nignored\n")

    chunks = load_knowledge_base_chunks(tmp_path)

    assert [chunk.chunk_id for chunk in chunks] == ["01-first.md#a", "02-second.md#b"]


def test_markdown_in_subdirectories_is_included_with_posix_paths(tmp_path: Path) -> None:
    write_file(tmp_path / "projects" / "sluice.md", "# Sluice\n\n## Overview\n\nGo tool.\n")
    write_file(tmp_path / "projects" / "README.md", "# Nested readme\n\n## Kept\n\nx\n")

    chunk_ids = [chunk.chunk_id for chunk in load_knowledge_base_chunks(tmp_path)]

    # Only the top-level README describes the knowledge base; a nested one is content.
    assert chunk_ids == ["projects/README.md#kept", "projects/sluice.md#overview"]


def test_same_heading_in_two_files_gives_distinct_ids(tmp_path: Path) -> None:
    write_file(tmp_path / "a.md", "## Overview\n\na\n")
    write_file(tmp_path / "b.md", "## Overview\n\nb\n")

    chunk_ids = [chunk.chunk_id for chunk in load_knowledge_base_chunks(tmp_path)]

    assert chunk_ids == ["a.md#overview", "b.md#overview"]


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseError, match="directory not found"):
        load_knowledge_base_chunks(tmp_path / "missing")


def test_directory_with_only_readme_is_rejected(tmp_path: Path) -> None:
    write_file(tmp_path / "README.md", "# Readme\n\n## Files\n\nx\n")

    with pytest.raises(KnowledgeBaseError, match="No Markdown files to ingest"):
        load_knowledge_base_chunks(tmp_path)


def test_one_malformed_file_fails_the_whole_load_naming_that_file(tmp_path: Path) -> None:
    write_file(tmp_path / "good.md", "## A\n\na\n")
    write_file(tmp_path / "bad.md", "# Only a title\n")

    with pytest.raises(KnowledgeBaseError, match=r"^bad\.md: "):
        load_knowledge_base_chunks(tmp_path)


def test_real_knowledge_base_loads() -> None:
    knowledge_base_dir = Path(__file__).resolve().parents[1] / "data"

    chunks = load_knowledge_base_chunks(knowledge_base_dir)

    assert chunks
    assert all(not chunk.source_path.startswith("README") for chunk in chunks)
