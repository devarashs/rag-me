"""Split the Markdown knowledge base into retrieval chunks, one per `##` section.

Why sections rather than fixed-size windows: the knowledge base is written so that
every `##` section is self-contained and sized to fit one chunk (see
`data/README.md`). Splitting on the author's own boundaries keeps each chunk a
complete thought, where a fixed window would cut sentences and mix topics.

Conventions this module enforces:

- The `#` heading is the document title. At most one per file.
- Each `##` section becomes one chunk. Deeper headings (`###`) stay inside it.
- Text between the title and the first `##` is an editor's note and is not
  ingested. A file with no `##` sections at all is an error, so content is never
  dropped silently.
- `README.md` describes the folder and is never ingested.

Headings are found with a CommonMark parser, not a regex, so a `## ` line inside a
fenced code block is not mistaken for a heading and setext headings are handled.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

KNOWLEDGE_BASE_README_NAME = "README.md"

_markdown_parser = MarkdownIt("commonmark")


class KnowledgeBaseError(Exception):
    """Raised when a knowledge base file breaks the conventions above.

    The message names the file and, where relevant, the heading, so the author
    can fix the source rather than the code.
    """


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable section of the knowledge base.

    Attributes:
        chunk_id: Stable identifier, `<source_path>#<heading-slug>`. Unchanged
            when the section's body is edited, so the store can update it in place.
        source_path: Path of the source file relative to the knowledge base
            directory, with forward slashes on every platform.
        document_title: The file's `#` heading, or its file name without the
            extension when it has none.
        section_heading: The `##` heading text, as written.
        body: The section's Markdown below its heading, trimmed.
    """

    chunk_id: str
    source_path: str
    document_title: str
    section_heading: str
    body: str

    @property
    def title(self) -> str:
        """Document title and section heading together, e.g. "Skills - Payments".

        Embedded alongside the body because a body alone often lacks the words a
        question would use: a section about "the economy service" may never say
        "Hypersonic", while its heading does.
        """
        return f"{self.document_title} - {self.section_heading}"


def slugify_heading(heading: str) -> str:
    """Turn a heading into a lowercase, hyphen-separated identifier.

    Letters and digits in any script are kept; every other run of characters
    becomes one hyphen. Returns an empty string when nothing is left.

    Example:
        >>> slugify_heading("Story: idempotent real-money payments (Hypersonic)")
        'story-idempotent-real-money-payments-hypersonic'
    """
    return re.sub(r"[\W_]+", "-", heading.lower()).strip("-")


def split_markdown_into_chunks(markdown_text: str, source_path: str) -> list[Chunk]:
    """Split one Markdown document into chunks, one per `##` section.

    Args:
        markdown_text: The document's full text.
        source_path: Identifier for the document, used in chunk IDs and error
            messages (normally its path relative to the knowledge base).

    Returns:
        The chunks in document order. Never empty.

    Raises:
        KnowledgeBaseError: If the document has more than one `#` heading, a `#`
            heading after its first section, no `##` sections, a section with an
            empty body, a heading that slugs to nothing, or two headings with the
            same slug.
    """
    # Normalise Windows line endings so bodies and hashes are identical whichever
    # OS the file was saved on.
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings = _find_headings(lines)

    title_headings = [heading for heading in headings if heading.level == 1]
    if len(title_headings) > 1:
        raise KnowledgeBaseError(
            f"{source_path}: has {len(title_headings)} '#' headings; "
            "a file has at most one, its title"
        )
    sections = [heading for heading in headings if heading.level == 2]
    if not sections:
        raise KnowledgeBaseError(
            f"{source_path}: has no '##' sections, so nothing in it would be ingested"
        )

    if title_headings and title_headings[0].start_line > sections[0].start_line:
        raise KnowledgeBaseError(
            f"{source_path}: '#' heading '{title_headings[0].text}' appears after a "
            "'##' section; the title must come first"
        )
    document_title = title_headings[0].text if title_headings else Path(source_path).stem

    chunks: list[Chunk] = []
    seen_slugs: dict[str, str] = {}
    for index, section in enumerate(sections):
        # A section runs until the next `##`, or the end of the file. A `#` after
        # the first `##` was rejected above, so only `##` can end a section.
        end_line = sections[index + 1].start_line if index + 1 < len(sections) else len(lines)
        body = "\n".join(lines[section.end_line : end_line]).strip()

        if not body:
            raise KnowledgeBaseError(f"{source_path}: section '{section.text}' has no body")

        slug = slugify_heading(section.text)
        if not slug:
            raise KnowledgeBaseError(
                f"{source_path}: heading '{section.text}' has no letters or digits "
                "to build an ID from"
            )
        if slug in seen_slugs:
            raise KnowledgeBaseError(
                f"{source_path}: headings '{seen_slugs[slug]}' and '{section.text}' "
                "produce the same ID; make them distinct"
            )
        seen_slugs[slug] = section.text

        chunks.append(
            Chunk(
                chunk_id=f"{source_path}#{slug}",
                source_path=source_path,
                document_title=document_title,
                section_heading=section.text,
                body=body,
            )
        )
    return chunks


def load_knowledge_base_chunks(knowledge_base_dir: Path) -> list[Chunk]:
    """Read every Markdown file in a directory tree and split it into chunks.

    Files are processed in sorted path order so output is deterministic. The
    directory's top-level `README.md` is skipped.

    Args:
        knowledge_base_dir: The knowledge base root, normally `data/`.

    Returns:
        All chunks, grouped by file in sorted path order. Never empty.

    Raises:
        KnowledgeBaseError: If the directory does not exist, contains no
            ingestible Markdown files, or any file breaks the conventions. Chunk
            IDs are unique across files because they include the source path.
    """
    if not knowledge_base_dir.is_dir():
        raise KnowledgeBaseError(f"Knowledge base directory not found: {knowledge_base_dir}")

    markdown_paths = sorted(
        path
        for path in knowledge_base_dir.rglob("*.md")
        if path.is_file() and path != knowledge_base_dir / KNOWLEDGE_BASE_README_NAME
    )
    # An empty result must be an error, never an empty list: ingest treats chunks
    # missing from this list as deleted, so an empty list would wipe the store.
    if not markdown_paths:
        raise KnowledgeBaseError(f"No Markdown files to ingest in {knowledge_base_dir}")

    chunks: list[Chunk] = []
    for path in markdown_paths:
        source_path = path.relative_to(knowledge_base_dir).as_posix()
        chunks.extend(split_markdown_into_chunks(path.read_text(encoding="utf-8"), source_path))
    return chunks


@dataclass(frozen=True, slots=True)
class _Heading:
    """A heading found by the parser, with its 0-based line range in the source."""

    level: int
    text: str
    start_line: int
    end_line: int


def _find_headings(lines: list[str]) -> list[_Heading]:
    """Return every heading in the document, in order, with its source lines."""
    tokens = _markdown_parser.parse("\n".join(lines))
    headings: list[_Heading] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        # The parser always emits heading_open, inline, heading_close; the inline
        # token holds the heading's raw text.
        inline_token = tokens[index + 1]
        start_line, end_line = token.map
        headings.append(
            _Heading(
                level=int(token.tag.removeprefix("h")),
                text=inline_token.content.strip(),
                start_line=start_line,
                end_line=end_line,
            )
        )
    return headings
