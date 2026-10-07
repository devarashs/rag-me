"""Command-line entry point: `uv run rag-me <command>`.

Commands:
    check-config  Validate configuration without calling any remote service.
    chunks        Split the knowledge base and print one line per chunk, to
                  review what would be embedded before spending any quota.
    ingest        Embed new and changed chunks and store them in Postgres;
                  remove chunks whose sections were deleted.
    ask           Answer a question from the stored chunks, citing sources.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg
from google.genai import errors as genai_errors

from rag_me.answering import InvalidQuestionError, answer_question, extract_cited_source_numbers
from rag_me.chunking import Chunk, KnowledgeBaseError, load_knowledge_base_chunks
from rag_me.config import ConfigurationError, Settings, load_settings
from rag_me.database import connect
from rag_me.embeddings import EmbeddingError, GeminiEmbedder, create_gemini_client
from rag_me.generation import GeminiGenerator
from rag_me.ingest import IngestReport, run_ingest
from rag_me.store import PostgresChunkStore, RetrievedChunk

DEFAULT_KNOWLEDGE_BASE_DIR = Path("data")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the chosen command, and return its exit code."""
    # Answers contain characters such as en dashes. When output is piped on
    # Windows, Python would otherwise encode with the ANSI code page and crash.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(prog="rag-me", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check-config", help="validate configuration from the environment and .env")

    chunks_parser = commands.add_parser(
        "chunks", help="list the chunks the knowledge base splits into"
    )
    _add_data_dir_argument(chunks_parser)

    ingest_parser = commands.add_parser(
        "ingest", help="embed new and changed chunks and sync them to the database"
    )
    _add_data_dir_argument(ingest_parser)
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change without embedding or writing anything",
    )

    ask_parser = commands.add_parser("ask", help="answer a question about Arash")
    ask_parser.add_argument("question", help="the question, in quotes")
    ask_parser.add_argument(
        "--verbose",
        action="store_true",
        help="also list every retrieved section with its similarity score",
    )

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "check-config":
            load_settings()
            print("Configuration OK.")
        elif arguments.command == "chunks":
            print(format_chunk_report(load_knowledge_base_chunks(arguments.data_dir)))
        elif arguments.command == "ingest":
            print(
                format_ingest_report(ingest_knowledge_base(arguments.data_dir, arguments.dry_run))
            )
        elif arguments.command == "ask":
            ask(arguments.question, load_settings(), verbose=arguments.verbose)
    except (ConfigurationError, KnowledgeBaseError, EmbeddingError, InvalidQuestionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except genai_errors.APIError as error:
        print(f"error: Gemini request failed ({error.code}): {error.message}", file=sys.stderr)
        return 1
    except psycopg.errors.UndefinedTable:
        print("error: table missing; run `uv run alembic upgrade head` first", file=sys.stderr)
        return 1
    except psycopg.Error as error:
        print(f"error: database: {error}", file=sys.stderr)
        return 1
    return 0


def ingest_knowledge_base(knowledge_base_dir: Path, dry_run: bool) -> IngestReport:
    """Wire real settings, Gemini and Postgres into `run_ingest`.

    The knowledge base is loaded first, so a malformed file fails before any
    connection is opened or quota is spent.
    """
    chunks = load_knowledge_base_chunks(knowledge_base_dir)
    settings = load_settings()
    embedder = GeminiEmbedder(
        client=create_gemini_client(settings.google_ai_api_key.get_secret_value()),
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    with connect(settings.database_url.get_secret_value()) as connection:
        return run_ingest(chunks, PostgresChunkStore(connection), embedder, dry_run=dry_run)


def ask(question: str, settings: Settings, *, verbose: bool) -> None:
    """Answer `question`, printing the answer as it streams, then its sources."""
    client = create_gemini_client(settings.google_ai_api_key.get_secret_value())
    with connect(settings.database_url.get_secret_value()) as connection:
        answer = answer_question(
            question,
            embedder=GeminiEmbedder(
                client, settings.embedding_model, settings.embedding_batch_size
            ),
            searcher=PostgresChunkStore(connection),
            generator=GeminiGenerator(client, settings.generation_model),
            top_k=settings.retrieval_top_k,
            min_similarity=settings.min_similarity,
            contact_email=settings.contact_email,
        )
        answer_text = ""
        for text_piece in answer.text_pieces:
            print(text_piece, end="", flush=True)
            answer_text += text_piece
    print()

    cited_numbers = extract_cited_source_numbers(answer_text, len(answer.sources))
    if cited_numbers:
        print("\nSources:")
        for number in cited_numbers:
            print(f"  [{number}] {_format_source(answer.sources[number - 1])}")
    if verbose:
        if answer.is_grounded:
            print("\nRetrieved (cosine similarity):")
            for number, source in enumerate(answer.sources, start=1):
                print(f"  [{number}] {_format_source(source)}")
        else:
            print(
                f"\nNo section reached the relevance threshold ({settings.min_similarity}); "
                "the model was not called."
            )


def _format_source(source: RetrievedChunk) -> str:
    return f"{source.similarity:.3f}  {source.chunk.chunk_id}"


def format_chunk_report(chunks: Sequence[Chunk]) -> str:
    """Render one line per chunk (ID and word count) followed by totals.

    Word counts are a rough size check: a chunk far larger than the rest is
    usually two topics that should be split into separate sections.
    """
    if not chunks:
        return "0 chunks."
    word_counts = [len(f"{chunk.title} {chunk.body}".split()) for chunk in chunks]
    id_column_width = max(len(chunk.chunk_id) for chunk in chunks)
    lines = [
        f"{chunk.chunk_id:<{id_column_width}}  {word_count:>5} words"
        for chunk, word_count in zip(chunks, word_counts, strict=True)
    ]
    source_count = len({chunk.source_path for chunk in chunks})
    lines.append("")
    lines.append(
        f"{len(chunks)} chunks from {source_count} files; "
        f"words per chunk: min {min(word_counts)}, max {max(word_counts)}, "
        f"mean {sum(word_counts) // len(word_counts)}"
    )
    return "\n".join(lines)


def format_ingest_report(report: IngestReport) -> str:
    """List each added, changed and deleted chunk, then a one-line summary."""
    plan = report.plan
    lines = [f"+ {planned.chunk.chunk_id}" for planned in plan.new]
    lines += [f"~ {planned.chunk.chunk_id}" for planned in plan.changed]
    lines += [f"- {chunk_id}" for chunk_id in plan.deleted_ids]
    if lines:
        lines.append("")
    verb = "Synced" if report.written else "Dry run, nothing written"
    lines.append(
        f"{verb}: {len(plan.new)} new, {len(plan.changed)} changed, "
        f"{len(plan.deleted_ids)} deleted, {len(plan.unchanged)} unchanged."
    )
    return "\n".join(lines)


def _add_data_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_KNOWLEDGE_BASE_DIR,
        help=f"knowledge base directory (default: {DEFAULT_KNOWLEDGE_BASE_DIR})",
    )


if __name__ == "__main__":
    sys.exit(main())
