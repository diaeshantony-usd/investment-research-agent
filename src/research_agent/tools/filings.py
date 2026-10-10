"""
tools/filings.py
=================
SEC EDGAR + local vector search over a company's most recent 10-K/10-Q.

Pipeline:

1. Resolve ticker -> CIK via SEC's ``company_tickers.json``.
2. Find the most recent filing of the requested form type via
   ``data.sec.gov/submissions``.
3. Download the filing document and strip it to plain text.
4. Split the text into overlapping chunks (the "local vector store").
5. Embed the chunks with a local Ollama embedding model
   (``nomic-embed-text``) and rank them against the query by cosine
   similarity.

Fallback: if embeddings are unavailable (Ollama not running, model not
pulled, any error), the same chunks are ranked by a simple keyword-overlap
score instead - the "Fallback: Keyword search" behaviour in the design doc.
The fetched-and-chunked filing itself is cached (see ``tools/cache.py``), so
a repeat search against the same filing doesn't re-download or re-chunk it.

SEC's fair-access policy requires a descriptive ``User-Agent`` on every
request; set ``EDGAR_USER_AGENT`` in ``.env`` to your own contact details.
"""

from __future__ import annotations

import json
import math
import os
import re
from html.parser import HTMLParser
from typing import Any

import requests

from research_agent.agent_framework import tool
from research_agent.config import DEFAULT_API_KEY, DEFAULT_OLLAMA_HOST, logger
from research_agent.tools.cache import ONE_DAY, cached

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{document}"

_DEFAULT_USER_AGENT = "AAI-520-Investment-Research-Agent research@example.com"
_USER_AGENT = os.getenv("EDGAR_USER_AGENT", _DEFAULT_USER_AGENT)
_EMBEDDING_MODEL = os.getenv("EDGAR_EMBEDDING_MODEL", "nomic-embed-text")

# Keep embedding time bounded: a full 10-K can run 100+ pages, but the
# earliest ~180K characters typically cover Items 1-7 (Business, Risk
# Factors, Legal Proceedings and MD&A) - the sections a research brief
# actually cites.
_MAX_FILING_CHARS = 180_000
_CHUNK_SIZE = 1_200
_CHUNK_OVERLAP = 150
_MAX_CHUNKS = 120

_VALID_FORM_TYPES = frozenset({"10-K", "10-Q"})


def _headers() -> dict[str, str]:
    return {"User-Agent": _USER_AGENT}


class _TextExtractor(HTMLParser):
    """Strips an HTML/inline-XBRL filing down to its human-readable text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs  # unused: only the tag name matters for skipping <script>/<style>
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        joined = " ".join(self._parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\s*\n\s*", "\n", joined)
        return joined.strip()


def _strip_html(html_text: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html_text)
    return extractor.text()


@cached("edgar_cik_lookup", ttl_seconds=7 * ONE_DAY)
def _ticker_to_cik_map() -> dict[str, str]:
    """Download and cache SEC's full ticker -> zero-padded CIK directory."""
    logger.info("EDGAR: downloading company_tickers.json")
    response = requests.get(_COMPANY_TICKERS_URL, headers=_headers(), timeout=20)
    response.raise_for_status()
    rows = response.json().values()
    return {row["ticker"].upper(): f"{row['cik_str']:010d}" for row in rows}


def _get_cik(ticker: str) -> str:
    """Resolve a ticker to its 10-digit zero-padded CIK.

    Raises:
        ValueError: If the ticker isn't in SEC's directory, or the
            directory can't be downloaded.
    """
    try:
        mapping = _ticker_to_cik_map()
    except requests.RequestException as exc:
        raise ValueError(f"Could not download the SEC ticker directory: {exc}") from exc

    cik = mapping.get(ticker)
    if not cik:
        raise ValueError(f"'{ticker}' is not in SEC's ticker directory")
    return cik


def _get_recent_filing(cik: str, form_type: str) -> dict[str, Any]:
    """Return metadata for the most recent filing of ``form_type`` for ``cik``.

    Raises:
        ValueError: If the request fails or no filing of that type exists.
    """
    url = _SUBMISSIONS_URL.format(cik=cik)
    try:
        response = requests.get(url, headers=_headers(), timeout=20)
        response.raise_for_status()
        recent = response.json()["filings"]["recent"]
    except (requests.RequestException, KeyError) as exc:
        raise ValueError(f"Could not fetch SEC filing history for CIK {cik}: {exc}") from exc

    for i, form in enumerate(recent["form"]):
        if form == form_type:
            return {
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_document": recent["primaryDocument"][i],
                "form": form,
            }

    raise ValueError(f"No {form_type} filing found for CIK {cik}")


def _fetch_filing_text(cik: str, filing: dict[str, Any]) -> str:
    """Download a filing document and return its stripped, truncated text.

    Raises:
        ValueError: If the document can't be downloaded or is empty.
    """
    accession_nodash = filing["accession_number"].replace("-", "")
    url = _ARCHIVE_URL.format(
        cik_int=int(cik), accession_nodash=accession_nodash, document=filing["primary_document"]
    )
    logger.info("EDGAR: downloading filing document %s", url)
    try:
        response = requests.get(url, headers=_headers(), timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Could not download filing document: {exc}") from exc

    text = _strip_html(response.text)
    if not text:
        raise ValueError(f"Filing document at {url} had no extractable text")
    return text[:_MAX_FILING_CHARS]


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping, size-bounded chunks, capped at ``_MAX_CHUNKS``."""
    chunks = []
    start = 0
    length = len(text)
    while start < length and len(chunks) < _MAX_CHUNKS:
        end = min(start + _CHUNK_SIZE, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - _CHUNK_OVERLAP if end < length else end
    return chunks


@cached("edgar_filing_chunks", ttl_seconds=ONE_DAY)
def _get_filing_chunks(ticker: str, form_type: str) -> dict[str, Any]:
    """Fetch, strip and chunk a ticker's most recent filing of ``form_type``.

    Raises:
        ValueError: If the ticker, filing or document can't be resolved.
    """
    cik = _get_cik(ticker)
    filing = _get_recent_filing(cik, form_type)
    text = _fetch_filing_text(cik, filing)
    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError(f"No usable text chunks extracted from {ticker}'s {form_type}")
    return {"filing": filing, "chunks": chunks}


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the local Ollama embedding model.

    Raises:
        ValueError: If Ollama (or the embedding model) is unavailable.
    """
    try:
        import ollama

        client = ollama.Client(
            host=DEFAULT_OLLAMA_HOST,
            headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"} if DEFAULT_API_KEY else {},
        )
        response = client.embed(model=_EMBEDDING_MODEL, input=texts)
    except Exception as exc:
        raise ValueError(f"Embedding via Ollama model '{_EMBEDDING_MODEL}' failed: {exc}") from exc

    embeddings = response.embeddings
    if not embeddings:
        raise ValueError(f"Ollama returned no embeddings from model '{_EMBEDDING_MODEL}'")
    return [list(vec) for vec in embeddings]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@cached("edgar_chunk_embeddings", ttl_seconds=ONE_DAY)
def _get_chunk_embeddings(
    ticker: str,  # noqa: ARG001 - unused in the body, but part of the @cached cache key
    form_type: str,  # noqa: ARG001 - ditto
    chunks: list[str],
) -> list[list[float]]:
    """Embed (and cache) a filing's chunks once, so repeat queries don't re-embed them.

    Raises:
        ValueError: If embeddings can't be computed (propagated from
            :func:`_embed_texts`), so the caller can fall back to keyword search.
    """
    return _embed_texts(chunks)


def _vector_rank(
    ticker: str, form_type: str, chunks: list[str], query: str, top_k: int
) -> list[tuple[str, float]]:
    """Rank chunks against ``query`` by embedding cosine similarity.

    Raises:
        ValueError: If embeddings can't be computed, so the caller can fall
            back to keyword search.
    """
    chunk_vectors = _get_chunk_embeddings(ticker, form_type, chunks)
    query_vector = _embed_texts([query])[0]
    scored = [
        (chunk, _cosine_similarity(query_vector, vec))
        for chunk, vec in zip(chunks, chunk_vectors, strict=True)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def _keyword_rank(chunks: list[str], query: str, top_k: int) -> list[tuple[str, float]]:
    """Fallback: rank chunks by how many query terms they contain."""
    terms = [t for t in re.findall(r"[A-Za-z']+", query.lower()) if len(t) > 2]
    if not terms:
        return [(chunk, 0.0) for chunk in chunks[:top_k]]

    scored = [(chunk, float(sum(chunk.lower().count(term) for term in terms))) for chunk in chunks]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = [pair for pair in scored if pair[1] > 0][:top_k]
    return top or scored[:top_k]


@tool(
    name="search_sec_filings",
    description=(
        "Searches a company's most recent 10-K or 10-Q filing from SEC EDGAR for passages "
        "relevant to a natural-language query (e.g. 'supply chain risk', 'segment revenue "
        "growth'), using local embedding-based vector search over the filing text."
    ),
)
def search_sec_filings(ticker: str, query: str, form_type: str = "10-K", top_k: int = 5) -> str:
    """Search a ticker's most recent SEC filing for passages relevant to ``query``.

    Args:
        ticker: Stock symbol (e.g. 'NVDA').
        query: What to look for, in plain language (e.g. 'export controls').
        form_type: '10-K' (annual) or '10-Q' (quarterly). Defaults to '10-K'.
        top_k: Maximum number of passages to return.

    Returns:
        JSON with the filing's form/date, which search method actually ran
        (``vector`` or ``keyword``), and a ``passages`` list of
        ``{text, score}`` ranked most-relevant first; an ``error`` field if
        the ticker/filing/document couldn't be resolved at all.
    """
    clean_ticker = (ticker or "").strip().upper()
    requested_form = form_type.strip().upper() if form_type else ""
    clean_form = requested_form if requested_form in _VALID_FORM_TYPES else "10-K"
    logger.info(
        "Executing search_sec_filings for %s (form=%s, query=%r)", clean_ticker, clean_form, query
    )

    try:
        data = _get_filing_chunks(clean_ticker, clean_form)
    except ValueError as exc:
        logger.warning("search_sec_filings failed for %s: %s", clean_ticker, exc)
        return json.dumps({"ticker": clean_ticker, "form_type": clean_form, "error": str(exc)})

    chunks = data["chunks"]
    method = "vector"
    try:
        ranked = _vector_rank(clean_ticker, clean_form, chunks, query, top_k)
    except ValueError as exc:
        logger.warning("Vector search unavailable (%s); falling back to keyword search", exc)
        method = "keyword"
        ranked = _keyword_rank(chunks, query, top_k)

    passages = [{"text": text, "score": round(score, 4)} for text, score in ranked]
    return json.dumps(
        {
            "ticker": clean_ticker,
            "form_type": data["filing"]["form"],
            "filing_date": data["filing"]["filing_date"],
            "search_method": method,
            "query": query,
            "passages": passages,
        },
        indent=2,
    )
