"""
test_tools_filings.py
======================
Offline tests for the SEC EDGAR filing search tool. ``requests.get`` and the local
Ollama embedding client are replaced with fakes, so these tests check CIK resolution,
text extraction/chunking, and the vector/keyword ranking fallback without network access.
"""

from __future__ import annotations

import json

import ollama
import pytest

from research_agent.tools import cache, filings


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


class FakeResponse:
    def __init__(self, payload=None, text: str = "", status: int = 200) -> None:
        self._payload = payload
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise filings.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class TestHeaders:
    def test_includes_user_agent(self):
        assert "User-Agent" in filings._headers()


class TestStripHtml:
    def test_removes_tags_and_script_style_content(self):
        html = "<html><body><script>var x=1;</script><p>Hello  World</p></body></html>"
        assert filings._strip_html(html) == "Hello World"

    def test_collapses_whitespace(self):
        html = "<p>Line one</p>\n\n<p>Line   two</p>"
        text = filings._strip_html(html)
        assert "Line one" in text
        assert "Line   two" not in text  # collapsed to single spaces


class TestTickerToCikMap:
    def test_builds_upper_cased_padded_map(self, monkeypatch):
        payload = {"0": {"ticker": "nvda", "cik_str": 1045810}}
        monkeypatch.setattr(filings.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
        mapping = filings._ticker_to_cik_map()
        assert mapping["NVDA"] == "0001045810"


class TestGetCik:
    def test_resolves_known_ticker(self, monkeypatch):
        monkeypatch.setattr(filings, "_ticker_to_cik_map", lambda: {"NVDA": "0001045810"})
        assert filings._get_cik("NVDA") == "0001045810"

    def test_raises_for_unknown_ticker(self, monkeypatch):
        monkeypatch.setattr(filings, "_ticker_to_cik_map", lambda: {"NVDA": "0001045810"})
        with pytest.raises(ValueError, match="not in SEC"):
            filings._get_cik("ZZZZ")

    def test_raises_when_directory_download_fails(self, monkeypatch):
        def boom():
            raise filings.requests.RequestException("down")

        monkeypatch.setattr(filings, "_ticker_to_cik_map", boom)
        with pytest.raises(ValueError, match="Could not download"):
            filings._get_cik("NVDA")


class TestGetRecentFiling:
    def test_finds_most_recent_matching_form(self, monkeypatch):
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-K", "8-K"],
                    "accessionNumber": ["0001-1", "0001-2", "0001-3"],
                    "filingDate": ["2026-05-01", "2026-02-01", "2026-01-01"],
                    "reportDate": ["2026-03-31", "2025-12-31", "2026-01-01"],
                    "primaryDocument": ["q.htm", "k.htm", "c.htm"],
                }
            }
        }
        monkeypatch.setattr(filings.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
        filing = filings._get_recent_filing("0001045810", "10-K")
        assert filing["accession_number"] == "0001-2"
        assert filing["primary_document"] == "k.htm"

    def test_raises_when_form_type_not_found(self, monkeypatch):
        payload = {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "accessionNumber": ["x"],
                    "filingDate": ["x"],
                    "reportDate": ["x"],
                    "primaryDocument": ["x"],
                }
            }
        }
        monkeypatch.setattr(filings.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
        with pytest.raises(ValueError, match="No 10-K filing found"):
            filings._get_recent_filing("0001045810", "10-K")

    def test_raises_on_request_failure(self, monkeypatch):
        def boom(*a, **k):
            raise filings.requests.RequestException("down")

        monkeypatch.setattr(filings.requests, "get", boom)
        with pytest.raises(ValueError, match="Could not fetch SEC filing history"):
            filings._get_recent_filing("0001045810", "10-K")


class TestFetchFilingText:
    def test_downloads_and_strips_html(self, monkeypatch):
        monkeypatch.setattr(
            filings.requests, "get", lambda *a, **k: FakeResponse(text="<p>Risk factors</p>")
        )
        filing = {"accession_number": "0001-23-000456", "primary_document": "doc.htm"}
        text = filings._fetch_filing_text("0001045810", filing)
        assert text == "Risk factors"

    def test_raises_when_text_is_empty(self, monkeypatch):
        monkeypatch.setattr(
            filings.requests, "get", lambda *a, **k: FakeResponse(text="<script>x</script>")
        )
        filing = {"accession_number": "0001-23-000456", "primary_document": "doc.htm"}
        with pytest.raises(ValueError, match="no extractable text"):
            filings._fetch_filing_text("0001045810", filing)

    def test_raises_on_request_failure(self, monkeypatch):
        def boom(*a, **k):
            raise filings.requests.RequestException("down")

        monkeypatch.setattr(filings.requests, "get", boom)
        filing = {"accession_number": "0001-23-000456", "primary_document": "doc.htm"}
        with pytest.raises(ValueError, match="Could not download filing document"):
            filings._fetch_filing_text("0001045810", filing)


class TestChunkText:
    def test_splits_into_overlapping_chunks(self):
        text = "x" * 3000
        chunks = filings._chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= filings._CHUNK_SIZE for c in chunks)

    def test_short_text_is_one_chunk(self):
        assert filings._chunk_text("hello world") == ["hello world"]

    def test_empty_text_is_no_chunks(self):
        assert filings._chunk_text("") == []

    def test_caps_at_max_chunks(self):
        text = "x" * (filings._CHUNK_SIZE * (filings._MAX_CHUNKS + 10))
        chunks = filings._chunk_text(text)
        assert len(chunks) == filings._MAX_CHUNKS


class TestGetFilingChunks:
    def test_raises_when_no_chunks_extracted(self, monkeypatch):
        monkeypatch.setattr(filings, "_get_cik", lambda ticker: "0001045810")
        monkeypatch.setattr(
            filings,
            "_get_recent_filing",
            lambda cik, form: {"accession_number": "a", "primary_document": "d"},
        )
        monkeypatch.setattr(filings, "_fetch_filing_text", lambda cik, filing: "")
        with pytest.raises(ValueError, match="No usable text chunks"):
            filings._get_filing_chunks("NVDA", "10-K")

    def test_returns_filing_and_chunks(self, monkeypatch):
        monkeypatch.setattr(filings, "_get_cik", lambda ticker: "0001045810")
        fake_filing = {
            "accession_number": "a",
            "primary_document": "d",
            "form": "10-K",
            "filing_date": "2026-01-01",
        }
        monkeypatch.setattr(filings, "_get_recent_filing", lambda cik, form: fake_filing)
        monkeypatch.setattr(filings, "_fetch_filing_text", lambda cik, filing: "some filing text")
        data = filings._get_filing_chunks("NVDA", "10-K")
        assert data["filing"] == fake_filing
        assert data["chunks"] == ["some filing text"]


class FakeEmbedResponse:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings


class FakeOllamaClient:
    def __init__(self, embeddings=None, error: Exception | None = None) -> None:
        self._embeddings = embeddings
        self._error = error

    def embed(self, model, input):
        if self._error:
            raise self._error
        return FakeEmbedResponse(self._embeddings)


class TestEmbedTexts:
    def test_returns_embeddings_as_lists(self, monkeypatch):
        monkeypatch.setattr(
            ollama, "Client", lambda host, headers: FakeOllamaClient(embeddings=[[0.1, 0.2]])
        )
        result = filings._embed_texts(["hello"])
        assert result == [[0.1, 0.2]]

    def test_raises_when_ollama_errors(self, monkeypatch):
        monkeypatch.setattr(
            ollama, "Client", lambda host, headers: FakeOllamaClient(error=RuntimeError("down"))
        )
        with pytest.raises(ValueError, match="Embedding via Ollama"):
            filings._embed_texts(["hello"])

    def test_raises_when_no_embeddings_returned(self, monkeypatch):
        monkeypatch.setattr(ollama, "Client", lambda host, headers: FakeOllamaClient(embeddings=[]))
        with pytest.raises(ValueError, match="no embeddings"):
            filings._embed_texts(["hello"])


class TestCosineSimilarity:
    def test_identical_vectors_are_similarity_one(self):
        assert filings._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_similarity_zero(self):
        assert filings._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert filings._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestVectorRank:
    def test_ranks_chunks_by_similarity(self, monkeypatch):
        monkeypatch.setattr(
            filings, "_get_chunk_embeddings", lambda ticker, form, chunks: [[1.0, 0.0], [0.0, 1.0]]
        )
        monkeypatch.setattr(filings, "_embed_texts", lambda texts: [[1.0, 0.0]])
        ranked = filings._vector_rank("NVDA", "10-K", ["a", "b"], "query", top_k=2)
        assert ranked[0][0] == "a"


class TestKeywordRank:
    def test_ranks_by_term_frequency(self):
        chunks = ["risk factors are significant", "revenue grew substantially", "no mention here"]
        ranked = filings._keyword_rank(chunks, "revenue growth", top_k=2)
        assert ranked[0][0] == "revenue grew substantially"

    def test_no_terms_returns_first_chunks_unscored(self):
        ranked = filings._keyword_rank(["a", "b", "c"], "a an", top_k=2)
        assert ranked == [("a", 0.0), ("b", 0.0)]

    def test_falls_back_to_top_chunks_when_nothing_matches(self):
        ranked = filings._keyword_rank(["alpha", "beta"], "zzzznomatch", top_k=2)
        assert len(ranked) == 2


class TestSearchSecFilings:
    def test_vector_search_success(self, monkeypatch):
        fake_filing = {"form": "10-K", "filing_date": "2026-01-01"}
        monkeypatch.setattr(
            filings,
            "_get_filing_chunks",
            lambda ticker, form: {"filing": fake_filing, "chunks": ["a", "b"]},
        )
        monkeypatch.setattr(
            filings, "_vector_rank", lambda ticker, form, chunks, query, top_k: [("a", 0.9)]
        )
        data = json.loads(filings.search_sec_filings.func("NVDA", "risk"))
        assert data["search_method"] == "vector"
        assert data["passages"][0]["text"] == "a"

    def test_falls_back_to_keyword_search(self, monkeypatch):
        fake_filing = {"form": "10-K", "filing_date": "2026-01-01"}
        monkeypatch.setattr(
            filings,
            "_get_filing_chunks",
            lambda ticker, form: {"filing": fake_filing, "chunks": ["a", "b"]},
        )

        def vector_fail(ticker, form, chunks, query, top_k):
            raise ValueError("ollama unavailable")

        monkeypatch.setattr(filings, "_vector_rank", vector_fail)
        monkeypatch.setattr(filings, "_keyword_rank", lambda chunks, query, top_k: [("b", 1.0)])
        data = json.loads(filings.search_sec_filings.func("NVDA", "risk"))
        assert data["search_method"] == "keyword"

    def test_filing_resolution_failure_returns_error(self, monkeypatch):
        def fail(ticker, form):
            raise ValueError("not in SEC directory")

        monkeypatch.setattr(filings, "_get_filing_chunks", fail)
        data = json.loads(filings.search_sec_filings.func("ZZZZ", "risk"))
        assert "error" in data

    def test_invalid_form_type_defaults_to_10k(self, monkeypatch):
        fake_filing = {"form": "10-K", "filing_date": "2026-01-01"}
        monkeypatch.setattr(
            filings,
            "_get_filing_chunks",
            lambda ticker, form: {"filing": fake_filing, "chunks": ["a"]},
        )
        monkeypatch.setattr(
            filings, "_vector_rank", lambda ticker, form, chunks, query, top_k: [("a", 1.0)]
        )
        data = json.loads(filings.search_sec_filings.func("NVDA", "risk", form_type="20-F"))
        assert data["form_type"] == "10-K"

    def test_non_numeric_top_k_falls_back_to_default(self, monkeypatch):
        fake_filing = {"form": "10-K", "filing_date": "2026-01-01"}
        captured = {}

        def fake_vector_rank(ticker, form, chunks, query, top_k):
            captured["top_k"] = top_k
            return [("a", 1.0)]

        monkeypatch.setattr(
            filings,
            "_get_filing_chunks",
            lambda ticker, form: {"filing": fake_filing, "chunks": ["a"]},
        )
        monkeypatch.setattr(filings, "_vector_rank", fake_vector_rank)
        filings.search_sec_filings.func("NVDA", "risk", top_k="not a number")
        assert captured["top_k"] == 5
