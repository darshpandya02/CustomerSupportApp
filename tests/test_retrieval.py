from supportbot.retrieval import BM25, get_retriever, tokenize


def test_tokenize_drops_stopwords():
    assert tokenize("How do I restore THE deleted files?") == ["restore", "deleted", "files"]


def test_bm25_prefers_matching_doc():
    bm = BM25([["trash", "bin", "restore"], ["calendar", "event"]])
    s = bm.scores(["restore"])
    assert s[0] > 0 and s[1] == 0


def test_hybrid_rerank_finds_the_2fa_section():
    res = get_retriever().search("my authenticator phone was stolen, how do I log in?", k=5)
    assert any(h.chunk["page"] == "user_2fa.html" for h in res.hits[:3])
    assert res.hits[0].rerank is not None
    assert "retrieval_ms" in res.timings_ms


def test_every_chunk_has_a_source_url():
    for c in get_retriever().chunks:
        assert c["url"].startswith("https://docs.nextcloud.com/server/stable/user_manual/en/")
