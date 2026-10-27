import asyncio

import pytest
from agent_framework import Agent, ChatResponse, Content, Message

from presslens.agents import llm as L


class CountingClient(L.OfflineChatClient):
    """Stands in for a real model: answers every request and counts the round trips."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls += 1
        return ChatResponse(messages=[Message(role="assistant", contents=["press broken"])], model=self.model)


def _agent(client):
    return Agent(client=client, name="Explainer", instructions="Explain the moment.")


def test_azure_endpoint_is_reduced_to_the_v1_base():
    want = "https://x.services.ai.azure.com/openai/v1/"
    for url in ("https://x.services.ai.azure.com/openai/v1/responses",
                "https://x.services.ai.azure.com/openai/v1/chat/completions",
                "https://x.services.ai.azure.com/openai/v1/",
                "https://x.services.ai.azure.com"):
        assert L._azure_base_url(url) == want


def test_request_key_ignores_per_run_ids():
    def msgs(run_id):
        call = Content.from_function_call(call_id="call_1", name="xt_for", arguments='{"team":"RIV"}')
        call.id = f"af-call-{run_id}"
        return [Message(role="user", contents=["q"]), Message(role="assistant", contents=[call])]
    assert L.request_key("m", msgs("a"), {}) == L.request_key("m", msgs("b"), {})
    assert L.request_key("m", msgs("a"), {}) != L.request_key("other", msgs("a"), {})
    assert L.request_key("m", msgs("a"), {}) != L.request_key("m", msgs("a"), {"temperature": 0.5})


def test_repeated_request_is_served_from_disk(tmp_path):
    cache = L.ResponseCache(tmp_path / "c.sqlite")
    mw = L.CacheMiddleware(cache, "fake")
    client = CountingClient(model="fake", middleware=[mw])
    for _ in range(2):
        r = asyncio.run(_agent(client).run("Why did the press break?"))
        assert r.text == "press broken"
    assert client.calls == 1 and (mw.stats.hits, mw.stats.misses) == (1, 1)

    replay = L.OfflineChatClient(model="fake", middleware=[L.CacheMiddleware(L.ResponseCache(tmp_path / "c.sqlite"), "fake")])
    r = asyncio.run(_agent(replay).run("Why did the press break?"))
    assert r.text == "press broken" and not L.is_offline_miss(r)


def test_offline_miss_is_flagged_and_not_cached(tmp_path):
    mw = L.CacheMiddleware(L.ResponseCache(tmp_path / "c.sqlite"), "fake")
    client = L.OfflineChatClient(model="fake", middleware=[mw])
    for _ in range(2):
        r = asyncio.run(_agent(client).run("Never asked before"))
        assert L.is_offline_miss(r) and r.text == ""
    assert mw.stats.offline_misses == 2 and mw.cache.stats() == []


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        L.chat_client("gpt-on-a-toaster", cache_path=tmp_path / "c.sqlite")
