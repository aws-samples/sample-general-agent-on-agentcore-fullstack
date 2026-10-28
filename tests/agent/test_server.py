"""The Runtime contract's gate: the verified caller, the payload it may name, the chat it answers."""

import asyncio
from unittest import mock

import server
from identity import caller
from runtime import sessions

from support import _bind


def test_a_web_users_account_comes_from_the_verified_token_and_nothing_else(_signer):
    web = _signer(client_id="web-client", username="carol@example.com", sub="s-carol")
    assert caller.identify([web]) == ("web:s-carol", "carol@example.com")
    assert caller.identify([_signer()]) == ("lark:ou_alice", "")     # Lark's comes from Lark
    payload = {"actorAccount": "mallory@example.com"}
    req = mock.Mock()
    req.headers.getall.return_value = [web]
    assert server._bind_caller(req, payload) is None
    assert payload["actorAccount"] == "carol@example.com"


def test_a_payload_may_only_agree_with_the_verified_caller(_signer):
    alice = sessions._session_id_for("lark:ou_alice")
    payload = {"action": "chat", "actorId": "lark:ou_alice", "userId": "user_abc",
               "memorySessionId": alice}
    assert _bind([("Authorization", _signer())], payload) is None
    # The router's userId is an internal key, never an actor, so it is not compared.
    assert payload["actorId"] == "lark:ou_alice"
    # Absent, the actor is filled in from the token — the A2A adapter names nobody.
    payload = {"action": "chat"}
    assert _bind([("Authorization", _signer())], payload) is None
    assert payload["actorId"] == "lark:ou_alice"
    for payload, status in (({"action": "chat", "actorId": "lark:ou_bob"}, 403),
                            ({"action": "history_stats"}, None)):
        r = _bind([("Authorization", _signer())], payload)
        assert (r.status if r else None) == status
    assert _bind([], {"action": "chat"}).status == 401


def test_a_thread_is_honoured_only_under_the_callers_own_prefix(_signer):
    """The checkpoint is keyed by thread id alone and the default id is computable by
    anyone, so naming another user's thread under your own identity has to fail too."""
    bob = sessions._session_id_for("lark:ou_bob")
    alice = sessions._session_id_for("lark:ou_alice")
    for thread in (bob, bob + "-0123456789ab", "sess-" + "0" * 32):
        r = _bind([("Authorization", _signer())], {"action": "clear_history",
                                                   "memorySessionId": thread})
        assert r is not None and r.status == 403, thread
    assert sessions.thread_belongs("lark:ou_alice", alice)
    assert sessions.thread_belongs("lark:ou_alice", alice + "-0123456789ab")
    assert not sessions.thread_belongs("lark:ou_alice", alice + "-xyz")
    assert not sessions.thread_belongs("lark:ou_alice", alice + "-0123456789abc")


def test_a_turn_is_answered_only_into_a_chat_the_caller_is_in():
    """The bot's token reaches every chat the bot is in, so a verified caller naming some
    other chat — or another chat's message to pull a picture from — is refused."""
    member = lambda chat_id, who: (chat_id, who) == ("oc_mine", "ou_alice")
    chat_of = {"om_mine": "oc_mine", "om_theirs": "oc_theirs"}.get
    with mock.patch.object(server.lark_card, "in_chat", side_effect=member), \
         mock.patch.object(server.lark_card, "chat_of", side_effect=lambda m: chat_of(m, "")):
        ok = lambda chat_id, imgs=(): server._chat_refusal("lark:ou_alice", chat_id, list(imgs))
        assert ok("oc_mine") == "" and ok("ou_alice") == ""
        assert ok("oc_theirs") == "foreign_chat" and ok("ou_bob") == "foreign_chat"
        assert ok("oc_mine", [{"message_id": "om_mine", "key": "k"}]) == ""
        assert ok("oc_mine", [{"message_id": "om_theirs", "key": "k"}]) == "foreign_message"
        assert ok("oc_mine", [{"key": "k"}]) == "foreign_message"
        # The message being answered is reacted to with the bot's token: it must be there too.
        msg = lambda mid: server._chat_refusal("lark:ou_alice", "oc_mine", [], mid)
        assert msg("om_mine") == "" and msg("om_theirs") == "foreign_message"


def test_only_the_probes_go_without_a_bearer():
    """Any user can take their own token from the web page and call the Runtime directly,
    so everything but the probes that read nothing of anybody's needs a verified bearer."""
    from multidict import CIMultiDict

    def status(payload):
        request = mock.Mock(headers=CIMultiDict())
        request.json = mock.AsyncMock(return_value=payload)
        return asyncio.run(server.handle_invocations(request)).status
    assert status({"action": "status"}) == 200
    assert status({"action": "chat", "message": "hi"}) == 401
    assert status({"action": "transcript"}) == 401
    assert status({"threadId": "t", "messages": []}) == 401          # AG-UI


def test_a_session_the_page_talks_on_counts_as_served_from_its_first_turn(_signer):
    """/status said 「服务本会话 0 秒」 for a session that had just served a web turn: the
    AG-UI branch returned before the session was recorded."""
    from aiohttp import web
    from multidict import CIMultiDict
    request = mock.Mock(headers=CIMultiDict({
        "Authorization": _signer(), "x-amzn-bedrock-agentcore-runtime-session-id": "sid-page"}))
    request.json = mock.AsyncMock(return_value={"threadId": "t", "messages": []})
    server._session_first_seen.pop("sid-page", None)
    with mock.patch.object(server.agui, "handle", mock.AsyncMock(return_value=web.Response())):
        asyncio.run(server.handle_invocations(request))
    assert "sid-page" in server._session_first_seen
