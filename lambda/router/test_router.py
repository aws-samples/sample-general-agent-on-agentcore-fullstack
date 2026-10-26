"""Router unit tests — Lark webhook crypto + event handling.

Run: uv run --with cryptography --with boto3 python -m pytest lambda/router/test_router.py -v

Env is set before importing modules so boto3 clients construct without error;
AWS calls are mocked.
"""

import hashlib
import json
import os
import sys
from unittest import mock

import pytest

os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("LARK_API_DOMAIN", "https://open.larksuite.com")
os.environ.setdefault("LARK_OAUTH_PROVIDER", "test-3lo")
os.environ.setdefault("AGENT_WORKLOAD_NAME", "test-wl")
os.environ.setdefault("IDENTITY_TABLE_NAME", "test-identity")
os.environ.setdefault("LARK_SECRET_ID", "test/lark")
os.environ.setdefault("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-west-2:1:runtime/x")

sys.path.insert(0, os.path.dirname(__file__))

ENCRYPT_KEY = "test-encrypt-key"

# Golden ciphertext: Lark's AES-256-CBC webhook framing (key=sha256(ENCRYPT_KEY),
# IV "0123456789abcdef" prepended, PKCS#7) of the payload below. Precomputed so the
# test never constructs a cipher itself — it exercises the production decrypt path
# against a fixed, real-format payload.
_GOLDEN_EVENT = {"type": "url_verification", "challenge": "abc123"}
_GOLDEN_CIPHERTEXT = (
    "MDEyMzQ1Njc4OWFiY2RlZjTCy6xCTjl4LV6d/+dDtlSdzPmqcTkTl1iyYaHzzRk+H4NOv2+gZhM3z9/EBitHGNigsYuEJZ1EgAioJmTx0ps="
)


@pytest.fixture
def lark_mod():
    import lark
    lark._creds_cache = {"appId": "cli_x", "appSecret": "s",
                         "verificationToken": "v", "encryptKey": ENCRYPT_KEY}
    lark._token_cache = {"token": "", "expires_at": 0.0}
    return lark


def test_decrypt_event_roundtrip(lark_mod):
    assert lark_mod.decrypt_event(_GOLDEN_CIPHERTEXT) == _GOLDEN_EVENT


def test_verify_signature_valid(lark_mod):
    import time
    body = b'{"hello":"world"}'
    ts = str(int(time.time()))
    nonce = "n1"
    sig = hashlib.sha256(f"{ts}{nonce}{ENCRYPT_KEY}".encode() + body).hexdigest()
    headers = {"X-Lark-Request-Timestamp": ts, "X-Lark-Request-Nonce": nonce,
               "X-Lark-Signature": sig}
    assert lark_mod.verify_signature(headers, body) is True


def test_verify_signature_wrong_sig_fails(lark_mod):
    import time
    ts = str(int(time.time()))
    headers = {"X-Lark-Request-Timestamp": ts, "X-Lark-Request-Nonce": "n",
               "X-Lark-Signature": "deadbeef"}
    assert lark_mod.verify_signature(headers, b"body") is False


def test_verify_signature_replay_window(lark_mod):
    old_ts = "1000000000"  # year 2001 — well outside window
    body = b"body"
    sig = hashlib.sha256(f"{old_ts}n{ENCRYPT_KEY}".encode() + body).hexdigest()
    headers = {"X-Lark-Request-Timestamp": old_ts, "X-Lark-Request-Nonce": "n",
               "X-Lark-Signature": sig}
    assert lark_mod.verify_signature(headers, body) is False


def test_verify_signature_fail_closed_without_key(lark_mod):
    lark_mod._creds_cache = {"encryptKey": ""}
    assert lark_mod.verify_signature({"x-lark-signature": "x"}, b"b") is False


def test_send_message_chunks_long_text(lark_mod):
    calls = []

    class FakeResp:
        def __init__(self, d): self._d = d
        def read(self): return json.dumps(self._d).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        calls.append(req.data)
        return FakeResp({"code": 0})

    lark_mod._token_cache = {"token": "t", "expires_at": 9e18}
    with mock.patch.object(lark_mod.urllib.request, "urlopen", fake_urlopen):
        ok = lark_mod.send_message("oc_x", "A" * 45000)
    assert ok is True
    assert len(calls) == 3  # 45000 / 20000 -> 3 chunks


def test_challenge_regex_rejects_xss():
    import index
    assert index._CHALLENGE_RE.match("safe_challenge-1.2") is not None
    assert index._CHALLENGE_RE.match("<script>") is None


# --- IdP registry ---------------------------------------------------------
# A malformed registry must degrade to the single-provider defaults rather than
# leaving the bot with no way to authorize anyone.

def _load_idps_with(raw: str) -> dict:
    import index
    with mock.patch.dict(os.environ, {"IDP_REGISTRY": raw}):
        return index._load_idps()


def test_idp_registry_parsed():
    idps = _load_idps_with(json.dumps([
        {"key": "lark", "provider": "p-lark", "scopes": ["a"], "label": "Lark"},
        {"key": "google", "provider": "p-goog", "scopes": ["b"], "label": "Google"},
    ]))
    assert set(idps) == {"lark", "google"}
    assert idps["google"]["provider"] == "p-goog"


def test_idp_registry_falls_back_when_malformed():
    idps = _load_idps_with("{not json")
    assert list(idps) == ["lark"]          # still usable
    assert idps["lark"]["provider"]        # and points somewhere


def test_idp_registry_falls_back_when_empty():
    assert list(_load_idps_with("")) == ["lark"]


# --- Memory thread rotation -----------------------------------------------
# Resetting a conversation rotates the thread id instead of deleting events:
# instant, and the old events stay put. So a rotation must yield a NEW id, and
# an unrotated lookup must be stable.

def test_memory_session_is_stable_then_rotates():
    import identity
    stored = {}

    def fake_get_item(Key):
        item = stored.get((Key["PK"], Key["SK"]))
        return {"Item": item} if item else {}

    def fake_put_item(Item):
        stored[(Item["PK"], Item["SK"])] = Item

    with mock.patch.object(identity._table, "get_item", side_effect=fake_get_item), \
         mock.patch.object(identity._table, "put_item", side_effect=fake_put_item):
        first = identity.get_or_create_memory_session("u1", "lark:ou_x")
        again = identity.get_or_create_memory_session("u1", "lark:ou_x")
        assert again == first                      # stable while not rotated
        rotated = identity.rotate_memory_session("u1", "lark:ou_x")
        assert rotated != first and rotated.startswith(first + "-")
        assert identity.get_or_create_memory_session("u1", "lark:ou_x") == rotated
        # An id rotated before threads carried the actor's prefix is refused by the agent,
        # so it is replaced once rather than sent.
        stored[("USER#u1", "MEMSESSION")]["memorySessionId"] = "sess-" + "a" * 32
        migrated = identity.get_or_create_memory_session("u1", "lark:ou_x")
        assert identity.thread_belongs("lark:ou_x", migrated) and migrated != first


# --- The status probe ------------------------------------------------------
# One probe, several lines: probe_agent asks the serving container what it is, and the
# _*_line helpers only format. They must never invoke anything themselves.
def _microvm_line_with(response, session_id="ses_x"):
    import index
    with mock.patch.object(index, "invoke_agent", return_value=response) as inv:
        data = index.probe_agent(session_id, "user_1", "lark:ou_x")
        return index._microvm_line(session_id, data), inv


def test_microvm_line_reports_kernel_age_and_session_age():
    import index
    line, inv = _microvm_line_with(
        {"instance": "abc12345", "kernelUptime": 1830.4, "sessionAge": 65.9, "uptime": 777.0})
    assert "abc12345" in line
    assert "1830 秒" in line and "65 秒" in line
    # The process's own uptime is untrustworthy (wall clock inherits the image's
    # base), so it must not appear.
    assert "777" not in line
    # Probing must not spend the whole Lambda budget on a cold start.
    assert inv.call_args.kwargs["action"] == "status"
    assert inv.call_args.kwargs["budget"] == index._PROBE_SECONDS


# --- Reading what the user sent ---------------------------------------------
# A caption sent with a screenshot arrives as `post`, not `text`. Reading content["text"]
# for every type dropped it, and the "hi" fallback then presented the empty result to the
# model as a greeting — so the agent cheerfully answered a message nobody had sent.


def test_a_caption_sent_with_a_screenshot_is_not_lost():
    """The payload Lark really delivers, taken from a message this bug was reported on."""
    import index
    content = {"title": "", "content": [
        [{"tag": "text", "text": "我给你添加了一个 deliver_file 工具", "style": []}],
        [{"tag": "img", "image_key": "img_v3_0216o_b46b56ab", "width": 531, "height": 641}]]}
    text, carried, found = index._words_and_attachments("post", content)
    assert text == "我给你添加了一个 deliver_file 工具"
    # A handle, not the word "img": the agent fetches the picture from Lark itself, and
    # bytes through here would not survive the parked state's 400 KB cap.
    assert found == [{"kind": "image", "key": "img_v3_0216o_b46b56ab"}]
    assert carried == []


def test_a_link_label_counts_as_words_but_a_mention_does_not():
    import index
    content = {"content": [[{"tag": "at", "user_id": "ou_x"},
                            {"tag": "text", "text": "see"},
                            {"tag": "a", "text": "this page", "href": "https://x"}]]}
    assert index._words_and_attachments("post", content) == ("see this page", [], [])


def test_a_file_sent_with_text_is_found_beside_the_paragraphs():
    """The payload Lark really delivers for a caption plus a file: the file is in a
    top-level `files` list, not a paragraph element, so a parser that only walked the
    paragraphs answered "no attachment" while the user was looking at one."""
    import index
    content = {"title": "",
               "content": [[{"tag": "text", "text": "总结这个文档内容", "style": []}]],
               "files": [{"file_key": "file_v3_0016o_715d4a6b",
                          "file_name": "GCR MAP Mobilize Phase - Technical Review V1.docx",
                          "is_folder": False}]}
    text, carried, found = index._words_and_attachments("post", content)
    assert text == "总结这个文档内容"
    assert found == [{"kind": "file", "key": "file_v3_0016o_715d4a6b",
                      "name": "GCR MAP Mobilize Phase - Technical Review V1.docx"}]
    assert carried == []


def test_a_folder_has_no_bytes_to_fetch_so_it_is_owned_up_to():
    import index
    content = {"content": [[{"tag": "text", "text": "看看"}]],
               "files": [{"file_key": "f", "file_name": "docs", "is_folder": True}]}
    assert index._words_and_attachments("post", content) == ("看看", ["folder"], [])


def test_an_attachment_is_carried_by_its_handle_whatever_it_is():
    """No caption is not nothing to answer, and a document is as fetchable as a picture —
    the kind only decides whether the model looks at it or opens it with code."""
    import index
    assert index._words_and_attachments("image", {"image_key": "img_1"}) == (
        "", [], [{"kind": "image", "key": "img_1"}])
    assert index._words_and_attachments(
        "file", {"file_key": "f_1", "file_name": "q3.pdf"}) == (
        "", [], [{"kind": "file", "key": "f_1", "name": "q3.pdf"}])
    assert index._words_and_attachments("text", {"text": "plain"}) == ("plain", [], [])
    # A sticker has nothing behind it worth reading, so it stays something to own up to.
    assert index._words_and_attachments("sticker", {}) == ("", ["sticker"], [])


def test_microvm_line_reports_platform_version():
    line, _ = _microvm_line_with(
        {"instance": "abc12345", "platformVersion": "V2", "kernelUptime": 30, "sessionAge": 5})
    assert "V2" in line


def test_microvm_line_omits_platform_version_when_unreported():
    """An older image answers status with no platformVersion key at all — must not
    print the literal "unknown" the agent falls back to when the env var is unset."""
    line, _ = _microvm_line_with(
        {"instance": "abc12345", "platformVersion": "unknown", "kernelUptime": 30, "sessionAge": 5})
    assert "unknown" not in line


def test_microvm_line_uses_one_unit_so_the_two_are_comparable():
    """Both durations in seconds: the point is to subtract them (equal → this microVM
    started for you; age >> sessionAge → an existing one took over)."""
    line, _ = _microvm_line_with(
        {"instance": "abc12345", "kernelUptime": 7200, "sessionAge": 30})
    assert "7200 秒" in line and "30 秒" in line
    assert "分钟" not in line and "小时" not in line


def test_microvm_line_without_a_session_does_not_probe():
    """No session id means no microVM to ask about — and probing would create one."""
    import index
    with mock.patch.object(index, "invoke_agent") as inv:
        line = index._microvm_line("", index.probe_agent("", "user_1", "lark:ou_x"))
    assert inv.call_count == 0
    assert "尚未建立会话" in line


def test_microvm_line_retries_a_provisioning_conflict():
    """AgentCore returns a retryable 409 while it provisions a session. The chat path
    disables retries on purpose (a timeout must not replay a turn), so this path has
    to handle it itself or /status reports "unknown" during a cold start."""
    import index

    class RetryableConflictException(Exception):
        pass

    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RetryableConflictException("Session operation in progress")
        return {"instance": "abc12345", "kernelUptime": 30, "sessionAge": 5}

    with mock.patch.object(index, "invoke_agent", side_effect=flaky), \
         mock.patch.object(index.time, "sleep"):
        line = index._microvm_line("ses_x", index.probe_agent("ses_x", "user_1", "lark:ou_x"))
    assert len(calls) == 2
    assert "abc12345" in line


def test_microvm_line_gives_up_after_one_retry():
    """Bounded: /status must not stall on a session stuck mid-provision."""
    import index

    class RetryableConflictException(Exception):
        pass

    with mock.patch.object(index, "invoke_agent",
                           side_effect=RetryableConflictException("busy")) as inv, \
         mock.patch.object(index.time, "sleep"):
        line = index._microvm_line("ses_x", index.probe_agent("ses_x", "user_1", "lark:ou_x"))
    assert inv.call_count == 2
    assert "未知" in line


def test_microvm_line_survives_a_probe_failure():
    """Diagnostics must never break the command that carries them."""
    import index
    with mock.patch.object(index, "invoke_agent", side_effect=RuntimeError("timeout")):
        line = index._microvm_line("ses_x", index.probe_agent("ses_x", "user_1", "lark:ou_x"))
    assert "未知" in line


def test_microvm_line_tolerates_an_older_image():
    """A microVM running a previous image reports no instance field."""
    line, _ = _microvm_line_with({"reply": "pong"})
    assert "旧镜像" in line


def test_model_reports_what_the_container_says_not_the_routers_config():
    """UpdateAgentRuntime replaces the environment, so the router's own MODEL_ID can name
    a model nobody is calling. Only the serving container's answer is shown."""
    import index
    text = index._model_panel({"settings": {
        "model": "global.anthropic.claude-sonnet-5", "region": "ap-northeast-1",
        "cache": "explicit 1h", "summarizeAt": 120000, "summarizeKeep": 20,
        "systemPromptCustom": False, "systemPromptChars": 180}})["text"]
    assert "global.anthropic.claude-sonnet-5" in text
    assert "Amazon Bedrock · ap-northeast-1" in text
    assert "TTL 1h" in text
    # The prompt and the summarisation threshold shape the context, not the model.
    assert "系统提示词：" not in text and "保留最近" not in text
    assert "关闭" in index._model_panel({"settings": {"model": "m", "cache": "off"}})["text"]
    assert "隐式" in index._model_panel({"settings": {"model": "m", "cache": "implicit"}})["text"]
    # An older image reports no settings, and a failed probe reports nothing at all.
    assert "探测未返回" in index._model_panel({"model": "m"})["text"]
    assert "探测未返回" in index._model_panel(None)["text"]


def test_context_holds_the_size_the_prompt_and_the_compaction_policy():
    import index
    stats = {"turns": 4, "toolCalls": 6, "contextTokens": 9147, "cachedTokens": 5049,
             "inputTotal": 19166, "cachedTotal": 14845, "modelCalls": 3}
    probe = {"settings": {"cache": "implicit", "summarizeAt": 120000, "summarizeKeep": 20,
                          "systemPromptCustom": False, "systemPromptChars": 165}}
    rows = dict(r for g in index._context_panel(stats, probe)["panel"]["groups"] for r in g)
    assert rows["对话"] == "4 轮 · 6 次工具调用"
    assert rows["当前大小"] == "9.1k token（最近一次请求，压缩阈值的 8%）"
    assert rows["系统提示词"] == "默认 · 165 字符"
    assert rows["自动压缩"] == "超过 120.0k token 时摘要，保留最近 20 条消息"
    # Figures from a real thread: the latest call half cached, the thread as a whole more.
    assert rows["缓存命中"] == "最近一次 55% · 本线程累计 77%（3 次模型调用）"
    off = {"settings": {**probe["settings"], "cache": "off", "summarizeAt": 0}}
    rows = dict(r for g in index._context_panel(stats, off)["panel"]["groups"] for r in g)
    assert rows["自动压缩"] == "关闭" and rows["缓存命中"].startswith("未启用")
    # Nothing has run yet, or the probe did not answer: says so rather than inventing zeros.
    text = index._context_panel({}, None)["text"]
    assert "发一条普通消息" in text and "探测未返回" in text


def test_a_panel_carries_its_fields_and_the_text_the_web_page_shows():
    import index
    r = index._panel("T", [("a", "1")], [], [("b", "2")], hint="h")
    assert r["panel"] == {"title": "T", "groups": [[("a", "1")], [("b", "2")]], "hint": "h"}
    assert r["text"] == "T\n\na：1\n\nb：2\n\nh"


def test_a_refused_card_still_answers_as_text():
    import lark
    with mock.patch.object(lark, "get_tenant_token", return_value="t"), \
         mock.patch.object(lark.urllib.request, "urlopen", side_effect=OSError("boom")), \
         mock.patch.object(lark, "send_message", return_value=True) as text:
        assert lark.send_panel("oc_1", {"title": "T", "groups": [[("a", "1")]]}, "T a")
    assert text.call_args.args[:2] == ("oc_1", "T a")


def test_tools_are_grouped_by_where_they_came_from_with_the_grant_they_need():
    """Grouped by source, so the approval tools stand apart from Lark's even though the
    Lark grant gates both; an ungranted system says which /auth to send."""
    import index
    with mock.patch.dict(index.IDPS, {"lark": {"label": "Lark"},
                                      "google": {"label": "Google"}}, clear=True):
        r = index._tools_panel({"tools": [
            {"name": "lark_whoami", "source": "lark", "idp": "lark"},
            {"name": "approval_list", "source": "approval", "idp": "lark"},
            {"name": "google_whoami", "source": "google", "idp": "google"},
            {"name": "run_code", "source": "code", "idp": ""},
        ], "ungranted": ["google"]})["panel"]
    assert r["title"] == "可用工具 · 共 4 个"
    rows = r["groups"][0]
    assert ("Lark", "lark_whoami") in rows and ("审批", "approval_list") in rows
    assert ("Google（未授权）", "google_whoami") in rows and ("代码执行", "run_code") in rows
    assert "/auth google" in r["hint"]


def test_compact_says_what_happened_in_every_case():
    import index
    assert "较早的 12 条" in index._compact_text({"compacted": 12, "kept": 6})
    for reason, words in (("busy", "回答中"), ("short", "很短"), ("failed", "没能生成摘要"),
                          ("identity", "身份校验"), ("error", "没有完成")):
        assert words in index._compact_text({"compacted": 0, "reason": reason}), reason


def test_tools_say_why_there_are_none():
    import index
    assert "身份校验未通过" in index._tools_panel({"tools": [], "identityError": "x"})["text"]
    assert "探测未返回" in index._tools_panel(None)["text"]




def test_add_reaction_returns_id_and_never_raises():
    """The progress marker is best-effort: the reaction_id is needed to remove it
    later, but a failure must not stop the turn."""
    import lark
    ok = {"code": 0, "data": {"reaction_id": "RE_1"}}

    class FakeResp:
        def __init__(self, d): self._d = d
        def read(self): return json.dumps(self._d).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with mock.patch.object(lark, "get_tenant_token", return_value="t"), \
         mock.patch.object(lark.urllib.request, "urlopen", lambda *a, **k: FakeResp(ok)):
        assert lark.add_reaction("om_1") == "RE_1"

    # Lark rejecting it, and the call blowing up, both degrade to "" rather than raise.
    with mock.patch.object(lark, "get_tenant_token", return_value="t"), \
         mock.patch.object(lark.urllib.request, "urlopen",
                           lambda *a, **k: FakeResp({"code": 99, "msg": "nope"})):
        assert lark.add_reaction("om_1") == ""
    with mock.patch.object(lark, "get_tenant_token", return_value="t"), \
         mock.patch.object(lark.urllib.request, "urlopen",
                           mock.Mock(side_effect=OSError("boom"))):
        assert lark.add_reaction("om_1") == ""
    # No message id (or no token) means nothing to react to.
    with mock.patch.object(lark, "get_tenant_token", return_value="t"):
        assert lark.add_reaction("") == ""


def test_status_shows_both_identities():
    """/status must name both identities — the whole point of this sample is that
    tools act as the *user* while replies go out as the *app*."""
    import index
    with mock.patch.object(index.lark, "get_credentials",
                           return_value=("cli_a1b2c3d4e5f6g7h8", "s", "v", "k")):
        assert index._app_identity() == "cli_a1b2c3d4e5f6g7h8"
    # A deployment without a Lark app has none to name, so /status leaves the row out.
    with mock.patch.object(index.lark, "get_credentials", return_value=("", "", "", "")), \
         mock.patch.object(index.identity, "session_info", return_value={}), \
         mock.patch.object(index.identity, "get_or_create_memory_session", return_value="m"), \
         mock.patch.object(index, "probe_agent", return_value=None):
        assert index._app_identity() == ""
        text = index.run_command("/status", "u-1", "web:s-1")["text"]
        assert "应用身份" not in text and "web:s-1" in text


# --------------------------- consent resume ---------------------------------
# When a Lark tool hits an auth wall in the async run, the message is parked; the
# shim's /return replays it after consent. These pin the parts that must not drift:
# take is once-only, expiry is honoured, and resume actually re-invokes the agent.

def test_take_pending_auth_claims_atomically_and_honours_ttl():
    """The claim IS the delete. Two paths race to replay a parked turn — the router
    polling the vault and the shim's post-consent callback — so a get-then-delete lets
    both read the same item and run the turn twice, with duplicated side effects."""
    import identity, time as _t
    store = {}
    def fake_put(Item): store[Item["SK"]] = Item
    def fake_del(Key, ReturnValues=None):
        old = store.pop(Key["SK"], None)
        return {"Attributes": old} if (old and ReturnValues == "ALL_OLD") else {}
    with mock.patch.object(identity._table, "put_item", side_effect=fake_put), \
         mock.patch.object(identity._table, "delete_item", side_effect=fake_del):
        identity.park_pending_auth("u1", "查待审批", "oc_1")
        assert identity.take_pending_auth("u1") == {"message": "查待审批", "chatId": "oc_1"}
        # The loser of the race gets nothing — replay must not happen twice.
        assert identity.take_pending_auth("u1") is None

        # An item past its ttl is treated as absent, even before DynamoDB sweeps it.
        store["PENDING_AUTH"] = {"message": "old", "chatId": "oc_1",
                                 "ttl": int(_t.time()) - 1}
        assert identity.take_pending_auth("u1") is None


def test_take_pending_auth_reads_the_item_from_the_delete_itself():
    """Guards the mechanism, not just the outcome: without ReturnValues=ALL_OLD the
    claim degrades to a blind delete and nothing is ever replayed."""
    import identity
    with mock.patch.object(identity._table, "delete_item",
                           return_value={"Attributes": {"message": "m", "chatId": "c",
                                                        "ttl": 9999999999}}) as dele:
        assert identity.take_pending_auth("u1") == {"message": "m", "chatId": "c"}
    assert dele.call_args.kwargs.get("ReturnValues") == "ALL_OLD"


# The synchronous consent-wait is gone (it polled a vault the router cannot read), so the
# shim's /return is the only replay path. The atomic claim above is what still stops a
# double replay if anything else ever races it.

def test_resume_replays_the_parked_message():
    import index, identity
    with mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "take_pending_auth",
                           return_value={"message": "查待审批", "chatId": "oc_1"}), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_x"), \
         mock.patch.object(identity, "get_or_create_memory_session", return_value="mem_x"), \
         mock.patch.object(index.lark, "send_message") as said, \
         mock.patch.object(index, "invoke_agent") as inv:
        index.resume_consented_turn("lark:ou_abc")
    said.assert_called_once_with("oc_1", "✅ 授权成功，正在继续处理刚才的消息…")
    assert inv.call_count == 1
    kw = inv.call_args.kwargs
    assert kw["action"] == "chat_async"
    assert kw["chat_id"] == "oc_1"
    assert inv.call_args.args[3] == "查待审批"   # message positional


def test_resume_is_a_noop_when_nothing_parked():
    """A user who ran /auth directly has no turn to resume — must not invoke."""
    import index, identity
    with mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "take_pending_auth", return_value=None), \
         mock.patch.object(index.lark, "send_message") as said, \
         mock.patch.object(index, "invoke_agent") as inv:
        index.resume_consented_turn("lark:ou_abc")
    assert inv.call_count == 0
    # Nothing to continue, so nothing is promised.
    said.assert_called_once_with("ou_abc", "✅ 授权成功")


def test_resume_ignores_malformed_actor():
    import index
    with mock.patch.object(index, "invoke_agent") as inv:
        index.resume_consented_turn("not-a-lark-id")
    assert inv.call_count == 0



# --------------------------- event-driven approval ---------------------------

def _approval_event(**over) -> dict:
    """A Lark `approval_task` event. Legacy 1.0 schema: no `header`, type inside
    `event`. Fields per the official doc, plus the `open_id` its sample payload shows
    but its field table omits."""
    ev = {"app_id": "cli_x", "type": "approval_task", "open_id": "ou_alice",
          "user_id": "b613t51g", "task_id": "t1", "instance_code": "i1",
          "approval_code": "AC1", "status": "PENDING", "operate_time": "1700000000000"}
    ev.update(over)
    return ev


def test_approval_event_dispatches_a_turn_addressed_to_the_approver():
    """An approval event carries no chat, so the approver's open_id is the delivery
    address — the senders read an `ou_` prefix as "DM this person"."""
    import index, identity
    with mock.patch.object(identity, "claim_approval_task", return_value=True), \
         mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "park_pending_auth"), \
         mock.patch.object(index, "_dispatch_turn") as disp:
        index.process_approval_event(_approval_event())
    assert disp.call_count == 1
    user_id, actor_id, message, chat_id = disp.call_args.args[:4]
    assert actor_id == "lark:ou_alice"
    assert chat_id == "ou_alice"
    # The ids are handed over rather than left to be discovered: an unattended turn has
    # nobody to ask, and user_id decides whose name the decision is recorded under.
    for token in ("i1", "t1", "AC1", "ou_alice"):
        assert token in message


def test_approval_event_ignores_a_settled_task():
    """The loop-breaker: the agent's own approve emits another event, and acting on it
    would decide the same instance again."""
    import index, identity
    for status in ("APPROVED", "REJECTED", "TRANSFERRED", "DONE"):
        with mock.patch.object(identity, "claim_approval_task", return_value=True), \
             mock.patch.object(index, "_dispatch_turn") as disp:
            index.process_approval_event(_approval_event(status=status))
        assert disp.call_count == 0, status


def test_approval_event_without_an_approver_is_skipped():
    """`approval_instance` events, and auto-approve tasks (documented as having an
    empty user), name nobody — there is no identity to act as."""
    import index, identity
    for missing in ({"open_id": ""}, {"task_id": ""}, {"instance_code": ""}):
        with mock.patch.object(identity, "claim_approval_task", return_value=True), \
             mock.patch.object(index, "_dispatch_turn") as disp:
            index.process_approval_event(_approval_event(**missing))
        assert disp.call_count == 0, missing


def test_approval_event_redelivery_decides_once():
    """Lark redelivers until acked, and the ack goes out long before the agent has
    decided. Without the claim the same task gets approved twice."""
    import index, identity
    with mock.patch.object(identity, "claim_approval_task", return_value=False), \
         mock.patch.object(index, "_dispatch_turn") as disp:
        index.process_approval_event(_approval_event())
    assert disp.call_count == 0


def test_approval_event_for_an_unknown_user_stays_silent():
    """Someone outside the allowlist: their approval is their own business."""
    import index, identity
    with mock.patch.object(identity, "claim_approval_task", return_value=True), \
         mock.patch.object(identity, "resolve_user", return_value=(None, False)), \
         mock.patch.object(index, "_dispatch_turn") as disp:
        index.process_approval_event(_approval_event())
    assert disp.call_count == 0


def test_approval_event_parks_the_turn_when_the_approver_has_not_consented():
    """The approval server refuses to decide without that person's own grant, so the turn
    may wall. Parking is what lets consent-resume replay it — otherwise an unattended turn
    is simply lost. Unconditional: only the agent knows whether a grant exists, and an
    unnecessary park expires by TTL."""
    import index, identity
    with mock.patch.object(identity, "claim_approval_task", return_value=True), \
         mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "park_pending_auth") as park, \
         mock.patch.object(index, "_dispatch_turn"):
        index.process_approval_event(_approval_event())
    assert park.call_count == 1
    assert park.call_args.args[2] == "ou_alice"      # DM target, not a chat


def test_legacy_schema_event_is_routed_to_the_approval_path():
    """Message events put the type in `header` (schema 2.0); approval events put it in
    `event` (1.0). Reading only `header` drops every approval event on the floor."""
    import index
    body = json.dumps({"uuid": "u", "type": "event_callback",
                       "event": _approval_event()})
    with mock.patch.object(index, "process_approval_event") as appr:
        index.process_lark_event(body, {})
    assert appr.call_count == 1


def test_message_events_still_take_the_message_path():
    """The 2.0 header must keep winning — a regression here breaks the whole bot."""
    import index
    body = json.dumps({"header": {"event_type": "im.message.receive_v1"},
                       "event": {"sender": {"sender_type": "bot"}}})
    with mock.patch.object(index, "process_approval_event") as appr:
        index.process_lark_event(body, {})
    assert appr.call_count == 0


def test_dm_target_is_inferred_from_the_id_shape(lark_mod):
    """`ou_` is a person, anything else a chat. This is what lets the same senders
    serve an event-driven turn, which only ever knows the person."""
    assert lark_mod._receive_id_type("ou_alice") == "open_id"
    assert lark_mod._receive_id_type("oc_room") == "chat_id"
    assert lark_mod._receive_id_type("ou_alice", "chat_id") == "chat_id"   # explicit wins


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------- signed inbound identity (CUSTOM_JWT cutover) ----------------

def test_invoke_agent_carries_the_users_signed_identity_not_a_string():
    """The whole point of the cutover: the runtime hop is authorised by the user's own
    JWT. A payload string naming a user would be forgeable; a signature is not — and the
    agent never receives an actorId it could substitute for someone else's."""
    import index
    captured = {}

    class FakeResp:
        def read(self): return b'{"reply": "ok"}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data)
        return FakeResp()

    with mock.patch.object(index.cognito, "user_jwt", return_value="JWT-FOR-U") as mint, \
         mock.patch.object(index.urllib.request, "urlopen", fake_urlopen), \
         mock.patch.object(index, "agentcore") as ac:
        out = index.invoke_agent("sess-0123456789abcdef0123456789abcdef",
                                 "user_1", "lark:ou_u", "hello")

    assert out == {"reply": "ok"}
    mint.assert_called_once_with("lark:ou_u")
    assert captured["headers"]["authorization"] == "Bearer JWT-FOR-U"
    # SigV4 must not be used at all — a CUSTOM_JWT runtime rejects it outright.
    ac.invoke_agent_runtime.assert_not_called()
    assert "/invocations" in captured["url"]
    assert captured["body"]["actorId"] == "lark:ou_u"


def test_complete_consent_names_the_user_by_token_not_by_id():
    """A consent session started from a JWT-derived workload token can only be completed
    with that identity. Passing userId fails as `Invalid or expired session`, which reads
    like a timing bug and is really a namespace mismatch (measured)."""
    import index
    with mock.patch.object(index.cognito, "consent_tokens", return_value=["JWT-FOR-U"]), \
         mock.patch.object(index, "agentcore") as ac:
        index.complete_consent("lark:ou_u", "sess-uri")
    ac.complete_resource_token_auth.assert_called_once_with(
        sessionUri="sess-uri", userIdentifier={"userToken": "JWT-FOR-U"})


def test_the_router_never_queries_the_vault_itself():
    """It cannot: a grant is vaulted against the workload identity the Runtime derives from
    the inbound JWT, and one re-derived here reads a different namespace and reports "no"
    forever — an endless consent loop while the agent's own calls succeed. Measured, see
    docs/agentcore-behavior.md. An earlier test asserted the opposite as correct."""
    import index, inspect
    src = inspect.getsource(index)
    assert "get_resource_oauth2_token" not in src
    assert "get_workload_access_token_for_jwt" not in src


def test_credential_load_failure_is_not_cached():
    """A failed secret read must not disable the container for its whole life: seeding or
    rotating the Lark secret would then need a cold start, presenting as
    "no encryptKey configured" long after the secret is correct."""
    import lark
    lark._creds_cache = None
    with mock.patch.object(lark, "_SECRET_ID", "s"), \
         mock.patch.object(lark, "_secrets") as sm:
        sm.get_secret_value.side_effect = ValueError("not json")
        assert lark.get_credentials() == ("", "", "", "")
        assert lark._creds_cache is None          # nothing poisoned
        sm.get_secret_value.side_effect = None
        sm.get_secret_value.return_value = {"SecretString": '{"encryptKey":"k"}'}
        assert lark.get_credentials()[3] == "k"   # next call recovers
    lark._creds_cache = None


def test_auth_status_is_answered_by_the_agent_not_the_vault():
    """The router cannot read the grant: it lives under the workload identity the Runtime
    derives from the inbound JWT, so a locally re-derived one always reports "no"."""
    import index
    with mock.patch.object(index, "invoke_agent",
                           return_value={"lark": True}) as inv:
        assert index.user_authorized("ses_x", "u1", "lark:ou_a", "lark") is True
    assert inv.call_args.kwargs["action"] == "auth_status"


def test_auth_status_treats_a_missing_key_as_unauthorized():
    """An agent error must not read as authorised."""
    import index
    with mock.patch.object(index, "invoke_agent", return_value={"error": "boom"}):
        assert index.user_authorized("ses_x", "u1", "lark:ou_a", "lark") is False


def test_web_session_requires_a_lark_code_and_never_trusts_an_open_id():
    """The browser proves identity with a single-use code Lark issued to this app inside the
    Lark client. Accepting an open_id instead would let anyone name anyone."""
    import index, inspect
    src = inspect.getsource(index._h5_identity) + inspect.getsource(index._h5_session)
    assert "open_id_from_auth_code" in src
    assert 'payload.get("openId")' not in src and 'payload.get("open_id")' not in src


def test_the_code_exchange_uses_the_v2_endpoint_and_app_credentials():
    """v1 oidc/access_token is deprecated and took a tenant token; v2 takes client
    credentials in the body and returns only a token, so open_id needs user_info."""
    import io
    import lark as larkmod
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, req.data, dict(req.headers)))
        body = ({"access_token": "u-tok"} if "oauth/token" in req.full_url
                else {"code": 0, "data": {"open_id": "ou_x"}})
        return io.BytesIO(json.dumps(body).encode())

    with mock.patch.object(larkmod, "get_credentials",
                           return_value=("app", "secret", "", "")), \
         mock.patch.object(larkmod.urllib.request, "urlopen",
                           side_effect=lambda req, timeout=None:
                           _ctx(fake_urlopen(req, timeout))):
        assert larkmod.open_id_from_auth_code("c") == "ou_x"
    assert "authen/v2/oauth/token" in seen[0][0]
    assert b'"client_secret": "secret"' in seen[0][1]
    assert "Authorization" not in seen[0][2]      # no tenant token on the redeem call
    assert "authen/v1/user_info" in seen[1][0]
    assert seen[1][2]["Authorization"] == "Bearer u-tok"


class _ctx:
    """urlopen is used as a context manager; wrap a plain BytesIO so the fake matches."""

    def __init__(self, inner):
        self.inner = inner

    def __enter__(self):
        return self.inner

    def __exit__(self, *a):
        return False


def test_web_session_refuses_an_unexchangeable_code():
    import index
    with mock.patch.object(index.lark, "open_id_from_auth_code", return_value=""):
        assert index._h5_session('{"code": "bad"}')["statusCode"] == 401


def test_run_command_returns_text_and_never_sends_it_itself():
    """Channel-agnostic on purpose: the same commands answer in Lark and on the web page.
    A branch that sent to Lark directly would work in chat and silently do nothing on the web."""
    import index, inspect
    src = inspect.getsource(index.run_command)
    assert "lark.send" not in src
    with mock.patch.object(index.lark, "send_message") as send:
        assert "/status" in index.run_command("/help", "u1", "lark:ou_a")["text"]
        assert index.run_command("你好", "u1", "lark:ou_a") is None   # not a command
    assert send.call_count == 0


def test_an_auth_link_travels_as_data_so_each_channel_renders_it_its_own_way():
    """Lark wants a rich-text post; the web page wants markdown. Returning the URL keeps
    that choice with the channel."""
    import index, identity
    with mock.patch.dict(index.IDPS, {"lark": {"label": "Lark"}}, clear=True), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_x"), \
         mock.patch.object(index, "invoke_agent", return_value={"auth_url": "https://c/x"}):
        r = index.run_command("/auth lark", "u1", "lark:ou_a")
    assert r["link"] == {"text": "点击授权", "url": "https://c/x"}


def test_auth_idp_says_who_granted_it_rather_than_asking_again():
    """A new consent is for a system not granted; switching accounts is /auth revoke first."""
    import index, identity
    calls = []

    def fake_invoke(sid, uid, actor, message, action="", **k):
        calls.append(action)
        return {"google": True, "accounts": {"google": "me@example.com"}} if action == "auth_status" \
            else {"auth_url": "https://c/x"}
    with mock.patch.dict(index.IDPS, {"google": {"label": "Google"}, "lark": {"label": "Lark"}},
                         clear=True), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_x"), \
         mock.patch.object(index, "invoke_agent", side_effect=fake_invoke):
        r = index.run_command("/auth google", "u1", "web:s-1")
        assert "已授权 Google（me@example.com）" in r["text"] and "/auth revoke google" in r["text"]
        assert "link" not in r and calls == ["auth_status"]
        calls.clear()
        r = index.run_command("/auth lark", "u1", "web:s-1")
        assert r["link"]["url"] == "https://c/x" and calls == ["auth_status", "reauth"]


def test_web_command_refuses_anything_that_is_not_a_command():
    """The command endpoint must not become a second way to talk to the agent — that path
    goes straight to the Runtime and is the only one carrying the user's JWT."""
    import index, identity
    with mock.patch.object(index.lark, "open_id_from_auth_code", return_value="ou_x"), \
         mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "is_user_allowed", return_value=True):
        assert index._h5_command('{"code":"c","command":"你好"}')["statusCode"] == 400
        r = index._h5_command('{"code":"c","command":"/help"}')
    assert r["statusCode"] == 200 and "/status" in json.loads(r["body"])["text"]


def test_web_command_establishes_identity_per_request():
    """It rotates session ids, so it cannot trust a token the page is holding."""
    import index
    with mock.patch.object(index.lark, "open_id_from_auth_code", return_value="") as ex:
        assert index._h5_command('{"code":"bad","command":"/status"}')["statusCode"] == 401
    assert ex.call_count == 1


def test_cors_is_configured_on_the_api_and_not_duplicated_in_the_handler():
    """The browser preflights a cross-origin JSON POST. With CORS on the HTTP API, API
    Gateway answers OPTIONS itself and ignores the integration's own CORS headers — so
    setting them here too is dead code, which is exactly how this shipped broken."""
    import pathlib
    stack = pathlib.Path(__file__).parents[2] / "stacks" / "router_stack.py"
    src = stack.read_text()
    assert "cors_preflight" in src and "web_allowed_origins.split" in src
    handler_src = pathlib.Path(__file__).with_name("index.py").read_text()
    assert "Access-Control-Allow-Origin" not in handler_src


def test_web_session_honours_the_allowlist():
    import index, identity
    with mock.patch.object(index.lark, "open_id_from_auth_code", return_value="ou_x"), \
         mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "is_user_allowed", return_value=False):
        assert index._h5_session('{"code": "c"}')["statusCode"] == 403


def test_web_session_hands_back_the_same_routing_key_lark_chat_uses():
    """Both surfaces must land on one microVM: AgentCore maps a session id to exactly one,
    which is what lets the agent serialise two surfaces' turns in process rather than
    needing a distributed lock. A page-local id would put them in different containers,
    where they corrupt one conversation."""
    import index, identity
    with mock.patch.object(index.lark, "open_id_from_auth_code", return_value="ou_x"), \
         mock.patch.object(identity, "resolve_user", return_value=("u1", False)), \
         mock.patch.object(identity, "is_user_allowed", return_value=True), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_shared"), \
         mock.patch.object(identity, "get_or_create_memory_session", return_value="sess_mem"), \
         mock.patch.object(index.cognito, "user_jwt", return_value="JWT"):
        r = index._h5_session('{"code": "c"}')
    body = json.loads(r["body"])
    assert r["statusCode"] == 200 and body["token"] == "JWT"
    assert body["runtimeSessionId"] == "ses_shared"
    assert body["runtimeArn"] and body["qualifier"]


def test_status_survives_a_missing_runtime_session():
    """rt_sid is empty until a turn has run. Passing it through produced a 400 that killed
    the whole handler, so /status silently answered nothing at all."""
    import index
    assert index.thread_stats("", "u1", "lark:ou_a", "mem-1") == {}
    assert index.user_authorized("", "u1", "lark:ou_a", "lark") is False


def test_a_failed_count_does_not_take_down_the_command():
    import index
    with mock.patch.object(index, "invoke_agent", side_effect=RuntimeError("boom")):
        assert index.thread_stats("ses_x", "u1", "lark:ou_a", "mem-1") == {}
        assert index.user_authorized("ses_x", "u1", "lark:ou_a", "lark") is False


def test_context_does_not_manufacture_the_session_it_reports_as_absent():
    """Counting needs a runtime session, so counting eagerly created one — while the line
    above still said "not established". A diagnostic must not cause what it reports."""
    import index, inspect
    src = inspect.getsource(index.run_command)
    block = src[src.index('cmd == "/context"'):src.index('cmd == "/model"')]
    assert "get_or_create_session" not in block
    assert "if rt_sid else" in block


def test_status_shows_who_serves_you_and_points_at_the_rest():
    """The runtime state stays in /status — the routing key, the microVM, the thread — and
    what the thread holds moves to /context. Neither may create a session to report on."""
    import index, inspect
    src = inspect.getsource(index.run_command)
    block = src[src.index('cmd == "/status"'):src.index('cmd == "/context"')]
    assert "get_or_create_session" not in block
    for line in ("会话路由键", "当前 microVM", "记忆线程", "最近活跃"):
        assert line in block, line
    for pointer in ("/context", "/model", "/tools", "/auth"):
        assert pointer in block, pointer
    assert "thread_stats" not in block, "turns and context size belong to /context"


# --------------------------- the web page outside Lark -------------------------
# A web user signs in at the hosted login and everything is done with their own token: the
# router can mint only for Lark users, and only the token a consent was started with can
# complete it.

def _web_event(claims, token="tok-web"):
    return {"requestContext": {"authorizer": {"jwt": {"claims": claims}}},
            "headers": {"authorization": f"Bearer {token}"}}


def test_a_web_caller_is_web_sub_and_only_from_the_web_client():
    import index, identity
    good = {"sub": "s-1", "client_id": "web-client", "username": "carol@example.com"}
    with mock.patch.object(index, "_WEB_CLIENT_ID", "web-client"), \
         mock.patch.object(identity, "resolve_user", return_value=("u-web", True)):
        assert index._web_caller(_web_event(good))[1:3] == ("web:s-1", "tok-web")
        assert index._web_caller(_web_event({**good, "client_id": "router"}))[3]["statusCode"] == 401
        assert index._web_caller(_web_event({**good, "username": "lark:ou_x"}))[3]["statusCode"] == 403
    # The pool is a web user's allowlist: no ALLOW# entry, registration closed or not.
    with mock.patch.object(identity, "_REGISTRATION_OPEN", False):
        assert identity.is_user_allowed("web", "s-1") is True


def test_auth_lists_each_system_with_the_account_it_was_granted_by_in_one_call():
    import index, identity
    calls = []

    def fake_invoke(*a, **k):
        calls.append(k.get("action"))
        return {"lark": True, "google": False, "accounts": {"lark": "XUESONG · me@myners.net"}}
    idps = {"lark": {"key": "lark", "label": "Lark"}, "google": {"key": "google", "label": "Google"}}
    with mock.patch.object(index, "IDPS", idps), \
         mock.patch.object(index, "invoke_agent", side_effect=fake_invoke), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_x"):
        text = index.run_command("/auth", "u-1", "web:s-1")["text"]
    assert calls == ["auth_status"]
    assert "✅ Lark（XUESONG · me@myners.net）" in text and "lark (Lark)" not in text
    assert "❌ Google — 发送 /auth google 授权" in text


def test_auth_revoke_clears_each_granted_system_by_starting_a_consent_it_discards():
    """AgentCore cannot delete one user's token; a forced consent drops it (measured)."""
    import index, identity
    calls = []

    def fake_invoke(sid, uid, actor, message, action="", **k):
        calls.append((action, message))
        if action == "auth_status":
            return {"lark": True, "google": True, "accounts": {}}
        return {"auth_url": "https://discarded"}
    idps = {"lark": {"key": "lark", "label": "Lark"}, "google": {"key": "google", "label": "Google"}}
    with mock.patch.object(index, "IDPS", idps), \
         mock.patch.object(index, "invoke_agent", side_effect=fake_invoke), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_x"):
        out = index.run_command("/auth revoke", "u-1", "web:s-1")
        assert calls[1:] == [("reauth", "lark"), ("reauth", "google")]
        assert "已清除 Lark、Google 的授权" in out["text"] and "link" not in out
        calls.clear()
        one = index.run_command("/auth revoke google", "u-1", "web:s-1")["text"]
        assert "已清除 Google" in one and "第三方应用" in one and "Lark" not in one
        assert calls[1:] == [("reauth", "google")]
        assert "未知的 IdP" in index.run_command("/auth revoke github", "u-1", "web:s-1")["text"]


def test_only_an_admin_manages_web_users_and_never_lark_users_or_themselves():
    """The admin page is gated on the admins group of the verified token, not on the page
    hiding a link; Lark users are the router's own, and an admin cannot lock everyone out."""
    import index, identity, json as _j
    admin = {"sub": "s-a", "client_id": "web-client", "username": "ann@example.com",
             "cognito:groups": "[admins]"}
    user = {**admin, "sub": "s-u", "username": "bob@example.com", "cognito:groups": ""}

    def call(claims, method, body=None):
        return index._web_admin(_web_event(claims), method, _j.dumps(body) if body else "")
    with mock.patch.object(index, "_WEB_CLIENT_ID", "web-client"), \
         mock.patch.object(identity, "resolve_user", return_value=("u-web", True)), \
         mock.patch.object(index.cognito, "web_users", return_value=[{"email": "bob@example.com"}]), \
         mock.patch.object(index.cognito, "invite") as invite, \
         mock.patch.object(index.cognito, "remove") as remove:
        assert call(user, "GET")["statusCode"] == 403
        assert call(user, "POST", {"email": "eve@example.com"})["statusCode"] == 403
        assert _j.loads(call(admin, "GET")["body"])["users"] == [{"email": "bob@example.com"}]
        assert call(admin, "POST", {"email": " Eve@Example.com "})["statusCode"] == 200
        invite.assert_called_once_with("eve@example.com")
        for bad in ("lark:ou_x", "not-an-email", ""):
            assert call(admin, "POST", {"email": bad})["statusCode"] == 400, bad
        assert call(admin, "DELETE", {"email": "ann@example.com"})["statusCode"] == 400
        assert call(admin, "DELETE", {"email": "bob@example.com"})["statusCode"] == 200
        remove.assert_called_once_with("bob@example.com")
    # The groups claim also arrives as several names in one string.
    with mock.patch.object(index, "_WEB_CLIENT_ID", "web-client"), \
         mock.patch.object(identity, "resolve_user", return_value=("u-web", True)), \
         mock.patch.object(index.cognito, "web_users", return_value=[]):
        assert call({**admin, "cognito:groups": "[staff admins]"}, "GET")["statusCode"] == 200
        assert call({**admin, "cognito:groups": "[administrators]"}, "GET")["statusCode"] == 403


def test_the_users_listed_are_the_web_ones_with_who_is_an_admin():
    import cognito
    client = mock.Mock()
    client.list_users_in_group.return_value = {"Users": [{"Username": "ann@example.com"}]}
    client.list_users.side_effect = [
        {"Users": [{"Username": "ann@example.com", "UserStatus": "CONFIRMED"},
                   {"Username": "lark:ou_x", "UserStatus": "CONFIRMED"}], "PaginationToken": "p2"},
        {"Users": [{"Username": "bob@example.com", "UserStatus": "FORCE_CHANGE_PASSWORD"}]}]
    with mock.patch.object(cognito, "_cognito", client):
        assert cognito.web_users() == [
            {"email": "ann@example.com", "status": "CONFIRMED", "admin": True},
            {"email": "bob@example.com", "status": "FORCE_CHANGE_PASSWORD", "admin": False}]
        for f in (cognito.invite, cognito.remove):
            with pytest.raises(ValueError):
                f("lark:ou_x")


def test_a_web_command_reaches_the_agent_with_the_callers_own_token():
    """Minting for `web:{sub}` would create a shadow Cognito user whose vault namespace is
    not the page's: /auth and /tools would read somebody else's grants."""
    import index, identity
    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req.headers["Authorization"])
        raise RuntimeError("stop here")
    claims = {"sub": "s-1", "client_id": "web-client", "username": "carol@example.com"}
    with mock.patch.object(index, "_WEB_CLIENT_ID", "web-client"), \
         mock.patch.object(identity, "resolve_user", return_value=("u-web", True)), \
         mock.patch.object(identity, "get_or_create_session", return_value="ses_x"), \
         mock.patch.object(identity, "session_info", return_value={"sessionId": "ses_x"}), \
         mock.patch.object(identity, "get_or_create_memory_session", return_value="m"), \
         mock.patch.object(index.cognito, "user_jwt", side_effect=AssertionError("minted")), \
         mock.patch.object(index.urllib.request, "urlopen", fake_urlopen):
        index._web_command(_web_event(claims), '{"command": "/model"}')
    assert seen and all(h == "Bearer tok-web" for h in seen)
    assert index._caller_bearer.get() == ""        # not left set for the next request


def test_a_consent_is_completed_only_with_a_token_it_can_have_been_started_with():
    import index
    calls = []

    def complete(sessionUri, userIdentifier):
        calls.append(userIdentifier["userToken"])
        if userIdentifier["userToken"] != "previous":
            raise RuntimeError("Invalid or expired session")
        return {}
    with mock.patch.object(index.agentcore, "complete_resource_token_auth", side_effect=complete), \
         mock.patch.object(index.cognito, "consent_tokens", return_value=["current", "previous"]), \
         mock.patch.object(index.cognito, "user_jwt", side_effect=AssertionError("minted")):
        index.complete_consent("lark:ou_x", "urn:s")             # a Lark user: both stored ones
        assert calls == ["current", "previous"]
        calls.clear()
        with pytest.raises(RuntimeError):
            index.complete_consent("web:s-1", "urn:s", bearer="page-token")
        assert calls == ["page-token"]                          # a web user: theirs alone


def test_a_lark_users_token_is_shared_and_renewed_by_a_conditional_write():
    import cognito, time as _t
    from botocore.exceptions import ClientError
    table = mock.Mock()
    fresh = {"current": "tok-a", "currentExp": int(_t.time()) + 3000}
    table.get_item.return_value = {"Item": fresh}
    with mock.patch.object(cognito, "_table", table), \
         mock.patch.object(cognito, "configured", return_value=True), \
         mock.patch.object(cognito, "_mint", side_effect=AssertionError("minted")):
        assert cognito.user_jwt("lark:ou_x") == "tok-a"          # every container, same token
        with pytest.raises(ValueError):
            cognito.user_jwt("web:s-1")
    stale = {"current": "tok-a", "currentExp": int(_t.time()) + 60}
    table.get_item.side_effect = [{"Item": stale}, {"Item": {"current": "tok-winner"}}]
    table.put_item.side_effect = ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
    with mock.patch.object(cognito, "_table", table), \
         mock.patch.object(cognito, "configured", return_value=True), \
         mock.patch.object(cognito, "_mint", return_value="x.eyJleHAiOjk5OTk5OTk5OTl9.y"):
        assert cognito.user_jwt("lark:ou_x") == "tok-winner"     # lost the race: the winner's
    kw = table.put_item.call_args.kwargs
    assert kw["Item"]["previous"] == "tok-a" and kw["ExpressionAttributeValues"] == {":seen": "tok-a"}
    table.get_item.side_effect = None
    table.get_item.return_value = {"Item": {"current": "c", "currentExp": int(_t.time()) + 100,
                                            "previous": "p", "previousExp": int(_t.time()) - 1}}
    with mock.patch.object(cognito, "_table", table):
        assert cognito.consent_tokens("lark:ou_x") == ["c"]      # an expired one is not tried
