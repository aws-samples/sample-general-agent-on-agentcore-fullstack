"""Code-execution tool tests.

What matters here is not that a sandbox runs code — the platform does that — but that the
right workspace is mounted and that everything the model writes lands in it. Both are ours
to get wrong, and either mistake is invisible in a passing conversation: files silently
vanish, or worse, belong to somebody else.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
# What a deploy injects; the modules read it at import and have no fallback.
os.environ.setdefault("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-5")
os.environ.setdefault("LARK_API_DOMAIN", "https://open.larksuite.com")

import code_tools  # noqa: E402
import delivery  # noqa: E402

_FS = "arn:aws:s3files:ap-northeast-1:1234:file-system/fs-abc"


@pytest.fixture(autouse=True)
def _deployed():
    with mock.patch.object(code_tools, "_INTERPRETER_ID", "ci-1"), \
         mock.patch.object(code_tools, "_FS_ARN", _FS), \
         mock.patch.object(code_tools, "_WORKSPACE", "/mnt/workspace"):
        yield


class _FakeBox:
    """Records what would have been sent to InvokeCodeInterpreter."""

    def __init__(self):
        self.calls = []

    def invoke(self, name, arguments):
        self.calls.append((name, arguments))
        return "ok"


def _tools(box=None):
    return {t.name: t for t in code_tools.tools_for("lark:ou_x", box or _FakeBox())}


def test_absent_not_broken_when_no_interpreter_is_deployed():
    """Same convention as search and long-term memory: the agent simply cannot run code."""
    with mock.patch.object(code_tools, "_INTERPRETER_ID", ""):
        assert code_tools.tools_for("lark:ou_x") == []
    with mock.patch.object(code_tools, "_FS_ARN", ""):
        assert code_tools.tools_for("lark:ou_x") == []


def test_every_tool_runs_in_the_mounted_workspace():
    """The sandbox's own working directory is elsewhere and dies with the session, so a
    tool that forgets to cd writes files that silently disappear between turns."""
    box = _FakeBox()
    t = _tools(box)
    t["run_code"].invoke({"code": "print(1)"})
    t["run_command"].invoke({"command": "ls"})
    t["list_files"].invoke({})
    assert len(box.calls) == 3
    for name, args in box.calls:
        payload = args.get("code") or args.get("command")
        assert "/mnt/workspace" in payload, (name, payload)


def test_the_workspace_is_never_named_by_the_model():
    """No tool takes a path outside the workspace or an actor id: the mount is chosen for
    the model, not by it."""
    for tool in _tools().values():
        fields = tool.args_schema.model_fields
        assert "actor_id" not in fields and "actorId" not in fields
        assert "mount_path" not in fields and "workspace" not in fields


def test_a_broken_sandbox_reports_instead_of_killing_the_turn():
    class _Boom:
        def invoke(self, name, arguments):
            raise RuntimeError("no session")
    out = _tools(_Boom())["run_code"].invoke({"code": "print(1)"})
    assert "could not run" in out and "RuntimeError" in out


def test_the_access_point_root_is_derived_from_the_actor_and_sanitised():
    """It becomes a filesystem path, and an actor id carries a colon. The shape also has
    to match what the previous implementation used, or existing files become unreachable."""
    assert code_tools._ap_path("lark:ou_abc") == "/users/lark_ou_abc"
    assert code_tools._ap_path("lark:../../etc") == "/users/lark_.._.._etc"


def test_an_existing_access_point_is_reused_rather_than_duplicated():
    """A second Access Point for the same user would split their files in two, and Access
    Points are a limited resource."""
    fake = mock.Mock()
    fake.list_access_points.return_value = {"accessPoints": [
        {"rootDirectory": {"path": "/users/other"}, "accessPointArn": "arn:other"},
        {"rootDirectory": {"path": "/users/lark_ou_x"}, "accessPointArn": "arn:mine"},
    ]}
    with mock.patch.object(code_tools.boto3, "client", return_value=fake):
        assert code_tools._access_point_for("lark:ou_x") == "arn:mine"
    assert fake.create_access_point.call_count == 0


def test_a_new_access_point_is_created_idempotently_and_rooted_server_side():
    fake = mock.Mock()
    fake.list_access_points.return_value = {"accessPoints": []}
    fake.create_access_point.return_value = {"accessPointArn": "arn:new"}
    with mock.patch.object(code_tools.boto3, "client", return_value=fake):
        assert code_tools._access_point_for("lark:ou_x") == "arn:new"
    kw = fake.create_access_point.call_args.kwargs
    # rootDirectory is what confines the mount, and the service enforces it.
    assert kw["rootDirectory"]["path"] == "/users/lark_ou_x"
    # clientToken makes a concurrent second call return the same Access Point.
    assert kw["clientToken"] and kw["clientToken"] == mock.ANY
    assert code_tools._access_point_for.__doc__


def test_the_session_mounts_this_actors_access_point_and_starts_lazily():
    """Starting a session costs a microVM, so it must not happen while merely building the
    tool list — only when the model actually runs something."""
    client = mock.Mock()
    client.start_code_interpreter_session.return_value = {"sessionId": "s-1"}
    with mock.patch.object(code_tools.boto3, "client", return_value=client):
        box = code_tools._Sandbox("lark:ou_x")
        assert client.start_code_interpreter_session.call_count == 0
        with mock.patch.object(code_tools, "_access_point_for", return_value="arn:mine"):
            assert box.session_id() == "s-1"
            assert box.session_id() == "s-1"          # reused, not restarted
    assert client.start_code_interpreter_session.call_count == 1
    kw = client.start_code_interpreter_session.call_args.kwargs
    mount = kw["filesystemConfigurations"][0]["s3FilesConfiguration"]
    assert mount["accessPointArn"] == "arn:mine"
    # Measured constraint: mountPath must match /mnt/[a-zA-Z0-9._-]+/?
    assert mount["mountPath"].startswith("/mnt/")


def test_closing_the_session_stops_the_sandbox():
    """A code session left running is billed until it times out, and _close_session is the
    only place that knows a cached entry is being discarded."""
    import agent_core
    src = inspect.getsource(agent_core._close_session)
    assert 'get("sandbox")' in src and ".close()" in src


def test_output_is_capped_from_the_middle_so_the_exception_survives():
    """A Python traceback puts the exception last, so trimming the tail throws away the one
    line that says what went wrong and keeps the call stack that does not."""
    body = "HEAD-MARKER\n" + "x" * 99999 + "\nValueError: the actual problem"
    client = mock.Mock()
    client.invoke_code_interpreter.return_value = {"stream": [
        {"result": {"structuredContent": {"stdout": body, "stderr": ""}}}]}
    with mock.patch.object(code_tools.boto3, "client", return_value=client):
        box = code_tools._Sandbox("lark:ou_x")
        box._session_id = "s-1"
        out = box.invoke("executeCommand", {"command": "boom"})
    assert len(out) < len(body)
    assert out.startswith("HEAD-MARKER")
    assert out.endswith("ValueError: the actual problem")
    assert "chars omitted" in out


def test_an_expired_session_is_replaced_instead_of_failing_every_later_call():
    """A code session times out well before the cached agent session does, so a held id
    goes stale mid-conversation. Retrying is safe here and only here: the error is raised
    before the code runs, so nothing executes twice."""
    client = mock.Mock()
    client.start_code_interpreter_session.side_effect = [
        {"sessionId": "s-old"}, {"sessionId": "s-new"}]
    client.invoke_code_interpreter.side_effect = [
        Exception("ValidationException: Code interpreter session s-old is not active"),
        {"stream": [{"result": {"structuredContent": {"stdout": "ok", "stderr": ""}}}]}]
    with mock.patch.object(code_tools.boto3, "client", return_value=client), \
         mock.patch.object(code_tools, "_access_point_for", return_value="arn:mine"):
        box = code_tools._Sandbox("lark:ou_x")
        assert box.invoke("executeCommand", {"command": "ls"}) == "ok"
    assert client.start_code_interpreter_session.call_count == 2
    assert client.invoke_code_interpreter.call_args_list[-1].kwargs["sessionId"] == "s-new"


def test_any_other_failure_is_not_retried():
    """Retrying a call that may already have run would duplicate a write."""
    client = mock.Mock()
    client.start_code_interpreter_session.return_value = {"sessionId": "s-1"}
    client.invoke_code_interpreter.side_effect = Exception("ThrottlingException: slow down")
    with mock.patch.object(code_tools.boto3, "client", return_value=client), \
         mock.patch.object(code_tools, "_access_point_for", return_value="arn:mine"):
        box = code_tools._Sandbox("lark:ou_x")
        with pytest.raises(Exception, match="Throttling"):
            box.invoke("executeCommand", {"command": "ls"})
    assert client.invoke_code_interpreter.call_count == 1


def test_a_failure_with_no_output_still_says_something():
    """"" back to the model reads as success. It is not."""
    client = mock.Mock()
    client.invoke_code_interpreter.return_value = {"stream": [
        {"result": {"isError": True, "content": [], "structuredContent": {}}}]}
    with mock.patch.object(code_tools.boto3, "client", return_value=client):
        box = code_tools._Sandbox("lark:ou_x")
        box._session_id = "s-1"
        assert "failed" in box.invoke("executeCode", {"language": "python", "code": ""})


# --- Delivering a file out of the workspace --------------------------------
# Two things are ours to get wrong: which surface a file is sent to, and the upload/message
# type pairing Lark enforces. The first one bit already — a target stored on the cached
# session sent a web user's file to whichever Lark chat was served last.


class _DeliverBox(_FakeBox):
    def __init__(self, cover: bool = True, size: int = 10):
        super().__init__()
        self._cover = cover
        self._size = size

    def invoke(self, name, arguments):
        super().invoke(name, arguments)
        return "ok" if self._cover else "(no output)"

    def file_size(self, path):
        return self._size

    def read_bytes(self, path):
        return b"\0" * self._size


@pytest.fixture
def _lark():
    with mock.patch.object(delivery.lark_notify, "upload_file",
                           return_value="file_k") as up, \
         mock.patch.object(delivery.lark_notify, "upload_image",
                           return_value="img_k"), \
         mock.patch.object(delivery.lark_notify, "send_media",
                           return_value=True) as media, \
         mock.patch.object(delivery.lark_notify, "send_file",
                           return_value=True) as attach:
        yield {"upload": up, "media": media, "file": attach}


def _deliver(box, channel, path="report.pdf"):
    """Call the tool the way a run does: the surface arrives in the config, not the args."""
    tools = _tools(box)
    return tools["deliver_file"].func(
        path, {"configurable": {"delivery": channel}})


def test_a_path_outside_the_workspace_is_refused_before_any_upload(_lark):
    for bad in ("/etc/passwd", "../../etc/passwd"):
        assert "without .." in _deliver(_DeliverBox(), delivery.LarkDelivery("oc_1"), bad)
    assert _lark["upload"].call_count == 0


def test_an_mp4_is_sent_as_media_so_it_plays(_lark):
    out = _deliver(_DeliverBox(cover=True), delivery.LarkDelivery("oc_1"), "clip.mp4")
    assert "plays" in out
    assert _lark["upload"].call_args.args[2] == "mp4"
    assert _lark["media"].called and not _lark["file"].called


def test_an_mp4_without_a_cover_is_re_uploaded_as_stream(_lark):
    """Lark refuses to send an mp4-typed upload as `file`, so the type has to change with
    the message. Measured: "The type of file upload does not match the type of message"."""
    out = _deliver(_DeliverBox(cover=False), delivery.LarkDelivery("oc_1"), "clip.mp4")
    assert "attachment" in out
    assert _lark["upload"].call_args.args[2] == "stream"
    assert _lark["file"].called and not _lark["media"].called


def test_the_file_goes_where_this_run_says_not_where_the_last_one_did(_lark):
    """The session is shared across surfaces, so the target has to travel with the run."""
    box = _DeliverBox()
    _deliver(box, delivery.LarkDelivery("oc_group"))
    assert _lark["file"].call_args.args[0] == "oc_group"
    _deliver(box, delivery.LarkDelivery("ou_x"))
    assert _lark["file"].call_args.args[0] == "ou_x"


def test_a_surface_that_cannot_carry_files_says_so_instead_of_using_lark(_lark):
    out = _deliver(_DeliverBox(), delivery.NoDelivery("the web page"))
    assert "the web page" in out and "Do not claim it was sent" in out
    assert not _lark["upload"].called and not _lark["file"].called


def test_a_run_with_no_surface_refuses_rather_than_guessing(_lark):
    """Forgetting to set the surface must cost a refusal, never a delivery to the wrong
    person — that is the whole reason the target is not kept on the session."""
    tools = _tools(_DeliverBox())
    out = tools["deliver_file"].func("report.pdf", {"configurable": {}})
    assert "cannot receive files" in out
    assert not _lark["upload"].called


def test_an_oversized_file_is_refused_before_it_is_read(_lark):
    box = _DeliverBox(size=31 * 1000 * 1000)
    out = _deliver(box, delivery.LarkDelivery("oc_1"), "big.mp4")
    assert "30 MB" in out
    assert not any("base64" in (a or {}).get("command", "") for _, a in box.calls)
    assert not _lark["upload"].called


def test_a_missing_file_is_named_rather_than_sent_as_empty(_lark):
    class _Missing(_DeliverBox):
        def file_size(self, path):
            return None

    out = _deliver(_Missing(), delivery.LarkDelivery("oc_1"), "nope.txt")
    assert "no nope.txt" in out
    assert not _lark["upload"].called


def test_larks_own_reason_reaches_the_model(_lark):
    """A bare failure reads as "try again". "The bot is not in that chat" does not — and
    only Lark knows which it is."""
    _lark["file"].side_effect = delivery.lark_notify.LarkError(
        "Lark refused the message: bot is not in the chat")
    out = _deliver(_DeliverBox(), delivery.LarkDelivery("oc_1"))
    assert "was not sent" in out and "bot is not in the chat" in out


def test_a_video_lark_will_not_take_a_cover_for_still_arrives(_lark):
    """A rejected cover says nothing about the video, so it falls back to an attachment."""
    with mock.patch.object(delivery.lark_notify, "upload_image",
                           side_effect=delivery.lark_notify.LarkError("bad cover")):
        out = _deliver(_DeliverBox(cover=True), delivery.LarkDelivery("oc_1"), "clip.mp4")
    assert "attachment" in out
    assert _lark["file"].called and not _lark["media"].called


def test_the_guard_does_not_hide_the_runs_config_from_the_tool(_lark):
    """The regression that made every send refuse: LangChain injects the run config by
    reading the callable's annotations, so the telemetry wrapper has to keep them. A bare
    `*a, **kw` wrapper leaves deliver_file with no target, and it then refuses silently —
    the tool returns a sentence, telemetry says ok, and nothing looks broken."""
    tools = {t.name: t for t in code_tools.tools_for("lark:ou_x", _DeliverBox())}
    assert "config" not in tools["deliver_file"].args, "config must stay out of the schema"
    # Invoked the way a run does, through the tool rather than the raw function.
    out = tools["deliver_file"].invoke(
        {"path": "report.pdf"},
        config={"configurable": {"delivery": delivery.LarkDelivery("oc_real")}})
    assert "attachment" in out
    assert _lark["file"].call_args.args[0] == "oc_real"


def test_save_attachment_takes_its_handle_from_the_run_not_from_the_model(_lark):
    """A tenant token can read every chat this bot is in, so a handle the model supplied
    would let a prompt injection fetch a stranger's file. The run config is the only
    source, and the model only ever names a position."""
    box = _DeliverBox()
    written = {}
    box.write_bytes = lambda name, data: written.update({"name": name, "size": len(data)})
    tools = {t.name: t for t in code_tools.tools_for("lark:ou_x", box)}
    assert list(tools["save_attachment"].args) == ["which"], \
        "no handle may come from the model"

    code_tools.images.add_to_turn("t-att", [{"kind": "file", "key": "file_real",
                                             "name": "report.pdf", "message_id": "om_real"}])
    with mock.patch.object(code_tools.images.lark_notify, "fetch_message_resource",
                           return_value=b"PK\x03\x04") as fetched:
        out = tools["save_attachment"].invoke(
            {"which": 1}, config={"configurable": {"thread_id": "t-att"}})
    code_tools.images.end_turn("t-att")
    assert "report.pdf" in out and written["name"] == "report.pdf"
    # The kind has to travel with the handle: Lark answers 500 if it does not match.
    assert fetched.call_args.args == ("om_real", "file_real", "file")

    # Nothing registered for this thread means nothing to fetch, whatever the model asks.
    out = tools["save_attachment"].invoke({"which": 1},
                                          config={"configurable": {"thread_id": "t-other"}})
    assert "Nothing came with this message" in out


def test_a_picture_keeps_a_usable_name_when_the_sender_gave_none(_lark):
    box = _DeliverBox()
    written = {}
    box.write_bytes = lambda name, data: written.update({"name": name})
    tools = {t.name: t for t in code_tools.tools_for("lark:ou_x", box)}
    with mock.patch.object(code_tools.images, "blocks_for", return_value=[]):
        code_tools.images.add_to_turn("t-pic", [{"kind": "image", "key": "img_1",
                                                 "message_id": "om_1"}])
    with mock.patch.object(code_tools.images.lark_notify, "fetch_message_resource",
                           return_value=b"\xff\xd8ab"):
        tools["save_attachment"].invoke({"which": 1},
                                        config={"configurable": {"thread_id": "t-pic"}})
    code_tools.images.end_turn("t-pic")
    assert written["name"] == "image-1.jpg"


def test_an_edited_image_is_sent_so_it_shows_rather_than_downloads(_lark):
    with mock.patch.object(delivery.lark_notify, "send_image", return_value=True) as shown:
        out = _deliver(_DeliverBox(), delivery.LarkDelivery("oc_1"), "cropped.png")
    assert "shows there" in out
    assert shown.called and not _lark["file"].called


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_the_access_point_token_is_per_file_system_not_per_actor():
    """S3 Files dedupes clientToken across the account and region; one keyed on the actor
    alone made a second deployment's create collide with the first's Access Point."""
    tokens = []
    fake = mock.Mock()
    fake.list_access_points.return_value = {"accessPoints": []}
    fake.create_access_point.side_effect = lambda **kw: tokens.append(kw["clientToken"]) or {
        "accessPointArn": "arn:ap"}
    with mock.patch.object(code_tools.boto3, "client", return_value=fake):
        for fs in ("arn:aws:s3files:r:1:file-system/fs-a", "arn:aws:s3files:r:1:file-system/fs-b"):
            with mock.patch.object(code_tools, "_FS_ARN", fs):
                code_tools._access_point_for("lark:ou_same")
        with mock.patch.object(code_tools, "_FS_ARN", "arn:aws:s3files:r:1:file-system/fs-a"):
            code_tools._access_point_for("lark:ou_same")
    assert tokens[0] != tokens[1], "same user, two file systems: tokens must differ"
    assert tokens[0] == tokens[2], "same user, same file system: the retry must be idempotent"
