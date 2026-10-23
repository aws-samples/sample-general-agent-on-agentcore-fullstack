"""Code generation and execution, with a workspace that belongs to one user.

Three tools — run_code, run_command, list_files — over an AgentCore Code Interpreter
session that mounts that user's own S3 Files Access Point at /mnt/workspace. So generated
code has somewhere durable to write, and a script from last week still runs against the
files it produced.

`actor_id` is bound when the tools are built, never taken as an argument: it decides which
Access Point gets mounted, and that is the only thing standing between two users' files.
The isolation itself is the platform's — one session mounts exactly one Access Point, at
the microVM boundary — but *choosing* it is ours, so nothing here accepts it from a caller.

Measured constraints that shape this file, all in docs/agentcore-behavior.md: the mount
needs VPC network mode (SANDBOX refuses it); `mountPath` must match
`/mnt/[a-zA-Z0-9._-]+/?`; `readFiles`/`writeFiles` cannot reach the mount, so files are
read and written by running code; write-through to S3 is asynchronous.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re

import lark_notify
import telemetry

import boto3
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger("agent.code")

_REGION = os.environ.get("AWS_REGION", "us-west-2")
_INTERPRETER_ID = os.environ.get("CODE_INTERPRETER_ID", "")
_FS_ARN = os.environ.get("FILES_FS_ARN", "")
_WORKSPACE = os.environ.get("WORKSPACE_PATH", "/mnt/workspace")
# Long enough to outlive a turn that thinks, short enough that an abandoned session is not
# billed for hours. The platform caps at 28,800.
_SESSION_TIMEOUT = int(os.environ.get("CODE_SESSION_TIMEOUT_SECONDS", "1800"))
# One tool result goes into the model's context, so a runaway print must not fill it. Big
# enough for a real traceback or a printed DataFrame; ~3k tokens at the ceiling.
_MAX_OUTPUT = int(os.environ.get("CODE_MAX_OUTPUT_CHARS", "12000"))


def available() -> bool:
    return bool(_INTERPRETER_ID and _FS_ARN)


def _fs_id() -> str:
    return _FS_ARN.rsplit("/", 1)[-1]


def _ap_path(actor_id: str) -> str:
    """The Access Point's server-side root for this actor.

    Sanitised because it becomes a path: an actor id is `lark:ou_…` and the colon has no
    business in one. Same shape the previous implementation used, so existing files stay
    reachable."""
    return "/users/" + re.sub(r"[^A-Za-z0-9_.-]", "_", actor_id)


def _access_point_for(actor_id: str) -> str:
    """Get or create this user's Access Point, returning its ARN.

    Idempotent by `clientToken`: a stable hash of the actor, so a second call returns the
    same Access Point instead of creating a rival one. `rootDirectory` is fixed here and
    enforced server-side, which is what keeps a mount from seeing another user's prefix
    however the sandbox behaves."""
    s3files = boto3.client("s3files", region_name=_REGION)
    path = _ap_path(actor_id)
    for ap in s3files.list_access_points(fileSystemId=_fs_id()).get("accessPoints", []):
        if (ap.get("rootDirectory") or {}).get("path") == path:
            return ap["accessPointArn"]
    r = s3files.create_access_point(
        fileSystemId=_fs_id(),
        clientToken=hashlib.sha256(actor_id.encode()).hexdigest()[:64],
        tags=[{"key": "actor", "value": actor_id}],
        # 1000/1000 rather than root: files land owned by an unprivileged uid, and the
        # sandbox sees only this subtree anyway.
        posixUser={"uid": 1000, "gid": 1000},
        rootDirectory={"path": path,
                       "creationPermissions": {"ownerUid": 1000, "ownerGid": 1000,
                                               "permissions": "0700"}})
    log.info("created workspace access point for %s at %s", actor_id, path)
    return r["accessPointArn"]


class _Sandbox:
    """One Code Interpreter session, started on first use and reused for the rest of it.

    Held by the caller (agent_core's session cache) rather than module state, so its
    lifetime matches the conversation's and two users can never share one."""

    def __init__(self, actor_id: str) -> None:
        self.actor_id = actor_id
        self._client = boto3.client("bedrock-agentcore", region_name=_REGION)
        self._session_id = ""

    def session_id(self) -> str:
        if self._session_id:
            return self._session_id
        mount = {"s3FilesConfiguration": {
            "accessPointArn": _access_point_for(self.actor_id),
            "fileSystemArn": _FS_ARN,
            "mountPath": _WORKSPACE,
        }}
        r = self._client.start_code_interpreter_session(
            codeInterpreterIdentifier=_INTERPRETER_ID,
            name="agent-" + hashlib.sha256(self.actor_id.encode()).hexdigest()[:16],
            sessionTimeoutSeconds=_SESSION_TIMEOUT,
            filesystemConfigurations=[mount])
        self._session_id = r["sessionId"]
        log.info("code session %s for %s", self._session_id, self.actor_id)
        return self._session_id

    def close(self) -> None:
        if not self._session_id:
            return
        try:
            self._client.stop_code_interpreter_session(
                codeInterpreterIdentifier=_INTERPRETER_ID, sessionId=self._session_id)
        except Exception:  # noqa: BLE001 — the session expires on its own anyway
            log.warning("could not stop code session %s", self._session_id, exc_info=True)
        self._session_id = ""

    def read_bytes(self, path: str) -> bytes:
        """The contents of a workspace file, as bytes, without passing through the model.

        `readFiles` is scoped to the sandbox's own directory and rejects the mount path as
        traversal, so the file is base64'd across first — which also sidesteps having to
        know how the API would encode raw binary. Measured: a 22,500-byte mp4 came back
        byte-identical, and the 30,000-char payload was not truncated, because this reply
        goes to this process rather than into a tool result."""
        staged = hashlib.sha256(path.encode()).hexdigest()[:16] + ".b64"
        out = self.invoke("executeCommand", {
            "command": f"base64 -w0 -- {json.dumps(_WORKSPACE + '/' + path)} > {staged}"})
        if out not in ("(no output)", ""):
            raise FileNotFoundError(out)
        r = self._client.invoke_code_interpreter(
            codeInterpreterIdentifier=_INTERPRETER_ID, sessionId=self.session_id(),
            name="readFiles", arguments={"paths": [staged]})
        encoded = ""
        for event in r["stream"]:
            result = event.get("result") or {}
            if result.get("isError"):
                raise OSError(" ".join(c.get("text", "") for c in result.get("content", [])))
            for c in result.get("content", []):
                resource = c.get("resource") or {}
                encoded += resource.get("text") or resource.get("blob") or ""
        self.invoke("executeCommand", {"command": f"rm -f -- {staged}"})
        return base64.b64decode(encoded)

    def invoke(self, name: str, arguments: dict) -> str:
        """One InvokeCodeInterpreter call, flattened to what the model should read."""
        try:
            r = self._client.invoke_code_interpreter(
                codeInterpreterIdentifier=_INTERPRETER_ID, sessionId=self.session_id(),
                name=name, arguments=arguments)
        except Exception as e:  # noqa: BLE001 — narrowed by the message check below
            # A code session times out (30 min) well before the cached agent session does
            # (50), so a held id goes stale mid-conversation and every later call fails.
            # Safe to retry with a fresh session: the error is raised before the code runs
            # (`ValidationException: … session … is not active`), so nothing ran twice.
            if "is not active" not in str(e):
                raise
            log.info("code session %s expired; starting a new one", self._session_id)
            telemetry.count("code_session_restarted")
            self._session_id = ""
            r = self._client.invoke_code_interpreter(
                codeInterpreterIdentifier=_INTERPRETER_ID, sessionId=self.session_id(),
                name=name, arguments=arguments)
        chunks, failed = [], False
        for event in r["stream"]:
            result = event.get("result") or {}
            failed = failed or bool(result.get("isError"))
            structured = result.get("structuredContent") or {}
            text = (structured.get("stdout", "") + structured.get("stderr", "")).strip()
            if not text:
                text = " ".join(c.get("text", "") for c in result.get("content", [])).strip()
            if text:
                chunks.append(text)
            if structured.get("exitCode"):
                chunks.append(f"[exit {structured['exitCode']}]")
        out = "\n".join(chunks).strip() or ("(failed, no output)" if failed else "(no output)")
        return _clamp(out)


def _clamp(out: str) -> str:
    """Trim from the middle, not the end. A Python traceback puts the exception — the one
    line that says what went wrong — last, so keeping only the head throws away the answer
    and leaves the call stack."""
    if len(out) <= _MAX_OUTPUT:
        return out
    head = int(_MAX_OUTPUT * 0.6)
    tail = _MAX_OUTPUT - head
    dropped = len(out) - _MAX_OUTPUT
    return f"{out[:head]}\n…({dropped} chars omitted)…\n{out[-tail:]}"


class _RunCodeArgs(BaseModel):
    code: str = Field(description="The complete program to run. It starts in the "
                                  "workspace, so plain relative paths land there.")
    language: str = Field(default="python",
                          description="python, javascript or typescript.")


class _RunCommandArgs(BaseModel):
    command: str = Field(description="A shell command, run in the workspace.")


class _ListFilesArgs(BaseModel):
    path: str = Field(default=".", description="A path relative to the workspace root.")


class _DeliverFileArgs(BaseModel):
    path: str = Field(description="The file to send, relative to the workspace root.")


# Lark's own ceiling for an uploaded file. Checked here so an oversized send fails with a
# sentence the model can act on rather than an API error it has to interpret.
_MAX_DELIVER_BYTES = 30 * 1000 * 1000
# Only mp4 plays inline; everything else is an attachment. Lark names a handful of types
# and treats the rest as `stream`.
_NAMED_TYPES = {"mp4": "mp4", "pdf": "pdf", "doc": "doc", "docx": "doc", "xls": "xls",
                "xlsx": "xls", "ppt": "ppt", "pptx": "ppt", "opus": "opus"}


def tools_for(actor_id: str, sandbox: _Sandbox | None = None,
              deliver_to: dict | None = None) -> list:
    """The three code tools bound to this user, or [] when code execution is not deployed.

    Absent-not-broken, like search and long-term memory: without an interpreter the agent
    simply cannot run code."""
    if not available():
        return []
    box = sandbox or _Sandbox(actor_id)
    # Falls back to a DM only when the turn has no chat — the web page and A2A, where
    # lark_notify._rid_type reads an `ou_` id as "send to the person".
    open_id = actor_id.split(":", 1)[1] if ":" in actor_id else actor_id

    def _target() -> str:
        return (deliver_to or {}).get("chat_id") or open_id

    def _guard(fn):
        # Counted by name and outcome: "did a tool actually run" is the one question a
        # plausible-looking reply cannot answer, and it cost this project a long evening.
        def run(*a, **kw):
            try:
                out = fn(*a, **kw)
                telemetry.count("tool_call", tool=fn.__name__, outcome="ok")
                return out
            except Exception as e:  # noqa: BLE001 — a broken sandbox must not kill the turn
                telemetry.count("tool_call", tool=fn.__name__, outcome=type(e).__name__)
                log.exception("code tool failed for %s", actor_id)
                return f"The sandbox could not run that ({type(e).__name__})."
        run.__name__ = fn.__name__
        return run

    # Every call is prefixed with a cd: the sandbox's own working directory is elsewhere
    # (/opt/amazon/genesis1p-tools/var) and anything written there dies with the session.
    # The model is not told about the mount — it just has a working directory that persists.
    def run_code(code: str, language: str = "python") -> str:
        if language == "python":
            return box.invoke("executeCode", {
                "language": "python", "code": f"import os\nos.chdir({_WORKSPACE!r})\n" + code})
        # JS/TS cannot go through executeCode: it runs them under Deno with no permissions
        # and no way to pass a flag, so the workspace is unreachable (`NotCapable: Requires
        # read access`) and `require` is undefined. Writing the file and running node gives
        # both. Measured; see docs/agentcore-behavior.md.
        name = "_run.ts" if language == "typescript" else "_run.js"
        runner = f"deno run --allow-all {name}" if language == "typescript" else f"node {name}"
        return box.invoke("executeCommand", {
            "command": f"cd {_WORKSPACE} && cat > {name} <<'AGENT_EOF'\n{code}\nAGENT_EOF\n{runner}"})

    def run_command(command: str) -> str:
        return box.invoke("executeCommand", {"command": f"cd {_WORKSPACE} && {command}"})

    def list_files(path: str = ".") -> str:
        return box.invoke("executeCommand",
                          {"command": f"cd {_WORKSPACE} && ls -la -- {json.dumps(path)}"})

    def deliver_file(path: str) -> str:
        # The workspace is this user's own mount, so a traversal reaches no one else's
        # files — but it does reach the sandbox's system paths, which are not ours to send.
        if path.startswith("/") or ".." in path.split("/"):
            return "Give a path inside the workspace, without .. or a leading /."
        name = os.path.basename(path)
        suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        data = box.read_bytes(path)
        if not data:
            return f"{name} is empty, so there is nothing to send."
        if len(data) > _MAX_DELIVER_BYTES:
            return (f"{name} is {len(data) / 1e6:.1f} MB and Lark accepts 30 MB. "
                    "Compress it or split it, then send again.")
        # A video plays inline only as `media`, which Lark will not render without a cover,
        # so a still frame is cut for it. The upload's file_type and the message type have
        # to agree — an mp4-typed upload refuses to be sent as `file` ("The type of file
        # upload does not match the type of message being sent") — hence the re-upload as
        # `stream` when there is no cover to send it with.
        if suffix == "mp4":
            cover = box.invoke("executeCommand", {"command": (
                f"cd {_WORKSPACE} && FF=$(python3 -c 'import imageio_ffmpeg as f;"
                f" print(f.get_ffmpeg_exe())') && \"$FF\" -y -i {json.dumps(name)}"
                " -frames:v 1 -f image2 _cover.jpg >/dev/null 2>&1 && echo ok")})
            if "ok" in cover:
                image = lark_notify.upload_image(box.read_bytes("_cover.jpg"))
                key = image and lark_notify.upload_file(name, data, "mp4")
                if key and lark_notify.send_media(_target(), key, image):
                    return f"Sent {name} to the chat; it plays there."
            log.info("no cover for %s, sending it as an attachment", name)
        key = lark_notify.upload_file(name, data, _NAMED_TYPES.get(suffix, "stream")
                                      if suffix != "mp4" else "stream")
        if not key:
            return f"Lark would not accept {name}."
        if lark_notify.send_file(_target(), key):
            return f"Sent {name} to the chat as an attachment."
        return f"{name} uploaded but the message would not send."

    return [
        StructuredTool.from_function(
            func=_guard(run_code), name="run_code",
            description=(  # nosemgrep: string-concat-in-list — one wrapped description, not two items
                "Run code and get its output. Use this for anything computed rather than "
                "recalled — data work, file conversion, checking your own arithmetic. "
                "Files you write persist between turns and across days, so build on them "
                "instead of regenerating them. Nothing can be installed; Python already has "
                "pandas, matplotlib, scikit-learn, opencv, moviepy and pydub (audio/video), "
                "python-docx, python-pptx, openpyxl, pypdf and reportlab."),
            args_schema=_RunCodeArgs),
        StructuredTool.from_function(
            func=_guard(run_command), name="run_command",
            description=(  # nosemgrep: string-concat-in-list — one wrapped description, not two items
                "Run one shell command in the workspace. Use it for the things a shell is "
                "better at than a program — inspecting files, moving them, checking sizes. "
                "No internet and no root, so nothing can be installed. Available: jq, "
                "sqlite3, gcc, make, ImageMagick `convert`. ffmpeg is not on PATH; its path "
                "is `python3 -c 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())'`. "
                "Not available: git, pandoc, LibreOffice, tesseract, any browser."),
            args_schema=_RunCommandArgs),
        StructuredTool.from_function(
            func=_guard(list_files), name="list_files",
            description=(  # nosemgrep: string-concat-in-list — one wrapped description, not two items
                "List what is already in the workspace. Worth doing before you claim a "
                "file does or does not exist, since it persists from earlier conversations."),
            args_schema=_ListFilesArgs),
        StructuredTool.from_function(
            func=_guard(deliver_file), name="deliver_file",
            description=(  # nosemgrep: string-concat-in-list — one wrapped description, not two items
                "Send a file from the workspace to the person you are talking to, so they "
                "can open it. The workspace is not visible to them, so a file they asked "
                "for is not delivered until you call this — say what you made, then send "
                "it. An mp4 plays in the chat; anything else arrives as an attachment. "
                "Limit 30 MB."),
            args_schema=_DeliverFileArgs),
    ]
