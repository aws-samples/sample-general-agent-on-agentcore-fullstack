"""Send a message back to a Lark chat from inside the agent.

Long turns can't be answered synchronously: the caller's request/response window
is bounded (15 min for InvokeAgentRuntime, less for a Lambda), while a real task —
research something, write several documents — may take longer. So the agent
accepts the work, returns immediately, and pushes the result here when it's done.

This uses the app's own tenant token (not the user's), because it is the bot
speaking to the user. The per-user Lark token is only ever used for reading and
writing that user's own data, via lark_3lo.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request

import boto3

log = logging.getLogger("agent.notify")

_REGION = os.environ.get("AWS_REGION", "us-west-2")
_API_DOMAIN = os.environ["LARK_API_DOMAIN"].rstrip("/")  # injected at deploy, from .env
_SECRET_ID = os.environ.get("LARK_SECRET_ID", "")
_MAX_TEXT_LEN = 4000  # Lark rejects longer bodies; split rather than truncate

_secrets = boto3.client("secretsmanager", region_name=_REGION)
_token: tuple[str, float] | None = None  # (token, expires_at)
_lock = threading.Lock()


def _call(method: str, url: str, body: dict, bearer: str = "") -> dict:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310 # nosemgrep: dynamic-urllib-use-detected — const scheme+host
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode() or "{}")
        except ValueError:          # a gateway's HTML error page
            return {"code": e.code, "msg": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001 — every caller reads the code; none expects a raise
        return {"code": -1, "msg": f"{type(e).__name__}: {e}"}


def _post(url: str, body: dict, bearer: str = "") -> dict:
    return _call("POST", url, body, bearer)


def _tenant_token() -> str:
    """Cached tenant token. Lark issues these for ~2h; refresh a little early."""
    global _token
    with _lock:
        if _token and _token[1] > time.time():
            return _token[0]
        if not _SECRET_ID:
            log.warning("LARK_SECRET_ID unset — cannot notify the chat")
            return ""
        secret = json.loads(_secrets.get_secret_value(SecretId=_SECRET_ID)["SecretString"])
        body = _post(f"{_API_DOMAIN}/open-apis/auth/v3/tenant_access_token/internal",
                     {"app_id": secret["appId"], "app_secret": secret["appSecret"]})
        tok = body.get("tenant_access_token", "")
        if not tok:
            log.error("tenant token request failed: %s", body.get("msg", body))  # nosemgrep: python-logger-credential-disclosure — logs ids and codes, never values
            return ""
        _token = (tok, time.time() + int(body.get("expire", 7200)) - 300)
        return tok


def _rid_type(receive_id: str) -> str:
    """`ou_` means an open_id — a DM to that person rather than a chat. Event-driven
    turns (an approval task landing) have no chat to reply in, only the person."""
    return "open_id" if receive_id.startswith("ou_") else "chat_id"


def send_text(chat_id: str, text: str) -> bool:
    """Post text to a chat, splitting to stay under Lark's length limit."""
    tok = _tenant_token()
    if not (tok and chat_id):
        return False
    url = f"{_API_DOMAIN}/open-apis/im/v1/messages?receive_id_type={_rid_type(chat_id)}"
    ok = True
    for i in range(0, len(text) or 1, _MAX_TEXT_LEN):
        chunk = text[i:i + _MAX_TEXT_LEN]
        body = _post(url, {"receive_id": chat_id, "msg_type": "text",
                           "content": json.dumps({"text": chunk})}, bearer=tok)
        if body.get("code") not in (0, "0"):
            log.error("send to %s failed: %s", chat_id, body.get("msg", body))
            ok = False
    return ok


def send_link(chat_id: str, text: str, link_text: str, url: str) -> bool:
    """Post text + a clickable hyperlink, so the user sees "点击授权" rather than a
    raw consent URL — mirrors the router's send_link_message. Used from the async
    turn when a tool hits an authorization wall, so the prompt matches what the
    router shows on the synchronous path."""
    tok = _tenant_token()
    if not (tok and chat_id):
        return False
    post = {"zh_cn": {"content": [[
        {"tag": "text", "text": (text + " ") if text else ""},
        {"tag": "a", "text": link_text, "href": url},
    ]]}}
    body = _post(f"{_API_DOMAIN}/open-apis/im/v1/messages?receive_id_type={_rid_type(chat_id)}",
                 {"receive_id": chat_id, "msg_type": "post",
                  "content": json.dumps(post)}, bearer=tok)
    if body.get("code") not in (0, "0"):
        log.error("send_link to %s failed: %s", chat_id, body.get("msg", body))
        return False
    return True


class LarkError(RuntimeError):
    """A refusal from Lark, carrying its own words.

    Raised by the file-delivery calls only. The reply path keeps returning booleans: a
    failed reply is logged and degraded, while a failed delivery has to reach the model,
    which otherwise tells the user a file was sent when Lark said why it was not."""


def _reason(body: dict) -> str:
    return str(body.get("msg") or body or "no response")


def _upload(path: str, fields: dict, file_name: str, data: bytes,
            part: str = "file", timeout: int = 120) -> dict:
    """One multipart/form-data POST, built by hand because this module speaks urllib.

    Lark's upload endpoints take the file as a form part, so neither `_call` nor a JSON
    body can reach them — this is why uploading cannot go through the `lark_api` tool.
    `part` is the field name, which differs per endpoint: `file` for im/v1/files, `image`
    for im/v1/images, and getting it wrong answers `Invalid request param` (measured)."""
    tok = _tenant_token()
    if not tok:
        return {}
    boundary = "----agent" + os.urandom(8).hex()
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                     f"{value}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{part}\"; "
                 f"filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                 .encode())
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{_API_DOMAIN}{path}", data=b"".join(parts), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {tok}"})
    try:
        # Longer than the 10 s the JSON calls use: this one carries the whole file.
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 # nosemgrep: dynamic-urllib-use-detected — const scheme+host
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}")


def upload_file(file_name: str, data: bytes, file_type: str = "stream",
                duration_ms: int = 0) -> str:
    """Upload a file for sending, returning its file_key. `stream` covers any type Lark
    does not name; `mp4` is required for a video to play inline rather than download."""
    fields = {"file_type": file_type, "file_name": file_name}
    if duration_ms:
        fields["duration"] = str(duration_ms)
    body = _upload("/open-apis/im/v1/files", fields, file_name, data)
    key = (body.get("data") or {}).get("file_key", "")
    if not key:
        raise LarkError(f"Lark would not accept {file_name}: {_reason(body)}")
    return key


def upload_image(data: bytes, name: str = "cover.jpg", timeout: int = 120) -> str:
    """Upload an image, returning its image_key — also how a video gets its cover, which
    Lark requires for `media` and offers no default for."""
    body = _upload("/open-apis/im/v1/images", {"image_type": "message"}, name, data,
                   part="image", timeout=timeout)
    key = (body.get("data") or {}).get("image_key", "")
    if not key:
        raise LarkError(f"Lark would not accept the cover image: {_reason(body)}")
    return key


def fetch_message_resource(message_id: str, key: str, kind: str = "image") -> bytes:
    """The bytes of something somebody sent, read back from the message that carried it.

    Lark keeps the original, so nothing here has to store one. Two things to know: the
    `im/v1/images/{key}` endpoint serves only keys this app uploaded and fails for
    anything received, so a resource has to be addressed through its message; and `kind`
    must match what the key is — asking for `image` on a file key answers 500."""
    tok = _tenant_token()
    if not tok:
        raise LarkError("no tenant token, so the attachment cannot be read")
    url = (f"{_API_DOMAIN}/open-apis/im/v1/messages/{message_id}"
           f"/resources/{key}?type={kind}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310 # nosemgrep: dynamic-urllib-use-detected — const scheme+host
            return r.read()
    except urllib.error.HTTPError as e:
        # Recalled and deleted messages both land here, and the user can act on that.
        raise LarkError(f"Lark would not return the attachment: {e.code}") from e


def send_image(chat_id: str, image_key: str) -> bool:
    """Post an image so it renders in the chat rather than arriving as a download."""
    return _send_media_message(chat_id, "image", {"image_key": image_key})


def send_file(chat_id: str, file_key: str) -> bool:
    """Post an uploaded file as an attachment."""
    return _send_media_message(chat_id, "file", {"file_key": file_key})


def send_media(chat_id: str, file_key: str, image_key: str) -> bool:
    """Post an uploaded video with its cover, so it plays in the chat."""
    return _send_media_message(chat_id, "media",
                               {"file_key": file_key, "image_key": image_key})


def _send_media_message(chat_id: str, msg_type: str, content: dict) -> bool:
    tok = _tenant_token()
    if not tok:
        raise LarkError("no tenant token, so nothing can be sent")
    if not chat_id:
        raise LarkError("this turn carries no chat to send to")
    body = _post(f"{_API_DOMAIN}/open-apis/im/v1/messages?receive_id_type={_rid_type(chat_id)}",
                 {"receive_id": chat_id, "msg_type": msg_type,
                  "content": json.dumps(content)}, bearer=tok)
    if body.get("code") not in (0, "0"):
        # Lark's own sentence, e.g. the bot not being a member of the target chat — the
        # one thing that tells the model what to say next.
        raise LarkError(f"Lark refused the message: {_reason(body)}")
    return True


def add_reaction(message_id: str, emoji: str = "OnIt") -> str:
    """React to the user's message so they know it was received. Returns the
    reaction_id needed to remove it, or "" on failure.

    Faster than any message: one call, no card entity to create. Uses the tenant
    token (the bot reacting as itself) and needs only `im:message`, which the bot
    already holds. A bot may react to a message it didn't send, but may only delete
    reactions it added itself."""
    tok = _tenant_token()
    if not (tok and message_id):
        return ""
    body = _post(f"{_API_DOMAIN}/open-apis/im/v1/messages/{message_id}/reactions",
                 {"reaction_type": {"emoji_type": emoji}}, bearer=tok)
    if body.get("code") not in (0, "0"):
        log.warning("add reaction failed: %s", body.get("msg", body))
        return ""
    return body.get("data", {}).get("reaction_id", "")


def remove_reaction(message_id: str, reaction_id: str) -> bool:
    """Drop the in-progress marker once the turn is done. Best-effort: if the
    container dies mid-turn nobody removes it, which is cosmetic only."""
    if not (message_id and reaction_id):
        return False
    tok = _tenant_token()
    if not tok:
        return False
    body = _call("DELETE",
                 f"{_API_DOMAIN}/open-apis/im/v1/messages/{message_id}"
                 f"/reactions/{reaction_id}", {}, bearer=tok)
    if body.get("code") not in (0, "0"):
        log.warning("remove reaction failed: %s", body.get("msg", body))
        return False
    return True


# --------------------------- CardKit streaming ------------------------------
# A turn is silent for several seconds before the first token (session assembly +
# MCP handshake + model latency, measured ~7.5 s), which reads as "did my message
# even send?". A streaming card fills that gap: post a placeholder at once, then
# type the answer into it. CardKit's streaming mode is the supported mechanism —
# its updates don't count against the QPS limit, unlike editing a message.

# Below this a step is noise — the model re-reading a tool result before it writes.
_MIN_STEP_SECONDS = 0.5
# Measured: a card holds 200 components. A status line costs up to five and the text after
# it one, so past this many the answer keeps writing into its last block, without lines.
_MAX_GAPS = 30


def _grey(text: str) -> str:
    return f"<font color='grey'>{text}</font>"


def _secs(seconds: float) -> int:
    return max(1, round(seconds))


def _times(n: int) -> str:
    return f" ×{n}" if n > 1 else ""


# --- the waiting animation --------------------------------------------------
# A live status line moves by itself instead of counting seconds, so a wait costs one card
# write per change of state rather than one every few seconds. Three grey dots, drawn at 4x
# so the client's downscale smooths them: rising in turn while thinking, swelling in turn
# while something else runs.

_ANIM_FILES = {"think": "thinking.gif", "work": "working.gif"}
_ANIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_ANIM_SIZE = "24px 12px"
_anim_keys: dict[str, str] = {}
_anim_lock = threading.Lock()
_anim_failed_at = 0.0
_ANIM_RETRY_SECONDS = 600
# The first card of a process waits for this; a slow upload must not hold up the answer.
_ANIM_UPLOAD_TIMEOUT = 3


def _animation_keys() -> dict[str, str]:
    """{kind: image_key}, uploaded once per process. Empty after a failure, and retried
    only after a while: a status line without its dots is fine, a slow turn start is not."""
    global _anim_failed_at
    if _anim_keys:
        return dict(_anim_keys)
    # A turn arriving while another uploads goes without dots rather than wait for it.
    if not _anim_lock.acquire(blocking=False):
        return {}
    try:
        if _anim_keys:
            return dict(_anim_keys)
        if _anim_failed_at and time.monotonic() - _anim_failed_at < _ANIM_RETRY_SECONDS:
            return {}
        try:
            keys = {}
            for kind, name in _ANIM_FILES.items():
                with open(os.path.join(_ANIM_DIR, name), "rb") as f:
                    keys[kind] = upload_image(f.read(), name, timeout=_ANIM_UPLOAD_TIMEOUT)
        except Exception:  # noqa: BLE001 — the dots are decoration
            log.warning("could not upload the status animation", exc_info=True)
            _anim_failed_at = time.monotonic()
            return {}
        _anim_keys.update(keys)
        return dict(_anim_keys)
    finally:
        _anim_lock.release()


def _pill(eid: str, text: str, image_key: str = "") -> dict:
    """A status line set apart from the answer: grey text on a grey chip sized to it,
    followed by the moving dots while it is live."""
    label = {"tag": "markdown", "text_size": "notation", "content": _grey(text)}
    inner = [label]
    if image_key:
        inner = [{"tag": "column_set", "flex_mode": "none", "horizontal_spacing": "6px",
                  "columns": [
                      {"tag": "column", "width": "auto", "vertical_align": "center",
                       "elements": [label]},
                      {"tag": "column", "width": "auto", "vertical_align": "center",
                       "elements": [{"tag": "img", "img_key": image_key,
                                     "scale_type": "crop_center", "size": _ANIM_SIZE,
                                     "transparent": True, "preview": False,
                                     "alt": {"tag": "plain_text", "content": ""}}]}]}]
    return {"tag": "column_set", "element_id": eid, "flex_mode": "none",
            "horizontal_align": "left",
            "columns": [{"tag": "column", "width": "auto", "background_style": "grey-50",
                         "padding": "2px 10px 2px 10px", "elements": inner}]}


def _band(eid: str, text: str) -> dict:
    """The turn's closing summary: a full-width grey band, so it reads as a footer."""
    return {"tag": "column_set", "element_id": eid, "background_style": "grey-50",
            "columns": [{"tag": "column", "width": "weighted", "weight": 1,
                         "padding": "4px 8px 4px 8px",
                         "elements": [{"tag": "markdown", "text_size": "notation",
                                       "content": _grey(text)}]}]}


class _Item:
    """One top-level element of the card, in order. `content` is what it should say — the
    text, or for a status line (text, animation) — and `created`/`written` are what CardKit
    has; only the writer touches those two."""
    __slots__ = ("id", "kind", "content", "dead", "created", "written")

    def __init__(self, eid: str, kind: str, content, created: bool = False):
        self.id, self.kind, self.content, self.dead = eid, kind, content, False
        self.created, self.written = created, (content if created else None)

    def element(self, content=None, keys: dict | None = None) -> dict:
        content = self.content if content is None else content
        if self.kind == "text":
            return {"tag": "markdown", "element_id": self.id, "content": content}
        if self.kind == "pill":
            text, anim = content
            return _pill(self.id, text, (keys or {}).get(anim, "") if anim else "")
        return _band(self.id, content)


class StreamingCard:
    """One streaming card's lifecycle: the answer's text with the turn's status lines in
    between, where they happened, and a one-line summary at the end.

    Between two runs of text there is one status line. While the turn waits it says what
    is happening now — thinking, a tool running, the thread being summarised — with moving
    dots instead of a clock, and when text arrives it freezes as a record of the wait
    ("🤔 思考 · 3 秒 · 🔧 调用工具 ×2 · 5 秒"). Phases are generic on purpose: naming each
    tool would be a mapping to maintain, and the chat is not a log.

    The card is kept as an ordered list of elements; the worker brings CardKit in line
    with it, inserting each new element after the one before. All CardKit writes on a
    card share a single strictly-increasing sequence, so it lives here. Every method
    returns a bool or nothing and never raises: streaming is an enhancement, and a failure
    must fall back to send_text rather than lose the answer. `ok` tracks whether the card
    is usable."""

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.card_id = ""
        self._seq = 0
        self.ok = False
        self.sent = False          # posted to the chat, so somebody is looking at it
        # Sending happens on its own thread. Measured, one CardKit write takes ~470 ms
        # (361–606 ms), so writing from the token loop stalled it for longer than the
        # flush interval it was tuned against. Now the loop only hands over state.
        self._lock = threading.Lock()       # guards the items, the phase and the counts
        self._send_lock = threading.Lock()  # one write in flight, so _seq stays ordered
        self._flush_lock = threading.Lock()  # one flusher, so created/written have one writer
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._t0 = time.monotonic()
        self._phase = "think"
        self._phase_since = self._t0
        self._tools_done = 0
        self._steps: list[tuple[str, float]] = []
        self._full = ""            # the whole answer so far, as the caller hands it over
        self._text_from = 0        # where in it the open (or next) text block starts
        self._gaps = 0
        self._gap_tools = 0
        self._gap_think_secs = 0.0
        self._gap_tool_secs = 0.0
        self._gap_compact_secs = 0.0
        self._n = 0
        self._keys: dict[str, str] = {}     # animation image keys, fetched on open()
        self._live: _Item | None = _Item("p0", "pill", self._live_text(), created=True)
        self._text: _Item | None = None
        self._items: list[_Item] = [self._live]

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _next_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # --- what the turn is doing ---------------------------------------------

    def _elapsed(self) -> int:
        return _secs(time.monotonic() - self._t0)

    def _enter(self, phase: str) -> None:
        """Close the running step and start the next. Caller holds _lock."""
        now = time.monotonic()
        spent = now - self._phase_since
        if self._phase == "tool":
            self._gap_tool_secs += spent
        elif self._phase == "compact":
            self._gap_compact_secs += spent
        elif self._phase == "think" and self._live is not None:
            self._gap_think_secs += spent
        if self._phase in ("think", "tool", "write", "compact") and (
                self._phase == "tool" or spent >= _MIN_STEP_SECONDS):
            self._steps.append((self._phase, spent))
        self._phase, self._phase_since = phase, now
        if self._live is not None:
            self._live.content = self._live_text()

    def _frozen_text(self, thinking: bool = True) -> str:
        """What a finished wait leaves behind: how long it thought and what ran in it.
        Caller holds _lock."""
        parts = []
        if thinking and self._gap_think_secs >= 1:
            parts.append(f"🤔 思考 · {_secs(self._gap_think_secs)} 秒")
        if self._gap_compact_secs:
            parts.append(f"🗜️ 整理较早的对话 · {_secs(self._gap_compact_secs)} 秒")
        if self._gap_tools:
            parts.append(f"🔧 调用工具{_times(self._gap_tools)} · "
                         f"{_secs(self._gap_tool_secs)} 秒")
        return " · ".join(parts)

    def _live_text(self) -> tuple[str, str]:
        """The status line of the wait in progress, and which dots go with it. Caller
        holds _lock."""
        if self._phase == "tool":
            return f"🔧 调用工具{_times(self._gap_tools + 1)}", "work"
        if self._phase == "compact":
            return "🗜️ 正在整理较早的对话", "work"
        done = self._frozen_text(thinking=False)
        return (f"{done} · 🤔 思考中" if done else "🤔 思考中"), "think"

    def _start_wait(self) -> None:
        """Text stopped and something else began: the open text block is finished, and a
        status line goes after it. Caller holds _lock."""
        if self._text is not None:
            if self._gaps >= _MAX_GAPS:
                return                      # keep writing into the last block
            self._gaps += 1
            self._text, self._text_from = None, len(self._full)
        if self._live is None:
            self._live = _Item(self._next_id("p"), "pill", ("", ""))
            self._items.append(self._live)
        self._live.content = self._live_text()

    def _end_wait(self) -> None:
        """The wait is over: its status line freezes as a record, or goes. Caller holds
        _lock."""
        if self._live is not None:
            done = self._frozen_text()
            if done:
                self._live.content = (done, "")
            else:
                self._live.dead = True      # under a second of thought: not worth a line
            self._live = None
        self._gap_tools = 0
        self._gap_think_secs = self._gap_tool_secs = self._gap_compact_secs = 0.0

    def tool_started(self) -> None:
        """The model began a tool call. Counted on the result instead: a streamed call
        names its tool on more than one chunk, so counting starts would overcount."""
        if not self.ok:
            return
        with self._lock:
            if self._phase != "tool":
                self._start_wait()
                self._enter("tool")
        self._wake.set()

    def tool_finished(self) -> None:
        if not self.ok:
            return
        with self._lock:
            self._tools_done += 1
            if self._live is not None:
                self._gap_tools += 1
            self._enter("think")
        self._wake.set()

    def compacting(self, started: bool) -> None:
        """The thread is being summarised before the model is called — a long, uncached
        call that streams nothing, so without this the card reads "thinking" for it."""
        if not self.ok:
            return
        with self._lock:
            if started:
                self._start_wait()
            self._enter("compact" if started else "think")
        self._wake.set()

    def summary(self) -> str:
        """The closing line: the whole turn's time, and where it went."""
        with self._lock:
            spent: dict[str, float] = {}
            for phase, t in self._steps:
                spent[phase] = spent.get(phase, 0.0) + t
            parts = []
            if "compact" in spent:
                parts.append(f"整理上下文 {_secs(spent['compact'])} 秒")
            if "think" in spent:
                parts.append(f"思考 {_secs(spent['think'])} 秒")
            if self._tools_done:
                parts.append(f"{self._tools_done} 次工具调用 "
                             f"{_secs(spent.get('tool', 0.0))} 秒")
            if "write" in spent:
                parts.append(f"撰写回答 {_secs(spent['write'])} 秒")
            line = f"⏱️ 耗时 {self._elapsed()} 秒"
        return f"{line}（{' · '.join(parts)}）" if parts else line

    # --- writing ------------------------------------------------------------

    def _set_text(self, segment: str) -> None:
        """Show `segment` as the open text block, opening one if needed. CardKit rejects
        empty text, so a block exists only once there are words for it. Caller holds
        _lock."""
        shown = segment.lstrip()
        if not shown:
            return
        if self._text is None:
            self._end_wait()
            self._text = _Item(self._next_id("t"), "text", shown)
            self._items.append(self._text)
        else:
            self._text.content = shown

    def _pump(self) -> None:
        """Bring the card in line with the items, repeatedly. Text superseded before it
        was written is never sent: each write carries a block's whole text, so the
        newest one subsumes every earlier one."""
        while not self._stop.is_set() and self.ok:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            self._flush()

    def _flush(self) -> None:
        with self._flush_lock:
            if self.ok:
                self._flush_once()

    def _flush_once(self) -> None:
        """Insert what is new, rewrite what changed, then delete what went. Deletions come
        last so a removed element can still anchor the one inserted after it."""
        with self._lock:
            plan = [(it, it.content, it.dead) for it in self._items]
        anchor = ""
        for it, content, dead in plan:
            if not dead and not it.created:
                if not self._put("/elements", {
                        "type": "insert_after", "target_element_id": anchor,
                        "elements": json.dumps([it.element(content, self._keys)])},
                        method="POST"):
                    if it.kind == "text":
                        self.ok = False
                        return
                    it.dead = True          # a missing status line costs nothing
                    continue
                it.created, it.written = True, content
            elif not dead and it.written != content:
                # Text is rewritten in place, which CardKit types out; a status line is
                # replaced whole, since its dots come and go with its state.
                if it.kind == "text":
                    ok = self._put(f"/elements/{it.id}/content", {"content": content})
                else:
                    ok = self._put(f"/elements/{it.id}", {
                        "element": json.dumps(it.element(content, self._keys))})
                if ok:
                    it.written = content
                elif it.kind == "text":
                    self.ok = False
                    return
            if it.created:
                anchor = it.id
        for it, _, dead in plan:
            if dead and it.created and self._put(f"/elements/{it.id}", {},
                                                 method="DELETE"):
                it.created = False
        with self._lock:
            self._items = [it for it in self._items if it.created or not it.dead]

    def _abandon(self) -> None:
        """The caller posts the answer as text instead. Best-effort, stop the card claiming
        it is still working: drop any status line still showing its dots or already given
        up, and end streaming."""
        if not self.sent:
            return
        with self._lock:
            stale = [it for it in self._items if it.created and (
                it.dead or (it.kind == "pill" and (it.written or ("", ""))[1]))]
        for it in stale:
            self._put(f"/elements/{it.id}", {}, method="DELETE")
        self._put("/settings",
                  {"settings": json.dumps({"config": {"streaming_mode": False}})},
                  method="PATCH")

    def _put(self, path: str, body: dict, method: str = "PUT") -> bool:
        with self._send_lock:
            tok = _tenant_token()
            if not tok:
                return False
            resp = _call(method, f"{_API_DOMAIN}/open-apis/cardkit/v1/cards/{self.card_id}{path}",
                         {**body, "sequence": self._next_seq()}, bearer=tok)
        if resp.get("code") not in (0, "0"):
            log.warning("card write %s failed: %s", path, resp.get("msg", resp))
            return False
        return True

    def open(self) -> bool:
        """Create the card entity and post it. False if CardKit is unavailable
        (e.g. the cardkit:card:write scope was never granted) — caller falls back."""
        tok = _tenant_token()
        if not (tok and self.chat_id):
            return False
        self._keys = _animation_keys()
        card = {"schema": "2.0",
                "config": {"streaming_mode": True, "update_multi": True},
                "body": {"elements": [it.element(keys=self._keys) for it in self._items]}}
        created = _post(f"{_API_DOMAIN}/open-apis/cardkit/v1/cards",
                        {"type": "card_json", "data": json.dumps(card)}, bearer=tok)
        if created.get("code") not in (0, "0"):
            log.warning("card create failed (falling back to text): %s",
                        created.get("msg", created))
            return False
        self.card_id = created.get("data", {}).get("card_id", "")
        if not self.card_id:
            return False
        sent = _post(f"{_API_DOMAIN}/open-apis/im/v1/messages?receive_id_type={_rid_type(self.chat_id)}",
                     {"receive_id": self.chat_id, "msg_type": "interactive",
                      "content": json.dumps({"type": "card",
                                             "data": {"card_id": self.card_id}})},
                     bearer=tok)
        if sent.get("code") not in (0, "0"):
            log.warning("card send failed (falling back to text): %s",
                        sent.get("msg", sent))
            return False
        self.ok = self.sent = True
        self._worker = threading.Thread(target=self._pump, daemon=True,
                                        name="cardkit-stream")
        self._worker.start()
        return True

    def update(self, full_text: str) -> bool:
        """Hand over the accumulated text and return at once — the write happens on the
        worker. Must be the full text so far, not a delta: the open block is cut from it,
        and CardKit renders the appended tail with a typewriter effect.

        The bool reports whether the card is still usable, which is now known one call
        late — a write that fails is discovered by the worker, so this returns False from
        the *next* call onwards. That is what the caller needs it for (stop streaming,
        fall back to text), and close() re-checks before finishing."""
        if not self.ok:
            return False
        with self._lock:
            self._full = full_text
            # Leading whitespace is not the answer starting: the wait goes on until words.
            if self._phase != "write" and full_text[self._text_from:].strip():
                self._enter("write")
            self._set_text(full_text[self._text_from:])
        self._wake.set()
        return True

    def close(self, final_text: str) -> bool:
        """Write the final text, freeze the last status line, add the summary, and turn
        streaming off so the card stops showing the typewriter indicator."""
        # Stop the worker first, then write the final state synchronously: the worker
        # drops superseded states, and the last one must not be dropped. Stopped on the
        # failure path too, or it would retry a dead card for the life of the process.
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
        if not self.ok:
            self._abandon()
            return False
        with self._lock:
            self._enter("done")
            # The final text is normally the accumulated one; when the caller replaced it
            # (a consent prompt), the replacement takes the open block's place.
            head = self._full[:self._text_from]
            self._set_text(final_text[len(head):] if final_text.startswith(head)
                           else final_text)
            self._end_wait()
        tail = _Item("tail", "band", self.summary())
        with self._lock:
            self._items.append(tail)
        self._flush()
        if not self.ok:
            self._abandon()
            return False
        if not self._put("/settings",
                         {"settings": json.dumps({"config": {"streaming_mode": False}})},
                         method="PATCH"):
            return False
        return True
