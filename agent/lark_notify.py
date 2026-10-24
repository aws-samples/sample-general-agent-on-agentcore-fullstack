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
        return json.loads(e.read().decode() or "{}")


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
            part: str = "file") -> dict:
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
        with urllib.request.urlopen(req, timeout=120) as r:  # nosec B310 # nosemgrep: dynamic-urllib-use-detected — const scheme+host
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


def upload_image(data: bytes) -> str:
    """Upload an image, returning its image_key — also how a video gets its cover, which
    Lark requires for `media` and offers no default for."""
    body = _upload("/open-apis/im/v1/images", {"image_type": "message"}, "cover.jpg", data,
                   part="image")
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

_ELEMENT_ID = "answer"
_STATUS_ID = "status"
# While waiting, the status line is refreshed this often so its seconds keep moving: a
# long tool call emits no event, and a card that sits still reads as stuck.
_HEARTBEAT_SECONDS = 3.0
# Below this a step is noise — the model re-reading a tool result before it writes.
_MIN_STEP_SECONDS = 0.5
_STEP_LABEL = {"think": "思考", "tool": "调用工具", "write": "撰写回答",
               "compact": "整理上下文"}


def _grey(text: str) -> str:
    return f"<font color='grey'>{text}</font>"


def _status_element(text: str) -> dict:
    return {"tag": "markdown", "element_id": _STATUS_ID, "text_size": "notation",
            "content": _grey(text)}


def _card_json(status: str, streaming: bool) -> dict:
    """A card that starts as a status line only. The answer element is inserted above it
    when the first words arrive: CardKit rejects an empty text update, and an empty
    element would leave a gap above the status while the model is still thinking."""
    return {
        "schema": "2.0",
        "config": {"streaming_mode": streaming, "update_multi": True},
        "body": {"elements": [_status_element(status)]},
    }


def _summary_panel(title: str, steps: list[str]) -> dict:
    """What the status line becomes when the turn ends: a collapsed, grey-filled panel,
    so the record of what ran reads as a footnote rather than as more of the answer."""
    icon = {"tag": "standard_icon", "token": "down-small-ccm_outlined",
            "size": "16px 16px", "color": "grey"}
    return {"tag": "collapsible_panel", "element_id": _STATUS_ID, "expanded": False,
            "background_color": "grey-50", "padding": "4px 8px 4px 8px",
            "header": {"title": {"tag": "markdown", "text_size": "notation",
                                 "content": _grey(title)},
                       "icon": icon, "icon_position": "right", "icon_expanded_angle": -180,
                       "padding": "4px 0px 4px 0px"},
            "elements": [{"tag": "markdown", "text_size": "notation",
                          "content": _grey("\n".join(steps))}]}


class StreamingCard:
    """One streaming card's lifecycle: an answer above, and below it a status line that
    tracks what the turn is doing and ends as a collapsed summary of it.

    The status follows events, not a clock — a tool starting, a result arriving, the
    first words of the answer — with a heartbeat only while nothing is happening, so the
    seconds move during a long tool call and stop once the answer streams. Phases are
    generic on purpose: naming each tool would be a mapping to maintain.

    All CardKit writes on a card share a single strictly-increasing sequence, so it lives
    here rather than in the caller. Every method returns a bool or nothing and never
    raises: streaming is an enhancement, and a failure must fall back to send_text rather
    than lose the answer. `ok` tracks whether the card is usable."""

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.card_id = ""
        self._seq = 0
        self.ok = False
        # Sending happens on its own thread. Measured, one CardKit write takes ~470 ms
        # (361–606 ms), so writing from the token loop stalled it for longer than the
        # flush interval it was tuned against — the loop spent over half its time
        # waiting on HTTP instead of reading the model, and the text arrived in visible
        # jerks. Now the loop only ever hands over the latest accumulated text.
        self._lock = threading.Lock()      # guards _pending, the phase and the steps
        self._send_lock = threading.Lock()  # one write in flight, so _seq stays ordered
        self._pending: str | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._answer_shown = False
        self._t0 = time.monotonic()
        self._phase = "think"
        self._phase_since = self._t0
        self._tools_done = 0
        self._steps: list[tuple[str, float]] = []
        self._status_due = False
        self._status_written_at = self._t0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # --- what the turn is doing ---------------------------------------------

    def _elapsed(self) -> int:
        return max(1, round(time.monotonic() - self._t0))

    def _enter(self, phase: str) -> None:
        """Close the running step and start the next. Caller holds _lock."""
        now = time.monotonic()
        spent = now - self._phase_since
        if self._phase in _STEP_LABEL and (self._phase == "tool"
                                           or spent >= _MIN_STEP_SECONDS):
            last = self._steps[-1] if self._steps else None
            # Consecutive thinking or writing is one step; each tool call is its own.
            if last and last[0] == self._phase and self._phase != "tool":
                self._steps[-1] = (self._phase, last[1] + spent)
            else:
                self._steps.append((self._phase, spent))
        self._phase, self._phase_since = phase, now
        self._status_due = True

    def _status_text(self) -> str:
        if self._phase == "tool":
            return f"🛠️ 正在调用工具(第 {self._tools_done + 1} 次)· {self._elapsed()} 秒"
        if self._phase == "write":
            return "✍️ 正在撰写回答"
        if self._phase == "compact":
            return f"🗜️ 正在整理较早的对话 · {self._elapsed()} 秒"
        return f"🤔 思考中 · {self._elapsed()} 秒"

    def tool_started(self) -> None:
        """The model began a tool call. Counted on the result instead: a streamed call
        names its tool on more than one chunk, so counting starts would overcount."""
        if not self.ok:
            return
        with self._lock:
            if self._phase != "tool":
                self._enter("tool")
        self._wake.set()

    def tool_finished(self) -> None:
        if not self.ok:
            return
        with self._lock:
            self._tools_done += 1
            self._enter("think")
        self._wake.set()

    def compacting(self, started: bool) -> None:
        """The thread is being summarised before the model is called — a long, uncached
        call that streams nothing, so without this the card reads "thinking" for it."""
        if not self.ok:
            return
        with self._lock:
            self._enter("compact" if started else "think")
        self._wake.set()

    def summary(self) -> tuple[str, list[str]]:
        """The panel's title and its steps, as they stand."""
        with self._lock:
            title = (f"⚙️ 执行过程 · 调用 {self._tools_done} 次工具 · 耗时 {self._elapsed()} 秒"
                     if self._tools_done else f"⚙️ 执行过程 · 耗时 {self._elapsed()} 秒")
            steps = [f"{_STEP_LABEL[p]} {max(1, round(t))} 秒" for p, t in self._steps]
        return title, steps

    # --- writing ------------------------------------------------------------

    def _pump(self) -> None:
        """Write the newest text and the current status, repeatedly. Superseded text is
        dropped rather than queued: CardKit takes the full accumulated text on every
        write, so the newest one subsumes every earlier one."""
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                text, self._pending = self._pending, None
                waiting = self._phase in ("think", "tool", "compact")
                stale = time.monotonic() - self._status_written_at >= _HEARTBEAT_SECONDS
                status = (self._status_text()
                          if self._status_due or (waiting and stale) else None)
                self._status_due = False
            if text is not None:
                self._write(text)
            if status is not None:
                self._write_status(status)

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

    def _write(self, full_text: str) -> bool:
        """The answer. Inserted above the status on its first write, updated after."""
        if not full_text:
            return True
        if self._answer_shown:
            ok = self._put(f"/elements/{_ELEMENT_ID}/content", {"content": full_text})
        else:
            ok = self._put("/elements", {
                "type": "insert_before", "target_element_id": _STATUS_ID,
                "elements": json.dumps([{"tag": "markdown", "element_id": _ELEMENT_ID,
                                         "content": full_text}])}, method="POST")
            self._answer_shown = ok
        if not ok:
            self.ok = False
        return ok

    def _write_status(self, text: str) -> None:
        # Best-effort: the status is decoration, so a failed write must not stop the
        # answer from streaming.
        if self._put(f"/elements/{_STATUS_ID}/content", {"content": _grey(text)}):
            self._status_written_at = time.monotonic()

    def open(self) -> bool:
        """Create the card entity and post it. False if CardKit is unavailable
        (e.g. the cardkit:card:write scope was never granted) — caller falls back."""
        tok = _tenant_token()
        if not (tok and self.chat_id):
            return False
        created = _post(f"{_API_DOMAIN}/open-apis/cardkit/v1/cards",
                        {"type": "card_json",
                         "data": json.dumps(_card_json(self._status_text(), True))},
                        bearer=tok)
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
        self.ok = True
        self._worker = threading.Thread(target=self._pump, daemon=True,
                                        name="cardkit-stream")
        self._worker.start()
        return True

    def update(self, full_text: str) -> bool:
        """Hand over the accumulated text and return at once — the write happens on the
        worker. Must be the full text so far, not a delta: CardKit renders the appended
        tail with a typewriter effect.

        The bool reports whether the card is still usable, which is now known one call
        late — a write that fails is discovered by the worker, so this returns False from
        the *next* call onwards. That is what the caller needs it for (stop streaming,
        fall back to text), and close() re-checks before finishing."""
        if not self.ok:
            return False
        with self._lock:
            self._pending = full_text
            if self._phase != "write":
                self._enter("write")
        self._wake.set()
        return True

    def close(self, final_text: str) -> bool:
        """Write the final text, turn the status line into the summary panel, and turn
        streaming off so the card stops showing the typewriter indicator."""
        if not self.ok:
            return False
        # Stop the worker first, then write the final text synchronously: the worker
        # drops superseded states, and the last one must not be dropped.
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
        with self._lock:
            self._enter("done")
        self._write(final_text)
        if not self.ok:
            return False
        title, steps = self.summary()
        # Losing the panel loses a footnote, not the answer, so it does not fail close().
        self._put(f"/elements/{_STATUS_ID}",
                  {"element": json.dumps(_summary_panel(title, steps))})
        if not self._put("/settings",
                         {"settings": json.dumps({"config": {"streaming_mode": False}})},
                         method="PATCH"):
            return False
        return True
