"""Where a file produced by the agent is sent, decided by the surface the turn came from.

The workspace is invisible to the user, so a generated file is undelivered until something
sends it. Which "something" differs per surface, and the tool that reads the bytes has no
business knowing: it hands over a name, the bytes, and a way to produce a video cover if
the surface turns out to want one.

Carried per run in `config["configurable"]["delivery"]`, never on the cached session — a
session is keyed by actor and thread and is reused across surfaces, so a stored target
would send a web user's file to whichever Lark chat was served last.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

from channels import web_files
from channels import lark_card

log = logging.getLogger("agent.delivery")


class Delivery(ABC):
    """One surface's answer to "can this file reach the person, and how"."""

    @abstractmethod
    def refusal(self, name: str, size: int, *, in_workspace: bool = True) -> str | None:
        """Why this file cannot be sent, or None if it can.

        Asked before the bytes are read so an impossible send costs one `stat` rather
        than pushing tens of megabytes of base64 across the control plane. `in_workspace`
        is whether a copy stays there — false for a browser screenshot — so no sentence
        points the model at a file that does not exist."""

    @abstractmethod
    def send_file(self, name: str, data: bytes, cover: Callable[[], bytes | None],
                  *, in_workspace: bool = True) -> str:
        """Send it, and report in a sentence the model can repeat to the user — or (sentence,
        artifact), the artifact being what the tool message keeps out of the model's sight,
        such as a stored file's key, for a surface that shows it again later.

        `cover` is only called if this surface needs a still frame to render a video."""


class LarkDelivery(Delivery):
    """Into a Lark chat: a video plays inline, everything else is an attachment."""

    # Lark's own ceiling for an uploaded file.
    MAX_BYTES = 30 * 1000 * 1000
    # Lark names a handful of types and treats the rest as `stream`.
    _TYPES = {"pdf": "pdf", "doc": "doc", "docx": "doc", "xls": "xls", "xlsx": "xls",
              "ppt": "ppt", "pptx": "ppt", "opus": "opus"}

    def __init__(self, target: str) -> None:
        # A chat id, or an open_id for the event-driven turns that have no chat to reply
        # in — lark_card._rid_type reads which it is.
        self.target = target

    def refusal(self, name: str, size: int, *, in_workspace: bool = True) -> str | None:
        if size > self.MAX_BYTES:
            return (f"{name} is {size / 1e6:.1f} MB and Lark accepts 30 MB. "
                    "Compress it or split it, then send again.")
        return None

    # Sent through im/v1/images so they render in the chat; a file attachment of a png
    # is something the user has to download before they can look at it.
    _IMAGES = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}

    def send_file(self, name: str, data: bytes, cover: Callable[[], bytes | None],
                  *, in_workspace: bool = True) -> str:
        suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix in self._IMAGES:
            try:
                key = lark_card.upload_image(data)
                if lark_card.send_image(self.target, key):
                    return f"Sent {name} to the chat; it shows there."
            except lark_card.LarkError as e:
                return f"{name} was not sent. {e}"
        # The upload's file_type and the message type have to agree: an mp4-typed upload
        # refuses to be sent as `file` ("The type of file upload does not match the type
        # of message being sent"), and `media` will not render without a cover. So a video
        # is only uploaded as mp4 once there is a frame to send with it.
        if suffix == "mp4":
            try:
                frame = cover()
                if frame:
                    image = lark_card.upload_image(frame)
                    key = lark_card.upload_file(name, data, "mp4")
                    if lark_card.send_media(self.target, key, image):
                        return f"Sent {name} to the chat; it plays there."
            except lark_card.LarkError:
                # Worth one attachment attempt: a cover Lark would not take says nothing
                # about the video itself.
                log.info("could not send %s as a video", name, exc_info=True)
            else:
                log.info("no cover for %s, sending it as an attachment", name)
        try:
            key = lark_card.upload_file(
                name, data,
                "stream" if suffix == "mp4" else self._TYPES.get(suffix, "stream"))
            if lark_card.send_file(self.target, key):
                return f"Sent {name} to the chat as an attachment."
            return f"{name} uploaded but the message would not send."
        except lark_card.LarkError as e:
            # Lark's own words, because the model has to tell the user something true:
            # "the bot is not in that chat" and "the file is too large" need different
            # next steps, and a bare failure reads as "try again".
            return f"{name} was not sent. {e}"


class WebDelivery(Delivery):
    """Into the web page, on the turn's own event stream: an image shows and a video plays
    inline, anything else downloads. Stored first (channels/web_files.py), so the event carries a link and a reload of
    the page shows it again; without a bucket the bytes ride on the event itself and are gone
    on reload. `out` collects CUSTOM events for the run's stream to write."""

    # One SSE frame through the Runtime's proxy, and base64 adds a third: ~10.5 MB measured
    # to pass intact. Kept for the stored path too, so the cap does not depend on storage.
    MAX_BYTES = 8 * 1000 * 1000

    def __init__(self, out: list, actor_id: str = "") -> None:
        self.out = out
        self.actor_id = actor_id

    def refusal(self, name: str, size: int, *, in_workspace: bool = True) -> str | None:
        if size > self.MAX_BYTES:
            return (f"{name} is {size / 1e6:.1f} MB and the web page takes 8 MB. "
                    "Compress it or split it, then send again.")
        return None

    def send_file(self, name: str, data: bytes, cover: Callable[[], bytes | None],
                  *, in_workspace: bool = True) -> str | tuple[str, dict]:
        # The key is returned, not kept on this object: one run's tool calls can run in
        # parallel, and a shared slot lost a file's record when two were sent at once.
        if web_files.configured() and self.actor_id:
            try:
                key = web_files.put(self.actor_id, name, data)
                self.out.append({"name": "file", "value": {
                    "name": name, "size": len(data),
                    "url": web_files.url(key, name, self.actor_id)}})
                return (f"Sent {name} to the page; it shows there and stays in the conversation.",
                        {"file": {"key": key, "name": name, "size": len(data)}})
            except Exception:  # noqa: BLE001 — the stream can still carry it
                log.warning("could not store %s; sending it inline", name, exc_info=True)
        self.out.append({"name": "file", "value": {
            "name": name, "size": len(data), "data": base64.b64encode(data).decode()}})
        # The stream is the only copy: what the model tells the user has to say so.
        return (f"Sent {name} to the page, where it shows while the page stays open; it is "
                "not kept there" + (", but stays in the workspace." if in_workspace else "."))


class NoDelivery(Delivery):
    """A surface with no way to carry a file yet — A2A.

    Says so instead of quietly using Lark, which would send a web user's file to their
    Lark account, and names the path so the model can tell them where it is rather than
    claim it was delivered."""

    def __init__(self, surface: str = "this surface") -> None:
        self.surface = surface

    def refusal(self, name: str, size: int, *, in_workspace: bool = True) -> str | None:
        if not in_workspace:
            return (f"{name} was not sent: {self.surface} cannot receive files yet. "
                    "Do not claim it was sent.")
        return (f"{name} ({size / 1e3:.0f} kB) stays in the workspace: {self.surface} "
                "cannot receive files yet. Say where it is and that it persists, or "
                "offer to show its contents. Do not claim it was sent.")

    def send_file(self, name: str, data: bytes, cover: Callable[[], bytes | None],
                  *, in_workspace: bool = True) -> str:
        return self.refusal(name, len(data), in_workspace=in_workspace) or ""


def for_run(config: dict | None) -> Delivery:
    """The delivery for the run in progress. Absent means no surface claimed the turn,
    which is a refusal — never the last surface that did, which is the bug this replaces."""
    delivery = ((config or {}).get("configurable") or {}).get("delivery")
    if isinstance(delivery, Delivery):
        return delivery
    log.warning("no delivery on this run; refusing to guess a destination")
    return NoDelivery()
