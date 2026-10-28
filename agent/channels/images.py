"""What a user attached: shown to the model when it is a picture, written to the
workspace when it is a file, and in neither case paid for on every later turn.

An image costs tokens by pixel area on every request that carries it, and this project
replays the whole thread each turn, so an inlined screenshot would be re-billed for the
life of the conversation — summarisation does not save it either, because the token
counter behind the trigger charges a flat 85 per image and so never fires on an
image-heavy thread.

So the bytes never enter the conversation. What is stored is a handle: Lark already keeps
the original, and the handle is how to ask for it again. The pixels are injected into the
model call for the turn the image arrived in, and that turn only.

The placeholder text is deliberately stable — no sizes, no timestamps. It sits in the
checkpoint, so anything that varies would move the prompt cache's prefix on every turn.
"""

from __future__ import annotations

import io
import logging
import threading
import uuid

from channels import lark_card

log = logging.getLogger("agent.images")

# Beyond this the model gains no detail, so it is wasted area — and area is what costs.
_MAX_EDGE = 1568
# One message can hold many; each one is tokens on the turn it arrives.
MAX_PER_MESSAGE = 3


def placeholder(index: int, message_id: str, image_key: str) -> str:
    """How an image appears in the stored conversation: a handle the model can name."""
    return f"[image {index} msg={message_id} key={image_key}]"




def downscale(data: bytes) -> tuple[bytes, str]:
    """Shrink to something the model can read in full, returning (bytes, mime).

    Pillow only: the sandbox has the richer imaging stack, but reaching it would cost a
    session round trip on the common path, which is a user simply asking what is in a
    screenshot."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover — Pillow ships in the image
        log.warning("Pillow missing; sending the image at its original size")
        return data, "image/jpeg"
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            if max(img.size) > _MAX_EDGE:
                ratio = _MAX_EDGE / max(img.size)
                img = img.resize((max(1, int(img.width * ratio)),
                                  max(1, int(img.height * ratio))))
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85, optimize=True)
            return out.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 — an unreadable image must not cost the turn
        log.warning("could not downscale the image; sending it as received", exc_info=True)
        return data, "image/jpeg"


def fetch(message_id: str, image_key: str) -> tuple[bytes, str]:
    """An image's bytes, downscaled, from Lark."""
    return downscale(lark_card.fetch_message_resource(message_id, image_key, "image"))


# Lark names its resource kinds, and asking for the wrong one answers 500. A picture is
# the only kind the model can look at; everything else is for code to open.
IMAGE = "image"
FILE = "file"


def save_to_workspace(sandbox, handle: dict, index: int) -> str:
    """Fetch an attachment and write it into the workspace, returning the file name.

    Unlike a picture, a document's contents have no business in the model's context: a
    long pdf would not fit, and code can read exactly the part that matters."""
    kind = handle.get("kind", IMAGE)
    # A web picture came with its bytes, held for the turn; a Lark one is fetched again.
    data = handle.get("data") or lark_card.fetch_message_resource(
        handle.get("message_id", ""), handle.get("key", ""), kind)
    name = handle.get("name") or (
        f"image-{index}.jpg" if kind == IMAGE else f"attachment-{index}")
    # Keep the name the sender used where there is one: the model is about to write code
    # against it, and a suffix is how it picks a library.
    sandbox.write_bytes(name, data)
    return name


def blocks_for(handles: list[tuple[str, str]]) -> list[dict]:
    """Content blocks for the model, skipping any image Lark will no longer hand over.

    Verified against langchain-aws 1.7: this shape becomes a Bedrock
    `{"image": {"format", "source": {"bytes"}}}` block.
    """
    import base64

    out = []
    for message_id, image_key in handles[:MAX_PER_MESSAGE]:
        try:
            data, mime = fetch(message_id, image_key)
        except Exception:  # noqa: BLE001 — a missing image is a fact to report, not a crash
            log.warning("image %s is not available", image_key, exc_info=True)
            continue
        out.append({"type": "image",
                    "base64": base64.b64encode(data).decode(),
                    "mime_type": mime})
    return out


# --- the attachments of the turn running on a thread --------------------------------
# A turn's own attachments and any steered into it while it runs share one registry, keyed
# by thread: a steered message joins a run whose config was fixed when it started, so the
# config cannot carry what that message brought. Sound because a thread runs one turn at a
# time. Pixels are bound to the message id they arrived with — attaching to "the last human
# message" would hand the original picture to whatever was steered in after it — and the
# handles are numbered across the turn, so save_attachment(3) can reach a file sent mid-turn.
_lock = threading.Lock()
_turns: dict[str, dict] = {}


def prepare(handles: list[dict]) -> list[dict]:
    """Fetch and downscale a message's pictures. Network, so done before any lock."""
    return blocks_for([(h.get("message_id", ""), h.get("key", ""))
                       for h in handles if h.get("kind") == IMAGE])


def register(thread_id: str, handles: list[dict], blocks: list[dict]) -> tuple[str, str]:
    """Join a message's attachments to the turn on this thread.

    Returns the placeholder text to append to the message, and the message id its pixels
    are bound to — the caller gives that id to the HumanMessage it sends. No network, so it
    can run under the caller's lock, which is what keeps registering and claiming the
    thread one step."""
    message_id = "msg-" + uuid.uuid4().hex
    if not handles:
        return "", message_id
    with _lock:
        turn = _turns.setdefault(thread_id, {"pixels": {}, "handles": []})
        start = len(turn["handles"]) + 1
        turn["handles"].extend(handles)
        if blocks:
            turn["pixels"][message_id] = blocks
    marks = []
    for i, h in enumerate(handles, start=start):
        if h.get("kind") == IMAGE and h.get("source") == "web":
            seen = "" if blocks else " (not a picture the model can read; save_attachment can still open it)"
            marks.append(f"[image {i}: {h.get('name') or 'image'}{seen}]")
        elif h.get("kind") == IMAGE:
            mark = placeholder(i, h.get("message_id", ""), h.get("key", ""))
            marks.append(mark if blocks else mark + " (no longer available)")
        else:
            marks.append(f"[attachment {i}: {h.get('name') or 'file'} — "
                         f"call save_attachment({i}) to open it]")
    return "\n".join(marks), message_id


def from_page(content: list) -> tuple[str, list[dict], list[dict]]:
    """A web message's AG-UI content as (text, handles, blocks): the page sends pictures
    inline, and like a Lark picture each is shown to the model on its own turn only — kept
    on a handle with its bytes for that turn, not in the conversation."""
    import base64
    texts, handles, blocks = [], [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text") or "")
            continue
        source = part.get("source") or {}
        if part.get("type") != "image" or source.get("type") != "data":
            continue
        if len(handles) >= MAX_PER_MESSAGE:
            break
        try:
            raw = base64.b64decode(source.get("value") or "")
        except Exception:  # noqa: BLE001 — a malformed picture is skipped, not fatal
            continue
        ext = _EXT.get(source.get("mimeType") or "", "jpg")
        handle = {"kind": IMAGE, "source": "web", "data": raw,
                  "name": f"image-{len(handles) + 1}.{ext}"}
        # Only what Pillow can read reaches the model: anything else (HEIC, SVG) labelled as
        # a jpeg would make Bedrock refuse every call of the turn. The file can still be saved.
        if _readable(raw):
            data, mime = downscale(raw)
            handle["block"] = {"type": "image", "base64": base64.b64encode(data).decode(),
                               "mime_type": mime}
        handles.append(handle)
    blocks = [h.pop("block") for h in handles if "block" in h]
    return "\n".join(t for t in texts if t), handles, blocks


_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp",
        "image/bmp": "bmp", "image/heic": "heic", "image/svg+xml": "svg"}


def _readable(data: bytes) -> bool:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001 — not a picture the model can be shown
        return False


def add_to_turn(thread_id: str, handles: list[dict]) -> tuple[str, str]:
    """prepare then register, for a caller that holds no lock."""
    return register(thread_id, handles, prepare(handles))


def pixels_for(thread_id: str) -> dict[str, list]:
    with _lock:
        return dict((_turns.get(thread_id) or {}).get("pixels") or {})


def handles_for(thread_id: str) -> list[dict]:
    with _lock:
        return list((_turns.get(thread_id) or {}).get("handles") or [])


def end_turn(thread_id: str) -> None:
    """Forget the turn's attachments. Called when the turn ends, however it ends."""
    with _lock:
        _turns.pop(thread_id, None)

