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

import lark_notify

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
    return downscale(lark_notify.fetch_message_resource(message_id, image_key, "image"))


# Lark names its resource kinds, and asking for the wrong one answers 500. A picture is
# the only kind the model can look at; everything else is for code to open.
IMAGE = "image"
FILE = "file"


def save_to_workspace(sandbox, handle: dict, index: int) -> str:
    """Fetch an attachment and write it into the workspace, returning the file name.

    Unlike a picture, a document's contents have no business in the model's context: a
    long pdf would not fit, and code can read exactly the part that matters."""
    kind = handle.get("kind", IMAGE)
    data = lark_notify.fetch_message_resource(
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
