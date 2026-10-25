"""Files sent to the web page, kept so a reload still shows them — as a Lark chat keeps them.

Stored in the checkpoint bucket under `deliveries/`, at a key no one can guess: the role has
no List, as for the checkpoints, so isolation is the key. The page gets a short-lived
presigned link, re-signed whenever the history is read; the key itself rides on the tool
message's artifact, which the model never sees. Only raster pictures are served inline: any
other file — an SVG or an HTML page included — comes as an attachment, so nothing a tool
made is ever rendered as a document of the page's origin or another.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.parse
import uuid

log = logging.getLogger("agent.deliveries")

_BUCKET = os.environ.get("CHECKPOINT_BUCKET", "")
_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", ""))
_PREFIX = "deliveries/"
# At most: a link signed with the role's temporary credentials also ends with them, which
# can be 10-15 minutes. Reading the history signs afresh; a page left open longer than that
# shows the file but its link no longer opens until a reload.
_LINK_SECONDS = 3600
PICTURES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
            "webp": "image/webp", "bmp": "image/bmp"}

_client = None


def configured() -> bool:
    return bool(_BUCKET)


def _s3():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client("s3", region_name=_REGION or None,
                               config=Config(signature_version="s3v4",
                                             s3={"addressing_style": "virtual"}))
    return _client


def _disposition(name: str) -> str:
    inline = name.rsplit(".", 1)[-1].lower() in PICTURES if "." in name else False
    quoted = urllib.parse.quote(name, safe="")
    return f"{'inline' if inline else 'attachment'}; filename*=UTF-8''{quoted}"


def _type(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return PICTURES.get(ext, "application/octet-stream")


def _owned(actor_id: str) -> str:
    return f"{_PREFIX}{hashlib.sha256(actor_id.encode()).hexdigest()[:32]}/"


def put(actor_id: str, name: str, data: bytes) -> str:
    """Store a delivered file; its key."""
    key = f"{_owned(actor_id)}{uuid.uuid4().hex}/{os.path.basename(name) or 'file'}"
    _s3().put_object(Bucket=_BUCKET, Key=key, Body=data, ContentType=_type(name),
                     ContentDisposition=_disposition(name))
    return key


def url(key: str, name: str, actor_id: str) -> str:
    """A link the page can open for up to an hour; "" for a key that is not this actor's."""
    if not key.startswith(_owned(actor_id)):
        return ""
    return _s3().generate_presigned_url(
        "get_object", ExpiresIn=_LINK_SECONDS,
        Params={"Bucket": _BUCKET, "Key": key, "ResponseContentType": _type(name),
                "ResponseContentDisposition": _disposition(name)})


def delete(keys: list[str], actor_id: str) -> int:
    """Remove this actor's delivered files, e.g. when their thread is cleared; how many."""
    keys = [k for k in set(keys) if k.startswith(_owned(actor_id))]
    for key in keys:
        try:
            _s3().delete_object(Bucket=_BUCKET, Key=key)
        except Exception:  # noqa: BLE001 — the lifecycle rule removes what this misses
            log.warning("could not delete %s", key, exc_info=True)
    return len(keys)
