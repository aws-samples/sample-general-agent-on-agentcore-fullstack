"""User identity resolution + allowlist over the DynamoDB identity table.

Identity is `lark:{open_id}` for every entrypoint (webhook + web UI), so a user's
profile/session is shared. Table layout (single-table):
  CHANNEL#lark:{open_id} / PROFILE   -> {userId}
  USER#{userId}          / PROFILE   -> profile
  USER#{userId}          / SESSION   -> {sessionId, lastActivity}
  USER#{userId}          / MEMSESSION-> {memorySessionId, createdAt}
  ALLOW#lark:{open_id}   / ALLOW     -> allowlist entry

The runtime session (which microVM serves a request) and the Memory thread (where
history is stored) are separate ids so they can be rotated independently — see
get_or_create_memory_session.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
import logging

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger("router.identity")

_REGION = os.environ.get("AWS_REGION", "us-west-2")
_TABLE_NAME = os.environ["IDENTITY_TABLE_NAME"]
_REGISTRATION_OPEN = os.environ.get("REGISTRATION_OPEN", "false").lower() == "true"

_table = boto3.resource("dynamodb", region_name=_REGION).Table(_TABLE_NAME)


def is_user_allowed(channel: str, channel_user_id: str) -> bool:
    # A web user exists only because an admin created them in the Cognito pool, and only a
    # token that pool signed for the web client gets this far: the pool is their allowlist.
    if _REGISTRATION_OPEN or channel == "web":
        return True
    key = f"{channel}:{channel_user_id}"
    resp = _table.get_item(Key={"PK": f"ALLOW#{key}", "SK": "ALLOW"})
    return "Item" in resp


def resolve_user(channel: str, channel_user_id: str, display_name: str = "") -> tuple[str | None, bool]:
    """Return (internal_user_id, is_new). None if not allowed and new."""
    channel_key = f"{channel}:{channel_user_id}"
    resp = _table.get_item(Key={"PK": f"CHANNEL#{channel_key}", "SK": "PROFILE"})
    if "Item" in resp:
        return resp["Item"]["userId"], False

    # New user — gate on allowlist
    if not is_user_allowed(channel, channel_user_id):
        return None, False

    user_id = "user_" + hashlib.sha256(channel_key.encode()).hexdigest()[:16]
    now = int(time.time())
    _table.put_item(Item={
        "PK": f"USER#{user_id}", "SK": "PROFILE",
        "displayName": display_name, "createdAt": now,
    })
    _table.put_item(Item={
        "PK": f"CHANNEL#{channel_key}", "SK": "PROFILE",
        "userId": user_id, "channel": channel,
    })
    _table.put_item(Item={
        "PK": f"USER#{user_id}", "SK": f"CHANNEL#{channel_key}",
        "channel": channel,
    })
    return user_id, True


def get_or_create_session(user_id: str) -> str:
    """Deterministic-ish session id per user; refreshed lastActivity on reuse."""
    resp = _table.get_item(Key={"PK": f"USER#{user_id}", "SK": "SESSION"})
    if "Item" in resp:
        _table.update_item(
            Key={"PK": f"USER#{user_id}", "SK": "SESSION"},
            UpdateExpression="SET lastActivity = :t",
            ExpressionAttributeValues={":t": int(time.time())},
        )
        return resp["Item"]["sessionId"]

    # AgentCore requires the session id to be >= 33 chars.
    session_id = f"ses_{user_id}_{uuid.uuid4().hex[:12]}"
    if len(session_id) < 33:
        session_id = session_id + "0" * (33 - len(session_id))
    _table.put_item(Item={
        "PK": f"USER#{user_id}", "SK": "SESSION",
        "sessionId": session_id, "lastActivity": int(time.time()),
    })
    return session_id


def drop_session(user_id: str) -> None:
    """Forget the stored session id — the next message starts a new AgentCore
    session (and therefore a new runtime instance)."""
    _table.delete_item(Key={"PK": f"USER#{user_id}", "SK": "SESSION"})


# When a turn hits a Lark authorization wall, the original message is parked here so
# the shim's /return can replay it once consent completes — the "durable intent"
# that makes callback-driven resume possible without a live task to wake. TTL-bounded
# so an abandoned consent doesn't linger.
_PENDING_AUTH_TTL = int(os.environ.get("PENDING_AUTH_TTL_SECONDS", "600"))  # 10 min


def park_pending_auth(user_id: str, message: str, chat_id: str,
                      attachments: list | None = None,
                      message_id: str = "") -> None:
    """Remember what the user was trying to do, to replay after they authorize.
    One per user (overwrites) — a second attempt supersedes the first.

    Handles, never bytes: an item here is capped at 400 KB, and the agent can fetch the
    resource from the message that carried it."""
    item = {
        "PK": f"USER#{user_id}", "SK": "PENDING_AUTH",
        "message": message, "chatId": chat_id,
        "createdAt": int(time.time()),
        "ttl": int(time.time()) + _PENDING_AUTH_TTL,
    }
    if attachments:
        item["attachments"] = list(attachments)
        item["attachmentMessageId"] = message_id
    _table.put_item(Item=item)


def take_pending_auth(user_id: str) -> dict | None:
    """Claim the parked turn: return {message, chatId} and remove it, atomically.

    Two paths race for this — the router polling the vault and the shim's callback
    after consent completes — and both replay whatever they claim. A get-then-delete
    lets both read the same item inside the same window and run the turn twice, with
    duplicated side effects. Deleting with ALL_OLD makes the delete itself the claim:
    DynamoDB returns the item to exactly one caller.

    None if nothing was parked (the user ran /auth directly, with no turn to resume),
    someone else already claimed it, or it expired — a stale item past its ttl counts
    as absent even if DynamoDB has not swept it yet."""
    resp = _table.delete_item(
        Key={"PK": f"USER#{user_id}", "SK": "PENDING_AUTH"},
        ReturnValues="ALL_OLD",
    )
    item = resp.get("Attributes")
    if not item:
        return None
    if int(item.get("ttl", 0) or 0) < int(time.time()):
        return None
    return {"message": item.get("message", ""), "chatId": item.get("chatId", "")}


# Lark redelivers an event until it is acked, and acking happens before the agent has
# decided anything — so without a claim the same approval task would be decided twice.
_APPROVAL_TASK_TTL = int(os.environ.get("APPROVAL_TASK_TTL_SECONDS", "86400"))


def claim_approval_task(task_id: str) -> bool:
    """True only for the first caller to claim this approval task. A conditional put
    is what makes it a claim rather than a check: two concurrent redeliveries both
    read "unseen" if you get-then-put, and both go on to decide."""
    now = int(time.time())
    try:
        _table.put_item(
            Item={"PK": f"APPROVALTASK#{task_id}", "SK": "CLAIM",
                  "claimedAt": now, "ttl": now + _APPROVAL_TASK_TTL},
            ConditionExpression="attribute_not_exists(PK)",
        )
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def session_info(user_id: str) -> dict:
    """Current session id + lastActivity, or {} when no session is stored."""
    item = _table.get_item(Key={"PK": f"USER#{user_id}", "SK": "SESSION"}).get("Item") or {}
    return {"sessionId": item.get("sessionId", ""),
            "lastActivity": int(item.get("lastActivity", 0) or 0)}


def _default_thread(actor_id: str) -> str:
    return "sess-" + hashlib.sha256(actor_id.encode()).hexdigest()[:32]


def thread_belongs(actor_id: str, mem_sid: str) -> bool:
    """The agent honours a thread only under the actor's own prefix — the checkpoint is
    keyed by thread id alone and the default is computable by anyone. Kept in step with
    sessions.thread_belongs in the agent."""
    default = _default_thread(actor_id)
    suffix = mem_sid[len(default) + 1:] if mem_sid.startswith(default + "-") else ""
    return mem_sid == default or (len(suffix) == 12
                                  and all(c in "0123456789abcdef" for c in suffix))


def get_or_create_memory_session(user_id: str, actor_id: str) -> str:
    """The Memory thread the agent writes history under. Stored separately from the
    runtime session so the two can be rotated independently: /reset rotates only
    this one, /reconnect rotates only the runtime one, /new rotates both. Rotating
    means "use a new id" — no events are deleted."""
    item = _table.get_item(Key={"PK": f"USER#{user_id}", "SK": "MEMSESSION"}).get("Item")
    if item and item.get("memorySessionId"):
        if thread_belongs(actor_id, item["memorySessionId"]):
            return item["memorySessionId"]
        # A rotated id from before threads carried the actor's prefix: the agent refuses it
        # now, so it is replaced once. The thread is kept, just no longer read.
        log.info("memory session for %s predates the actor prefix; rotating", actor_id)
        return rotate_memory_session(user_id, actor_id)
    # Seed with the agent's legacy per-user id so existing history stays visible.
    mem_sid = _default_thread(actor_id)
    _table.put_item(Item={"PK": f"USER#{user_id}", "SK": "MEMSESSION",
                          "memorySessionId": mem_sid, "createdAt": int(time.time())})
    return mem_sid


def rotate_memory_session(user_id: str, actor_id: str) -> str:
    """Point the user at a brand-new Memory thread (old events are kept, just no
    longer read). Returns the new id, under the actor's prefix."""
    mem_sid = f"{_default_thread(actor_id)}-{uuid.uuid4().hex[:12]}"
    _table.put_item(Item={"PK": f"USER#{user_id}", "SK": "MEMSESSION",
                          "memorySessionId": mem_sid, "createdAt": int(time.time())})
    return mem_sid
