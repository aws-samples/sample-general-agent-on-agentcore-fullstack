"""AgentCore container server: the HTTP contract on 8080.

  GET  /ping          -> {"status":"Healthy"}   (must respond within seconds)
  POST /invocations   -> the router's JSON protocol, or an AG-UI RunAgentInput
      chat       : {action,actorId,message,email?} -> {reply}
      chat_async : same + chatId -> {accepted:true}; result is pushed to the chat
                   (messageId/reactionId: the router's progress marker, cleared at the end)
      warmup     : {action} -> {ready:true}
      status     : {action} -> {ready, instance, uptime, sessionAge, model, + clock diags}
      history_stats / clear_history / transcript / auth_status / reauth
      no `action` -> AG-UI, answered as an SSE event stream (see agui.py)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid

_T_IMPORT_START = time.monotonic()

from aiohttp import web

import agent_core   # pulls in langchain/langgraph, boto3 and mcp — the bulk of start-up
import agui
import lark_3lo

_T_IMPORT_DONE = time.monotonic()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("agent.server")

# Identity and age of the compute serving a request, both established on the first
# request rather than at import.
#
# Deriving either at import is wrong on Runtime V2, which starts an instance by restoring
# a snapshot taken once per version: anything computed before the snapshot is copied
# identically into every instance. Measured — two V2 instances that were demonstrably
# separate (independent per-process counters, neither seeing the other's marker file)
# reported the same import-time uuid, and /proc/uptime read ~92.6 s on both because it
# dates the snapshot's kernel rather than the restore. On V1 the same code reported
# distinct uuids and a true ~18.5 s kernel age. State written after the restore is the
# only thing that distinguishes instances, so that is when these are taken.
#
# The cost on V1 is nil: the first request arrives moments after start-up, so process age
# still includes the imports it used to be measured from. What neither value can do is
# prove isolation; concurrent sessions not seeing each other's files is what does that.
_instance_id: str | None = None
_serving_since: float | None = None
_identity_lock = threading.Lock()


def _mark_serving() -> None:
    global _instance_id, _serving_since
    if _instance_id is not None:
        return
    with _identity_lock:
        if _instance_id is None:
            _serving_since = time.monotonic()
            _instance_id = uuid.uuid4().hex[:8]


def _instance() -> str:
    """This instance's id, generated on first use and fixed for its lifetime."""
    _mark_serving()
    return _instance_id  # type: ignore[return-value]


def _kernel_uptime() -> float | None:
    """Seconds since the kernel booted, i.e. the microVM's own age — maintained by
    the kernel, not by this process. Comparing it with the process's `uptime` is what
    tells the two apart: process age can never exceed it, so a process claiming to be
    older than its kernel would mean the probe (or the guest clock) is wrong."""
    try:
        with open("/proc/uptime", encoding="utf-8") as f:
            return round(float(f.read().split()[0]), 1)
    except Exception:
        return None

# When each session was first seen here, so /status can report how long this compute
# has served this session — distinct from the microVM's own age (/proc/uptime) and
# from this process's age. Bounded: a long-lived process can see many sessions and
# this is only a diagnostic.
_SESSIONS_TRACKED = 256
_session_first_seen: dict[str, float] = {}
_HTTP_PORT = int(os.environ.get("PORT", "8080"))


# ----------------------------- HTTP contract --------------------------------

async def handle_ping(request: web.Request) -> web.Response:
    # HealthyBusy tells AgentCore a background turn is still running, so the
    # container isn't reclaimed out from under it.
    status = "HealthyBusy" if agent_core.busy() else "Healthy"
    return web.json_response({"status": status})


async def handle_invocations(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    action = payload.get("action", "chat")


    # AgentCore passes the runtime session id on every request (verified), which is
    # what lets this process report per-session age rather than only its own.
    sid = request.headers.get("x-amzn-bedrock-agentcore-runtime-session-id", "")
    # Under CUSTOM_JWT inbound the Runtime validates the caller's JWT and hands us a
    # workload access token derived from it. Three header aliases carry the same value
    # (measured); read them in order and pass it down — using it instead of naming a
    # user is what makes this turn's identity unforgeable. Empty under SigV4 inbound.
    workload_token = next(
        (request.headers[h] for h in (
            "x-amzn-bedrock-agentcore-runtime-workload-accesstoken",
            "x-amz-bedrock-agentcore-identity-wat",
            "workloadaccesstoken",
        ) if request.headers.get(h)), "")
    # A browser on the AG-UI path posts a RunAgentInput, which carries no `action`. It is
    # answered as an SSE event stream; everything below is the router's JSON protocol. Placed
    # after workload_token so both paths use the one value — reading the headers separately
    # here is how the two drifted.
    if agui.is_agui_request(payload):
        log.info("dispatch: agui  instance=%s session=%s", _instance(), sid[:24])
        return await agui.handle(request, payload, workload_token)

    now = time.monotonic()
    if sid:
        if sid not in _session_first_seen and len(_session_first_seen) >= _SESSIONS_TRACKED:
            _session_first_seen.pop(next(iter(_session_first_seen)))  # oldest inserted
        _session_first_seen.setdefault(sid, now)

    if action == "status":
        return web.json_response({
            "ready": True,
            # Set by provision.sh at deploy time — the running container has no other way to know 
            # its own platformVersion, which is a control-plane field, not an AgentCore-injected one.
            "platformVersion": os.environ.get("AGENT_PLATFORM_VERSION", "unknown"),
            "instance": _instance(),
            # How long this instance has been serving, from its first request. Not the
            # microVM's age, and not "how long you have been served" either.
            "uptime": round(now - (_serving_since or now), 1),
            # The kernel's own age. True on V1; on V2 it dates the snapshot's kernel
            # rather than this restore, so read it as a lower bound there.
            "kernelUptime": _kernel_uptime(),
            # How long THIS session has been served by THIS process, counted from its
            # first request. Independent of when the microVM booted.
            "sessionAge": round(now - _session_first_seen[sid], 1) if sid else None,
            # Reported by the container that actually calls Bedrock: the router's own
            # config can disagree with what was baked into this Runtime.
            "model": agent_core.model_id(),
            "cacheTtl": agent_core.cache_ttl(),
            "settings": agent_core.model_settings(),
        })

    if action == "warmup":
        return web.json_response({"ready": True})


    # Thread bookkeeping for /status and /clear. Here rather than in the router because
    # the conversation is LangGraph state now — reading or deleting it needs the
    # checkpointer, and agent_core is the only module allowed to know about the framework.
    if action in ("history_stats", "clear_history"):
        actor_id = payload.get("actorId") or payload.get("userId") or "anonymous"
        mem_sid = payload.get("memorySessionId", "")
        fn = (agent_core.history_stats if action == "history_stats"
              else agent_core.clear_history)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, fn, actor_id, mem_sid)
            return web.json_response(result)
        except Exception as e:
            log.exception("%s failed", action)
            return web.json_response({"error": str(e)})

    # The transcript for a reloaded browser tab. The claimed actorId cannot be trusted here
    # the way it can on the chat path: a thread is addressed by the actor alone, so believing
    # the claim would hand any valid token holder somebody else's conversation. Building the
    # session first is what checks it — lark_3lo asks Lark whose vaulted token this is and
    # refuses a mismatch — and it is the one identity path proven end to end.
    if action == "transcript":
        actor_id = payload.get("actorId") or ""
        if not actor_id:
            return web.json_response({"messages": [], "error": "actorId required"})
        session = await agent_core.aget_session(actor_id, workload_token=workload_token)
        if session.get("identity_error") or session.get("auth_url"):
            return web.json_response({"messages": [], "unverified": True})
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent_core.transcript, actor_id, session["mem_sid"])
            return web.json_response(result)
        except Exception as e:
            log.exception("transcript failed")
            return web.json_response({"messages": [], "error": str(e)})

    # The tool list is per session — it depends on which servers answered and which systems
    # the user has granted — so only the container serving this user can say what it is.
    if action == "tools":
        actor_id = payload.get("actorId") or payload.get("userId") or "anonymous"
        try:
            return web.json_response(await agent_core.atools_overview(
                actor_id, payload.get("memorySessionId", ""), workload_token))
        except Exception as e:
            log.exception("tools failed")
            return web.json_response({"tools": [], "error": str(e)})

    if action == "auth_status":
        actor_id = payload.get("actorId") or payload.get("userId") or "anonymous"
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent_core.auth_status, actor_id, workload_token)
            return web.json_response(result)
        except Exception as e:
            log.exception("auth_status failed")
            return web.json_response({"error": str(e)})

    if action == "reauth":
        actor_id = payload.get("actorId") or payload.get("userId") or "anonymous"
        idp = payload.get("message", "") or "lark"   # router sends the idp key here
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent_core.reauth, actor_id, idp, workload_token
            )
            return web.json_response(result)
        except Exception as e:
            log.exception("reauth failed")
            return web.json_response({"error": str(e)})

    if action == "chat_async":
        actor_id = payload.get("actorId") or payload.get("userId") or "anonymous"
        message = payload.get("message", "")
        chat_id = payload.get("chatId", "")
        email = payload.get("email", "")
        mem_sid = payload.get("memorySessionId", "")
        # The router's in-progress reaction, for us to clear when the turn ends.
        msg_id = payload.get("messageId", "")
        reaction_id = payload.get("reactionId", "")
        # Set on the consent-resume replay: rebuild the session so the just-vaulted
        # token is picked up instead of the cached unauthorized one.
        fresh_session = bool(payload.get("freshSession"))
        # Handles, not bytes: the agent fetches the image itself, so nothing large travels
        # through the router's parked state or its async self-invoke.
        image_handles = payload.get("images") or []
        # An image with no caption is a real message, so only the chat is required.
        if not (message or image_handles) or not chat_id:
            return web.json_response({"error": "chatId and a message or image required"},
                                     status=400)
        try:
            # Returns as soon as the work is accepted; the result is pushed to the
            # chat later. Consent still comes back inline (see chat_async).
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent_core.chat_async, actor_id, message, chat_id, email,
                mem_sid, msg_id, reaction_id, fresh_session, workload_token,
                image_handles,
            )
            return web.json_response(result)
        except Exception as e:
            log.exception("chat_async failed")
            return web.json_response({"error": str(e)})

    if action == "chat":
        log.info("dispatch: chat  instance=%s session=%s", _instance(), sid[:24])
        # A caller that cannot name the user — the A2A adapter forwards a bearer and nothing
        # else — gets the actor derived from the token instead of defaulting to "anonymous",
        # which would fail the ownership check and ask for consent forever.
        actor_id = payload.get("actorId") or payload.get("userId") or ""
        if not actor_id:
            kind, derived = await asyncio.get_event_loop().run_in_executor(
                None, lark_3lo.actor_from_workload_token, workload_token)
            if kind != "actor":
                return web.json_response(
                    {"reply": "无法确认这次请求代表谁。请先在 Lark 里完成一次授权。",
                     "needs_auth": True}, status=200)
            actor_id = derived
        message = payload.get("message", "")
        email = payload.get("email", "")
        # Memory thread id is chosen by the caller (router) — that's what makes
        # /reset and /new able to start a new thread without deleting anything.
        mem_sid = payload.get("memorySessionId", "")
        if not message:
            return web.json_response({"error": "message required"}, status=400)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent_core.chat_result, actor_id, message, email, mem_sid,
                workload_token,
            )
            # {reply, needs_auth, auth_url?} — router drives the consent wait.
            return web.json_response(result)
        except Exception as e:
            log.exception("chat failed")
            # Return 200: AgentCore wraps non-2xx as RuntimeClientError and drops
            # the body, hiding the real error from callers. The Router surfaces
            # the error field instead.
            return web.json_response({"error": str(e)})

    return web.json_response({"error": f"unknown action: {action}"}, status=400)


# ------------------------------- bootstrap ----------------------------------

def build_http_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/ping", handle_ping)
    app.router.add_post("/invocations", handle_invocations)
    return app


def _log_startup_breakdown() -> None:
    """Break down the cold start, since AgentCore documents none of it and X-Ray
    can't see it (its trace root is this process, so everything before OTel starts is
    invisible). Log-only on purpose: /status stays readable, and whoever is measuring
    cold starts is reading CloudWatch anyway.

    kernel age is the microVM's; the gap between it and this process's age is what
    AgentCore spent booting and pulling the image before we got control.

    On Runtime V2 this describes the snapshot build, not a restore: it runs before the
    snapshot is taken, so every restored instance inherits the figures rather than
    producing its own. No instance id is reported for the same reason — one generated
    here would be identical across every instance. Restored instances report their own
    identity through /status instead."""
    kernel = _kernel_uptime()
    proc = time.monotonic() - _T_IMPORT_START
    imports = _T_IMPORT_DONE - _T_IMPORT_START
    before_us = round(kernel - proc, 1) if kernel is not None else None
    log.info(
        "startup breakdown: microVM=%ss process=%.1fs imports=%.1fs before-our-code=%ss",
        kernel, proc, imports, before_us,
    )


async def main() -> None:
    http_runner = web.AppRunner(build_http_app())
    await http_runner.setup()
    await web.TCPSite(http_runner, "0.0.0.0", _HTTP_PORT).start()  # nosec B104 # nosemgrep: avoid-bind-to-all-interfaces — the Runtime routes to this port
    log.info("HTTP contract on :%d", _HTTP_PORT)
    _log_startup_breakdown()

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
