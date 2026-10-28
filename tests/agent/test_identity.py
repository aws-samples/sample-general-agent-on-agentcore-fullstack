"""Who is calling, and whose grants the tools use: tokens, vault ownership, consent, /auth."""

import json
import os
import re
from unittest import mock


import tools
from identity import caller, grants, lark_3lo
from runtime import sessions
from tools import web_search


from support import AGENT, _dummy_tool, _make_tool


def test_a_second_downstream_system_is_absent_not_broken():
    """Google unconfigured must cost nothing: no session build failure, no tool, and /auth
    google says so instead of pretending. The same convention as search and long-term
    memory — an optional system that is not deployed simply is not there."""
    with mock.patch.dict(grants._DOWNSTREAMS, {}, clear=True):
        r = sessions.reauth("lark:ou_x", "google")
    assert r["needs_auth"] is False
    assert "未配置" in r["reply"]


def test_reauth_still_refuses_an_idp_nobody_wired_up():
    """The list is explicit on purpose: a typo should not silently start a consent flow for
    a provider that does not exist."""
    r = sessions.reauth("lark:ou_x", "dropbox")
    assert r["needs_auth"] is False and "尚未接入" in r["reply"]


# Search is optional; with no gateway configured the agent must simply run
# without the tool rather than failing to build a session.


# A user without a vaulted Lark token still gets a working session: lark-mcp lists
# its tools without one and only rejects tools/call. Consent is raised when a tool
# is actually reached, so plain chat ("hello") is never gated on it.

def test_auth_marker_matches_the_mcp_server():
    """The client detects the auth wall by matching lark-mcp's rejection text. It is
    a cross-process string contract: if the server's wording changes and this does
    not, the user silently stops being offered a consent link."""
    here = AGENT
    src = open(os.path.join(here, "identity", "grants.py"), encoding="utf-8").read()
    marker = re.search(r'_NEEDS_TOKEN_MARKER = "([^"]+)"', src).group(1)
    for name in ("lark-cli", "approval", "google"):
        server = open(os.path.join(here, "..", "mcp-servers", name, "server.js"), encoding="utf-8").read()
        assert marker in server, (
            f"grants expects {marker!r} but mcp-servers/{name}/server.js no longer says it")


def test_hit_auth_wall_reads_tool_results_not_the_final_reply():
    """The check must inspect tool results, not the model's text answer: the model
    paraphrases errors, so 'no user token is available' in the reply slips past a string
    check on the reply — verified end-to-end before this fix."""
    hit = grants._hit_auth_wall_from_tool_results
    lark = {"consent": {"lark": "https://consent"}, "tool_idp": {"lark_docs": "lark"}}

    # 1. Nothing collected → nothing to inspect, so no auth wall.
    assert hit(lark) is None
    assert hit({**lark, "tool_texts": []}) is None
    # 2. A paraphrase of the error, not the tool's own words → NOT a wall. This is
    #    exactly the case that fooled the first version of this function.
    assert hit({**lark, "tool_texts": [
        ("lark_docs", "It looks like no user token is available — please authorize")]}) is None
    # 3. The tool result carrying the marker verbatim → wall, for that tool's system.
    assert hit({**lark, "tool_texts": [("lark_docs", "no user token (authorize first)")]}) == "lark"
    # 4. Same tool result but the system is granted → no wall, because then the refusal
    #    means something else and consent won't fix it.
    assert hit({"tool_idp": {"lark_docs": "lark"}, "consent": {},
                "tool_texts": [("lark_docs", "no user token (authorize first)")]}) is None


def test_a_web_users_lark_is_granted_on_demand_like_google():
    """A web user has no Lark identity to match, so the Lark-user path — which mints a link at
    build and refuses a grant whose open_id differs (wrong_owner) — would lock them out for
    good. For them Lark is one more system: read without minting, its tools listed, and a
    link minted only when they ask to authorise."""
    lark_user_path, opened = [], []

    async def fake_open(stack, connection):
        opened.append(connection.get("url"))
        return [_dummy_tool]

    with mock.patch.object(lark_3lo, "get_user_lark_token",
                           side_effect=lambda *a, **k: lark_user_path.append(a)), \
         mock.patch.object(lark_3lo, "peek_user_token_for", return_value=""), \
         mock.patch.object(lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"url": url or "lark-mcp"}), \
         mock.patch.object(tools, "_aopen_tools", fake_open), \
         mock.patch.object(web_search, "available", return_value=False):
        s = sessions._build_session("web:sub-carol", "",
                                    sessions._session_id_for("web:sub-carol"))
        assert lark_user_path == []
        assert "lark-mcp" in opened and s["consent"]["lark"] == ""
        assert not s["auth_url"] and not s["identity_error"]
        assert grants.auth_status("web:sub-carol")["lark"] is False
        with mock.patch.object(lark_3lo, "get_user_token_for",
                               return_value=("auth_url", "https://lark-consent")) as generic:
            r = sessions.reauth("web:sub-carol", "lark")
    assert r["auth_url"] == "https://lark-consent" and generic.call_count == 1
    assert lark_user_path == []


def test_a_vaulted_lark_token_is_used_only_by_its_own_actor():
    """Consent completion binds the token to whatever userId the return-url was told, not to
    the account that signed in — so forwarding a consent link vaults someone else's token
    under your name. Checked at the point of use, failing closed, once per token, in a
    bounded cache."""
    asked = []

    def owner(token):
        asked.append(token)
        return {"tok-a": "ou_alice"}.get(token, "")
    with mock.patch.object(lark_3lo, "_token_owner", owner), \
         mock.patch.dict(lark_3lo._VERIFIED, clear=True):
        assert lark_3lo._belongs_to("tok-a", "lark:ou_alice") is True
        assert lark_3lo._belongs_to("tok-a", "lark:ou_alice") is True
        assert asked == ["tok-a"], "the owner is read once per token, not every turn"
        assert lark_3lo._belongs_to("tok-a", "lark:ou_bob") is False     # Alice's grant, Bob's slot
        assert lark_3lo._belongs_to("tok-x", "lark:ou_alice") is False   # owner unknown: refused
    with mock.patch.object(lark_3lo, "_token_owner", lambda t: t.replace("tok-", "ou_")), \
         mock.patch.dict(lark_3lo._VERIFIED, clear=True):
        for i in range(lark_3lo._VERIFIED_MAX + 20):
            lark_3lo._belongs_to(f"tok-{i}", f"lark:ou_{i}")
        assert len(lark_3lo._VERIFIED) <= lark_3lo._VERIFIED_MAX


def test_a_mismatched_lark_grant_is_refused_unforced_and_an_empty_vault_is_forced():
    """The vault is keyed by the caller's workload identity, so a mismatch means the claimed
    actor is wrong; forcing a fresh flow there destroyed the caller's working grant —
    measured: a token at 13:16:40, gone one second later. With nothing vaulted the reverse
    holds: an unforced authorizationUrl cannot be completed ("Invalid or expired session")."""
    vault, forced = {"token": "tok-a"}, []

    def fetch(wat, actor, force=False, **kw):
        forced.append(force)
        return {"accessToken": vault["token"]} if vault["token"] else {
            "authorizationUrl": f"https://consent/{force}"}
    with mock.patch.object(lark_3lo, "_workload_token_for", return_value="wat"), \
         mock.patch.object(lark_3lo, "_fetch_vaulted", side_effect=fetch), \
         mock.patch.object(lark_3lo, "_belongs_to", return_value=False):
        assert lark_3lo.get_user_lark_token("lark:ou_bob") == ("wrong_owner", "")
        assert forced == [False]
        vault["token"], forced[:] = "", []
        assert lark_3lo.get_user_lark_token("lark:ou_bob") == ("auth_url", "https://consent/True")
        assert forced == [False, True]


def test_auth_status_names_the_account_each_grant_came_from():
    """Read with the grant itself, so a user can tell which of their accounts it is; a lookup
    that fails costs the name, never the status."""
    tokens = {"lark-prov": "lark-tok", "google-prov": "g-tok"}
    with mock.patch.object(lark_3lo, "configured", return_value=True), \
         mock.patch.object(lark_3lo, "lark_provider", return_value=("lark-prov", [])), \
         mock.patch.object(grants, "_DOWNSTREAMS", {"google": ("u", "google-prov", [])}), \
         mock.patch.object(lark_3lo, "peek_user_token_for",
                           side_effect=lambda actor, prov, scopes, wt: tokens.get(prov, "")), \
         mock.patch.dict(grants._ACCOUNT_OF, {"lark": lambda t: "XUESONG · me@myners.net",
                                              "google": lambda t: "me@myners.net"}):
        st = grants.auth_status("web:s-1")
        assert st["lark"] is True and st["google"] is True
        assert st["accounts"] == {"lark": "XUESONG · me@myners.net", "google": "me@myners.net"}
        tokens.pop("google-prov")
        with mock.patch.dict(grants._ACCOUNT_OF, {"lark": lambda t: ""}):
            st = grants.auth_status("web:s-1")
        assert st["lark"] is True and st["google"] is False and st["accounts"] == {}


def test_reauthorising_a_system_already_granted_starts_a_new_consent():
    """How a user switches accounts: /auth google answered "Already authorized." while a grant
    was vaulted, so there was no way to consent as anybody else."""
    fetched = []

    def fake_fetch(wat, actor, force=False, provider="", scopes=None):
        fetched.append(force)
        return {"authorizationUrl": "https://new-consent"} if force else {"accessToken": "old"}
    with mock.patch.dict(grants._DOWNSTREAMS, {"google": ("https://g", "p", ["openid"])}), \
         mock.patch.object(lark_3lo, "_workload_token_for", return_value="wat"), \
         mock.patch.object(lark_3lo, "_fetch_vaulted", side_effect=fake_fetch):
        r = sessions.reauth("web:s-1", "google")
        assert r["auth_url"] == "https://new-consent" and fetched == [True]
        # An ordinary fetch still hands back the vaulted grant.
        assert lark_3lo.get_user_token_for("web:s-1", "p", ["openid"]) == ("token", "old")


def test_whoami_says_how_they_signed_in_as_whom_and_what_they_granted():
    with mock.patch.object(grants, "auth_status", return_value={
            "lark": True, "google": False, "accounts": {"lark": "XUESONG"}}):
        tool = grants._whoami_tool("web:s-carol", "wat")
        out = json.loads(tool.invoke({}, config={"configurable": {"account": "carol@example.com"}}))
        assert out == {"signed_in_with": "the web page (its own sign-in)",
                       "account": "carol@example.com", "user_id": "web:s-carol",
                       "authorised": {"lark": "XUESONG"}, "not_authorised": ["google"]}
        lark = json.loads(grants._whoami_tool("lark:ou_alice", "wat").invoke({}))
        assert lark["signed_in_with"] == "Lark" and lark["account"] == "XUESONG"
    # The account is never a parameter the model could fill in.
    assert "account" not in (tool.args or {})


def test_the_actor_is_the_username_of_a_verified_access_token(_signer):
    import time as _t
    from cryptography.hazmat.primitives.asymmetric import rsa
    assert caller.actor([_signer()]) == "lark:ou_alice"
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    for bad in (_signer(signing_key=other_key),          # not the pool's signature
                _signer(client_id="someone-else"),       # another app's token
                _signer(token_use="id"),                 # an ID token
                _signer(iss="https://evil.example"),
                _signer(exp=int(_t.time()) - 60)):
        assert caller.actor([bad]) == "", bad
    # Which of two headers the platform checked is unknown, so two are refused outright.
    assert caller.actor([_signer(), _signer(username="lark:ou_bob")]) == ""
    assert caller.actor([]) == "" and caller.actor(["Basic abc"]) == ""
    # A container clock a little behind Cognito's must not lock out a token minted just now.
    assert caller.actor([_signer(iat=int(_t.time()) + 30)]) == "lark:ou_alice"
    assert caller.actor(["bearer" + _signer()[len("Bearer"):]]) == "lark:ou_alice"


def test_the_issuing_client_decides_the_actors_namespace(_signer):
    """Cognito signs client_id, so a caller cannot choose it. A web user is `web:{sub}`, and
    the router's client yields only `lark:` actors, so a stray user of it is nobody."""
    web = dict(client_id="web-client", sub="sub-carol")
    assert caller.actor([_signer(**web, username="carol@example.com")]) == "web:sub-carol"
    # A Lark user's own account signed in at the hosted login carries their sub: refused.
    assert caller.actor([_signer(**web, username="lark:ou_victim")]) == ""
    assert caller.actor([_signer(username="service:websearch")]) == ""
    with mock.patch.object(caller, "_WEB_CLIENT", ""):          # no web login deployed
        assert caller.actor([_signer(**web)]) == ""


def test_github_without_a_grant_offers_one_tool_that_asks_for_it():
    """GitHub's server lists nothing without the user's token, so an ungranted session would
    have no GitHub tool to reach and never be asked; a stand-in walls on first use instead."""
    opened = []

    async def fake_open(stack, connection):
        opened.append(connection)
        return [_make_tool("get_me"), _make_tool("list_branches")]
    with mock.patch.dict(grants._DOWNSTREAMS, {"github": ("https://gh/mcp/", "gh-prov", ["repo"])},
                         clear=True), \
         mock.patch.object(lark_3lo, "get_user_lark_token", return_value=("token", "lark")), \
         mock.patch.object(lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"url": url or "lark-mcp"}), \
         mock.patch.object(tools, "_aopen_tools", fake_open), \
         mock.patch.object(web_search, "available", return_value=False):
        with mock.patch.object(lark_3lo, "peek_user_token_for", return_value=""):
            s = sessions._build_session("lark:ou_x", "", "mem-gh1")
        assert s["consent"]["github"] == "" and s["tool_idp"]["github"] == "github"
        assert all("gh/mcp" not in c.get("url", "") for c in opened)   # nothing to list yet
        assert grants._walled_idp(s, "github") == "github"
        opened.clear()
        with mock.patch.object(lark_3lo, "peek_user_token_for", return_value="gho_tok"), \
             mock.patch.dict(grants._VENDOR_MCP, {"github": (grants._github_connection,
                                                             lambda tok: False)}):
            s = sessions._build_session("lark:ou_x", "", "mem-gh2")
        gh = [c for c in opened if c.get("url") == "https://gh/mcp/"][0]
        assert gh["headers"]["Authorization"] == "Bearer gho_tok"
        assert gh["headers"]["X-MCP-Readonly"] == "true" and "repos" in gh["headers"]["X-MCP-Toolsets"]
        assert "github" not in s["consent"] and s["tool_idp"]["get_me"] == "github"
        assert "github" not in s["tool_idp"]                              # the stand-in is gone


def test_a_github_grant_github_refuses_is_asked_for_again_past_the_vault():
    """Revoked on GitHub's side, the vault still holds it: an ordinary consent would answer
    "already granted" for ever, so the new link is forced."""
    with mock.patch.dict(grants._DOWNSTREAMS, {"github": ("https://gh/mcp/", "gh-prov", ["repo"])},
                         clear=True), \
         mock.patch.dict(grants._VENDOR_MCP, {"github": (grants._github_connection,
                                                         lambda tok: True)}), \
         mock.patch.object(lark_3lo, "configured", return_value=False), \
         mock.patch.object(lark_3lo, "peek_user_token_for", return_value="gho_dead"), \
         mock.patch.object(tools, "_aopen_tools", side_effect=AssertionError("opened")), \
         mock.patch.object(web_search, "available", return_value=False):
        s = sessions._build_session("web:s-1", "", "mem-gh3")
        assert s["tool_idp"]["github"] == "github" and s["consent"]["github"] == ""
        with mock.patch.object(lark_3lo, "get_user_token_for",
                               return_value=("auth_url", "https://consent")) as get:
            assert grants._consent_link(s, "github") == "https://consent"
        assert get.call_args.kwargs["force"] is True


def test_auth_status_counts_a_grant_github_refuses_as_none():
    with mock.patch.dict(grants._DOWNSTREAMS, {"github": ("https://gh/mcp/", "gh-prov", ["repo"])},
                         clear=True), \
         mock.patch.object(lark_3lo, "configured", return_value=False), \
         mock.patch.object(lark_3lo, "peek_user_token_for", return_value="gho_tok"), \
         mock.patch.dict(grants._ACCOUNT_OF, {"github": lambda tok: "aleck31"}):
        with mock.patch.dict(grants._VENDOR_MCP, {"github": (None, lambda tok: True)}):
            assert grants.auth_status("web:s-1") == {"github": False, "accounts": {}}
        with mock.patch.dict(grants._VENDOR_MCP, {"github": (None, lambda tok: False)}):
            assert grants.auth_status("web:s-1") == {"github": True,
                                                     "accounts": {"github": "aleck31"}}


def test_github_refuses_only_a_401():
    with mock.patch("httpx.get", return_value=mock.Mock(status_code=401)):
        assert grants._github_refuses("t") is True
    with mock.patch("httpx.get", return_value=mock.Mock(status_code=503)):
        assert grants._github_refuses("t") is False
    with mock.patch("httpx.get", side_effect=OSError):
        assert grants._github_refuses("t") is False


def test_auth_shows_the_github_login_that_granted_it():
    resp = mock.Mock()
    resp.json.return_value = {"login": "aleck31"}
    with mock.patch("httpx.get", return_value=resp) as get:
        assert grants._github_account("gho_tok") == "aleck31"
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer gho_tok"
    assert grants._label("github") == "GitHub"


def test_consent_is_asked_for_the_system_the_tool_acts_on():
    """A Google grant without a Lark one was walled with the Lark card for google_whoami:
    the wall keyed on Lark alone. Each tool now walls only on its own system's grant."""
    tool_idp = {"lark_docs": "lark", "google_whoami": "google"}
    s = {"consent": {"lark": "https://lark"}, "tool_idp": tool_idp}
    assert grants._walled_idp(s, "lark_docs") == "lark"
    assert grants._walled_idp(s, "google_whoami") is None      # Google is granted
    assert grants._walled_idp(s, "memory_save") is None        # untagged never walls
    s = {"consent": {"google": ""}, "tool_idp": tool_idp}
    assert grants._walled_idp(s, "google_whoami") == "google"
    assert grants._walled_idp(s, "lark_docs") is None


def test_downstream_consent_is_recorded_at_build_and_minted_only_when_reached():
    """Minting at build would start a consent flow nobody asked for and kill any link the
    user is already holding; the build only peeks, the wall mints."""
    async def fake_open(stack, connection):
        return [_make_tool("google_whoami")] if connection["url"] else [_dummy_tool]

    with mock.patch.dict(grants._DOWNSTREAMS, {"google": ("https://g", "p", ["openid"])},
                         clear=True), \
         mock.patch.object(lark_3lo, "get_user_lark_token",
                           return_value=("token", "lark-tok")), \
         mock.patch.object(lark_3lo, "peek_user_token_for", return_value=""), \
         mock.patch.object(lark_3lo, "get_user_token_for",
                           return_value=("auth_url", "https://google-consent")) as mint, \
         mock.patch.object(lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"url": url}), \
         mock.patch.object(tools, "_aopen_tools", fake_open), \
         mock.patch.object(web_search, "available", return_value=False):
        s = sessions._build_session("lark:ou_x", "", "mem1")
        assert s["consent"] == {"google": ""} and not s["auth_url"]
        assert s["tool_idp"]["google_whoami"] == "google"
        mint.assert_not_called()
        assert grants._consent_link(s, "google") == "https://google-consent"
        mint.assert_called_once()
