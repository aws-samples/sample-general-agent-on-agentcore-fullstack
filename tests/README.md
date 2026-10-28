# Tests

Two layers, deliberately separate.

## Unit tests — `tests/run.sh`

```bash
tests/run.sh
```

No AWS needed; everything is mocked. One directory per deployed component — `agent/` (a file per package, sharing `conftest.py` and `support.py`), `router/`, `shim/`, `mcp/` — and `run.sh` gives each suite its own process: the router and shim both define `index.py`, and the agent and router both define `identity`, so a single pytest session would import the wrong module for one of them. Each suite runs on the Python and requirements its code ships with: Python 3.13 throughout, the agent suites with `agent/requirements.txt` (the image's).

CI (`.github/workflows/test.yml`) runs `tests/run.sh` on every push to main and every pull request, and `bash -n` plus `shellcheck -S warning` over every tracked shell script. The deploy scripts have no unit tests: they are a sequence of AWS calls, so a real deploy is what verifies them.

Two constraints shape what these can cover:

- The agent modules are imported for real and the turn loop driven through an actual compiled graph with a scripted model, so `tests/run.sh` installs the LangGraph stack for that suite. This replaced an exec-a-slice-of-the-source pattern that existed only because Strands' wheels target ARM64 and wouldn't import on an x86 test host. Constructing the model touches no credentials; nothing calls Bedrock.
- Handlers that are mostly a sequence of AWS calls (`/reset`, `/new`, `/reconnect`, `/clear`) are left to the e2e path — mocking them would largely assert the mocks. The parts that actually carry risk are covered directly instead: Memory-thread rotation, conversational-only event counting, and the IdP registry's fallback.

## E2E smoke tests — need a deployed stack

These hit real resources and skip themselves unless the required env vars are set, so they're safe to leave alongside the unit runner.

| File | Verifies | Requires |
|---|---|---|
| `test_webhook_smoke.py` | Router accepts a signed Lark event and 200s fast; url_verification challenge echoes. | `WEBHOOK_URL`, `LARK_ENCRYPT_KEY` |

Run after `./deploy.sh`:

```bash
WEBHOOK_URL=... LARK_ENCRYPT_KEY=... \
  uv run --with cryptography --with pytest python -m pytest tests/e2e/test_webhook_smoke.py -v
```
