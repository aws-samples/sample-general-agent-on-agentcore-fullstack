#!/usr/bin/env bash
# Run every unit suite: tests/<component>/ per deployed component. The e2e smoke tests in
# tests/e2e/ need a deployed stack and are run separately — see README.md here.
#
# Each suite gets its own process: the router and shim both define index.py, and the agent
# and router both define identity, so a single pytest session would import the wrong module
# for one of them.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# Each suite runs on the Python and the requirements its code is deployed with, outside the
# root project (the CDK app). The agent image and the Lambdas are all Python 3.13.
agent_py()  { uv run --no-project --python 3.13 --with-requirements agent/requirements.txt --with pytest "$@"; }
lambda_py() { uv run --no-project --python 3.13 --with boto3 --with pytest "$@"; }

echo "== agent =="
# The LangGraph stack is pure Python, so the turn loop is tested through a real compiled
# graph rather than a copy of its source. The A2A adapter stubs the agent loop out, so it
# gets a process of its own.
agent_py python -m pytest tests/agent -q --ignore=tests/agent/test_a2a.py

echo "== a2a adapter =="
agent_py python -m pytest tests/agent/test_a2a.py -q

echo "== router =="
lambda_py --with-requirements lambda/router/requirements.txt python -m pytest tests/router/test_router.py -q

echo "== shim =="
lambda_py python -m pytest tests/shim/test_shim.py -q

echo "== google mcp (node) =="
# Node's own runner: this server has no dependencies, so a toolchain for one file would be
# the wrong trade. Pointed at the file — the server listens on import.
node --test tests/mcp/test_google.mjs

echo "== approval guards (node) =="
# Node's own runner — the guards are dependency-free by design, so no new toolchain.
# Pointed at the file, not the directory: server.js starts listening on load.
node --test tests/mcp/test_approval_guards.mjs

echo "== all green =="
