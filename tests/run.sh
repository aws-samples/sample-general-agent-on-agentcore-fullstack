#!/usr/bin/env bash
# Run every unit suite. These live next to the code they test (the agent and each
# Lambda), so this walks them; the e2e smoke tests in this directory need a
# deployed stack and are run separately — see README.md here.
#
# Each suite gets its own process: the router and shim dirs both define index.py,
# and the agent and router both define identity.py, so a single pytest session
# would import the wrong module for one of them.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# Each suite runs on the Python and the requirements its code is deployed with, outside the
# root project (the CDK app). The agent image and the Lambdas are all Python 3.13.
agent_py()  { uv run --no-project --python 3.13 --with-requirements agent/requirements.txt --with pytest "$@"; }
lambda_py() { uv run --no-project --python 3.13 --with boto3 --with pytest "$@"; }

echo "== agent =="
# The LangGraph stack is pure Python, so the turn loop is tested through a real compiled
# graph rather than a copy of its source.
agent_py python -m pytest agent/test_agent.py -q

echo "== code tools =="
# Needs the agent's stack because it asserts against agent_core's session teardown.
agent_py python -m pytest agent/test_code_tools.py -q

echo "== a2a adapter =="
agent_py python -m pytest agent/test_a2a.py -q

echo "== router =="
lambda_py --with-requirements lambda/router/requirements.txt python -m pytest lambda/router/test_router.py -q

echo "== shim =="
lambda_py python -m pytest lambda/shim/test_shim.py -q

echo "== google mcp (node) =="
# Node's own runner: this server has no dependencies, so a toolchain for one file would be
# the wrong trade. Pointed at the file — the server listens on import.
node --test mcp-servers/google/test_server.mjs

echo "== approval guards (node) =="
# Node's own runner — the guards are dependency-free by design, so no new toolchain.
# Pointed at the file, not the directory: server.js starts listening on load.
node --test mcp-servers/approval/test_guards.mjs

echo "== all green =="
