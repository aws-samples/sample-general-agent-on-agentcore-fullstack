"""Generate doc_routes.json: every Lark OpenAPI that reaches Drive documents, with its method and
path template, read from the official Node SDK (@larksuiteoapi/node-sdk). The guard (guard.js)
checks a `lark_api` call against this table; a document-family path not in it is refused.

Regenerate when the SDK adds APIs:
    d=$(mktemp -d) && (cd "$d" && npm pack @larksuiteoapi/node-sdk && tar xzf *.tgz)
    python3 gen_doc_routes.py "$d/package/lib/index.js"
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The API families whose calls can reach a Drive document's content or metadata.
DOC_FAMILIES = ["base", "bitable", "board", "doc", "docs", "docx", "drive", "minutes", "search", "sheets", "suite", "wiki"]
# Older APIs Lark still serves that the SDK does not carry: Drive search and metadata, user search, Sheets v2
# (the only cell read/write), legacy Docs, and the Drive explorer. `*` is any method, and a
# template ending in `/*` covers every path beneath it.
EXTRA = [
    {"method": "POST", "template": "/open-apis/suite/docs-api/search/object"},
    {"method": "POST", "template": "/open-apis/suite/docs-api/meta"},
    {"method": "GET", "template": "/open-apis/search/v1/user"},
    {"method": "*", "template": "/open-apis/sheets/v2/spreadsheets/:spreadsheet_token/*"},
    {"method": "POST", "template": "/open-apis/sheets/v2/import"},
    {"method": "GET", "template": "/open-apis/sheets/v2/import/result"},
    {"method": "*", "template": "/open-apis/doc/v2/:doc_token/*"},
    {"method": "GET", "template": "/open-apis/doc/v2/meta/:doc_token"},
    {"method": "POST", "template": "/open-apis/doc/v2/create"},
    {"method": "GET", "template": "/open-apis/drive/explorer/v2/root_folder/meta"},
    {"method": "GET", "template": "/open-apis/drive/explorer/v2/folder/:folder_token/meta"},
    {"method": "GET", "template": "/open-apis/drive/explorer/v2/folder/:folder_token/children"},
    {"method": "POST", "template": "/open-apis/drive/explorer/v2/folder/:folder_token"},
    {"method": "POST", "template": "/open-apis/drive/explorer/v2/file/:folder_token"},
    {"method": "POST", "template": "/open-apis/drive/explorer/v2/file/copy/files/:file_token"},
    {"method": "DELETE", "template": "/open-apis/drive/explorer/v2/file/docs/:doc_token"},
    {"method": "DELETE", "template": "/open-apis/drive/explorer/v2/file/spreadsheets/:spreadsheet_token"},
]


def main(sdk_js: str) -> None:
    src = Path(sdk_js).read_text(errors="ignore")
    pairs = re.findall(r'url: fillApiPath\(`\$\{this\.domain\}(/open-apis/[^`]+)`, path\),\s*method: "(\w+)"', src)
    routes = sorted({(m.upper(), p) for p, m in pairs if p.split("/")[2] in DOC_FAMILIES})
    table = [{"method": m, "template": p} for m, p in routes] + EXTRA
    out = Path(__file__).with_name("doc_routes.json")
    out.write_text(json.dumps({"source": "@larksuiteoapi/node-sdk", "families": DOC_FAMILIES, "routes": table},
                              indent=1) + "\n")
    print(f"{len(table)} document routes → {out.name}")


if __name__ == "__main__":
    main(sys.argv[1])
