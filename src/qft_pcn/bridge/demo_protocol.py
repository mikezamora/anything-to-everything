"""End-to-end demo via subprocess JSON-RPC.

Spawns `python -m src.qft_pcn.bridge --once <request>` and feeds the
response to MockLLM.verbalize.
"""

from __future__ import annotations

import json
import subprocess
import sys

from .llm import MockLLM
from .demo import CANNED_PROMPT, CANNED_DSL, render_transcript


def main() -> int:
    llm = MockLLM(responses={CANNED_PROMPT: CANNED_DSL})
    dsl = llm.emit_dsl(CANNED_PROMPT)

    request = {"jsonrpc": "2.0", "id": 1, "method": "problem.run",
               "params": {"dsl": dsl}}
    proc = subprocess.run(
        [sys.executable, "-m", "src.qft_pcn.bridge", "--once",
         json.dumps(request)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    sys.stdout.write("JSON-RPC REQUEST:  " + json.dumps(request) + "\n")
    sys.stdout.write("JSON-RPC RESPONSE: " + proc.stdout.strip() + "\n\n")
    response = json.loads(proc.stdout.strip().splitlines()[-1])
    result = response["result"]
    verbalization = llm.verbalize(CANNED_PROMPT, result)
    render_transcript(CANNED_PROMPT, dsl, result, verbalization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
