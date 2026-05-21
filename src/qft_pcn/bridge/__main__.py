"""CLI entry point: `python -m src.qft_pcn.bridge`."""

from __future__ import annotations

import argparse
import logging
import sys

from .api import serve_stdio, serve_once


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qft_pcn.bridge",
                                description="QPCN LLM bridge (stdio JSON-RPC)")
    p.add_argument("--once", metavar="JSON",
                   help="Process one request from this JSON string and exit.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging on stderr.")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s qft_pcn.bridge %(message)s",
    )
    if args.once is not None:
        serve_once(args.once)
        return 0
    serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
