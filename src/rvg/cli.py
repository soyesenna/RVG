"""`rvg`: the product command line. `universal-query` and `ingest` are the Universal path's own
commands (`rvg.universal.cli`); `policy` and `capabilities` describe the build. Relative paths and
the `.env` file are resolved against `--root` (default: the current directory), never against the
installed package or a source checkout."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from .api import capabilities
from .config import load_settings
from .errors import ConfigurationError, ProviderError
from .transport import JsonTransport
from .universal import PATH, universal, universal_hash
from .universal.cli import InputError, add_ingest_command
from .universal.cli import add_commands as add_universal_commands
from .universal.cli import execute as execute_universal


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="rvg", description="RVG (Retrieval-Verified Generation) runtime"
    )
    result.add_argument(
        "--root",
        type=Path,
        default=None,
        help="base directory of relative paths and of .env (default: the current directory)",
    )
    result.add_argument("--env-file", type=Path, help="read this settings file instead of .env")
    result.add_argument("--mode", choices=["mock", "live"], help="override RVG_MODE")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("policy", help="the one policy's sha256, version and location")
    commands.add_parser(
        "capabilities", help="the components of this build and their verification level"
    )
    add_universal_commands(commands)
    add_ingest_command(commands)
    return result


async def query(args, settings):
    async with httpx.AsyncClient(trust_env=False) as client:
        transport = JsonTransport(
            client, timeout=settings.http_timeout_seconds, max_attempts=settings.http_max_attempts
        )
        return await execute_universal(args, settings, transport)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "policy":
        result = {"sha256": universal_hash(), "version": universal()["version"], "path": str(PATH)}
    elif args.command == "capabilities":
        result = capabilities()
    else:
        overrides = {"deepjev_mode": args.mode} if args.mode else {}
        try:
            settings = load_settings(args.root or Path.cwd(), env_file=args.env_file, **overrides)
            result = asyncio.run(query(args, settings))
        except (ConfigurationError, ProviderError, InputError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except (ValueError, OSError):
            print("Check the input and the local files (input not shown).", file=sys.stderr)
            return 2
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if getattr(args, "out", None):
        target = settings.resolve_path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
    print(output, end="")
    return 2 if result.get("status") == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
