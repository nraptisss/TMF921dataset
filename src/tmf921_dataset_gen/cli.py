from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import Settings
from .logging import configure_logging
from .preflight import run_preflight

LOGGER = logging.getLogger(__name__)


def _print_messages(messages: list[str], prefix: str) -> None:
    if not messages:
        print(f"{prefix}: none")
        return
    for message in messages:
        print(f"{prefix}: {message}")


def cmd_preflight(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    report = run_preflight(settings)
    _print_messages(report.errors, "ERROR")
    _print_messages(report.warnings, "WARNING")
    return 0 if report.ok else 1


def cmd_build_corpus(_: argparse.Namespace) -> int:
    from .ingestion.corpus_builder import build_corpus

    settings = Settings.from_env()
    report = run_preflight(settings)
    if not report.ok:
        _print_messages(report.errors, "ERROR")
        return 1
    corpus = build_corpus(settings)
    print(json.dumps({"document_count": len(corpus)}, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .graph.workflow import run_generation

    settings = Settings.from_env()
    report = run_preflight(settings)
    if not report.ok:
        _print_messages(report.errors, "ERROR")
        return 1
    records = run_generation(settings, count=args.count, output_dir=None)
    print(json.dumps({"generated_records": len(records)}, indent=2))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from .graph.workflow import run_generation

    settings = Settings.from_env()
    report = run_preflight(settings)
    if not report.ok:
        _print_messages(report.errors, "ERROR")
        return 1
    if args.count >= 1000 and not settings.repo.idan_reference_dir.exists():
        warning_path = Path(args.out) / "blocked_run_manifest.json"
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "idan-reference directory is missing",
                    "requested_count": args.count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Blocked sample run. Manifest written to {warning_path}")
        return 2
    records = run_generation(settings, count=args.count, output_dir=Path(args.out))
    print(json.dumps({"generated_records": len(records), "output_dir": args.out}, indent=2))
    return 0


def cmd_dashboard(_: argparse.Namespace) -> int:
    from .dashboard.streamlit_app import launch_dashboard

    launch_dashboard()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TMF921 dataset generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.set_defaults(func=cmd_preflight)

    build_corpus_parser = subparsers.add_parser("build-corpus")
    build_corpus_parser.set_defaults(func=cmd_build_corpus)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--count", type=int, default=100)
    generate_parser.set_defaults(func=cmd_generate)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--count", type=int, default=1000)
    sample_parser.add_argument("--out", default="output/test_dataset")
    sample_parser.set_defaults(func=cmd_sample)

    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.set_defaults(func=cmd_dashboard)

    return parser


def main() -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
