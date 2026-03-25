from pathlib import Path

from tmf921_dataset_gen.cli import build_parser, cmd_sample


def test_cli_parser_has_expected_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["sample", "--count", "1000"])
    assert args.command == "sample"
    assert args.count == 1000


def test_sample_blocks_without_idan_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd())
    out_dir = tmp_path / "sample"
    parser = build_parser()
    args = parser.parse_args(["sample", "--count", "1000", "--out", str(out_dir)])
    exit_code = cmd_sample(args)
    assert exit_code == 2
    assert (out_dir / "blocked_run_manifest.json").exists()
