"""Tests for the AlphaGenome command-line interface."""

from __future__ import annotations

# External
import pytest

# Internal
from alphagenome_pt import FOLD_NAMES, fold_filename
from alphagenome_pt.cli import download
from alphagenome_pt.cli import main as cli


MOCK_PACKAGE_VERSION = "-1.0.0"  # purposefully something we'll never set in pyproject.toml


def test_version_uses_package_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "package_version", lambda: MOCK_PACKAGE_VERSION)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"alphagenome-pt {MOCK_PACKAGE_VERSION}\n"


def test_no_arguments_prints_root_help(monkeypatch, capsys):
    monkeypatch.setattr(cli, "package_version", lambda: MOCK_PACKAGE_VERSION)

    cli.main([])

    output = capsys.readouterr().out
    assert output.startswith("usage: alphagenome-pt")
    assert "{download}" in output
    assert "--version" in output


def test_download_rejects_token_argument(monkeypatch, capsys):
    monkeypatch.setattr(cli, "package_version", lambda: MOCK_PACKAGE_VERSION)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["download", "--token", "secret"])

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --token secret" in capsys.readouterr().err


def test_legacy_download_entry_point_uses_standalone_parser(
    monkeypatch,
    tmp_path,
):
    parsed_args = []
    monkeypatch.setattr(download, "run", parsed_args.append)

    download.main([
        "--local-dir",
        str(tmp_path),
        "--fold",
        "fold_2",
        "--repo-id",
        "owner/repo",
        "--repo-dir",
        "vtest",
        "--force-download",
    ])

    assert len(parsed_args) == 1
    args = parsed_args[0]
    assert args.local_dir == tmp_path
    assert args.fold == "fold_2"
    assert args.repo_id == "owner/repo"
    assert args.repo_dir == "vtest"
    assert args.force_download


def test_download_subcommand_routes_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "package_version", lambda: MOCK_PACKAGE_VERSION)
    calls = {}

    # Record the arguments forwarded to each downloader and return
    # fake paths to mimic the returns of the real download functions.
    metadata_path = tmp_path / "alphagenome_metadata.json"
    state_paths = [
        tmp_path / fold_filename(fold_name)
        for fold_name in FOLD_NAMES
    ]

    def fake_download_metadata(local_dir, **kwargs):
        calls["metadata"] = (local_dir, kwargs)
        return metadata_path

    def fake_download_state(local_dir, **kwargs):
        calls["state"] = (local_dir, kwargs)
        return state_paths

    monkeypatch.setattr(
        download,
        "download_deepmind_metadata",
        fake_download_metadata,
    )
    monkeypatch.setattr(
        download,
        "download_deepmind_state",
        fake_download_state,
    )

    cli.main([
        "download",
        "--local-dir",
        str(tmp_path),
        "--repo-id",
        "owner/repo",
        "--repo-dir",
        "vtest",
        "--force-download",
    ])
    expected_options = {
        "repo_id": "owner/repo",
        "repo_dir": "vtest",
        "force_download": True,
    }

    assert calls["metadata"] == (tmp_path, expected_options)
    assert calls["state"] == (
        tmp_path,
        {
            "fold": "all_folds",
            "download_all_folds": True,
            **expected_options,
        },
    )


def test_download_subcommand_routes_one_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "package_version", lambda: MOCK_PACKAGE_VERSION)
    calls = {}

    monkeypatch.setattr(
        download,
        "download_deepmind_metadata",
        lambda local_dir, **kwargs: tmp_path / "alphagenome_metadata.json",
    )

    def fake_download_state(local_dir, **kwargs):
        calls["state"] = (local_dir, kwargs)
        return tmp_path / fold_filename(kwargs["fold"])

    monkeypatch.setattr(download, "download_deepmind_state", fake_download_state)

    cli.main([
        "download",
        "--local-dir",
        str(tmp_path),
        "--fold",
        "fold_2",
    ])

    assert calls["state"] == (
        tmp_path,
        {
            "fold": "fold_2",
            "download_all_folds": False,
            "repo_id": download.DEFAULT_ALPHAGENOME_REPO_ID,
            "repo_dir": None,
            "force_download": False,
        },
    )
