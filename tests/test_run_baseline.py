"""Chunked and unchunked baseline runs must agree exactly."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_baseline.py"
SPECIES = ("alkalinity", "dic", "hco3", "co3", "boh4")

# Small but not trivial: enough curves to span several chunks, few enough to
# keep the suite quick.
N_CURVES, N_POINTS = 12, 30


def _run(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Invoke the script in a subprocess, as a user would."""
    return subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--n-curves", str(N_CURVES),
            "--n-points", str(N_POINTS),
            "--instrument", "laboratory",
            "--nuisance", "offset",
            "--output", str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def unchunked(tmp_path_factory) -> dict:
    """One-shot run, the reference."""
    output = tmp_path_factory.mktemp("unchunked") / "baseline.json"
    result = _run(output)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf8"))["cases"][0]


class TestChunkingIsExact:
    """Chunk size must not touch a single reported digit."""

    @pytest.mark.parametrize("chunk_size", [1, 5, 12, 100])
    def test_statistics_match_the_unchunked_run(
        self, chunk_size, unchunked, tmp_path
    ):
        """Every summary statistic is identical at any chunk size."""
        output = tmp_path / f"chunked_{chunk_size}.json"
        result = _run(output, "--chunk-size", str(chunk_size))
        assert result.returncode == 0, result.stderr

        case = json.loads(output.read_text(encoding="utf8"))["cases"][0]
        assert case["n_curves"] == unchunked["n_curves"] == N_CURVES
        for name in SPECIES:
            assert case[name] == unchunked[name], (
                f"{name} differs at chunk_size={chunk_size}"
            )

    def test_resuming_across_processes_matches(self, unchunked, tmp_path):
        """One chunk per invocation, restarted repeatedly, still agrees."""
        output = tmp_path / "resumed.json"
        for _ in range(N_CURVES + 2):  # generous bound; loop breaks on success
            result = _run(
                output, "--chunk-size", "3", "--max-chunks-per-run", "1"
            )
            if result.returncode == 0:
                break
            assert result.returncode == 2, (
                f"expected 2 while incomplete, got {result.returncode}: "
                f"{result.stderr}"
            )
        else:
            pytest.fail("never completed")

        case = json.loads(output.read_text(encoding="utf8"))["cases"][0]
        for name in SPECIES:
            assert case[name] == unchunked[name], f"{name} differs after resume"

    def test_part_files_are_cleaned_up(self, tmp_path):
        """A completed run leaves the merged JSON and nothing else."""
        output = tmp_path / "tidy.json"
        assert _run(output, "--chunk-size", "5").returncode == 0
        assert output.exists()
        assert not list(tmp_path.glob("*.part*.json"))

    def test_keep_parts_retains_them(self, tmp_path):
        """``--keep-parts`` is honoured, for debugging a partial run."""
        output = tmp_path / "kept.json"
        assert _run(
            output, "--chunk-size", "5", "--keep-parts"
        ).returncode == 0
        assert list(tmp_path.glob("*.part*.json"))
