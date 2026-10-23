"""Pin the estimate commands' help text, output, and estimator calls.

``cli_golden.json`` was recorded from the CLI before its estimate commands
were rebuilt on shared options and a shared runner. The estimators are
replaced by a stub, so the comparison covers only the CLI and needs no model
downloads.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from ethnicolr import cli

GOLDEN_FILE = Path(__file__).with_name("cli_golden.json")
ESTIMATOR_NAMES = (
    "estimate_census_surname",
    "estimate_florida_voter_surname",
    "estimate_wikipedia_surname",
)
INPUT_CSV = "surname,id\nsmith,1\nzhang,2\nunknownname,3\n"
SCENARIOS = {
    "estimate-help": ["estimate", "--help"],
    "quick-estimate-help": ["quick-estimate", "--help"],
}
for _command in ("census-surname", "florida-voter-surname", "wikipedia-surname"):
    SCENARIOS[f"{_command}-help"] = ["estimate", _command, "--help"]
    SCENARIOS[f"{_command}-default-output"] = [
        "estimate",
        _command,
        "names.csv",
        "-l",
        "surname",
    ]
    SCENARIOS[f"{_command}-verbose"] = [
        "-v",
        "estimate",
        _command,
        "names.csv",
        "-l",
        "surname",
        "-o",
        "out/result.csv",
        "-u",
        "0.9",
        "-m",
        "50",
        "--overwrite",
    ]
    SCENARIOS[f"{_command}-missing-column"] = [
        "estimate",
        _command,
        "names.csv",
        "-l",
        "last",
    ]
    SCENARIOS[f"{_command}-quick"] = [
        "-v",
        "quick-estimate",
        "names.csv",
        "-l",
        "surname",
        "--model",
        _command,
    ]
SCENARIOS["census-surname-year"] = [
    "estimate",
    "census-surname",
    "names.csv",
    "-l",
    "surname",
    "-y",
    "2000",
    "-o",
    "census-2000.csv",
]


def _stub_estimator(estimator_name, calls):
    def estimate(data, surname_column, **options):
        calls.append({"estimator": estimator_name, **options})
        known_names = data[surname_column] != "unknownname"
        return data.assign(
            race=data[surname_column]
            .where(known_names, None)
            .map({"smith": "white", "zhang": "api"}),
            white=0.5,
            black=0.1,
            api=0.2,
            hispanic=0.15,
            asian=0.25,
            nh_black=0.05,
            nh_white=0.6,
            other=0.1,
        )

    return estimate


def run_scenario(arguments, monkeypatch, directory):
    """Run one CLI invocation in an empty directory and record what it did."""
    monkeypatch.chdir(directory)
    calls = []
    for estimator_name in ESTIMATOR_NAMES:
        monkeypatch.setattr(cli, estimator_name, _stub_estimator(estimator_name, calls))
    Path("names.csv").write_text(INPUT_CSV)
    result = CliRunner().invoke(cli.cli, arguments, terminal_width=80)
    written_files = {
        str(path): pd.read_csv(path).to_csv(index=False)
        for path in sorted(Path().rglob("*.csv"))
        if path.name != "names.csv"
    }
    return {
        "exit_code": result.exit_code,
        "output": result.output,
        "calls": calls,
        "written_files": written_files,
    }


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_cli_matches_recorded_behavior(scenario, monkeypatch, tmp_path):
    expected = json.loads(GOLDEN_FILE.read_text())[scenario]
    assert run_scenario(SCENARIOS[scenario], monkeypatch, tmp_path) == expected


def test_golden_file_covers_every_scenario():
    assert sorted(json.loads(GOLDEN_FILE.read_text())) == sorted(SCENARIOS)
