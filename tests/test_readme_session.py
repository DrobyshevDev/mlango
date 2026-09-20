"""The terminal session in the README has to be one the program could print.

The front page opens with a `manage.py diff` run: agreement, the transitions,
the accuracy pair, fixed against broken, and a verdict with a p-value. It is the
first thing anyone reads, it is the argument for the whole tool, and nothing
checked any part of it.

Three things can go wrong, and they go wrong silently:

  the numbers stop adding up   somebody edits one figure in the sample and the
                               others no longer follow from it -- 500 rows at
                               92.0% agreement is 40 changed rows, and 40
                               changed rows is 29 fixed plus 11 broken

  the two READMEs drift        the session is a terminal transcript, identical
                               in both languages; one gets updated and the other
                               does not

  the program stops saying it  a label gets renamed, a column gets rewidened,
                               "row(s) wrong in" becomes "case(s) failing in" --
                               and the README goes on showing output the program
                               no longer produces

The first two are here, and need nothing but the files. The third lives beside
the diff tests in test_cli_inprocess.py, because that is where the fixtures that
train a model are, and it compares shapes rather than figures: the sample's data
is invented, but the lines it is laid out in are not.
"""

from __future__ import annotations

import re
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
READMES = ("README.md", "README.ru.md")

SESSION = re.compile(r"```bash\n(\$ python manage\.py diff.*?)\n```", re.S)

#: Every line of the sample that is "two spaces, a label, a value".
LABELLED = re.compile(r"^  (\w+) {2,}(.*)$", re.M)


def session(name: str) -> str:
    found = SESSION.search((ROOT / name).read_text(encoding="utf-8"))
    assert found, f"{name}: the `manage.py diff` sample session is gone"
    return found.group(1)


def figures(text: str) -> dict[str, str]:
    return dict(LABELLED.findall(text))


def test_the_two_readmes_show_the_same_session() -> None:
    """It is a transcript of a command, so translating it would be inventing it."""
    assert session("README.md") == session("README.ru.md")


def test_the_sample_session_adds_up() -> None:
    """Every figure in the sample follows from the others, so check that it does.

    A sample that contradicts itself is worse than no sample: it is an
    advertisement for a tool whose whole pitch is that aggregate numbers hide
    what moved.
    """
    text = session("README.md")
    values = figures(text)

    rows = int(re.search(r"on (\d+) rows", text).group(1))
    agreement = float(values["agreement"].rstrip("%")) / 100
    changed = int(values["changed"].split()[0])
    assert round(rows * (1 - agreement)) == changed, "agreement and changed disagree"

    transitions = [int(n) for n in re.findall(r"^    \S+ → \S+ +(\d+)$", text, re.M)]
    assert transitions, "the transition breakdown is gone"
    assert sum(transitions) == changed, "the transitions do not sum to the changed rows"

    accuracies = [float(a) for a in re.findall(r"accuracy +([\d.]+)", text)]
    assert len(accuracies) == 2, "the sample no longer shows an accuracy for each version"
    delta = float(re.search(r"accuracy +[\d.]+ +\+([\d.]+)", text).group(1))
    assert round(accuracies[1] - accuracies[0], 4) == delta, "the accuracy delta is wrong"

    fixed = int(values["fixed"].split()[0])
    broke = int(values["broke"].split()[0])
    assert fixed + broke == changed, "fixed and broke do not account for every changed row"
    assert round(rows * accuracies[1]) - round(rows * accuracies[0]) == fixed - broke, (
        "the accuracy gain does not match fixed minus broken"
    )

    # McNemar, exact two-sided binomial over the discordant pairs -- which is
    # what the verdict's p-value is, and the one figure here that cannot be
    # checked by squinting.
    quoted = float(re.search(r"\(p=([\d.]+)\)", values["verdict"]).group(1))
    n, k = fixed + broke, min(fixed, broke)
    exact = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    assert round(exact, 3) == quoted, f"verdict says p={quoted}, McNemar gives p={exact:.6f}"
