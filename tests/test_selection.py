"""The model's paragraph numbers are checked before they index the context.

`numbered[n - 1]` with n = 0 would not raise: Python reads index -1 and cites the LAST
paragraph, often from another document. These tests pin the guard in `_selection` that
refuses such a reply, so a later refactor cannot remove it silently.
"""

from __future__ import annotations

import pytest

from agent import _selection

COUNT = 4  # four numbered paragraphs in the context


@pytest.mark.parametrize(
    "raw",
    [
        '{"paragraphs": [0], "confidence": 0.9}',  # would read numbered[-1]
        '{"paragraphs": [-1], "confidence": 0.9}',  # would read numbered[-2]
        '{"paragraphs": [5], "confidence": 0.9}',  # past the end: IndexError
        '{"paragraphs": [2, 0], "confidence": 0.9}',  # one bad number spoils the reply
        '{"paragraphs": ["two"], "confidence": 0.9}',  # not a number
        '{"paragraphs": 2, "confidence": 0.9}',  # not a list
        "I think paragraph 2 answers it.",  # no JSON at all
    ],
)
def test_an_unusable_selection_is_refused(raw: str) -> None:
    assert _selection(raw, COUNT) is None


def test_a_valid_selection_keeps_order_and_drops_duplicates() -> None:
    assert _selection('{"paragraphs": ["P3", 1, 3], "confidence": 0.7}', COUNT) == ([3, 1], 0.7)


def test_confidence_is_clamped_and_defaults() -> None:
    assert _selection('{"paragraphs": [1], "confidence": 7}', COUNT) == ([1], 1.0)
    assert _selection('{"paragraphs": [1], "confidence": "high"}', COUNT) == ([1], 0.8)
