import math

import pytest

from onepass.prompt import LETTERS, options, render
from onepass.score import score_one, summarize


def fake_encode(text):
    """One id per character keeps positions easy to check."""
    return [ord(c) for c in text]


QUESTIONS = {
    "refund": {"type": "noul", "instructions": "The customer wants a refund.",
               "criteria": {"false": "No refund.", "true": "Refund wanted."}},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Money.", "tech": "Bugs.", "sales": "Buying."}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Low.", "Medium.", "High."]},
}


def test_slots_sit_on_the_colon_before_each_placeholder():
    ids, slots = render({"text": "hi"}, QUESTIONS, fake_encode)
    text = "".join(map(chr, ids))
    assert list(slots) == ["refund", "team", "urgency"]
    for pos, _ in slots.values():
        assert text[pos - 6: pos + 3] == "Answer: ?"


def test_option_labels_and_order():
    _, slots = render("hi", QUESTIONS, fake_encode)
    assert slots["refund"][1] == ["true", "false"]
    assert slots["team"][1] == ["billing", "tech", "sales"]
    assert slots["urgency"][1] == ["0", "1", "2"]


def test_later_questions_never_see_earlier_answers():
    a, _ = render("hi", QUESTIONS, fake_encode)
    b, _ = render("hi", dict(list(QUESTIONS.items())[:1]), fake_encode)
    assert a[: len(b) - len(fake_encode("<|im_end|>\n"))] == b[: -len(fake_encode("<|im_end|>\n"))]


def test_too_many_options_is_an_error():
    q = {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "" for i in range(len(LETTERS) + 1)}}
    with pytest.raises(ValueError):
        render("hi", {"q": q}, fake_encode)


def test_noul_options_are_true_then_false():
    assert [label for label, _ in options(QUESTIONS["refund"])] == ["true", "false"]


def test_score_matches_benchmark_formulas():
    gold = {"type": "noul", "probabilities": {"false": 0.25, "true": 0.75}}
    ok, kl, brier, conf = score_one({"yes": 0.5, "no": 0.5}, gold)
    assert not ok  # a tie goes to the first gold option, "false"
    assert kl == pytest.approx(0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5))
    assert brier == pytest.approx(0.125)
    assert summarize([(True, 0.0, 0.0, 0.9), (False, 0.0, 0.0, 0.9)])["ece"] == pytest.approx(0.4)
