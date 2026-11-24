"""Held-out human-labelled sets, never trained on, turned into System One requests.
Each case is (state, questions, gold), where gold maps a question id to
{"type", "probabilities"} in the same shape typed-decisions uses."""
import ast
import json
import math
import random

from datasets import load_dataset

from onepass.build import clip
from onepass.score import score_one, summarize


def hard(labels, gold, kind):
    return {"type": kind, "probabilities": {label: float(label == gold) for label in labels}}


def condaqa(n, seed=0):
    """Yes / no / don't know about a passage that contains a negation."""
    crit = {"yes": "The passage implies yes.", "no": "The passage implies no.",
            "dont_know": "The passage does not say."}
    gold_of = {"YES": "yes", "NO": "no", "DON'T KNOW": "dont_know"}
    rows = [r for r in load_dataset("lasha-nlp/CONDAQA", split="test") if r["label"] in gold_of]
    random.Random(seed).shuffle(rows)
    for r in rows[:n]:
        q = {"type": "choice", "instructions": r["sentence2"], "criteria": crit}
        yield {"passage": clip(r["sentence1"])}, {"answer": q}, {"answer": hard(crit, gold_of[r["label"]], "choice")}


def cosmos_qa(n, seed=0):
    """Four-way reading comprehension that needs common sense."""
    rows = list(load_dataset("Samsoup/cosmos_qa", split="validation"))
    random.Random(seed).shuffle(rows)
    for r in rows[:n]:
        crit = {f"option_{i}": r[f"answer{i}"] for i in range(4)}
        q = {"type": "choice", "instructions": r["question"], "criteria": crit}
        yield {"context": clip(r["context"])}, {"answer": q}, {"answer": hard(crit, f"option_{r['label']}", "choice")}


SUMMEVAL = {
    "coherence": "How well-structured and coherent is the summary?",
    "consistency": "How factually consistent is the summary with the article?",
    "fluency": "How fluent and well-written is the summary?",
    "relevance": "How well does the summary capture the important points of the article?",
}
LEVELS = ["very poor", "poor", "fair", "good", "excellent"]


def level_mass(mean):
    """Spread an expert mean on the 1-5 scale over its two nearest levels, keeping the mean."""
    x = min(max(mean, 1.0), 5.0) - 1
    lo = math.floor(x)
    hi = min(lo + 1, 4)
    p = {str(i): 0.0 for i in range(5)}
    p[str(lo)] += 1 - (x - lo)
    p[str(hi)] += x - lo
    return p


def summeval(n, seed=0):
    """Expert ratings (mean of 3) of machine summaries on four aspects, 1-5."""
    cases = []
    for r in load_dataset("mteb/summeval", split="test"):
        for i, summary in enumerate(r["machine_summaries"]):
            cases.append(({"article": clip(r["text"], 3000), "summary": clip(summary)},
                          {k: {"type": "score", "instructions": v, "criteria": LEVELS} for k, v in SUMMEVAL.items()},
                          {k: {"type": "score", "probabilities": level_mass(r[k][i])} for k in SUMMEVAL}))
    random.Random(seed).shuffle(cases)
    yield from cases[:n]


def ragtruth(n, seed=0):
    """Does an LLM response conflict with, or add to, the context it was given?"""
    rows = list(load_dataset("wandb/RAGTruth-processed", split="test"))
    random.Random(seed).shuffle(rows)
    for r in rows[:n]:
        labels = r["hallucination_labels_processed"]
        labels = labels if isinstance(labels, dict) else ast.literal_eval(labels)
        qs = {
            "conflict": {"type": "noul", "instructions": "The response states something that conflicts with the context.",
                         "criteria": {"true": "It contradicts the context.", "false": "Nothing contradicts the context."}},
            "unsupported": {"type": "noul", "instructions": "The response adds information that is not in the context.",
                            "criteria": {"true": "It adds unsupported information.", "false": "Everything is supported."}},
        }
        gold = {"conflict": hard(["true", "false"], "true" if labels["evident_conflict"] else "false", "noul"),
                "unsupported": hard(["true", "false"], "true" if labels["baseless_info"] else "false", "noul")}
        state = {"task": clip(r["query"], 300), "context": clip(r["context"], 2500), "response": clip(r["output"], 1200)}
        yield state, qs, gold


SETS = {"condaqa": (condaqa, 1000), "cosmos_qa": (cosmos_qa, 1000), "summeval": (summeval, 400), "ragtruth": (ragtruth, 500)}


def evaluate_all(model, tok, letter_ids, device, temperature=1.0, sizes=None):
    from onepass.evaluate import predict
    results = {}
    for name, (fn, n) in SETS.items():
        cases = list(fn((sizes or {}).get(name, n)))
        preds = predict(model, tok, letter_ids, [(s, q) for s, q, _ in cases], device, temperature)
        rows = [score_one(answers[qid], g) for (_, _, gold), answers in zip(cases, preds) for qid, g in gold.items()]
        results[name] = summarize(rows)
        print(name, json.dumps({k: round(v, 4) for k, v in results[name].items()}), flush=True)
    return results
