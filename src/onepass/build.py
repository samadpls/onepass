"""Build the training mix: open, human-labelled datasets turned into System One
requests (state + typed questions) with target distributions.

Every record is {"src", "state", "questions", "targets"}, where targets maps a
question id to {option label: probability}. Soft labels (rater fractions or
vote counts) are kept as probabilities, since calibration is the point.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 0
MAX_CHARS = 2000


def clip(text, n=MAX_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def wrap(rng, text, keys=("text",)):
    """State as a plain string or as a small JSON object, so both shapes are seen."""
    return clip(text) if rng.random() < 0.5 else {rng.choice(keys): clip(text)}


def noul(instructions, true_desc, false_desc):
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true_desc, "false": false_desc}}


def binary(p):
    p = min(max(float(p), 0.0), 1.0)
    return {"true": p, "false": 1.0 - p}


def onehot(labels, gold):
    return {label: float(label == gold) for label in labels}


def normalize(counts):
    s = sum(counts.values())
    return {k: v / s for k, v in counts.items()}


def choice_with_distractors(rng, names, gold, describe, k_max=12):
    """A choice over the gold label plus sampled distractors, shuffled."""
    others = [n for n in names if n != gold]
    picked = rng.sample(others, min(len(others), rng.randint(3, k_max - 1))) + [gold]
    rng.shuffle(picked)
    return {name: describe(name) for name in picked}


def humanize(label):
    return label.replace("_", " ").strip()


# ---------------------------------------------------------------- yes / no

def boolq(rng, cap):
    for r in load_dataset("google/boolq", split="train").shuffle(seed=SEED).select(range(cap)):
        q = r["question"].rstrip("?") + "?"
        ask = rng.choice([f"Based on the passage, the answer to this question is yes: {q}",
                          f"The passage answers this question with yes: {q}",
                          f"Yes or no, according to the passage: {q}"])
        yield {"state": wrap(rng, r["passage"], ("passage", "document", "article")),
               "questions": {"answer_is_yes": noul(ask, "The answer is yes.", "The answer is no.")},
               "targets": {"answer_is_yes": binary(r["answer"])}}


def unli(rng, cap):
    """UNLI: humans rated how likely the hypothesis is, given the premise."""
    for r in load_dataset("Zhengping/UNLI", split="train").shuffle(seed=SEED).select(range(cap)):
        ask = rng.choice([f"Given the text, this statement is true: {r['hypothesis']}",
                          f"{r['hypothesis']}",
                          f"The text implies: {r['hypothesis']}"])
        yield {"state": wrap(rng, r["premise"], ("text", "observation", "description")),
               "questions": {"statement_true": noul(ask, "The statement is true.", "The statement is false.")},
               "targets": {"statement_true": binary(r["label"])}}


NLI = {"entailment": "The text shows the statement is true.",
       "neutral": "The text neither confirms nor contradicts the statement.",
       "contradiction": "The text shows the statement is false."}


def nli_record(rng, premise, hypothesis, gold):
    state = ({"text": clip(premise), "statement": clip(hypothesis)} if rng.random() < 0.5
             else f"Text: {clip(premise)}\nStatement: {clip(hypothesis)}")
    if rng.random() < 0.7:
        opts = list(NLI.items())
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": rng.choice(["How does the text relate to the statement?",
                                                           "Does the text support the statement?"]),
             "criteria": dict(opts)}
        return {"state": state, "questions": {"relation": q}, "targets": {"relation": onehot(NLI, gold)}}
    q = noul("The statement follows from the text.", "The text shows the statement is true.",
             "The statement is not shown by the text.")
    return {"state": state, "questions": {"follows": q}, "targets": {"follows": binary(gold == "entailment")}}


def mnli(rng, cap):
    names = ["entailment", "neutral", "contradiction"]
    for r in load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=SEED).select(range(cap)):
        if r["label"] in (0, 1, 2):
            yield nli_record(rng, r["premise"], r["hypothesis"], names[r["label"]])


def wanli(rng, cap):
    for r in load_dataset("alisawuffles/WANLI", split="train").shuffle(seed=SEED).select(range(cap)):
        yield nli_record(rng, r["premise"], r["hypothesis"], r["gold"])


CIVIL = {
    "toxicity": ("This comment is toxic: rude, disrespectful or unreasonable.", "toxic"),
    "severe_toxicity": ("This comment is severely toxic.", "severely toxic"),
    "obscene": ("This comment is obscene.", "obscene"),
    "threat": ("This comment contains a threat.", "threatening"),
    "insult": ("This comment is insulting.", "insulting"),
    "identity_attack": ("This comment attacks someone for their identity.", "an identity attack"),
    "sexual_explicit": ("This comment is sexually explicit.", "sexually explicit"),
}


def civil_comments(rng, cap):
    """Targets are the share of raters who marked each attribute."""
    ds = load_dataset("google/civil_comments", split="train").shuffle(seed=SEED)
    toxic = ds.filter(lambda r: r["toxicity"] >= 0.2).select(range(cap // 2))
    calm = ds.filter(lambda r: r["toxicity"] < 0.2).select(range(cap - cap // 2))
    for r in list(toxic) + list(calm):
        keys = rng.sample(list(CIVIL), rng.randint(2, 5))
        qs = {k: noul(CIVIL[k][0], f"Raters would call it {CIVIL[k][1]}.", f"Raters would not call it {CIVIL[k][1]}.")
              for k in keys}
        yield {"state": wrap(rng, r["text"], ("comment", "text", "post")),
               "questions": qs, "targets": {k: binary(r[k]) for k in keys}}


def paws(rng, cap):
    for r in load_dataset("google-research-datasets/paws", "labeled_final", split="train").shuffle(seed=SEED).select(range(cap)):
        state = {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}
        q = noul(rng.choice(["The two sentences mean the same thing.", "Sentence 2 is a paraphrase of sentence 1."]),
                 "They have the same meaning.", "Their meanings differ.")
        yield {"state": state, "questions": {"paraphrase": q}, "targets": {"paraphrase": binary(r["label"] == 1)}}


def vitaminc(rng, cap):
    names = {"SUPPORTS": "supports", "REFUTES": "refutes", "NOT ENOUGH INFO": "not_enough_info"}
    crit = {"supports": "The evidence shows the claim is true.",
            "refutes": "The evidence shows the claim is false.",
            "not_enough_info": "The evidence does not settle the claim."}
    for r in load_dataset("tals/vitaminc", split="train").shuffle(seed=SEED).select(range(cap)):
        opts = list(crit.items())
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": "What does the evidence say about the claim?", "criteria": dict(opts)}
        yield {"state": {"claim": clip(r["claim"]), "evidence": clip(r["evidence"])},
               "questions": {"verdict": q}, "targets": {"verdict": onehot(crit, names[r["label"]])}}


def sms_spam(rng, cap):
    for r in load_dataset("codesignal/sms-spam-collection", split="train").shuffle(seed=SEED).select(range(cap)):
        q = noul(rng.choice(["This message is spam.", "This text message is unwanted marketing or a scam."]),
                 "It is spam.", "It is a normal personal message.")
        yield {"state": wrap(rng, r["message"], ("message", "sms")), "questions": {"spam": q},
               "targets": {"spam": binary(r["label"] == "spam")}}


# ---------------------------------------------------------------- choice

def intents(rng, cap, name, config, split, text_key, label_key, what):
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    names = ds.features[label_key].names
    for r in ds.shuffle(seed=SEED).select(range(min(cap, len(ds)))):
        gold = names[r[label_key]]
        crit = choice_with_distractors(rng, names, gold,
                                       lambda n: "Not one of the listed requests." if n == "oos" else f"The {what} is about: {humanize(n)}.")
        q = {"type": "choice", "instructions": rng.choice([f"What does the {what} want?", f"Which intent fits this {what}?"]),
             "criteria": crit}
        yield {"state": wrap(rng, r[text_key], ("message", "query", "request")), "questions": {"intent": q},
               "targets": {"intent": onehot(crit, gold)}}


def banking77(rng, cap):
    yield from intents(rng, cap, "PolyAI/banking77", None, "train", "text", "label", "customer message")


def clinc(rng, cap):
    yield from intents(rng, cap, "clinc/clinc_oos", "plus", "train", "text", "intent", "user request")


def dbpedia(rng, cap):
    ds = load_dataset("fancyzhx/dbpedia_14", split="train")
    names = ds.features["label"].names
    for r in ds.shuffle(seed=SEED).select(range(cap)):
        gold = names[r["label"]]
        crit = choice_with_distractors(rng, names, gold, lambda n: f"The article is about: {re.sub(r'(?<!^)(?=[A-Z])', ' ', n).lower()}.", k_max=14)
        q = {"type": "choice", "instructions": "What kind of thing is this article about?", "criteria": crit}
        yield {"state": {"title": r["title"], "content": clip(r["content"])}, "questions": {"topic": q},
               "targets": {"topic": onehot(crit, gold)}}


def hate_offensive(rng, cap):
    crit = {"hate_speech": "It attacks a group of people.", "offensive": "It is offensive but not hate speech.",
            "neither": "It is neither offensive nor hate speech."}
    for r in load_dataset("tdavidson/hate_speech_offensive", split="train").shuffle(seed=SEED).select(range(cap)):
        votes = {"hate_speech": r["hate_speech_count"], "offensive": r["offensive_language_count"], "neither": r["neither_count"]}
        if sum(votes.values()) == 0:
            continue
        opts = list(crit.items())
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": "How would annotators classify this tweet?", "criteria": dict(opts)}
        yield {"state": wrap(rng, r["tweet"], ("tweet", "post")), "questions": {"label": q},
               "targets": {"label": normalize(votes)}}


def go_emotions(rng, cap):
    """Raw rater rows, grouped per comment: each emotion's share of raters."""
    ds = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
    emotions = [c for c in ds.column_names if c not in
                ("text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear", "neutral")]
    votes, texts = defaultdict(lambda: defaultdict(int)), {}
    for r in ds:
        if r["example_very_unclear"]:
            continue
        texts[r["id"]] = r["text"]
        votes[r["id"]]["_n"] += 1
        for e in emotions:
            votes[r["id"]][e] += r[e]
    ids = [i for i in texts if votes[i]["_n"] >= 2]
    rng.shuffle(ids)
    for i in ids[:cap]:
        v = votes[i]
        felt = [e for e in emotions if v[e] > 0]
        keys = list(dict.fromkeys(rng.sample(felt, min(len(felt), 2)) + rng.sample(emotions, rng.randint(1, 3))))
        qs = {e: noul(f"The writer of this comment expresses {e}.", f"It expresses {e}.", f"It does not express {e}.")
              for e in keys}
        yield {"state": wrap(rng, texts[i], ("comment", "text")), "questions": qs,
               "targets": {e: binary(v[e] / v["_n"]) for e in keys}}


# ---------------------------------------------------------------- score

LEVELS5 = ["not at all", "slightly", "moderately", "very", "extremely"]
MHS = {"insult": "How insulting is this comment?", "humiliate": "How much does this comment humiliate its target?",
       "dehumanize": "How much does this comment dehumanize its target?",
       "violence": "How much does this comment call for violence?", "genocide": "How much does this comment call for killing a group?"}


def measuring_hate_speech(rng, cap):
    """One row per annotator; targets are the vote share for each level."""
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    votes, texts = defaultdict(lambda: defaultdict(lambda: [0] * 5)), {}
    for r in ds.select_columns(["comment_id", "text"] + list(MHS)):
        texts[r["comment_id"]] = r["text"]
        for k in MHS:
            if r[k] is not None:
                votes[r["comment_id"]][k][int(r[k])] += 1
    ids = list(texts)
    rng.shuffle(ids)
    for i in ids[:cap]:
        keys = rng.sample(list(MHS), rng.randint(2, 4))
        qs = {k: {"type": "score", "instructions": MHS[k], "criteria": LEVELS5} for k in keys}
        yield {"state": wrap(rng, texts[i], ("comment", "text", "post")), "questions": qs,
               "targets": {k: normalize({str(j): c for j, c in enumerate(votes[i][k])}) for k in keys}}


HELPSTEER = {
    "helpfulness": ("How helpful is the response to the prompt?", ["not helpful", "slightly helpful", "partly helpful", "mostly helpful", "fully helpful"]),
    "correctness": ("How correct and complete is the response?", ["wrong", "mostly wrong", "partly correct", "mostly correct", "fully correct"]),
    "coherence": ("How clear and coherent is the response?", ["incoherent", "hard to follow", "partly clear", "mostly clear", "perfectly clear"]),
    "complexity": ("How much expertise does it take to write this response?", ["anyone could write it", "basic", "some education", "specialist", "expert"]),
    "verbosity": ("How long is the response compared with what the prompt asks for?", ["far too short", "short", "about right", "long", "far too long"]),
}


def helpsteer2(rng, cap):
    for r in load_dataset("nvidia/HelpSteer2", split="train").shuffle(seed=SEED).select(range(cap)):
        keys = rng.sample(list(HELPSTEER), rng.randint(2, 4))
        qs = {k: {"type": "score", "instructions": HELPSTEER[k][0], "criteria": HELPSTEER[k][1]} for k in keys}
        yield {"state": {"prompt": clip(r["prompt"], 1200), "response": clip(r["response"], 1800)}, "questions": qs,
               "targets": {k: onehot([str(j) for j in range(5)], str(r[k])) for k in keys}}


EMO4 = ["none", "low", "moderate", "high"]


def brighter(rng, cap):
    ds = load_dataset("brighter-dataset/BRIGHTER-emotion-intensities", "eng", split="train")
    emotions = [e for e in ("anger", "fear", "joy", "sadness", "surprise") if e in ds.column_names]
    for r in ds.shuffle(seed=SEED).select(range(min(cap, len(ds)))):
        keys = rng.sample(emotions, rng.randint(2, 3))
        qs = {e: {"type": "score", "instructions": f"How much {e} does the writer express?", "criteria": EMO4} for e in keys}
        yield {"state": wrap(rng, r["text"], ("text", "post")), "questions": qs,
               "targets": {e: onehot([str(j) for j in range(4)], str(r[e])) for e in keys}}


SOURCES = {  # name: (builder, number of texts)
    "boolq": (boolq, 6000), "unli": (unli, 8000), "mnli": (mnli, 5000), "wanli": (wanli, 5000),
    "civil_comments": (civil_comments, 8000), "paws": (paws, 4000), "vitaminc": (vitaminc, 5000),
    "sms_spam": (sms_spam, 3000), "banking77": (banking77, 5000), "clinc": (clinc, 5000), "dbpedia": (dbpedia, 4000),
    "hate_offensive": (hate_offensive, 6000), "go_emotions": (go_emotions, 8000),
    "measuring_hate_speech": (measuring_hate_speech, 8000), "helpsteer2": (helpsteer2, 6000), "brighter": (brighter, 2700),
}


def check(rec):
    """Every question has a target over exactly its options, summing to 1."""
    from onepass.prompt import options
    assert set(rec["questions"]) == set(rec["targets"]), rec["src"]
    for qid, q in rec["questions"].items():
        labels = [label for label, _ in options(q)]
        t = rec["targets"][qid]
        assert set(t) == set(labels), (rec["src"], qid, labels, list(t))
        assert abs(sum(t.values()) - 1) < 1e-6 and min(t.values()) >= 0, (rec["src"], qid, t)


def build(out_dir="data/mix", val_share=0.02, only=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out / "train.jsonl", "a" if only else "w") as tr, open(out / "val.jsonl", "a" if only else "w") as va:
        for name, (fn, cap) in SOURCES.items():
            if only and name not in only:
                continue
            rng = random.Random(f"{SEED}-{name}")
            n = 0
            for rec in fn(rng, cap):
                rec = {"src": name, **rec}
                check(rec)
                (va if rng.random() < val_share else tr).write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            counts[name] = n
            print(f"{name}: {n}", flush=True)
    return counts


if __name__ == "__main__":
    import sys
    build(only=set(sys.argv[1:]) or None)
