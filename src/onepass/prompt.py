"""Turn a System One request (state + typed questions) into token ids with one
answer slot per question, so every answer is read from a single forward pass."""
import json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = "You are a decision model. Read the state, then answer every question with the letter of one option."
KIND = {"noul": "true or false", "choice": "choose one", "score": "pick a level"}


def options(question):
    """(label, description) pairs in the order the letters are given."""
    kind, criteria = question["type"], question.get("criteria") or {}
    if kind == "noul":
        return [("true", criteria.get("true", "")), ("false", criteria.get("false", ""))]
    if kind == "score":
        return [(str(i), d) for i, d in enumerate(criteria)]
    return list(criteria.items())


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)


def render(state, questions, encode):
    """Token ids and {question id: (slot index, option labels)}.

    The logits at a slot predict that question's answer letter. A fixed " ?"
    follows each slot, so no question ever sees another question's answer.
    Ids are built segment by segment so tokenization never merges across a slot.
    """
    ids = encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nState:\n{state_text(state)}\n")
    slots = {}
    for n, (qid, q) in enumerate(questions.items(), 1):
        opts = options(q)
        if len(opts) > len(LETTERS):
            raise ValueError(f"question {qid!r} has {len(opts)} options; the limit is {len(LETTERS)}")
        lines = "".join(f"{LETTERS[i]}) {label}: {desc}\n" for i, (label, desc) in enumerate(opts))
        ids += encode(f"\nQuestion {n} ({KIND[q['type']]}): {q['instructions']}\n{lines}Answer:")
        slots[qid] = (len(ids) - 1, [label for label, _ in opts])
        ids += encode(" ?\n")
    ids += encode("<|im_end|>\n")
    return ids, slots
