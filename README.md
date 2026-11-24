# onepass

![onepass: typed decisions from one forward pass](https://raw.githubusercontent.com/samadpls/onepass/main/assets/hero.png)

A tiny open decision model. You give it a piece of text (the "state") and typed
questions: yes/no, choose one option, or pick a level on a scale. It returns a
probability for every possible answer, for all questions at once, from a single
forward pass. It generates no tokens, so the output can never be malformed.

The model is a small Qwen3 with a LoRA adapter, trained only on open datasets
labelled by people.

## Status

Work in progress for the [Qwen Local Hackathon 2026](https://github.com/junaiddshaukat/finetune-anywhere).
Results go here once they are measured.

Measured so far, on the 400-case test split of
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(2,000 decisions, scored with the benchmark's own formulas):

| Model | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| Qwen3-0.6B, no fine-tuning | 0.311 | 0.888 | 0.428 | 0.306 |
| Prior: label frequencies, ignores the input | 0.481 | 0.327 | 0.181 | 0.036 |

The fine-tuned model is training now. Its scores go here when they are measured.

### Why read answers instead of generating them

We asked base Qwen3-0.6B (8-bit GGUF, llama.cpp, thinking off) to *write* a JSON
answer for each of the 400 test cases:

| | Writes its answers | onepass (same 8-bit file) |
|---|---|---|
| Cases with a question missing | 37 of 400 (9.3%) | none, by design |
| Generated tokens per case | 49 | 0 |
| Time per case (5 questions, Apple M3 Pro GPU) | 497 ms | about 156 ms |
| Probabilities for every option | no | yes |

## Run it locally

With [uv](https://docs.astral.sh/uv/), one command runs the server, with nothing to install first:

```bash
uvx --from "onepass[serve] @ git+https://github.com/samadpls/onepass" onepass-serve path/to/onepass.gguf
```

To use it inside your own project:

```bash
uv add "onepass[serve] @ git+https://github.com/samadpls/onepass"
```

Then open http://127.0.0.1:8000 for the demo page, or call the API:

```bash
curl -s http://127.0.0.1:8000/v1/systemone -d '{
  "state": "My card was charged twice for one order, please refund the extra charge.",
  "questions": {
    "refund": {"type": "noul", "instructions": "The customer wants a refund.",
               "criteria": {"true": "Yes.", "false": "No."}},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments and refunds.", "tech": "Bugs and setup."}}
  }
}'
```

From Python:

```python
from onepass.model import Model

model = Model("onepass.gguf")
model.ask(state, questions)  # {"refund": {"label": "true", "probabilities": {...}}, ...}
```

## Use it without onepass

You don't need our package. The model is a normal Qwen3 checkpoint, and the only
thing that matters is the prompt layout it was trained on. Copy this, and you
depend on nothing but `transformers`:

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "samadpls/onepass-qwen3-0.6b"  # not published yet
TEMPERATURE = 1.0  # the calibration temperature from the model card
SYSTEM = "You are a decision model. Read the state, then answer every question with the letter of one option."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
KIND = {"noul": "true or false", "choice": "choose one", "score": "pick a level"}

def options(q):
    c = q.get("criteria") or {}
    if q["type"] == "noul":  # yes/no: always true, then false
        return [("true", c.get("true", "")), ("false", c.get("false", ""))]
    if q["type"] == "score":  # levels 0..n-1, in order
        return [(str(i), d) for i, d in enumerate(c)]
    return list(c.items())

def build_prompt(state, questions, tok):
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # encode piece by piece, never the whole string
    text = state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)
    ids = enc(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nState:\n{text}\n")
    slots = {}
    for n, (qid, q) in enumerate(questions.items(), 1):
        opts = options(q)
        lines = "".join(f"{LETTERS[i]}) {label}: {desc}\n" for i, (label, desc) in enumerate(opts))
        ids += enc(f"\nQuestion {n} ({KIND[q['type']]}): {q['instructions']}\n{lines}Answer:")
        slots[qid] = (len(ids) - 1, [label for label, _ in opts])  # the next token here is the answer letter
        ids += enc(" ?\n")
    return ids + enc("<|im_end|>\n"), slots

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
state = "My card was charged twice for one order. Please refund the extra charge today."
questions = {
    "refund": {"type": "noul", "instructions": "The customer wants a refund.", "criteria": {"true": "Yes.", "false": "No."}},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments and refunds.", "tech_support": "Bugs and setup.", "sales": "New orders."}},
}
ids, slots = build_prompt(state, questions, tok)
letters = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0]  # one forward pass, nothing generated
for qid, (pos, labels) in slots.items():
    p = (logits[pos, letters[: len(labels)]] / TEMPERATURE).softmax(-1)
    print(qid, dict(zip(labels, p.tolist())))
```

Three rules keep it working:
- Encode the prompt piece by piece, as above. Encoding it as one string can merge tokens across an answer slot.
- Yes/no options are always `true` then `false`. Score levels are `0` to `n-1`, in order.
- Use at most 26 options per question.

This `build_prompt` produces exactly the same token ids as `onepass.prompt.render`. We checked all 2,530 prompts in the benchmark sample and our validation set.

The model file is not published yet.

## How it works

1. **One slot per question.** The prompt lists the state, then each question with
   its options as letters (A, B, C ...), each followed by `Answer:`. The model's
   scores for the option letters at that position become the answer
   probabilities.
2. **No answer leaks into the next question.** A fixed placeholder follows every
   slot, so each answer depends only on the state and the questions so far.
3. **Trained on probabilities.** Where a dataset has several human raters, the
   training target is the share of raters who chose each answer, not a single
   label.
4. **One decode call at run time.** `onepass.model` puts the whole prompt in one
   llama.cpp batch and asks for scores only at the answer positions, so the
   full vocabulary is never stored for every token.

| Part | File |
|---|---|
| Prompt layout | [src/onepass/prompt.py](src/onepass/prompt.py) |
| Training data | [src/onepass/build.py](src/onepass/build.py) |
| Training (LoRA, loss at the answer slots only) | [src/onepass/train.py](src/onepass/train.py) |
| Scoring (typed-decisions formulas) and held-out sets | [src/onepass/score.py](src/onepass/score.py), [src/onepass/heldout.py](src/onepass/heldout.py) |
| GGUF export and quantization | [src/onepass/export.py](src/onepass/export.py) |
| Local runtime, server and demo page | [src/onepass/model.py](src/onepass/model.py), [src/onepass/serve.py](src/onepass/serve.py) |

## Training data

16 open, human-labelled datasets turned into typed questions: about 87,000 texts and
157,000 questions. Sources include BoolQ, UNLI, MultiNLI, WANLI, Civil Comments,
PAWS, VitaminC, SMS Spam, Banking77, CLINC150, DBpedia, the Davidson hate-speech
set, GoEmotions, Measuring Hate Speech, HelpSteer2 and BRIGHTER.
See [src/onepass/build.py](src/onepass/build.py). The model is never trained on
typed-decisions.

## Credit

Built by Abdul Samad ([samadpls.com](https://samadpls.com), abdulsamadsid1@gmail.com).
Licensed under Apache-2.0.
