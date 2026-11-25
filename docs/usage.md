# Using onepass

## Run it locally

With [uv](https://docs.astral.sh/uv/) and nothing else installed, download the 8-bit
0.6B model and start the server:

```bash
uvx --from huggingface_hub hf download samadpls/onepass-qwen3-0.6b-GGUF onepass-qwen3-0.6b-Q8_0.gguf --local-dir .
uvx --from "onepass[serve] @ git+https://github.com/samadpls/onepass" onepass-serve onepass-qwen3-0.6b-Q8_0.gguf
```

For the 1.7B, download `onepass-qwen3-1.7b-Q8_0.gguf` from
[samadpls/onepass-qwen3-1.7b-GGUF](https://huggingface.co/samadpls/onepass-qwen3-1.7b-GGUF)
and add `--temperature 1.05`.

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

model = Model("onepass-qwen3-0.6b-Q8_0.gguf")  # temperature=1.05 for the 1.7B
model.ask(state, questions)  # {"refund": {"label": "true", "probabilities": {...}}, ...}
```

## Demo app

[demo/streamlit_app.py](../demo/streamlit_app.py) is a small web app with five examples
and editable questions. It downloads the 8-bit 0.6B model once and runs it on your CPU.
From the repo root:

```bash
uv run --extra serve --with streamlit --with huggingface_hub streamlit run demo/streamlit_app.py
```

## Use it without onepass

You don't need our package. The model is a normal Qwen3 checkpoint, and the only
thing that matters is the prompt layout it was trained on. Copy this, and you
depend on nothing but `transformers`:

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "samadpls/onepass-qwen3-0.6b"
TEMPERATURE = 1.0  # the calibration temperature: 1.0 for the 0.6B, 1.05 for the 1.7B
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
    p = (logits[pos, letters[: len(labels)]].float() / TEMPERATURE).softmax(-1)
    print(qid, dict(zip(labels, p.tolist())))
```

Three rules keep it working:
- Encode the prompt piece by piece, as above. Encoding it as one string can merge tokens across an answer slot.
- Yes/no options are always `true` then `false`. Score levels are `0` to `n-1`, in order.
- Use at most 26 options per question.

This `build_prompt` produces exactly the same token ids as `onepass.prompt.render`. We checked all 2,530 prompts in the benchmark sample and our validation set.

