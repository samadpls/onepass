# onepass

![onepass: typed decisions from one forward pass](https://raw.githubusercontent.com/samadpls/onepass/main/assets/hero.png)

A tiny open decision model. You give it a piece of text (the "state") and typed
questions: yes/no, choose one option, or pick a level on a scale. It returns a
probability for every possible answer, for all questions at once, from a single
forward pass. It generates no tokens, so the output can never be malformed.

The model is Qwen3-0.6B (or Qwen3-1.7B) with a LoRA adapter, trained on open
datasets. It was never trained on the benchmark it is scored on.

**Try it in the browser:** [onepassqwen3.streamlit.app](https://onepassqwen3.streamlit.app/)
(free CPU; the first visit wakes the app up).

All models are in the [onepass collection](https://huggingface.co/collections/samadpls/onepass-6b06dee3eabd91e4de4df427) on Hugging Face:

| Size | transformers | GGUF (llama.cpp) | LoRA adapter |
|---|---|---|---|
| 0.6B | [onepass-qwen3-0.6b](https://huggingface.co/samadpls/onepass-qwen3-0.6b) | [onepass-qwen3-0.6b-GGUF](https://huggingface.co/samadpls/onepass-qwen3-0.6b-GGUF) | [onepass-qwen3-0.6b-lora](https://huggingface.co/samadpls/onepass-qwen3-0.6b-lora) |
| 1.7B | [onepass-qwen3-1.7b](https://huggingface.co/samadpls/onepass-qwen3-1.7b) | [onepass-qwen3-1.7b-GGUF](https://huggingface.co/samadpls/onepass-qwen3-1.7b-GGUF) | [onepass-qwen3-1.7b-lora](https://huggingface.co/samadpls/onepass-qwen3-1.7b-lora) |

## What it does

![A support message, two questions, and the answer probabilities onepass returned](https://raw.githubusercontent.com/samadpls/onepass/main/assets/example.png)

The probabilities above are the real output of the 8-bit 0.6B model for this message.
Every question gets a probability for each of its options, read from one forward pass.

## Results

Scored on the 400-case test split of
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(2,000 decisions), with the formulas from the benchmark's own scoring script.
Accuracy: higher is better. KL from gold, Brier and ECE: lower is better.

| Model | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| Qwen3-0.6B, no fine-tuning | 0.311 | 0.888 | 0.428 | 0.306 |
| Qwen3-1.7B, no fine-tuning | 0.278 | 1.914 | 0.812 | 0.611 |
| Label frequencies from the benchmark's train split (ignores the input) | 0.481 | 0.327 | 0.181 | **0.036** |
| **onepass 0.6B** | 0.601 | 0.270 | 0.154 | 0.049 |
| **onepass 1.7B** | **0.646** | **0.236** | **0.129** | 0.076 |

<details>
<summary>Accuracy by question type and by workflow</summary>

| | onepass 0.6B | onepass 1.7B |
|---|---|---|
| Yes/no | 0.737 | 0.755 |
| Choose one | 0.518 | 0.605 |
| Pick a level | 0.561 | 0.594 |
| Agent traces | 0.564 | 0.574 |
| Customer service | 0.630 | 0.686 |
| Invoices | 0.614 | 0.654 |
| Security incidents | 0.596 | 0.668 |

</details>

An earlier 0.6B run, trained only on the human-labelled datasets, scored 0.429:
below the label-frequency baseline. Adding the three open typed-question datasets
(see [How it was trained](#how-it-was-trained)) raised it to 0.601.

### Held-out human-labelled sets

Four public sets that the model never saw in training, turned into typed questions.

| Set | Always the most common answer | 0.6B before → after | 1.7B before → after |
|---|---|---|---|
| CondaQA (negation, yes/no) | 0.483 | 0.455 → 0.563 | 0.495 → 0.476 |
| Cosmos QA (common sense, choose one) | 0.260 | 0.282 → 0.373 | 0.529 → 0.506 |
| SummEval (summary quality, pick a level) | 0.651 | 0.504 → 0.265 | 0.073 → 0.376 |
| RAGTruth (is the answer supported?, yes/no) | 0.768 | 0.704 → 0.752 | 0.476 → 0.760 |

Accuracy is mixed. The probabilities improve on every set: KL from gold falls
for both models on all four (for example CondaQA 1.31 → 0.92 for the 0.6B).

On SummEval the 0.6B answers "good" almost every time. The experts rated most
summaries "excellent" for consistency and fluency, so its accuracy falls there,
but 87% of its answers are within one level of theirs.

### Smaller files

The same benchmark, run through llama.cpp with each GGUF file. "Shift" is how
far each answer probability moves from the 16-bit file, on average.

| Model | File | Size | Accuracy | KL from gold | Brier | ECE | Shift | Time per case |
|---|---|---|---|---|---|---|---|---|
| 0.6B | 16-bit | 1,198 MB | 0.601 | 0.270 | 0.154 | 0.048 | | 180 ms |
| 0.6B | **8-bit** | 639 MB | 0.605 | 0.270 | 0.154 | 0.054 | 0.004 | 185 ms |
| 0.6B | 4-bit (Q4_K_M) | 397 MB | 0.603 | 0.268 | 0.153 | 0.064 | 0.030 | 191 ms |
| 1.7B | 16-bit | 3,447 MB | 0.645 | 0.236 | 0.129 | 0.076 | | 418 ms |
| 1.7B | **8-bit** | 1,834 MB | 0.650 | 0.235 | 0.128 | 0.080 | 0.003 | 432 ms |
| 1.7B | 4-bit (Q4_K_M) | 1,107 MB | 0.632 | 0.237 | 0.131 | 0.073 | 0.028 | 478 ms |

8-bit halves the file and changes almost nothing. 4-bit keeps the scores close,
but single answers can move a lot (up to 0.22 for the 0.6B and 0.29 for the
1.7B). Use 8-bit when the exact probabilities matter. Times are medians on a
MacBook Pro (M3 Pro) GPU. For prompts this short, the smaller files save memory
but are not faster.

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

### Demo app

[demo/streamlit_app.py](demo/streamlit_app.py) is a small web app with five examples
and editable questions. It runs the 8-bit 0.6B model on CPU. The live copy is at
[onepassqwen3.streamlit.app](https://onepassqwen3.streamlit.app/), on Streamlit Community Cloud. To run it locally:

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

## How it was trained

### Data

68,061 texts with 138,928 questions, from two kinds of open data. We wrote no
training examples ourselves; [src/onepass/build.py](src/onepass/build.py) turns
existing datasets into typed questions.

| Part | Texts | What it is |
|---|---|---|
| 16 human-labelled datasets | 40,000 (sampled from 86,871) | BoolQ, UNLI, MultiNLI, WANLI, Civil Comments, PAWS, VitaminC, SMS Spam, Banking77, CLINC150, DBpedia, the Davidson hate-speech set, GoEmotions, Measuring Hate Speech, HelpSteer2 and BRIGHTER. Where several people rated a text, the target is the share of raters who chose each answer. |
| [n4ze3m/typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | 6,523 | Typed questions with probability labels from DeepSeek V4.1 Flash (MIT). |
| [decisionmodels/sn92-specialists](https://huggingface.co/datasets/decisionmodels/sn92-specialists) | 12,730 | Typed questions labelled by open-weight models and exact rules (CC BY 4.0). |
| [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | 8,808 | Typed questions whose answers follow from exact rules (Apache-2.0). |

Before using the three typed-question datasets we checked each one's license and
who made its labels. We also compared every text with the benchmark's 1,600
train and test cases: exact matches, any shared 10-word run, and identical
questions. We found no overlap. We also drop records that look like the
benchmark's own four workflows.

```bash
uv run python -m onepass.build        # human-labelled datasets -> data/mix
uv run python -m onepass.build open   # typed-question datasets -> data/mix-open
uv run python -m onepass.build v2     # the training mix above -> data/mix-v2
```

### Training

- **LoRA on every linear layer.** The Qwen weights stay frozen. Small trainable
  matrices (rank 32) are added to the attention layers (`q_proj`, `k_proj`,
  `v_proj`, `o_proj`) and the feed-forward layers (`gate_proj`, `up_proj`,
  `down_proj`) of all 28 blocks: 20M trainable weights for the 0.6B, 35M for the
  1.7B.
- **Loss only at the answer slots.** The model is scored only on the answer letter after each
  `Answer:`, against the target probabilities. The rest of the prompt is not trained on.
- **One pass over the data,** learning rate 2e-4 with a cosine schedule, on two
  free Kaggle T4 GPUs: 4.5 hours for the 0.6B, 8.8 hours for the 1.7B, side by
  side. Notebook: [kaggle/train-v2/train.ipynb](kaggle/train-v2/train.ipynb).
- **Calibration.** One temperature is fitted on our own validation set, never on
  the benchmark: 1.0 for the 0.6B, 1.05 for the 1.7B.

## Credit

Built by Abdul Samad ([samadpls.com](https://samadpls.com), abdulsamadsid1@gmail.com).
Licensed under Apache-2.0.
