# How it works and how it was trained

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
| Prompt layout | [src/onepass/prompt.py](../src/onepass/prompt.py) |
| Training data | [src/onepass/build.py](../src/onepass/build.py) |
| Training (LoRA, loss at the answer slots only) | [src/onepass/train.py](../src/onepass/train.py) |
| Scoring (typed-decisions formulas) and held-out sets | [src/onepass/score.py](../src/onepass/score.py), [src/onepass/heldout.py](../src/onepass/heldout.py) |
| GGUF export and quantization | [src/onepass/export.py](../src/onepass/export.py) |
| Local runtime, server and demo page | [src/onepass/model.py](../src/onepass/model.py), [src/onepass/serve.py](../src/onepass/serve.py) |

## Data

68,061 texts with 138,928 questions, from two kinds of open data. We wrote no
training examples ourselves; [src/onepass/build.py](../src/onepass/build.py) turns
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

## How the data was cleaned

[src/onepass/build.py](../src/onepass/build.py) turns every dataset into the same record: a state,
typed questions, and a target probability for every option. On the way it:

- **Tidies the text.** Runs of spaces and line breaks become one space, and texts longer than
  2,000 characters are cut at a word boundary.
- **Turns votes into probabilities.** Two of three raters choosing "offensive" becomes 0.67. A
  dataset with a single label gives 1.0 to that answer and 0 to the rest.
- **Shuffles the options.** In generated data the right answer came first 51% of the time, so a
  model could learn "pick A". Score levels keep their order.
- **Limits long option lists.** Banking77 has 77 labels, so a question shows the right one plus
  3 to 11 random others.
- **Mixes two shapes of state.** Half the texts are plain strings and half are small JSON objects,
  because the benchmark uses JSON.
- **Drops risky rows:** non-English rows, rows nobody voted on, and anything that looks like the
  benchmark's own four workflows (invoices, customer support, security incidents, agent traces).
- **Checks every record.** Each target must cover exactly its question's options and sum to 1,
  or the build stops.

It does not fix spelling, decode HTML codes such as `&#128514;`, or remove duplicates across datasets.

## Training

- **LoRA on every linear layer.** The Qwen weights stay frozen. Small trainable
  matrices (rank 32) are added to the attention layers (`q_proj`, `k_proj`,
  `v_proj`, `o_proj`) and the feed-forward layers (`gate_proj`, `up_proj`,
  `down_proj`) of all 28 blocks: 20M trainable weights for the 0.6B, 35M for the
  1.7B.
- **Loss only at the answer slots.** The model is scored only on the answer letter after each
  `Answer:`, against the target probabilities. The rest of the prompt is not trained on.
- **One pass over the data,** learning rate 2e-4 with a cosine schedule, on two
  free Kaggle T4 GPUs: 4.5 hours for the 0.6B, 8.8 hours for the 1.7B, side by
  side. Notebook: [kaggle/train-v2/train.ipynb](../kaggle/train-v2/train.ipynb).
- **Calibration.** One temperature is fitted on our own validation set, never on
  the benchmark: 1.0 for the 0.6B, 1.05 for the 1.7B.

