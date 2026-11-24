# onepass

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
