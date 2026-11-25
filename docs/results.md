# Results

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
(see [training.md](training.md)) raised it to 0.601.

## Held-out human-labelled sets

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

## Smaller files

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

## Why read answers instead of generating them

We asked base Qwen3-0.6B (8-bit GGUF, llama.cpp, thinking off) to *write* a JSON
answer for each of the 400 test cases:

| | Writes its answers | onepass (same 8-bit file) |
|---|---|---|
| Cases with a question missing | 37 of 400 (9.3%) | none, by design |
| Generated tokens per case | 49 | 0 |
| Time per case (5 questions, Apple M3 Pro GPU) | 497 ms | about 156 ms |
| Probabilities for every option | no | yes |

