"""Score a model on LocalLLaMA/typed-decisions (one forward pass per case) and
fit a single temperature on our own validation set (never on the benchmark)."""
import json

import torch
import torch.nn.functional as F

from onepass.prompt import render
from onepass.score import score_one, summarize
from onepass.train import batches, collate, slot_logits


def load_cases(parquet_path):
    import pandas as pd
    return [(json.loads(r.state), json.loads(r.questions), json.loads(r.gold), r.workflow)
            for r in pd.read_parquet(parquet_path).itertuples()]


@torch.no_grad()
def predict(model, tok, letter_ids, requests, device, temperature=1.0, max_tokens=16384):
    """requests: [(state, questions)] -> [{question id: {label: probability}}]."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    items, labels = [], []
    for state, questions in requests:
        ids, slots = render(state, questions, enc)
        items.append((ids, [(pos, [0.0] * len(lab)) for pos, lab in slots.values()], ""))
        labels.append({qid: lab for qid, (_, lab) in slots.items()})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [None] * len(items)
    model.eval()
    for idx in batches(items, max_tokens, shuffle=False):
        ids, att, pos, _ = collate(items, idx, pad_id, device)
        z = slot_logits(model, letter_ids, ids, att, pos) / temperature
        k = 0
        for i in idx:
            answers = {}
            for qid, lab in labels[i].items():
                p = F.softmax(z[k, : len(lab)], dim=-1).tolist()
                answers[qid] = dict(zip(lab, p))
                k += 1
            out[i] = answers
    return out


def typed_decisions(model, tok, letter_ids, parquet_path, device, temperature=1.0):
    cases = load_cases(parquet_path)
    preds = predict(model, tok, letter_ids, [(s, q) for s, q, _, _ in cases], device, temperature)
    rows, by_type, by_workflow = [], {}, {}
    for (_, _, gold, workflow), answers in zip(cases, preds):
        for qid, g in gold.items():
            s = score_one(answers[qid], g)
            rows.append(s)
            by_type.setdefault(g["type"], []).append(s)
            by_workflow.setdefault(workflow, []).append(s)
    result = summarize(rows)
    result["accuracy_by_type"] = {t: summarize(v)["accuracy"] for t, v in by_type.items()}
    result["accuracy_by_workflow"] = {w: summarize(v)["accuracy"] for w, v in by_workflow.items()}
    return result, preds


@torch.no_grad()
def fit_temperature(model, tok, letter_ids, val_items, device, max_tokens=16384):
    """One temperature for all slots, by minimising validation cross-entropy over a grid."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    logits, targets = [], []
    model.eval()
    for idx in batches(val_items, max_tokens, shuffle=False):
        ids, att, pos, tgt = collate(val_items, idx, pad_id, device)
        z = slot_logits(model, letter_ids, ids, att, pos)
        logits += [row[: len(t)].cpu() for row, t in zip(z, tgt)]
        targets += tgt

    def nll(T):
        return sum(-(torch.tensor(t) * F.log_softmax(z / T, -1)).sum().item() for z, t in zip(logits, targets)) / len(targets)

    grid = [0.5 + 0.05 * i for i in range(51)]
    best = min(grid, key=nll)
    return best, nll(1.0), nll(best)


def heldout(model, tok, letters, device, temperature=1.0):
    """Held-out human sets; a failure here must not lose the typed-decisions results."""
    try:
        from onepass.heldout import evaluate_all
        return evaluate_all(model, tok, letters, device, temperature)
    except Exception as e:  # noqa: BLE001
        print("held-out evaluation failed:", repr(e), flush=True)
        return {"error": repr(e)}


def run(base, data_dir, out_dir, test_path, device="cuda", **train_kwargs):
    """Train, then score base, tuned and temperature-calibrated on typed-decisions."""
    from pathlib import Path

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from onepass.prompt import LETTERS
    from onepass.train import encode_records, train

    history = train(base, data_dir, out_dir, device=device, **train_kwargs)
    out = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(base)
    letters = torch.tensor([tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS], device=device)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device)
    results = {"base_model": base, "history": history}
    results["base"], _ = typed_decisions(model, tok, letters, test_path, device)
    print("base", {k: v for k, v in results["base"].items() if not isinstance(v, dict)}, flush=True)
    results["heldout_base"] = heldout(model, tok, letters, device)
    model = PeftModel.from_pretrained(model, out / "adapter")
    val_items, _ = encode_records(Path(data_dir) / "val.jsonl", tok, 1536)
    T, nll1, nllT = fit_temperature(model, tok, letters, val_items, device)
    results["temperature"] = {"value": T, "val_nll_at_1": nll1, "val_nll_at_T": nllT}
    results["tuned"], _ = typed_decisions(model, tok, letters, test_path, device)
    results["tuned_calibrated"], preds = typed_decisions(model, tok, letters, test_path, device, temperature=T)
    for k in ("tuned", "tuned_calibrated"):
        print(k, {a: b for a, b in results[k].items() if not isinstance(b, dict)}, flush=True)
    json.dump(results, open(out / "results.json", "w"), indent=1)
    results["heldout_tuned_calibrated"] = heldout(model, tok, letters, device, temperature=T)
    json.dump(results, open(out / "results.json", "w"), indent=1)
    cases = load_cases(test_path)
    import pandas as pd
    ids = pd.read_parquet(test_path)["id"].tolist()
    with open(out / "predictions.jsonl", "w") as f:
        for case_id, (_, _, gold, workflow), answers in zip(ids, cases, preds):
            for qid in gold:
                f.write(json.dumps({"id": case_id, "workflow": workflow, "question": qid,
                                    "type": gold[qid]["type"], "probabilities": answers[qid]}) + "\n")
    return results
