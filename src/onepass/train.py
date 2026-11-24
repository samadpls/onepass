"""LoRA fine-tune: cross-entropy at the answer slots only, against the target
distribution over each question's option letters. The LM head is applied to
the slot positions and the letter rows only, so the full vocabulary is never
materialised."""
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from onepass.prompt import LETTERS, options, render


def encode_records(path, tok, max_len):
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    out, skipped = [], 0
    for line in open(path):
        rec = json.loads(line)
        ids, slots = render(rec["state"], rec["questions"], enc)
        if len(ids) > max_len:
            skipped += 1
            continue
        qs = []
        for qid, (pos, labels) in slots.items():
            qs.append((pos, [rec["targets"][qid][label] for label in labels]))
        out.append((ids, qs, rec["src"]))
    return out, skipped


def batches(items, max_tokens, shuffle, seed=0):
    """Length-bucketed batches capped by padded token count."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    groups, cur, cur_max = [], [], 0
    for i in order:
        n = len(items[i][0])
        if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        groups.append(cur)
    if shuffle:
        random.Random(seed).shuffle(groups)
    return groups


def collate(items, idx, pad_id, device):
    rows = [items[i] for i in idx]
    width = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), width), dtype=torch.long)
    pos, tgt = [], []
    for b, (seq, qs, _) in enumerate(rows):
        ids[b, : len(seq)] = torch.tensor(seq)
        att[b, : len(seq)] = 1
        for p, t in qs:
            pos.append((b, p))
            tgt.append(t)
    return ids.to(device), att.to(device), pos, tgt


def slot_logits(model, letter_ids, ids, att, pos):
    """Letter logits [n_slots, 26] at the given (batch, position) pairs."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = core.model(input_ids=ids, attention_mask=att).last_hidden_state
    b = torch.tensor([p[0] for p in pos], device=ids.device)
    t = torch.tensor([p[1] for p in pos], device=ids.device)
    h = hidden[b, t]
    w = core.lm_head.weight[letter_ids]
    return (h.float() @ w.float().T)


def slot_loss(logits, targets):
    """Mean cross-entropy against each slot's target over its first n letters."""
    losses, hits = [], []
    for z, t in zip(logits, targets):
        n = len(t)
        logp = F.log_softmax(z[:n], dim=-1)
        tt = torch.tensor(t, device=z.device, dtype=logp.dtype)
        losses.append(-(tt * logp).sum())
        hits.append(int(z[:n].argmax().item() == max(range(n), key=t.__getitem__)))
    return torch.stack(losses).mean(), hits


@torch.no_grad()
def evaluate(model, letter_ids, items, pad_id, device, max_tokens, autocast):
    model.eval()
    total, n, hits = 0.0, 0, []
    for idx in batches(items, max_tokens, shuffle=False):
        ids, att, pos, tgt = collate(items, idx, pad_id, device)
        with autocast():
            z = slot_logits(model, letter_ids, ids, att, pos)
        loss, h = slot_loss(z, tgt)
        total += loss.item() * len(tgt)
        n += len(tgt)
        hits += h
    model.train()
    return total / n, sum(hits) / len(hits)


def train(base, data_dir, out_dir, *, rank=32, alpha=64, lr=2e-4, epochs=1, max_tokens=8192, max_len=1536,
          warmup=100, eval_every=500, max_steps=None, max_minutes=None, dtype="fp16", checkpointing=True, device=None,
          seed=0, log=print):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    letter_ids = torch.tensor([tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS], device=device)
    train_items, skipped = encode_records(Path(data_dir) / "train.jsonl", tok, max_len)
    val_items, _ = encode_records(Path(data_dir) / "val.jsonl", tok, max_len)
    log(f"train {len(train_items)} (skipped {skipped} over {max_len} tokens), val {len(val_items)}")

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device)
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.train()  # from_pretrained returns eval mode, which silently disables gradient checkpointing

    use_amp = dtype == "fp16" and device == "cuda"
    if use_amp:
        autocast = lambda: torch.autocast("cuda", dtype=torch.float16)
    elif dtype == "bf16" and device in ("cuda", "mps"):
        autocast = lambda: torch.autocast(device, dtype=torch.bfloat16)
    else:
        autocast = lambda: torch.autocast(device, enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    plan = [b for e in range(epochs) for b in batches(train_items, max_tokens, shuffle=True, seed=seed + e)]
    if max_steps:
        plan = plan[:max_steps]
    total = len(plan)
    t0 = time.time()

    def progress(s):
        """Share of the run done: by steps, or by the time budget if that runs out first."""
        done = s / total
        if max_minutes:
            done = max(done, (time.time() - t0) / (60 * max_minutes))
        return min(done, 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * progress(s))))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for step, idx in enumerate(plan, 1):
        if progress(step - 1) >= 1.0:
            total = step - 1
            log(f"time budget reached after {total} steps")
            break
        ids, att, pos, tgt = collate(train_items, idx, pad_id, device)
        with autocast():
            z = slot_logits(model, letter_ids, ids, att, pos)
        loss, _ = slot_loss(z, tgt)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 50 == 0:
            log(f"step {step}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        if step % eval_every == 0 or step == total:
            vl, va = evaluate(model, letter_ids, val_items, pad_id, device, max_tokens, autocast)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": vl, "val_acc": va, "secs": time.time() - t0})
            log(f"eval step {step}: val loss {vl:.4f} acc {va:.4f}")
            if vl < best:
                best = vl
                model.save_pretrained(out / "adapter")
                tok.save_pretrained(out / "adapter")
    if not history or history[-1]["step"] != total:
        vl, va = evaluate(model, letter_ids, val_items, pad_id, device, max_tokens, autocast)
        history.append({"step": total, "train_loss": loss.item(), "val_loss": vl, "val_acc": va, "secs": time.time() - t0})
        log(f"final eval step {total}: val loss {vl:.4f} acc {va:.4f}")
        if vl < best:
            model.save_pretrained(out / "adapter")
            tok.save_pretrained(out / "adapter")
    json.dump(history, open(out / "history.json", "w"), indent=1)
    return history
