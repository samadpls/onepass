"""Answer typed questions with a GGUF model through llama.cpp, in one decode call.

One llama.cpp batch holds the whole prompt, and logits are requested only at the
answer slots, so the full vocabulary is never kept for every token."""
import numpy as np

from onepass.prompt import LETTERS, render


class Model:
    def __init__(self, path, *, n_ctx=4096, threads=None, gpu_layers=-1, temperature=1.0):
        from llama_cpp import Llama

        self.llm = Llama(path, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=512, n_threads=threads,
                         n_gpu_layers=gpu_layers, logits_all=False, verbose=False)
        self.n_ctx, self.temperature = n_ctx, temperature
        self.n_vocab = self.llm.n_vocab()
        self.letters = [self._encode(" " + L)[0] for L in LETTERS]

    def _encode(self, text):
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def ask(self, state, questions):
        """{question id: {"type", "label", "probabilities", ...}} for one state.

        Yes/no answers also carry "probability_true"; score answers carry
        "expected", the probability-weighted level.
        """
        ids, slots = render(state, questions, self._encode)
        if len(ids) > self.n_ctx:
            raise ValueError(f"the request is {len(ids)} tokens; this model was loaded with n_ctx={self.n_ctx}")
        ctx, batch = self.llm._ctx, self.llm._batch
        ctx.kv_cache_seq_rm(-1, 0, -1)
        batch.set_batch(ids, n_past=0, logits_all=False)
        for pos, _ in slots.values():
            batch.batch.logits[pos] = True
        ctx.decode(batch)
        self.llm.n_tokens = 0
        answers = {}
        for qid, (pos, labels) in slots.items():
            row = np.ctypeslib.as_array(ctx.get_logits_ith(pos), shape=(self.n_vocab,))
            z = row[self.letters[: len(labels)]].astype(np.float64) / self.temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            kind = questions[qid]["type"]
            a = {"type": kind, "label": labels[int(p.argmax())],
                 "probabilities": {label: float(v) for label, v in zip(labels, p)}}
            if kind == "noul":
                a["probability_true"] = a["probabilities"]["true"]
            elif kind == "score":
                a["expected"] = float(sum(i * v for i, v in enumerate(p)))
            answers[qid] = a
        return answers

    def close(self):
        self.llm.close()

