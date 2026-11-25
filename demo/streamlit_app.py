"""onepass demo for Streamlit Community Cloud: typed questions about a text, answered as
probabilities from one forward pass of the 8-bit 0.6B model on CPU."""
import html
import json
import threading
import time

import streamlit as st
from huggingface_hub import hf_hub_download

from onepass.model import Model
from onepass.prompt import KIND, options

REPO, FILE = "samadpls/onepass-qwen3-0.6b-GGUF", "onepass-qwen3-0.6b-Q8_0.gguf"
LINKS = ("[GitHub](https://github.com/samadpls/onepass) · "
         "[Model](https://huggingface.co/samadpls/onepass-qwen3-0.6b-GGUF) · "
         "[Collection](https://huggingface.co/collections/samadpls/onepass-6b06dee3eabd91e4de4df427)")

EXAMPLES = {
    "Support message": ("My card was charged twice for one order. Please refund the extra charge today.", {
        "refund": {"type": "noul", "instructions": "The customer wants a refund.",
                   "criteria": {"true": "Yes.", "false": "No."}},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments and refunds.", "tech_support": "Bugs and setup.", "sales": "New orders."}},
        "mood": {"type": "score", "instructions": "How upset is the customer?",
                 "criteria": ["calm", "a little upset", "upset", "very upset"]},
    }),
    "Login alert (JSON)": ({"event": "login", "user": "j.khan", "country": "Romania", "usual_country": "Pakistan",
                            "local_time": "03:12", "mfa": "failed 3 times, then passed", "new_device": True}, {
        "suspicious": {"type": "noul", "instructions": "This login looks like an account takeover.",
                       "criteria": {"true": "Likely takeover.", "false": "Likely the real user."}},
        "severity": {"type": "score", "instructions": "How severe is this alert?",
                     "criteria": ["low: routine", "medium: worth a look", "high: likely compromise",
                                  "critical: confirmed breach"]},
        "action": {"type": "choice", "instructions": "What should happen next?",
                   "criteria": {"lock_account": "Lock the account and reset the password.",
                                "ask_user": "Ask the user to confirm the login.", "ignore": "No action needed."}},
    }),
    "App review": ("The new update is great, the app finally opens fast. But now my saved playlists are gone and "
                   "support has not replied in four days.", {
        "bug": {"type": "noul", "instructions": "Something in the app is broken.",
                "criteria": {"true": "Yes, something is broken.", "false": "No, everything works."}},
        "sentiment": {"type": "choice", "instructions": "What is the overall sentiment?",
                      "criteria": {"positive": "Mostly happy.", "mixed": "Both happy and unhappy.",
                                   "negative": "Mostly unhappy."}},
        "urgency": {"type": "score", "instructions": "How urgently should the team reply?",
                    "criteria": ["not urgent", "soon", "today", "right now"]},
    }),
    "Order record (JSON)": ({"order_id": "A-1042", "status": "delivered", "days_since_delivery": 41,
                             "return_window_days": 30, "item": "headphones",
                             "request": "return, the item stopped working"}, {
        "eligible": {"type": "noul", "instructions": "The order can still be returned under the return window.",
                     "criteria": {"true": "Within the window.", "false": "Outside the window."}},
    }),
}
as_text = lambda x: x if isinstance(x, str) else json.dumps(x, indent=1, ensure_ascii=False)


@st.cache_resource(show_spinner="Loading the 8-bit 0.6B model (639 MB), only on the first visit ...")
def load():
    # one llama.cpp context is shared by every session, so requests take turns
    return Model(hf_hub_download(REPO, FILE), n_ctx=2048, gpu_layers=0), threading.Lock()


def parse(state_text, questions_text):
    """Validate user input before it reaches the model; problems become ValueErrors with a plain message."""
    state = state_text.strip()
    if not state:
        raise ValueError("Write a state: the text the questions are about.")
    if state[:1] in "{[":
        try:
            state = json.loads(state)
        except ValueError:
            pass  # not JSON after all: treat it as plain text
    try:
        questions = json.loads(questions_text)
    except ValueError as e:
        raise ValueError(f"Questions are not valid JSON: {e}")
    if not isinstance(questions, dict) or not questions:
        raise ValueError('Questions must be a JSON object like {"refund": {...}}.')
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in KIND:
            raise ValueError(f"Question '{qid}': type must be one of noul, choice, score.")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError(f"Question '{qid}': add the instructions (the question itself).")
        crit = q.get("criteria")
        if q["type"] == "choice" and not (isinstance(crit, dict) and len(crit) >= 2):
            raise ValueError(f"Question '{qid}': a choice needs criteria with at least 2 options, as an object.")
        if q["type"] == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            raise ValueError(f"Question '{qid}': a score needs criteria as a list of at least 2 levels, lowest first.")
        if q["type"] == "noul" and crit is not None and not isinstance(crit, dict):
            raise ValueError(f"Question '{qid}': yes/no criteria must be an object with true and false.")
        if len(options(q)) > 26:
            raise ValueError(f"Question '{qid}': use at most 26 options.")
    return state, questions


def render(questions, answers, ms):
    cards = []
    for qid, q in questions.items():
        probs = answers[qid]["probabilities"]
        names = {str(i): f"{i} · {d}" for i, d in enumerate(q["criteria"])} if q["type"] == "score" else {}
        top = max(probs, key=probs.get)
        rows = "".join(
            f'<div class="op-row{" op-top" if k == top else ""}"><span>{html.escape(names.get(k, k))}</span>'
            f'<div class="op-track"><div class="op-fill" style="width:{max(p * 100, 1):.1f}%"></div></div>'
            f'<span class="op-pct">{p * 100:.0f}%</span></div>'
            for k, p in sorted(probs.items(), key=lambda kv: -kv[1]))
        cards.append(f'<div class="op-q"><div class="op-kind">{html.escape(qid)} · {KIND[q["type"]]}</div>'
                     f'<div class="op-ins">{html.escape(q["instructions"])}</div>{rows}</div>')
    n = len(answers)
    stats = (f'<div class="op-stats"><b>1</b> forward pass <b>0</b> generated tokens '
             f'<b>{n}</b> answer{"s" if n != 1 else ""} <b>{ms:.0f} ms</b> on CPU</div>')
    return stats + "".join(cards)


CSS = """<style>
.op-stats { margin-bottom: 14px; opacity: .75; }
.op-stats b { margin: 0 4px 0 10px; } .op-stats b:first-child { margin-left: 0; }
.op-q { border: 1px solid rgba(127, 127, 127, .35); border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; }
.op-kind { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; opacity: .65; }
.op-ins { font-weight: 600; margin: 4px 0 8px; }
.op-row { display: grid; grid-template-columns: minmax(90px, 38%) 1fr 48px; gap: 10px; align-items: center; margin-top: 6px; }
.op-track { height: 10px; border-radius: 5px; background: rgba(127, 127, 127, .2); overflow: hidden; }
.op-fill { height: 100%; border-radius: 5px; background: #9aa3f7; }
.op-top { font-weight: 600; } .op-top .op-fill { background: #1d2cf3; }
.op-pct { text-align: right; font-variant-numeric: tabular-nums; }
</style>"""


def use_example():
    state, questions = EXAMPLES[st.session_state.example]
    st.session_state.state, st.session_state.questions = as_text(state), as_text(questions)


st.set_page_config(page_title="onepass", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("onepass")
st.markdown("Ask typed questions about a text: **yes/no**, **choose one**, or **pick a level**. The model returns a "
            "probability for every option, for all questions at once, from one forward pass. Nothing is generated, "
            f"so the answer can never be malformed.\n\n{LINKS} · 8-bit Qwen3-0.6B fine-tune, running on a free CPU")

if "state" not in st.session_state:
    st.session_state.example = next(iter(EXAMPLES))
    use_example()
st.radio("Example", list(EXAMPLES), key="example", horizontal=True, on_change=use_example)

left, right = st.columns(2, gap="large")
with left, st.form("ask", border=False):
    st.text_area("State (plain text or JSON)", key="state", height=130)
    st.text_area("Questions (JSON)", key="questions", height=400)
    st.form_submit_button("Ask", type="primary", use_container_width=True)

with right:
    try:
        state, questions = parse(st.session_state.state, st.session_state.questions)
        model, lock = load()
        with lock:
            t = time.perf_counter()
            answers = model.ask(state, questions)
            ms = (time.perf_counter() - t) * 1000
    except ValueError as e:
        st.error(str(e))
    else:
        st.markdown(render(questions, answers, ms), unsafe_allow_html=True)
        with st.expander("Raw answer (JSON)"):
            st.json(answers)
