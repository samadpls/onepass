"""onepass demo web app (run locally with streamlit): typed questions about a text, answered as
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
    "Agent tool choice": ({"user": "Book me something from Karachi to Dubai on 12 December, cheapest please."}, {
        "tool": {"type": "choice", "instructions": "Which tool should the assistant call next?",
                 "criteria": {"get_weather": "Current weather or forecast for a city.",
                              "search_flights": "Find flights between two cities on a date.",
                              "send_email": "Send an email to someone.", "none": "No tool is needed; just reply."}},
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
    stats = (f'<div class="op-stats"><span><b>1</b>pass</span><span><b>0</b>generated tokens</span>'
             f'<span><b>{n}</b>answer{"s" if n != 1 else ""}</span><span><b>{ms:.0f}</b>ms on CPU</span></div>')
    return stats + "".join(cards)


FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800"
         "&family=Geist:wght@400;500;600&family=Geist+Mono:wght@500&display=swap")
CSS = f"""<style>
@import url('{FONTS}');
:root {{ --blue: #1d2cf3; --yellow: #ffc61a; --navy: #0f1452; --lav: #eef0fd; --bar: #e2e5fb; --muted: #5b6190; }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input {{ font-family: "Geist", sans-serif; }}
.stApp textarea {{ font-family: "Geist Mono", monospace; font-size: 13px; }}
.block-container {{ padding-top: 3.6rem; max-width: 1280px; }}
.op-hero {{ background: var(--blue); color: #fff; border-radius: 28px; padding: 34px 40px; display: grid;
           grid-template-columns: 1fr 210px; align-items: center; gap: 24px; margin-bottom: 18px; }}
.op-kicker {{ color: var(--yellow); font-weight: 600; letter-spacing: .16em; text-transform: uppercase; font-size: 14px; }}
.op-word {{ font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 76px; line-height: .95;
           letter-spacing: -.035em; margin-top: 10px; }}
.op-line {{ color: var(--yellow); font-size: 24px; font-weight: 500; margin-top: 12px; }}
.op-links {{ margin-top: 18px; font-size: 15px; }}
.op-links a {{ color: #fff !important; text-decoration: none; border-bottom: 2px solid rgba(255, 255, 255, .35); margin-right: 18px; }}
.op-links a:hover {{ border-color: var(--yellow); }}
.op-hero svg {{ width: 100%; height: auto; }}
@media (max-width: 760px) {{ .op-hero {{ grid-template-columns: 1fr; padding: 26px; }} .op-hero svg {{ display: none; }}
                            .op-word {{ font-size: 56px; }} }}
.op-intro {{ color: var(--navy); font-size: 17px; margin: 6px 2px 10px; }}
.op-stats {{ display: flex; flex-wrap: wrap; gap: 2px 20px; align-items: baseline; color: var(--muted); font-size: 15px;
            margin: 4px 0 16px; }}
.op-stats b {{ color: var(--blue); font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 30px;
              margin-right: 6px; }}
.op-q {{ background: #fff; border-radius: 22px; padding: 20px 24px; margin-bottom: 14px; }}
.op-kind {{ font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
.op-ins {{ font-family: "Bricolage Grotesque", sans-serif; font-weight: 600; font-size: 21px; color: var(--navy);
          margin: 4px 0 10px; }}
.op-row {{ display: grid; grid-template-columns: minmax(90px, 38%) 1fr 52px; gap: 14px; align-items: center;
          margin-top: 8px; color: var(--navy); }}
.op-track {{ height: 12px; border-radius: 6px; background: var(--bar); overflow: hidden; }}
.op-fill {{ height: 100%; border-radius: 6px; background: #9aa3f7; }}
.op-top {{ font-weight: 600; }} .op-top .op-fill {{ background: var(--blue); }}
.op-pct {{ text-align: right; font-family: "Geist Mono", monospace; font-size: 14px; }}
.op-foot {{ color: var(--muted); font-size: 14px; margin-top: 28px; }}
.op-foot a {{ color: var(--blue) !important; }}
</style>"""
HERO = """<div class="op-hero"><div>
  <div class="op-kicker">Open decision model</div>
  <div class="op-word">onepass</div>
  <div class="op-line">One pass. Every answer.</div>
  <div class="op-links"><a href="https://github.com/samadpls/onepass" target="_blank">GitHub</a><a
    href="https://huggingface.co/samadpls/onepass-qwen3-0.6b-GGUF" target="_blank">Model</a><a
    href="https://huggingface.co/collections/samadpls/onepass-6b06dee3eabd91e4de4df427" target="_blank">Collection</a></div>
</div>
<svg viewBox="0 0 560 520" aria-hidden="true">
  <rect x="10" y="130" width="230" height="260" rx="28" fill="#fff"/>
  <g fill="#d9dcf5"><rect x="42" y="172" width="166" height="16" rx="8"/><rect x="42" y="208" width="138" height="16" rx="8"/>
    <rect x="42" y="244" width="160" height="16" rx="8"/><rect x="42" y="280" width="110" height="16" rx="8"/>
    <rect x="42" y="316" width="150" height="16" rx="8"/><rect x="42" y="352" width="90" height="16" rx="8"/></g>
  <circle cx="400" cy="260" r="82" fill="none" stroke="#fff" stroke-width="22"/>
  <path d="M262 260 H508" stroke="#ffc61a" stroke-width="22" stroke-linecap="round"/>
  <path d="M482 228 L520 260 L482 292" fill="none" stroke="#ffc61a" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>
</svg></div>"""


def use_example():
    state, questions = EXAMPLES[st.session_state.example]
    st.session_state.state, st.session_state.questions = as_text(state), as_text(questions)


st.set_page_config(page_title="onepass", page_icon="https://raw.githubusercontent.com/samadpls/onepass/main/assets/logo.png",
                   layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)
st.markdown('<div class="op-intro">Ask typed questions about a text: <b>yes/no</b>, <b>choose one</b> or '
            "<b>pick a level</b>. The 8-bit Qwen3-0.6B fine-tune returns a probability for every option, for all "
            "questions at once, from one forward pass. Nothing is generated, so the JSON it returns is always "
            "complete and valid.</div>", unsafe_allow_html=True)

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

st.markdown('<div class="op-foot">Built by Abdul Samad (<a href="https://samadpls.com" target="_blank">samadpls.com</a>)'
            ". On the benchmark it picks the wrong top answer about 2 times in 5, "
            "so read the probabilities, not only the top label.</div>", unsafe_allow_html=True)
