import json, os
import pandas as pd
import streamlit as st
import core

st.set_page_config(page_title="College Copilot", page_icon="🎓", layout="wide")
PROFILE_FILE = os.path.join(os.path.dirname(__file__), "profile.json")
DEFAULT = {"name": "Friend", "branch": "CSE", "sem": 5, "req": 75, "interests": ["internship", "data analytics"],
           "attendance": [{"subject": "DBMS", "attended": 18, "total": 25},
                          {"subject": "OS", "attended": 20, "total": 22},
                          {"subject": "CN", "attended": 14, "total": 20}]}

def load_profile():
    try:
        if os.path.exists(PROFILE_FILE):
            return json.load(open(PROFILE_FILE, encoding="utf-8"))
    except Exception:
        pass
    return json.loads(json.dumps(DEFAULT))

if "profile" not in st.session_state:
    st.session_state["profile"] = load_profile()
    st.session_state["items"] = []
    st.session_state["brief"] = ""
P = st.session_state["profile"]

# Groq key: Streamlit Cloud -> Settings -> Secrets  (GROQ_API_KEY = "...")
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@st.cache_data(ttl=600, show_spinner=False)
def get_models(key):
    return core.list_groq_models(key)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎓 College Copilot")
    st.caption("Powered by Groq Cloud API ⚡  ·  use sample data only on this public link")

    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY missing! Add it in Streamlit Cloud -> Settings -> Secrets.")

    groq_models = (get_models(GROQ_API_KEY) if GROQ_API_KEY else []) or ["openai/gpt-oss-120b"]
    model = st.selectbox("Model", groq_models, index=groq_models.index(core.pick_default(groq_models)))
    vision = st.selectbox("Vision model (screenshots, optional)", [""] + groq_models,
                          help="Leave blank if no vision model is available. Paste the notice text instead.")

    st.subheader("Student profile")
    P["name"] = st.text_input("Name", P["name"])
    c1, c2 = st.columns(2)
    P["branch"] = c1.text_input("Branch", P["branch"])
    P["sem"] = c2.number_input("Sem", 1, 8, int(P["sem"]))
    P["req"] = st.slider("Required attendance %", 50, 90, int(P["req"]))
    P["interests"] = [x.strip() for x in st.text_input("Interests (comma separated)", ", ".join(P["interests"])).split(",") if x.strip()]

    st.subheader("Feed it your college chaos")
    chats = st.file_uploader("WhatsApp chat exports (.txt)", type="txt", accept_multiple_files=True)
    pdfs = st.file_uploader("Notice / assignment PDFs", type="pdf", accept_multiple_files=True)
    imgs = st.file_uploader("Notice screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    pasted = st.text_area("Or paste any notice text")
    days = st.slider("Look back (days)", 7, 120, 45)
    fast = st.checkbox("Fast mode (only scan messages with keywords)", True)
    go = st.button("🔍 Analyze", type="primary", width="stretch")

# attendance table lives in the Attendance tab, but subjects are needed for extraction
att_df = pd.DataFrame(P["attendance"])
P["subjects"] = list(att_df["subject"])

if go:
    if not GROQ_API_KEY:
        st.error("Please add your GROQ_API_KEY before running analysis.")
    else:
        try:
            blocks = []
            for f in chats:
                msgs = core.relevant_messages(core.parse_chat(f.getvalue().decode("utf-8", "ignore")), days, fast)
                blocks += core.chunk_messages(msgs)
            for f in pdfs:
                t = core.pdf_to_text(f)
                blocks += [t[i:i + 3500] for i in range(0, len(t), 3500)]
            if imgs and not vision:
                st.warning("Screenshots skipped: pick a vision model in the sidebar, or paste the notice text instead.")
            for f in (imgs if vision else []):
                blocks.append(core.image_to_text(f.getvalue(), vision, api_key=GROQ_API_KEY))
            if pasted.strip():
                blocks.append(pasted)
            blocks = [b for b in blocks if b and b.strip()]
            if not blocks:
                st.warning("Upload at least one chat, PDF, screenshot or pasted text.")
            else:
                bar = st.progress(0, text="Groq AI is analyzing your files…")
                raw = core.extract_items(blocks, P, model, progress=lambda x: bar.progress(x), api_key=GROQ_API_KEY)
                bar.empty()
                st.session_state["items"] = core.rank(raw, P)
                alerts_now = core.attendance_actions(P["attendance"], P["req"])
                st.session_state["brief"] = (core.morning_brief(st.session_state["items"], alerts_now, P, model,
                                                                api_key=GROQ_API_KEY) if raw or alerts_now else "")
                if not raw:
                    st.info("The AI read your files but found nothing that needs action.")
                try:
                    json.dump(P, open(PROFILE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Analysis failed: {e}")

items = st.session_state["items"]
alerts = core.attendance_actions(P["attendance"], P["req"])
ICON = {"deadline": "⏰", "exam": "📝", "notice": "📢", "internship": "💼", "event": "🎤", "fee": "💳"}

def when(i):
    d = i.get("days")
    return "date unclear" if d is None else "TODAY" if d == 0 else "tomorrow" if d == 1 else f"in {d} days ({i['date']})"

def card(i):
    link = f" · [open link]({i['link']})" if i.get("link") else ""
    st.markdown(f"**{ICON.get(i['type'], '📢')} {i['title']}** — _{when(i)}_  \n➜ {i.get('action', '')}{link}")

tab_today, tab_att, tab_dl, tab_int, tab_ask, tab_exp = st.tabs(
    ["📌 Today", "📊 Attendance", "⏰ Deadlines & Exams", "💼 Internships", "💬 Ask", "📤 Export"])

with tab_today:
    st.header(f"Hi {P['name']}, here's what matters now")
    if not items and not alerts:
        st.info("Upload your chats / notices in the sidebar and press Analyze.")
    for a in alerts:
        (st.error if a.startswith("⚠️") else st.warning)(a)
    top = items[:3]
    if top:
        st.subheader("🔥 Do these first")
        for i in top: card(i)
    rest = [i for i in items[3:] if i.get("days") is not None and i["days"] <= 7]
    if rest:
        st.subheader("This week")
        for i in rest: card(i)
    if st.session_state["brief"]:
        st.subheader("📱 Ready-to-send morning message")
        st.code(st.session_state["brief"], language=None)

with tab_att:
    st.header("Attendance planner")
    st.caption("Edit the table with your latest numbers. Math is done in code, not by the AI.")
    edited = st.data_editor(att_df, num_rows="dynamic", width="stretch")
    P["attendance"] = edited.dropna().to_dict("records")
    rows = []
    for r in P["attendance"]:
        s = core.attendance_status(r["attended"], r["total"], P["req"])
        rows.append({"Subject": r["subject"], "Attendance %": round(s["pct"], 1),
                     "Status": {"danger": "🔴 Low", "edge": "🟡 Edge", "safe": "🟢 Safe"}[s["state"]],
                     "Must attend next": s["need"], "Safe bunks left": s["skip"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.bar_chart(pd.DataFrame(rows).set_index("Subject")["Attendance %"])
        st.subheader("🎯 Bunk simulator")
        sub = st.selectbox("Subject", [r["subject"] for r in P["attendance"]])
        n = st.slider("If I skip this many more classes…", 0, 10, 2)
        r = next(x for x in P["attendance"] if x["subject"] == sub)
        new = core.after_skipping(r["attended"], r["total"], n)
        (st.success if new >= P["req"] else st.error)(f"{sub} would become {new:.1f}% (required {P['req']}%)")

with tab_dl:
    st.header("Deadlines & exams (sorted by priority)")
    sel = [i for i in items if i["type"] in ("deadline", "exam", "fee", "event", "notice")]
    for i in sel: card(i)
    if not sel: st.info("Nothing yet.")

with tab_int:
    st.header("Internships & opportunities")
    sel = [i for i in items if i["type"] == "internship"]
    for i in sel: card(i)
    if not sel: st.info("No internships found in your chats yet.")

with tab_ask:
    st.header("Ask your copilot")
    q = st.text_input("e.g. 'Is hafte kya submit karna hai?' / 'Kaunsa internship apply karun?'")
    if q:
        if not GROQ_API_KEY:
            st.error("Please add your GROQ_API_KEY to secrets to use chat.")
        else:
            try:
                with st.spinner("Thinking on Groq Cloud…"):
                    st.write(core.ask(q, items, alerts, P, model, api_key=GROQ_API_KEY))
            except Exception as e:
                st.error(f"Could not get an answer: {e}")

with tab_exp:
    st.header("Export")
    st.download_button("📅 Download calendar (.ics) - import to Google Calendar", core.to_ics(items),
                       "college_copilot.ics", "text/calendar", disabled=not items)
    st.download_button("🧾 Download items (JSON)", json.dumps(items, ensure_ascii=False, indent=1),
                       "items.json", disabled=not items)
